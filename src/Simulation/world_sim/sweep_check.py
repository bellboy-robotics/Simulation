"""Runs a sweep's scenarios through the planner and flags every command with a problem.

Problem 1, the planner misses a solution that exists. Every target in a sweep scenario comes from a
configuration that is valid by construction (scenario["sweep"]["goal_joints_deg"]), so:
- "ik_miss": a pose command's IK lands farther than POSE_TOL from a target the goal configuration reaches;
- "solver_miss": a command failed (detour refused, no route, replay aborted) although the oracle
  (oracle.py) finds a valid joint path from the command's start to its goal;
- "no_path_found": a command failed and the oracle found no path either: probably no solution
  (kept separately, so they do not hide the misses);
- "end_state_invalid": the planner left the arm in self-collision, in an obstacle or below the floor.
Not judged: "goal_invalid" (the goal breaks a constraint on this arm) and "untestable_start" (the
command started from an invalid state, see end_state_invalid of the command before).
Problem 2, time on the robot:
- "slow_single_move": a joints/pose command spends more than SLOW_SINGLE_MOVE_S in the planner;
- "replay_stall": a replay batch takes longer than the playback of its frames, so the arm waits;
- "slow_replay_start": planning before a replay moves (detour + reroute) above SLOW_REPLAY_START_S;
- "recompile": JAX compiles more than RECOMPILE_S worth of programs inside one command.
Each flagged command becomes a stand-alone scenario: the same world, the arm's joints just before the
command, and that command only. It is re-run right away to check that it reproduces.
"""

import json
import logging
import os
import re
import time

import numpy as np
from billie_utils.messages.pyroki_node import BATCH_SIZE

from Simulation.world_sim.oracle import Oracle
from Simulation.world_sim.plan import _to_jsonable, run_scenario, save_results
from Simulation.world_sim.planner import SimPlanner
from Simulation.world_sim.recording import BRAIN_TICK_RATE
from Simulation.world_sim.world import scenario_from_data

# [mm, deg] How close a pose command must get to a reachable target (the planner usually gets < 1mm).
POSE_TOL = (5.0, 3.0)
# [s] Planner time of a joints/pose command the operator would notice as a pause before the arm moves.
SLOW_SINGLE_MOVE_S = 2.0
# [s] A replay batch slower than playing its BATCH_SIZE frames at the brain's tick rate drains the buffer.
REPLAY_BATCH_BUDGET_S = BATCH_SIZE / BRAIN_TICK_RATE
# [s] Planning a replay does before the arm starts moving (detour to frame 0 + reroute).
SLOW_REPLAY_START_S = 2.0
# [s] JAX compile time inside one command: tiny shape-specific kernels are normal, whole solvers are not.
RECOMPILE_S = 0.5
_COMPILE_LINE = re.compile(r"Finished XLA compilation of (.+) in ([0-9.eE+-]+) sec")


class CompileLog(logging.Handler):
    """Collects JAX's "Finished XLA compilation" log lines, tagged with the command running at the time."""

    def __init__(self, planner: SimPlanner):
        """Starts listening (enables jax_log_compiles). planner: Its .command says which command is running."""
        super().__init__(level=logging.DEBUG)
        import jax  # noqa: PLC0415

        jax.config.update("jax_log_compiles", True)
        jax_logger = logging.getLogger("jax")
        jax_logger.addHandler(self)
        jax_logger.propagate = False  # keeps the compile lines out of the console
        self.planner = planner
        self.scenario: int | None = None  # scenario whose commands run now; None = not recording
        self.events: list[dict] = []  # {"tag": (scenario, command), "function", "seconds"}

    def emit(self, record: logging.LogRecord) -> None:
        """Keeps one compile line while a scenario runs. record: A log record from a jax logger."""
        match = _COMPILE_LINE.search(record.getMessage())
        if match and self.scenario is not None:
            tag = (self.scenario, self.planner.command)
            self.events.append({"tag": tag, "function": match.group(1), "seconds": float(match.group(2))})


def _command_starts(results: dict, n_commands: int) -> list[np.ndarray]:
    """The arm's joints when each command started (the last target sent before it).

    results: run_scenario output. n_commands: Commands in the scenario.
    Returns: n_commands (6,) arrays, degrees.
    """
    starts, current = [], np.asarray(results["start_joints_deg"], dtype=np.float64)
    points = results["points"]
    k = 0
    for c in range(n_commands):
        while k < len(points) and points[k]["command"] < c:
            current = np.asarray(points[k]["joints_deg"], dtype=np.float64)
            k += 1
        starts.append(current)
    return starts


def _timing_flags(cmd: str, calls: list[dict], compile_s: float) -> tuple[list[str], dict]:
    """Problem-2 flags of one command from its planner calls.

    cmd: The command name. calls: Its planner calls (planner.timings entries). compile_s: JAX compile time in it.
    Returns: (flags, {"planner_s", "max_batch_s", "before_motion_s", "compile_s"}).
    """
    planner_s = sum(t["seconds"] for t in calls)
    batches = [t["seconds"] for t in calls if t["call"] == "batch_solve"]
    before = sum(t["seconds"] for t in calls if t["call"] in ("detour", "transit"))
    flags = []
    if cmd in ("joints", "pose") and planner_s > SLOW_SINGLE_MOVE_S:
        flags.append("slow_single_move")
    if batches and max(batches) > REPLAY_BATCH_BUDGET_S:
        flags.append("replay_stall")
    if cmd == "replay_policy" and before > SLOW_REPLAY_START_S:
        flags.append("slow_replay_start")
    if compile_s > RECOMPILE_S:
        flags.append("recompile")
    stats = {"planner_s": planner_s, "max_batch_s": max(batches) if batches else None,
             "before_motion_s": before, "compile_s": compile_s}  # fmt: skip
    return flags, stats


def _solution_flags(
    planner: SimPlanner, oracle: Oracle, spec: dict, outcome: dict, start: np.ndarray, end: np.ndarray,
    goal: np.ndarray | None, seed: int,
) -> tuple[list[str], dict]:  # fmt: skip
    """Problem-1 flags of one command: did it reach a goal that is known to be reachable?

    planner: For the TCP pose of the final joints. oracle: The scenario's oracle.
    spec: The command. outcome: Its run outcome (ok, error). start, end: (6,) joints before / after it.
    goal: (6,) valid configuration the command should be able to reach, or None if unknown.
    seed: Oracle seed.
    Returns: (flags, {"pose_error": [mm, deg] or None, "goal_broken" / "end_broken": constraints the goal /
        the arm's final joints break, "at_limits": joints ending on a limit, "oracle": plan status,
        "oracle_path_deg": path or None}).
    """
    info = {"pose_error": None, "goal_broken": [], "end_broken": oracle.broken(end), "at_limits": oracle.at_limits(end),
            "oracle": None, "oracle_path_deg": None}  # fmt: skip
    flags = ["end_state_invalid"] if info["end_broken"] else []
    if goal is not None:
        info["goal_broken"] = oracle.broken(goal)
        if info["goal_broken"]:
            return flags + ["goal_invalid"], info  # nothing to judge the command against
    if spec["cmd"] == "pose" and outcome["ok"]:
        from scipy.spatial.transform import Rotation  # noqa: PLC0415

        reached, target = planner.tcp_poses(end[None])[0], np.asarray(spec["pose"], dtype=np.float64)
        angle = (Rotation.from_rotvec(target[3:]).inv() * Rotation.from_rotvec(reached[3:])).magnitude()
        info["pose_error"] = [float(np.linalg.norm(reached[:3] - target[:3])), float(np.rad2deg(angle))]
        if goal is not None and (info["pose_error"][0] > POSE_TOL[0] or info["pose_error"][1] > POSE_TOL[1]):
            flags.append("ik_miss")
    if outcome["ok"] or goal is None:
        return flags, info
    path, status = oracle.plan(start, goal, seed=seed)
    info.update(oracle=status, oracle_path_deg=None if path is None else path.round(2).tolist())
    verdict = {"found": "solver_miss", "direct": "solver_miss", "not_found": "no_path_found", "start_invalid": "untestable_start"}
    return flags + [verdict[status]], info


def check_scenario(planner: SimPlanner, log: CompileLog, data: dict, index: int, seeds: list[int] | None = None) -> list[dict]:
    """Runs one scenario (every command, even after a failure) and flags each command.

    planner: The simulated planner. log: The compile log. data: Parsed sweep scenario. index: Its number.
    seeds: Oracle seed per command; default index * 100 + command.
    Returns: Per command {"command", "cmd", "ok", "error", "flags", "timing", "solution", "seed",
        "start_joints_deg", "end_joints_deg"} plus "results" (run_scenario output) on the first entry.
    """
    scenario = scenario_from_data(data, data.get("name", str(index)))
    goals = data.get("sweep", {}).get("goal_joints_deg", [None] * len(scenario["commands"]))
    first_timing = len(planner.timings)
    log.scenario = index
    try:
        results = run_scenario(planner, scenario["world"], scenario["start_joints_deg"], scenario["commands"],
                               stop_on_error=False)  # fmt: skip
    finally:
        log.scenario, planner.command = None, -1
    starts = _command_starts(results, len(scenario["commands"])) + [None]
    ends = _command_starts(results, len(scenario["commands"]) + 1)[1:]
    oracle = Oracle(planner, scenario["world"].capsules("avoid"), scenario["world"].floor_z_m)
    rows = []
    for c, (spec, outcome) in enumerate(zip(scenario["commands"], results["commands"])):
        calls = [t for t in planner.timings[first_timing:] if t["command"] == c]
        compile_s = sum(e["seconds"] for e in log.events if e["tag"] == (index, c))
        timing_flags, timing = _timing_flags(spec["cmd"], calls, compile_s)
        goal = None if goals[c] is None else np.asarray(goals[c], dtype=np.float64)
        seed = seeds[c] if seeds else index * 100 + c
        solution_flags, solution = _solution_flags(planner, oracle, spec, outcome, starts[c], ends[c], goal, seed)
        rows.append({"command": c, "cmd": spec["cmd"], "ok": outcome["ok"], "error": outcome["error"],
                     "flags": solution_flags + timing_flags, "timing": timing, "solution": solution, "seed": seed,
                     "start_joints_deg": starts[c], "end_joints_deg": ends[c]})  # fmt: skip
    rows[0]["results"] = results
    return rows


def repro_scenario(data: dict, row: dict, label: str) -> dict:
    """The stand-alone scenario of one flagged command: same world, its start joints, that command only.

    data: The sweep scenario. row: The command's check_scenario entry. label: Name prefix, e.g. "sweep 3 #12".
    Returns: Scenario data (editor / plan.py format) with a "flag" block describing the problem.
    """
    goals = data.get("sweep", {}).get("goal_joints_deg", [])
    goal = goals[row["command"]] if row["command"] < len(goals) else None
    out = {k: v for k, v in data.items() if k not in ("sweep", "commands", "start_joints_deg", "name")}
    out.update(
        name=f"{label} cmd {row['command']} {row['cmd']}: {', '.join(row['flags'])}",
        start_joints_deg=np.round(row["start_joints_deg"], 3).tolist(),
        commands=[data["commands"][row["command"]]],
    )
    out["flag"] = _to_jsonable({
        "problems": row["flags"], "error": row["error"], "timing": row["timing"], "goal_joints_deg": goal,
        "pose_error_mm_deg": row["solution"]["pose_error"], "oracle": row["solution"]["oracle"],
        "goal_broken": row["solution"]["goal_broken"], "end_broken": row["solution"]["end_broken"],
        "at_limits": row["solution"]["at_limits"],
        "oracle_path_deg": row["solution"]["oracle_path_deg"], "end_joints_deg": row["end_joints_deg"],
        "source": {"scenario": data.get("name"), "command": row["command"]},
    })  # fmt: skip
    return out


def reproduces(planner: SimPlanner, log: CompileLog, repro: dict, row: dict) -> bool:
    """Runs a stand-alone scenario again and checks that the same problems show up.

    planner, log: As check_scenario. repro: From repro_scenario. row: The original command's entry.
    Returns: True if every problem-1 flag, and every time flag except recompile (compiled programs are
        cached by then), shows up again.
    """
    sweep = {"goal_joints_deg": [repro["flag"]["goal_joints_deg"]]}
    again = check_scenario(planner, log, {**repro, "sweep": sweep}, -1, seeds=[row["seed"]])[0]
    expected = {f for f in row["flags"] if f != "recompile"}
    return expected <= set(again["flags"])


def run_sweep(folder: str) -> None:
    """Checks every scenario of a sweep folder and writes what it found.

    folder: Sweep folder with scenarios/*.json (from sweep.py generate). Writes into it:
        results/<scenario>.json (plan results, for reports), flagged/<scenario>-c<k>.json (stand-alone
        repro per flagged command) and sweep.json (every command's outcome, flags and timing, the planner
        build times, and "finished": true at the end).
    """
    names = sorted(f for f in os.listdir(os.path.join(folder, "scenarios")) if f.endswith(".json"))
    for sub in ("results", "flagged"):
        os.makedirs(os.path.join(folder, sub), exist_ok=True)
    t0 = time.time()
    planner = SimPlanner()
    log = CompileLog(planner)
    summary = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "build": planner.timings[:2], "rows": [], "finished": False}
    for index, name in enumerate(names):
        with open(os.path.join(folder, "scenarios", name)) as f:
            data = json.load(f)
        rows = check_scenario(planner, log, data, index)
        results = rows[0].pop("results")
        results["name"] = data.get("name", name)
        save_results(results, os.path.join(folder, "results", name))
        for row in rows:
            row["scenario"] = name
            if row["flags"]:
                repro = repro_scenario(data, row, f"{os.path.basename(folder)} {name[:-5]}")
                row["reproduced"] = reproduces(planner, log, repro, row)
                repro["flag"]["reproduced"] = row["reproduced"]
                with open(os.path.join(folder, "flagged", f"{name[:-5]}-c{row['command']}.json"), "w") as f:
                    json.dump(repro, f, indent=1)
            summary["rows"].append(_to_jsonable({k: v for k, v in row.items() if k != "solution"} |
                                                {"oracle": row["solution"]["oracle"], "pose_error": row["solution"]["pose_error"],
                                                 "goal_broken": row["solution"]["goal_broken"], "end_broken": row["solution"]["end_broken"],
        "at_limits": row["solution"]["at_limits"],
                                                 "at_limits": row["solution"]["at_limits"]}))  # fmt: skip
        summary["elapsed_s"] = time.time() - t0
        with open(os.path.join(folder, "sweep.json"), "w") as f:
            json.dump(summary, f, indent=1)
        flagged = sum(bool(r["flags"]) for r in summary["rows"])
        print(f"[{index + 1}/{len(names)}] {name}: {flagged} flagged commands so far ({time.time() - t0:.0f}s)", flush=True)
    summary["finished"] = True
    summary["machine"] = results.get("machine")
    with open(os.path.join(folder, "sweep.json"), "w") as f:
        json.dump(summary, f, indent=1)
