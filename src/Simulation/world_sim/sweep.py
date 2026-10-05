"""Problem finder: many random scenarios through the planner, every problem flagged as a stand-alone JSON.

    python -m Simulation.world_sim.sweep generate --count 100 --seed 1      # scenarios, on this machine
    python -m Simulation.world_sim.sweep robot output/world_sim/sweeps/<id>  # run on billie-29, fetch, summarize
    python -m Simulation.world_sim.sweep run output/world_sim/sweeps/<id>    # or run here (CPU timings)

A sweep folder holds scenarios/ (generated), results/ (plan results per scenario), flagged/ (one
stand-alone scenario per flagged command, see sweep_check.py for the problems) and sweep.json. The
summary step writes index.html, a report per flagged command, and copies the flagged scenarios into
src/Simulation/world_sim/scenarios/ as sweep-<id>-... so the editor opens them.

Every target of a generated scenario is a valid configuration (randomize.py: inside the limits,
clear of the objects, the floor and Billie's body), so a solution exists for each joints and pose
command. Some scenarios also replay a recording with an obstacle placed across its path, clear of
its first and last frames.
"""

import argparse
import json
import os
import re
import time

import numpy as np
from billie_utils.world_collision_check_poc import CollisionModel, link_obstacle_distances, link_transforms

from Simulation.world_sim import robot
from Simulation.world_sim.randomize import random_motions, random_objects
from Simulation.world_sim.recording import RECORDINGS_DIR, export_recording, load_recording
from Simulation.world_sim.world import object_capsules

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SWEEPS_DIR = os.path.join(_REPO, "output", "world_sim", "sweeps")
SCENARIOS_DIR = os.path.join(os.path.dirname(__file__), "scenarios")
# Billie in the map for map-frame random objects: at the origin, with billie-29's ARM_TO_BASE_CALIBRATION.
_BILLIE = {"pose": [0, 0, 0], "arm_to_base": [-180, 0, 0]}
_FLOOR_Z_MM = -440.5
# Share of scenarios that also replay a recording through an obstacle.
_REPLAY_SHARE = 0.35
# [m] Clearance a replay obstacle keeps from the recording's first / last frames and the start joints
# (more than the reroute's 3cm end clearance, so the replay is solvable).
_REPLAY_END_CLEARANCE_M = 0.06
# Fraction of the recording, at each end, whose frames must stay clear of the replay obstacle.
_REPLAY_END_SHARE = 0.1
# The planner's joint limits (pyroki_planner.urdf.REPLACEMENT_JOINT_LIMITS), degrees.
_LIMITS_DEG = [[-359.9, 360.0], [-116.9, 116.0], [-219.0, 10.0], [-360.0, 360.0], [-97.0, 180.0], [-360.0, 360.0]]
# [m] Self-collision distance a generated target must keep, so it is clearly valid on any xArm6 of the
# fleet (the run re-checks every target on its own arm with the oracle's tolerant rules).
_GOAL_SELF_MARGIN_M = 0.005
# Tries for a valid random configuration before a command is left out.
_GOAL_TRIES = 30
# Lines of the robot's sweep log worth showing here: progress, and anything that went wrong.
_PROGRESS_LINE = re.compile(r"^\[\d+/\d+\]|Traceback|Error")


def _tcp_poses(model: CollisionModel, joints_deg: np.ndarray) -> np.ndarray:
    """TCP poses [x, y, z mm, rotation vector rad] of configurations. joints_deg: (S, 6). Returns: (S, 6)."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    T = link_transforms(model, np.deg2rad(np.atleast_2d(joints_deg)))[:, model.link_names.index("link_tcp")]
    return np.concatenate([T[:, :3, 3] * 1000.0, Rotation.from_matrix(T[:, :3, :3]).as_rotvec()], axis=1)


def _path_obstacle(model: CollisionModel, joints_deg: np.ndarray, start_deg: np.ndarray, rng: np.random.Generator) -> dict | None:
    """An arm-frame obstacle across the middle of a recorded path, clear of its ends and the start joints.

    model: The arm model. joints_deg: (N, 6) recorded joints. start_deg: (6,) scenario start joints.
    rng: Random generator.
    Returns: The object description, or None if no placement was found.
    """
    n = len(joints_deg)
    ends = np.vstack([joints_deg[: max(1, int(n * _REPLAY_END_SHARE))], joints_deg[-max(1, int(n * _REPLAY_END_SHARE)) :], start_deg[None]])
    middle = joints_deg[int(n * 0.3) : int(n * 0.7) + 1]
    for _ in range(60):
        x, y, z = _tcp_poses(model, middle[rng.integers(len(middle))][None])[0, :3] + rng.uniform(-40, 40, 3)
        if rng.random() < 0.5:
            spec = {"type": "sphere", "center_mm": [round(x), round(y), round(z)], "radius_mm": int(rng.integers(40, 90))}
        else:  # a post from the floor up past the tool
            spec = {"type": "capsule", "start_mm": [round(x), round(y), round(_FLOOR_Z_MM)],
                    "end_mm": [round(x), round(y), round(z + rng.uniform(50, 250))], "radius_mm": int(rng.integers(30, 60))}  # fmt: skip
        starts, ends_m, radii = object_capsules(spec, _FLOOR_Z_MM / 1000.0)
        movable = model.movable_links
        clear_ends = link_obstacle_distances(model, np.deg2rad(ends), starts, ends_m, radii)[:, movable].min()
        hits_path = link_obstacle_distances(model, np.deg2rad(middle), starts, ends_m, radii)[:, movable].min() < 0
        if clear_ends > _REPLAY_END_CLEARANCE_M and hits_path:
            return {"name": f"path {spec['type']}", "role": "avoid", **spec}
    return None


def _ends_clear(model: CollisionModel, joints_deg: np.ndarray, spec: dict) -> bool:
    """Whether an object keeps clear of a recording's first and last frames.

    model: The arm model. joints_deg: (N, 6) recorded joints. spec: Object (arm or map frame).
    Returns: True if both end frames keep _REPLAY_END_CLEARANCE_M.
    """
    from Simulation.world_sim.world import object_in_arm_frame  # noqa: PLC0415

    capsules = object_capsules(object_in_arm_frame(spec, _BILLIE, -_FLOOR_Z_MM / 1000.0), _FLOOR_Z_MM / 1000.0)
    distance = link_obstacle_distances(model, np.deg2rad(joints_deg[[0, -1]]), *capsules)[:, model.movable_links]
    return bool(distance.min() > _REPLAY_END_CLEARANCE_M)


def _valid_joints(data: dict, planner, rng: np.random.Generator) -> np.ndarray | None:
    """A random configuration that randomize.py accepts and that also passes the oracle's checks.

    The oracle adds the planner's own self-collision model, which randomize.py only approximates.
    data: The scenario so far (its avoid objects are avoided). planner: The local SimPlanner.
    rng: Random generator.
    Returns: (6,) joints in degrees, or None if none was found in _GOAL_TRIES draws.
    """
    from billie_utils.messages.pyroki_world_poc import WORLD_COL_MARGIN_M  # noqa: PLC0415

    from Simulation.world_sim.oracle import Oracle  # noqa: PLC0415
    from Simulation.world_sim.world import scenario_from_data  # noqa: PLC0415

    world = scenario_from_data(data, "sweep")["world"]
    oracle = Oracle(planner, world.capsules("avoid"), world.floor_z_m, WORLD_COL_MARGIN_M, _GOAL_SELF_MARGIN_M)
    for _ in range(_GOAL_TRIES):
        motion = random_motions(data, planner.model, _LIMITS_DEG, rng, 1, "joints")
        if motion and oracle.valid(np.asarray(motion[0]["joints"])[None])[0]:
            return np.asarray(motion[0]["joints"])
    return None


def generate_scenario(planner, index: int, seed: int, recordings: list[str], sweep_id: str) -> dict:
    """One random sweep scenario: Billie's start joints, 1-3 objects, 2-3 joints/pose moves, sometimes a replay.

    planner: The local SimPlanner (arm model and self-collision model). index: Scenario number.
    seed: Sweep seed. recordings: repo_ids to replay from. sweep_id: Sweep name, for the scenario name.
    Returns: Scenario data (editor / plan.py format) with "sweep": {"index", "seed", "goal_joints_deg"}.
    """
    model = planner.model
    rng = np.random.default_rng([seed, index])
    data = {"name": f"{sweep_id} s{index:03d}", "billie": _BILLIE, "floor_z_mm": _FLOOR_Z_MM,
            "start_joints_deg": [0.0] * 6, "objects": [], "commands": []}  # fmt: skip
    start = _valid_joints(data, planner, rng)
    if start is None:
        raise RuntimeError(f"Scenario {index}: no valid start joints found")
    data["start_joints_deg"] = start.round(1).tolist()
    replay, recorded = None, None
    if recordings and rng.random() < _REPLAY_SHARE:
        repo_id = recordings[rng.integers(len(recordings))]
        recorded = np.array([s.recorded_joints for s in load_recording(repo_id)[0]])
        obstacle = _path_obstacle(model, recorded, np.asarray(data["start_joints_deg"]), rng)
        if obstacle is not None:
            data["objects"].append(obstacle)
            replay = {"cmd": "replay_policy", "repo_id": repo_id, "speed": 1.0, "transform": "poses"}
    added = random_objects(data, model, rng, int(rng.integers(1, 4)), None)
    data["objects"] += [o for o in added if replay is None or _ends_clear(model, recorded, o)]
    goals = []
    for _ in range(int(rng.integers(2, 4))):
        q = _valid_joints(data, planner, rng)
        if q is None:
            continue
        if rng.random() < 0.5:
            data["commands"].append({"cmd": "joints", "joints": q.round(1).tolist()})
        else:
            p = _tcp_poses(model, q[None])[0]
            data["commands"].append({"cmd": "pose", "pose": [*np.round(p[:3], 1).tolist(), *np.round(p[3:], 4).tolist()]})
        goals.append(q.tolist())
    if replay is not None:
        at = int(rng.integers(len(data["commands"]) + 1))
        data["commands"].insert(at, replay)
        goals.insert(at, recorded[-1].round(2).tolist())
    data["sweep"] = {"index": index, "seed": seed, "goal_joints_deg": goals}
    return data


def generate(count: int, seed: int, replays: bool) -> str:
    """Writes a new sweep folder with count random scenarios (builds the local planner for its models).

    count: Scenarios. seed: Sweep seed (same seed + count = same scenarios on the same arm).
    replays: Include replay scenarios (needs the recordings in the local Hugging Face cache).
    Returns: The sweep folder.
    """
    from Simulation.world_sim.planner import SimPlanner  # noqa: PLC0415 (imports JAX)

    sweep_id = f"sweep-{time.strftime('%Y%m%d-%H%M')}-seed{seed}"
    folder = os.path.join(SWEEPS_DIR, sweep_id)
    os.makedirs(os.path.join(folder, "scenarios"))
    planner = SimPlanner()
    recordings = []
    if replays:
        cache = os.path.join(os.environ["BILLIE_DIR"], ".cache", "huggingface", "lerobot", "bellboy-robotics")
        for name in sorted(os.listdir(cache)) if os.path.isdir(cache) else []:
            export_recording(f"bellboy-robotics/{name}", RECORDINGS_DIR)
            recordings.append(f"bellboy-robotics/{name}")
    for index in range(count):
        data = generate_scenario(planner, index, seed, recordings, sweep_id)
        with open(os.path.join(folder, "scenarios", f"s{index:03d}.json"), "w") as f:
            json.dump(data, f, indent=1)
    print(f"{count} scenarios in {folder} ({len(recordings)} recordings for replays)")
    return folder


def run_on_robot(host: str, folder: str, robot_checkout: str, detour: str, poll_s: float = 30.0) -> None:
    """Runs a sweep on a robot, detached (a dropped ssh does not stop it), then fetches the folder back.

    host: ssh destination. folder: Local sweep folder. robot_checkout: See robot.sync.
    detour: What routes blocked single moves, see commands.DETOUR_PLANNER.
    poll_s: Seconds between progress checks.
    Raises: RuntimeError if the sweep stops before finishing (its log tail says why).
    """
    remote = f"{robot.ROBOT_DIR}/sweeps/{os.path.basename(folder)}"
    robot.sync(host, [], robot_checkout)
    robot._sh(["ssh", host, "mkdir", "-p", f"{remote}/scenarios"])
    robot._sh([*robot._RSYNC, f"{folder}/scenarios/", f"{host}:{remote}/scenarios/"])
    runner = f"{robot.ROBOT_DIR}/src/Simulation/world_sim/robot_run_plan.sh"
    launch = f"bash {runner} {robot.ROBOT_DIR} run {remote} > {remote}/sweep.log 2>&1"
    robot._sh(["ssh", host, "docker", "exec", "-d", "-e", "WORLD_SIM_MODULE=Simulation.world_sim.sweep",
               "-e", f"WORLD_SIM_DETOUR={detour}", "billie", "bash", "-c", f"'{launch}'"])  # fmt: skip
    shown = 0
    while True:
        time.sleep(poll_s)
        try:
            lines = robot._sh(["ssh", host, f"cat {remote}/sweep.log"], capture=True).splitlines()
            done = '"finished": true' in robot._sh(["ssh", host, f"cat {remote}/sweep.json 2>/dev/null || true"], capture=True)
            # "[w]" keeps pgrep from matching this ssh shell's own command line (the container sees host processes).
            alive = robot._sh(["ssh", host, "docker exec billie pgrep -f '[w]orld_sim.sweep run' || true"], capture=True).strip()
        except Exception as e:  # a network blip: the sweep keeps running on the robot
            print(f"(robot not reachable: {e}; retrying)")
            continue
        for line in lines[shown:]:
            if _PROGRESS_LINE.search(line):
                print(line, flush=True)
        shown = len(lines)
        if done or not alive:
            break
    robot._sh(["rsync", "-a", f"{host}:{remote}/", f"{folder}/"])
    if not done:
        raise RuntimeError("The sweep stopped before finishing; see sweep.log:\n" + "\n".join(lines[-30:]))


def main() -> None:
    """Command line: generate, run (here), robot (run on a robot and summarize) or summarize a sweep."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    gen = sub.add_parser("generate", help="write a new sweep folder of random scenarios")
    gen.add_argument("--count", type=int, default=100)
    gen.add_argument("--seed", type=int, default=1)
    gen.add_argument("--no-replays", action="store_true", help="only joints and pose commands")
    for name in ("run", "robot", "summarize"):
        step = sub.add_parser(name)
        step.add_argument("folder", help="sweep folder")
        if name in ("run", "robot"):
            step.add_argument("--detour", choices=("transit", "batch_ik"), default="transit",
                              help="what routes blocked single moves: the fix (transit) or the robot today (batch_ik)")
        if name == "robot":
            step.add_argument("--host", default="bellboy@billie-29.bellboy")
            step.add_argument("--robot-checkout", default=robot.DEFAULT_ROBOT_CHECKOUT)
    args = parser.parse_args()

    if args.step == "generate":
        generate(args.count, args.seed, not args.no_replays)
        return
    folder = os.path.abspath(args.folder)
    if args.step == "run":
        os.environ["WORLD_SIM_DETOUR"] = args.detour  # read when commands.py is imported, below
        from Simulation.world_sim.sweep_check import run_sweep  # noqa: PLC0415 (imports JAX)

        run_sweep(folder)
        return
    if args.step == "robot":
        run_on_robot(args.host, folder, args.robot_checkout, args.detour)
    from Simulation.world_sim.sweep_summary import summarize  # noqa: PLC0415

    summarize(folder)


if __name__ == "__main__":
    main()
