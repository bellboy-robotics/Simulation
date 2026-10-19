"""Runs kitchen scenarios through the ESDF planner (billie-onboard feat/collision-alerts-pyroki-analysis), as is.

Runs on the robot inside the `billie` container, with that branch's planner code synced into its own
folder (see compare_run.py). The planner is the branch's own IKResolver with its WorldCollisionGuard
enforced; this file only plays the parts of the robot around it:
- the mapper: a collision_world.json snapshot (static office-2 ESDF, the scenario's base pose as
  T_map_from_arm_base, fresh timestamps) rewritten every SNAPSHOT_PERIOD_S, in a private BILLIE_DIR
  so the robot's live mapper files are never touched;
- the brain: `joints` -> resolve_joints, `pose` -> resolve, `replay_policy` -> resolve_joints to the
  recording's first joints, then resolve_batch on BATCH_SIZE frames at a time (like BufferingPlayer).
The only setting changed from the branch's defaults is the minimum clearance, 20mm (the user's rule
for both planners) instead of its 50mm, through the branch's own minimum_clearance_mm option (and its
guard's default, which the batch call uses).

    robot_run_plan.sh <dir> (with WORLD_SIM_MODULE=Simulation.world_sim.esdf_runner) <scenarios...> --map <esdfbin> --out <dir>
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time

import numpy as np

from Simulation.world_sim.frames import map_from_arm

# [mm] The user's clearance rule for both planners (the branch's default is 50mm).
MIN_CLEARANCE_MM = 20.0
# [s] How often the fake mapper rewrites the snapshot; the guard wants sources under 1s old.
SNAPSHOT_PERIOD_S = 0.1
# Replay frames per planner batch (billie_utils.messages.pyroki_node.BATCH_SIZE, the brain's batch).
BATCH_SIZE = 50
# [s] Longest wait for the branch's background solver preparation before giving up.
PREPARE_TIMEOUT_S = 1800.0


class _FakeMapper:
    """Keeps the branch's collision_world.json fresh: the static map and the current base pose."""

    def __init__(self, billie_dir: str, map_path: str):
        """billie_dir: Private BILLIE_DIR for the planner. map_path: The office-2 esdf_grid.esdfbin."""
        from billie_utils.messages.collision_world import collision_world_path  # noqa: PLC0415

        self.path = collision_world_path(billie_dir)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.map_path = os.path.abspath(map_path)
        with open(self.map_path, "rb") as f:
            self.checksum = hashlib.sha256(f.read()).hexdigest()
        self.transform = np.eye(4)
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True, name="fake-mapper").start()

    def set_billie(self, billie: dict) -> None:
        """Moves Billie: the map -> arm transform for the next snapshots.

        billie: Scenario "billie": {"pose": [x_mm, y_mm, yaw_deg], "arm_to_base": [x_mm, y_mm, yaw_deg]}.
        """
        self.transform = map_from_arm(billie)
        self._write()
        time.sleep(2 * SNAPSHOT_PERIOD_S)  # a new base pose must be in the file the guard reads next

    def _write(self) -> None:
        """Writes one snapshot atomically (the guard reads it from another thread)."""
        now = time.time()
        snapshot = {
            "schema_version": 1, "timestamp": now, "map_epoch": f"office2-{self.checksum[:8]}", "available": True,
            "map_source": "static", "grid_path": self.map_path, "grid_checksum": self.checksum,
            "frame_id": "autoxing_map", "pose_timestamp": now, "arm_timestamp": now, "base_timestamp": now,
            "T_map_from_arm_base": self.transform.tolist(),
        }  # fmt: skip
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(snapshot))
        os.replace(tmp, self.path)

    def _loop(self) -> None:
        """Rewrites the snapshot every SNAPSHOT_PERIOD_S until the process ends."""
        while not self._stop.wait(SNAPSHOT_PERIOD_S):
            self._write()


def gpu_memory() -> dict | None:
    """JAX's device memory counters (bytes in use now and the peak), or None on a device without them.

    Returns: {"bytes_in_use", "peak_bytes_in_use", "bytes_limit"} in bytes.
    """
    import jax  # noqa: PLC0415

    stats = jax.devices()[0].memory_stats() or {}
    keys = ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
    return {k: int(stats[k]) for k in keys if k in stats} or None


def build_resolver() -> tuple[object, dict]:
    """Builds the branch's IKResolver and waits for all its solvers, timing each stage.

    Returns: (the resolver, {"ik_ready_s", "prepared_s" per solver, "total_s", "memory"}).
    """
    from pyroki_planner.resolver import IKResolver  # noqa: PLC0415

    class _Node:
        """Stands in for the Dora node; the resolver only sends cloud messages through it."""

        def send_output(self, *args, **kwargs) -> None:
            """Drops the message."""

    started = time.monotonic()
    resolver = IKResolver(_Node())
    resolver._ready.wait()
    if resolver._build_error is not None:
        raise RuntimeError("ESDF planner build failed") from resolver._build_error
    build = {"ik_ready_s": time.monotonic() - started, "prepared_s": {}}
    wanted = {"pose-32": (resolver._trajectory_planner, 32), "joint-32": (resolver._joint_trajectory_planner, 32),
              "pose-64": (resolver._trajectory_planner, 64)}  # fmt: skip
    while len(build["prepared_s"]) < len(wanted):
        for name, (planner, size) in wanted.items():
            prep = planner.preparation
            if name not in build["prepared_s"] and (size in prep.solvers or size in prep.errors):
                build["prepared_s"][name] = time.monotonic() - started
                if size in prep.errors:
                    raise RuntimeError(f"ESDF planner {name} preparation failed") from prep.errors[size]
        if time.monotonic() - started > PREPARE_TIMEOUT_S:
            raise TimeoutError(f"ESDF planner not prepared after {PREPARE_TIMEOUT_S}s")
        time.sleep(0.2)
    build["total_s"] = time.monotonic() - started
    build["memory"] = gpu_memory()
    # The batch call takes no clearance argument; it uses the guard's default.
    begin = resolver.world_guard.begin
    resolver.world_guard.begin = lambda minimum_clearance_m=MIN_CLEARANCE_MM / 1000: begin(minimum_clearance_m=minimum_clearance_m)
    return resolver, build


def _joined(paths) -> list[np.ndarray]:
    """One waypoint list from the planner's path segments (each segment repeats its start).

    paths: List of (K, 6) joint arrays in radians, or None.
    Returns: Waypoints in radians, without the repeated segment starts.
    """
    out = []
    for segment in paths or []:
        segment = np.asarray(segment)
        out.extend(segment[1:] if out else segment)
    return out


def _run_command(resolver, spec: dict, current: np.ndarray, calls: list) -> list[np.ndarray]:
    """Plans one scenario command the way the branch's brain would, and returns the planned waypoints.

    resolver: The branch's IKResolver. spec: The scenario command. current: (6,) start joints, radians.
    calls: Receives {"call", "seconds"} for every planner call.
    Returns: The planned path's waypoints (radians), start first.
    Raises: The planner's error when it refuses or fails.
    """
    from Simulation.world_sim.recording import load_recording, select_poses  # noqa: PLC0415

    def timed(name, fn, *args, **kwargs):
        """Calls the planner and records how long it took."""
        started = time.monotonic()
        try:
            return fn(*args, **kwargs)
        finally:
            calls.append({"call": name, "seconds": time.monotonic() - started})

    if spec["cmd"] == "joints":
        return list(timed("resolve_joints", resolver.resolve_joints, current, np.deg2rad(spec["joints"]),
                          minimum_clearance_mm=MIN_CLEARANCE_MM))  # fmt: skip
    if spec["cmd"] == "pose":
        solution, base_hit, _, _, paths = timed("resolve", resolver.resolve, current, np.asarray(spec["pose"], float),
                                                minimum_clearance_mm=MIN_CLEARANCE_MM)  # fmt: skip
        if base_hit:
            raise RuntimeError("target collides with Billie's base (the planner's base check)")
        return _joined(paths) or [current, solution]
    if spec["cmd"] != "replay_policy":
        raise ValueError(f"Unknown command {spec['cmd']!r}")
    states, fps = load_recording(spec["repo_id"])
    states = select_poses(states, fps, spec.get("speed", 1.0))
    path = list(timed("resolve_joints", resolver.resolve_joints, current, np.deg2rad(states[0].recorded_joints),
                      minimum_clearance_mm=MIN_CLEARANCE_MM))  # fmt: skip
    for begin in range(0, len(states), BATCH_SIZE):
        batch = states[begin : begin + BATCH_SIZE]
        poses = np.stack([s.position for s in batch])
        recorded = np.deg2rad(np.stack([s.recorded_joints for s in batch]))
        solved, *_, paths = timed("resolve_batch", resolver.resolve_batch, path[-1], poses, recorded)
        path += _joined(paths)[1:] if paths else list(solved)
    return path


def run_scenario(resolver, mapper: _FakeMapper, scenario: dict) -> dict:
    """Runs one scenario's commands from its start joints.

    resolver: The branch's IKResolver. mapper: The fake mapper (gets the scenario's base pose).
    scenario: Scenario data (kitchen_tasks format).
    Returns: Results: per command ok / error / seconds / planner calls, and the planned path in degrees.
    """
    mapper.set_billie(scenario["billie"])
    current = np.deg2rad(np.asarray(scenario["start_joints_deg"], float))
    commands, path = [], []
    for index, spec in enumerate(scenario["commands"]):
        calls, started = [], time.monotonic()
        try:
            waypoints = _run_command(resolver, spec, current, calls)
            ok, error = True, None
        except Exception as exc:  # the planner's refusal is a result, not a crash
            waypoints, ok, error = [current], False, f"{type(exc).__name__}: {exc}"
        commands.append({"spec": spec, "ok": ok, "error": error, "seconds": time.monotonic() - started, "calls": calls})
        path += [{"command": index, "joints_deg": np.rad2deg(q).round(4).tolist()} for q in waypoints]
        current = np.asarray(waypoints[-1])
    return {"name": scenario["name"], "planner": "esdf", "start_joints_deg": scenario["start_joints_deg"],
            "commands": commands, "path": path, "memory": gpu_memory()}  # fmt: skip


def main() -> None:
    """Command line: build the branch's planner once, then run every scenario and save its results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenarios", nargs="+", help="scenario files, or one folder of them")
    parser.add_argument("--map", required=True, help="office-2 esdf_grid.esdfbin")
    parser.add_argument("--out", required=True, help="results folder")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")
    billie_dir = os.path.abspath(os.path.join(args.out, "..", "esdf_billie_dir"))
    os.environ["BILLIE_DIR"] = billie_dir  # never the robot's own: its mapper/live files stay untouched
    os.environ["PYROKI_WORLD_COLLISION"] = "enforce"
    os.makedirs(args.out, exist_ok=True)
    if len(args.scenarios) == 1 and os.path.isdir(args.scenarios[0]):
        folder = args.scenarios[0]
        args.scenarios = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json"))
    mapper = _FakeMapper(billie_dir, args.map)
    resolver, build = build_resolver()
    import jax  # noqa: PLC0415

    machine = {"jax_version": jax.__version__, "jax_devices": [str(d) for d in jax.devices()],
               "jax_cache_dir": os.environ.get("PYROKI_JAX_CACHE_DIR"), "map_checksum": mapper.checksum}  # fmt: skip
    print(f"ESDF planner ready: IK {build['ik_ready_s']:.1f}s, all solvers {build['total_s']:.1f}s", flush=True)
    for i, path in enumerate(args.scenarios):
        with open(path) as f:
            scenario = json.load(f)
        result = run_scenario(resolver, mapper, scenario)
        result.update(build=build, machine=machine)
        name = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(result, f)
        status = "ok" if all(c["ok"] for c in result["commands"]) else result["commands"][-1]["error"][:100]
        print(f"[{i + 1}/{len(args.scenarios)}] {name}: {status} ({sum(c['seconds'] for c in result['commands']):.1f}s)", flush=True)
    with open(os.path.join(args.out, "..", "finished.json"), "w") as f:
        json.dump({"finished": True, "scenarios": len(args.scenarios), "build": build}, f)


if __name__ == "__main__":
    main()
