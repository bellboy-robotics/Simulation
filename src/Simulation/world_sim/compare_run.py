"""Runs the kitchen comparison on a robot: the same scenarios through each planner, one after the other.

Each planner side has its own folder on the robot (code, JAX cache, results), so neither overwrites
the other, the robot's live planner and its cache are never touched, and the first run of a side
starts from an empty JAX cache (cold) while later runs reuse it (warm):
- "capsules": this repo's billie-onboard (arm_planner_test), through the sweep runner (sweep.py),
  with the scenarios' 32 nearest room capsules;
- "room": this repo's billie-onboard (arm_plan_esdf: the map's ESDF inside the solvers), through the
  sweep runner, with the scenarios' map (kitchen.map_file) instead of the room capsules;
- "room25": the room side with the map refined to 2.5cm (fine_map) and the planner built for 2.5cm, to time
  the planner's grid size (the 2.5cm occupancy, the user 2026-10-14);
- "esdf": billie-onboard feat/collision-alerts-pyroki-analysis (a detached worktree in
  output/esdf_branch), through esdf_runner.py, with the office-2 ESDF.
Results land in <sweep>/runs/<side>-<label>/ (results/*.json plus the run's log).

    python -m Simulation.world_sim.compare_run <sweep folder> --side capsules --label cold [--tasks 10]
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from billie_utils.esdf_grid import load_esdf_grid, serialize_esdf_grid

from Simulation.world_sim import robot, sweep
from Simulation.world_sim.office_map import DEFAULT_MAP

# Robot folders per side (mounted at the same path in the billie container).
ROBOT_DIRS = {
    "capsules": "/home/bellboy/billie/world_sim_cmp_capsules",
    "esdf": "/home/bellboy/billie/world_sim_cmp_esdf",
    "room": "/home/bellboy/billie/world_sim_cmp_room",
    "room25": "/home/bellboy/billie/world_sim_cmp_room25",
}
# [m] The room25 side's grid: the live occupancy's voxel (nvblox_mapper NVBLOX_OCCUPANCY_VOXEL_SIZE_M).
FINE_VOXEL_M = 0.025
# Local checkout of the ESDF branch the esdf side's planner code comes from.
ESDF_CHECKOUT = os.path.join(robot._REPO, "output", "esdf_branch")


def run_folder(sweep_folder: str, side: str, label: str, tasks: int | None) -> str:
    """A fresh local run folder with the scenarios to run (all, or the first `tasks` tasks).

    sweep_folder: The kitchen sweep (kitchen_tasks.py). side: "capsules" or "esdf". label: e.g. "cold".
    tasks: Only tasks t000..t<tasks-1>; None for all.
    Returns: The run folder (<sweep>/runs/<side>-<label>), with scenarios/ filled.
    """
    folder = os.path.join(sweep_folder, "runs", f"{side}-{label}")
    if os.path.exists(folder):
        raise FileExistsError(f"{folder} exists; pick another --label")
    os.makedirs(os.path.join(folder, "scenarios"))
    for name in sorted(os.listdir(os.path.join(sweep_folder, "scenarios"))):
        if tasks is None or int(name[1:4]) < tasks:
            shutil.copy(os.path.join(sweep_folder, "scenarios", name), os.path.join(folder, "scenarios", name))
            if side in ("room", "room25"):
                room_scenario(os.path.join(folder, "scenarios", name), fine=side == "room25")
    return folder


def room_scenario(path: str, fine: bool = False) -> None:
    """Rewrites a kitchen scenario for the room side: the map instead of its room capsules.

    The kitchen scenarios' objects are the room capsules nearest the arm base (kitchen_tasks.py); the
    room side plans against the map they were made from, so they are dropped.
    path: The scenario file, rewritten in place. fine: Use the map refined to FINE_VOXEL_M (fine_map).
    """
    with open(path) as f:
        data = json.load(f)
    data["objects"] = []
    data["map"] = data["kitchen"].get("map_file") or str(DEFAULT_MAP)
    if fine:
        data["map"] = fine_map(data["map"])
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def fine_map(room_map: str) -> str:
    """The map refined to FINE_VOXEL_M by linear interpolation along each axis (trilinear), made once.

    Every original grid point keeps its value, so the room is the same; only the planner's grid is finer.
    A new point between unknown neighbours stays unknown (NaN).
    room_map: The .esdfbin, relative to the repo root (or absolute).
    Returns: The refined map's path, in the same form (a sibling folder named <map>-2.5cm).
    """
    source = Path(room_map)
    target = source.parent.with_name(f"{source.parent.name}-{FINE_VOXEL_M * 100:g}cm") / source.name
    full = target if target.is_absolute() else Path(robot._REPO) / target
    if full.exists():
        return str(target)
    grid = load_esdf_grid(source if source.is_absolute() else Path(robot._REPO) / source)
    factor = round(grid.voxel_size_m / FINE_VOXEL_M)
    if factor != 2:
        raise ValueError(f"{room_map}: {grid.voxel_size_m}m voxel; fine_map halves 5cm maps only")
    values = grid.distances_m.astype(np.float32)
    for axis in range(3):
        shape = list(values.shape)
        shape[axis] = 2 * shape[axis] - 1
        fine = np.empty(shape, np.float32)
        even = [slice(None)] * 3
        odd = [slice(None)] * 3
        even[axis], odd[axis] = slice(0, None, 2), slice(1, None, 2)
        fine[tuple(even)] = values
        count = values.shape[axis]
        fine[tuple(odd)] = 0.5 * (np.take(values, range(count - 1), axis) + np.take(values, range(1, count), axis))
        values = fine
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(serialize_esdf_grid(name=f"{grid.name}-fine", frame_id=grid.frame_id, origin_m=grid.origin_m,
                                         voxel_size_m=FINE_VOXEL_M, distances_m=values))  # fmt: skip
    print(f"Refined {room_map} to {FINE_VOXEL_M}m: {values.shape} -> {target}")
    return str(target)


def sync_map(host: str, folder: str) -> None:
    """Copies the run's map to the robot under the synced repo root, where the scenarios name it.

    host: ssh destination. folder: The run folder (its scenarios name the map).
    """
    local_map = scenario_map(folder)
    relative = os.path.relpath(local_map, robot._REPO) if local_map.is_absolute() else str(local_map)
    remote = f"{robot.ROBOT_DIR}/{relative}"
    robot._sh(["ssh", host, "mkdir", "-p", os.path.dirname(remote)])
    robot._sh(["rsync", "-a", os.path.join(robot._REPO, relative), f"{host}:{remote}"])


def scenario_map(folder: str) -> Path:
    """The map file a run folder's scenarios use (kitchen.map_file; the original map when unset).

    folder: Run or sweep folder with scenarios/. Returns: Local path of its esdf_grid.esdfbin.
    """
    first = sorted(os.listdir(os.path.join(folder, "scenarios")))[0]
    with open(os.path.join(folder, "scenarios", first)) as f:
        data = json.load(f)
    return Path(data.get("map") or data["kitchen"].get("map_file") or DEFAULT_MAP)


def run_esdf(host: str, folder: str, poll_s: float = 30.0) -> None:
    """Runs a run folder through the ESDF branch's planner on the robot, detached, and fetches the results.

    host: ssh destination. folder: Local run folder (run_folder). poll_s: Seconds between progress checks.
    Raises: RuntimeError if the run stops before finishing (its log tail says why).
    """
    os.environ["WORLD_SIM_BILLIE_ONBOARD"] = ESDF_CHECKOUT  # robot.sync takes the planner code from here
    robot.sync(host, [], "")
    base = robot.ROBOT_DIR
    remote = f"{base}/runs/{os.path.basename(folder)}"
    local_map = scenario_map(folder)
    remote_map = f"{base}/map/{local_map.parent.name}/{local_map.name}"  # one folder per map version
    robot._sh(["ssh", host, "mkdir", "-p", f"{remote}/scenarios", os.path.dirname(remote_map)])
    robot._sh(["rsync", "-a", str(local_map), f"{host}:{remote_map}"])
    robot._sh([*robot._RSYNC, f"{folder}/scenarios/", f"{host}:{remote}/scenarios/"])
    runner = f"{base}/src/Simulation/world_sim/robot_run_plan.sh"
    launch = (f"bash {runner} {base} {remote}/scenarios --map {remote_map} "
              f"--out {remote}/results > {remote}/run.log 2>&1")  # fmt: skip
    robot._sh(["ssh", host, "docker", "exec", "-d", "-e", "WORLD_SIM_MODULE=Simulation.world_sim.esdf_runner",
               "billie", "bash", "-c", f"'{launch}'"])  # fmt: skip
    shown, done, lines = 0, False, []
    while not done:
        time.sleep(poll_s)
        try:
            lines = robot._sh(["ssh", host, f"cat {remote}/run.log"], capture=True).splitlines()
            done = robot._sh(["ssh", host, f"test -f {remote}/finished.json && echo yes || true"], capture=True).strip() == "yes"
            alive = robot._sh(["ssh", host, "docker exec billie pgrep -f '[w]orld_sim.esdf_runner' || true"], capture=True).strip()
        except Exception as e:  # a network blip: the run keeps going on the robot
            print(f"(robot not reachable: {e}; retrying)")
            continue
        for line in lines[shown:]:
            if line.startswith("[") or "ready:" in line or "Traceback" in line:
                print(line, flush=True)
        shown = len(lines)
        if not done and not alive:
            break
    robot._sh(["rsync", "-a", f"{host}:{remote}/", f"{folder}/"])
    if not done:
        raise RuntimeError("The ESDF run stopped before finishing; see run.log:\n" + "\n".join(lines[-30:]))


def main() -> None:
    """Command line: make the run folder, run one side on the robot, fetch the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep", help="kitchen sweep folder (kitchen_tasks.py)")
    parser.add_argument("--side", choices=sorted(ROBOT_DIRS), required=True)
    parser.add_argument("--label", required=True, help='run name, e.g. "cold" (first run of a side) or "warm"')
    parser.add_argument("--tasks", type=int, default=None, help="only the first N tasks (a pilot)")
    parser.add_argument("--host", default=robot.DEFAULT_HOST)
    parser.add_argument("--share-gpu", action="store_true",
                        help="JAX takes GPU memory as needed (a robot whose live processes use the GPU, e.g. its mapper)")
    args = parser.parse_args()
    folder = run_folder(os.path.abspath(args.sweep), args.side, args.label, args.tasks)
    robot.ROBOT_DIR = ROBOT_DIRS[args.side]  # robot.py and sweep.py read it at call time
    print(f"{args.side} side, {len(os.listdir(os.path.join(folder, 'scenarios')))} scenarios -> {folder}")
    if args.side == "capsules":
        sweep.run_on_robot(args.host, folder, "", None, share_gpu=args.share_gpu)
    elif args.side in ("room", "room25"):
        sync_map(args.host, folder)
        # The planner's crop takes the voxel of the map it is built with (pyroki_planner esdf_world).
        built_with = {"WORLD_SIM_ESDF_MAP": f"{robot.ROBOT_DIR}/{scenario_map(folder)}"} if args.side == "room25" else {}
        sweep.run_on_robot(args.host, folder, "", None, share_gpu=args.share_gpu, env=built_with)
    else:
        run_esdf(args.host, folder)
    print(f"Done: {folder}")


if __name__ == "__main__":
    main()
