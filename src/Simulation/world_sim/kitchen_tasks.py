"""Random kitchen tasks for the planner comparison (capsules vs. the ESDF planner), as sweep scenarios.

A task is one Billie base pose in the kitchen (office_map.base_positions, random heading), one start
configuration and three commands from it: `joints`, `pose` and `replay_policy`. Each command becomes
its own scenario (tNNN_joints / tNNN_pose / tNNN_replay), so all three start from the same state and
the success rates per command type are independent.

Every start and target is valid against the real map (map_judge with the real link meshes: 2cm clear, observed space, TCP
at most WORK_BAND_M[1] above the floor), against the floor, Billie's body and self-collision (the
sweep's own checks). They are NOT required to be clear of the capsules: if a capsule's bulge hides
a valid target, that is a cost of the capsule model the comparison should show. A replay needs its
recording's first and last frames valid the same way at that base pose.

The capsule planner's objects are the 32 room capsules (room_capsules_B-50.json) nearest the arm base.

    python -m Simulation.world_sim.kitchen_tasks --count 100 --seed 1
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from Simulation.world_sim import map_capsules as mc
from Simulation.world_sim.map_judge import CLEARANCE_M, map_from_arm, mesh_clearance
from Simulation.world_sim.mesh_check import mesh_points
from Simulation.world_sim.office_map import DEFAULT_MAP, WORK_BAND_M, base_positions, load_office_map
from Simulation.world_sim.recording import RECORDINGS_DIR, load_recording
from Simulation.world_sim.sweep import _FLOOR_Z_MM, SWEEPS_DIR, _tcp_poses, _valid_joints

# The capsule set the user chose (2026-10-07): unseen space may hold capsules, 50mm allowed bulge.
CAPSULE_SET = "B-50"
# Planner capsule slots (MAX_WORLD_CAPSULES); the nearest ones to the arm base are sent.
SLOTS = 32
# billie-29's ARM_TO_BASE_CALIBRATION [x_mm, y_mm, yaw_deg].
ARM_TO_BASE = [-180, 0, 0]
# Configurations drawn (each through the sweep's checks) before a task's base pose is drawn again.
_CONFIG_TRIES = 40
# Base poses drawn for one task before giving up (a replay needs clear first/last frames there).
_POSE_TRIES = 30


def _capsules(folder) -> mc.Capsules:
    """The chosen room capsule set.

    folder: The map folder holding room_capsules_<set>.json.
    Returns: The capsules, map frame, meters.
    """
    objects = json.loads((folder / f"room_capsules_{CAPSULE_SET}.json").read_text())["capsules"]
    s, e, r = (np.array([o[k] for o in objects], float) / 1000 for k in ("start_mm", "end_mm", "radius_mm"))
    return mc.Capsules(s, e, r)


def _valid_on_map(office, model, joints_deg: np.ndarray, T_map_arm: np.ndarray) -> bool:
    """Whether a configuration is clear of the real map, in observed space, with its TCP low enough.

    office: The map. model: Arm model. joints_deg: (6,) degrees. T_map_arm: map_from_arm of the task.
    Returns: True when valid.
    """
    clearance, _ = mesh_clearance(office, model, joints_deg[None], T_map_arm, mesh_points())
    tcp_z = (T_map_arm @ np.array([*_tcp_poses(model, joints_deg[None])[0, :3] / 1000, 1.0]))[2]
    return bool(not np.isnan(clearance).any() and clearance.min() >= CLEARANCE_M and tcp_z <= WORK_BAND_M[1])


def _config(data: dict, planner, office, T_map_arm: np.ndarray, rng) -> np.ndarray | None:
    """A random configuration valid for the sweep's checks (floor, body, self) and on the real map.

    data: Scenario without objects. planner: Local SimPlanner. office: The map.
    T_map_arm: map_from_arm of the task. rng: Random generator.
    Returns: (6,) degrees, or None after _CONFIG_TRIES draws.
    """
    for _ in range(_CONFIG_TRIES):
        q = _valid_joints(data, planner, rng)
        if q is not None and _valid_on_map(office, planner.model, q, T_map_arm):
            return q
    return None


def _replay_for(office, model, T_map_arm: np.ndarray, recordings: dict, rng) -> str | None:
    """A recording whose first and last frames are valid on the real map at this base pose.

    office: The map. model: Arm model. T_map_arm: map_from_arm of the task.
    recordings: repo_id -> (F, 6) recorded joints, degrees. rng: Random generator (order of trying).
    Returns: The repo_id, or None.
    """
    for repo_id in rng.permutation(sorted(recordings)):
        joints = recordings[repo_id]
        if all(_valid_on_map(office, model, joints[i], T_map_arm) for i in (0, -1)):
            return str(repo_id)
    return None


def _task(index: int, seed: int, planner, office, bases, capsules, recordings) -> list[dict] | None:
    """One task: base pose, start joints and its three scenarios (joints, pose, replay).

    index: Task number. seed: Sweep seed. planner: Local SimPlanner. office: The map.
    bases: (B, 3) base positions. capsules: Room capsules. recordings: See _replay_for.
    Returns: The three scenarios, or None if no valid pose was found in _POSE_TRIES.
    """
    rng = np.random.default_rng([seed, index])
    for _ in range(_POSE_TRIES):
        x, y, unseen = bases[rng.integers(len(bases))]
        heading = float(rng.uniform(-np.pi, np.pi))
        # The sim's units (world.object_in_arm_frame): map x, y in mm, heading in degrees.
        billie = {"pose": [round(float(x) * 1000, 1), round(float(y) * 1000, 1), round(float(np.degrees(heading)), 3)],
                  "arm_to_base": ARM_TO_BASE}
        T_map_arm = map_from_arm(billie)
        data = {"billie": billie, "floor_z_mm": _FLOOR_Z_MM, "start_joints_deg": [0.0] * 6, "objects": [], "commands": []}
        replay = _replay_for(office, planner.model, T_map_arm, recordings, rng)
        start = _config(data, planner, office, T_map_arm, rng) if replay else None
        goal_q = _config(data, planner, office, T_map_arm, rng) if start is not None else None
        pose_q = _config(data, planner, office, T_map_arm, rng) if goal_q is not None else None
        if pose_q is not None:
            break
    else:
        return None
    arm = T_map_arm[:3, 3]
    chosen, in_reach = mc.select_capsules(capsules, arm, SLOTS)
    objects = [o for k, o in enumerate(capsules.to_objects()) if k in set(chosen.tolist())]
    pose = _tcp_poses(planner.model, pose_q[None])[0]
    commands = {
        "joints": ({"cmd": "joints", "joints": goal_q.round(1).tolist()}, goal_q),
        "pose": ({"cmd": "pose", "pose": [*np.round(pose[:3], 1).tolist(), *np.round(pose[3:], 4).tolist()]}, pose_q),
        "replay": ({"cmd": "replay_policy", "repo_id": replay, "speed": 1.0, "transform": "poses"}, recordings[replay][-1]),
    }
    kitchen = {"base_xy": [float(x), float(y)], "heading_rad": heading, "unseen_share": float(unseen),
               "capsules_in_reach": int(in_reach), "map": office.checksum, "capsule_set": CAPSULE_SET}  # fmt: skip
    return [
        {**data, "name": f"t{index:03d}_{kind}", "start_joints_deg": start.round(1).tolist(), "objects": objects,
         "commands": [cmd], "kitchen": {**kitchen, "task": index, "kind": kind},
         "sweep": {"index": index, "seed": seed, "goal_joints_deg": [np.round(goal, 2).tolist()]}}  # fmt: skip
        for kind, (cmd, goal) in commands.items()
    ]


def generate(count: int, seed: int) -> str:
    """Writes a sweep folder with count tasks (3 scenarios each).

    count: Tasks. seed: Sweep seed (same seed + count = same tasks on the same arm and map).
    Returns: The sweep folder.
    """
    from Simulation.world_sim.planner import SimPlanner  # noqa: PLC0415 (imports JAX)

    folder = os.path.join(SWEEPS_DIR, f"kitchen-{time.strftime('%Y%m%d-%H%M')}-seed{seed}")
    os.makedirs(os.path.join(folder, "scenarios"))
    office = load_office_map()
    bases, capsules, planner = base_positions(office), _capsules(DEFAULT_MAP.parent), SimPlanner()
    recordings = {}
    for name in sorted(os.listdir(RECORDINGS_DIR)):
        repo_id = name.removesuffix(".json").replace("__", "/")
        recordings[repo_id] = np.array([s.recorded_joints for s in load_recording(repo_id)[0]])
    made = 0
    for index in range(count):
        scenarios = _task(index, seed, planner, office, bases, capsules, recordings)
        if scenarios is None:
            print(f"task {index}: no valid base pose found, skipped")
            continue
        for s in scenarios:
            with open(os.path.join(folder, "scenarios", f"{s['name']}.json"), "w") as f:
                json.dump(s, f, indent=1)
        made += 1
        print(f"task {index}: base ({s['kitchen']['base_xy'][0]:.2f}, {s['kitchen']['base_xy'][1]:.2f}), "
              f"{s['kitchen']['capsules_in_reach']} capsules in reach, replay {s['commands'][0]['repo_id'].split('/')[-1]}")  # fmt: skip
    print(f"{made} tasks ({3 * made} scenarios) in {folder}")
    return folder


def retarget(folder: str, map_path: str, suffix: str) -> str:
    """The same tasks on another version of the map: a copy of a sweep with that map's capsules.

    Starts, targets and base poses stay; each scenario gets the 32 capsules of the map's CAPSULE_SET
    nearest its arm base, and records the map file (compare_run / compare_report use it).
    folder: Sweep folder (generate). map_path: The other map's esdf_grid.esdfbin (its folder holds
        room_capsules_<set>.json). suffix: Appended to the new folder's name, e.g. "free12".
    Returns: The new sweep folder.
    """
    out = f"{folder.rstrip('/')}-{suffix}"
    os.makedirs(os.path.join(out, "scenarios"))
    capsules = _capsules(Path(map_path).parent)
    checksum = load_office_map(Path(map_path)).checksum
    for name in sorted(os.listdir(os.path.join(folder, "scenarios"))):
        with open(os.path.join(folder, "scenarios", name)) as f:
            data = json.load(f)
        chosen, in_reach = mc.select_capsules(capsules, map_from_arm(data["billie"])[:3, 3], SLOTS)
        data["objects"] = [o for k, o in enumerate(capsules.to_objects()) if k in set(chosen.tolist())]
        data["kitchen"].update(capsules_in_reach=int(in_reach), map=checksum, map_file=str(map_path))
        with open(os.path.join(out, "scenarios", name), "w") as f:
            json.dump(data, f, indent=1)
    print(f"{len(os.listdir(os.path.join(out, 'scenarios')))} scenarios on {map_path} in {out}")
    return out


def main() -> None:
    """Command line: generate tasks, or --retarget a sweep onto another version of the map."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--retarget", help="existing sweep folder to copy onto --map")
    parser.add_argument("--map", help="the other map's esdf_grid.esdfbin (with --retarget)")
    parser.add_argument("--suffix", default="free12", help="new folder suffix (with --retarget)")
    args = parser.parse_args()
    if args.retarget:
        retarget(args.retarget, args.map, args.suffix)
    else:
        generate(args.count, args.seed)


if __name__ == "__main__":
    main()
