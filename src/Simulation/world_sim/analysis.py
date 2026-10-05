"""Reads a plan step's results and measures the arm's path against the world objects (NumPy only)."""

import json
from dataclasses import dataclass

import numpy as np
from billie_utils.world_collision_check_poc import (
    CollisionModel,
    link_obstacle_distances,
    link_transforms,
    sample_joint_path,
)
from scipy.spatial.transform import Rotation

# Links that count as the end effector for "eef_touch" objects (the gripper/camera body and the TCP).
EEF_LINKS = ("link_gripper_and_camera", "link_tcp")
# [deg] Joint step between path samples for measuring and drawing; 2deg moves the TCP <= ~3cm.
SAMPLE_STEP_DEG = 2.0


@dataclass
class Run:
    """A plan step's results in array form."""

    name: str  # scenario name
    model: CollisionModel  # the planner's arm model
    start_deg: np.ndarray  # (6,) joints before the first command
    joints_deg: np.ndarray  # (N, 6) every joint target sent to the arm, in order
    command: np.ndarray  # (N,) index of the command that sent each target
    kind: list[str]  # (N,) what produced each target ("pose", "detour", "replay", ...)
    target_pose: np.ndarray  # (N, 6) TCP pose asked of the planner [mm, rad]; NaN rows for none
    blocked: list[str | None]  # (N,) guard refusal message when the move was played anyway
    objects: list[dict]  # world objects: name, role, spec, starts_m, ends_m, radii_m
    events: list[dict]  # operator messages: command, level, message
    commands: list[dict]  # per command: spec, ok, error, seconds
    timings: list[dict]  # per planner call: call, command, n (poses), seconds; build_* entries first
    machine: dict  # where the plan step ran: host, jax_devices, planner_code


def load_run(path: str) -> Run:
    """Loads a results file written by plan.py.

    path: The results JSON.
    Returns: The run, with NumPy arrays.
    """
    with open(path) as f:
        data = json.load(f)
    points = data["points"]
    for obj in data["objects"]:
        obj["starts_m"] = np.asarray(obj["starts_m"], dtype=np.float64).reshape(-1, 3)
        obj["ends_m"] = np.asarray(obj["ends_m"], dtype=np.float64).reshape(-1, 3)
        obj["radii_m"] = np.asarray(obj["radii_m"], dtype=np.float64).reshape(-1)
    return Run(
        name=data.get("name", path),
        model=CollisionModel.from_json(data["model_json"]),
        start_deg=np.asarray(data["start_joints_deg"], dtype=np.float64),
        joints_deg=np.asarray([p["joints_deg"] for p in points], dtype=np.float64).reshape(-1, 6),
        command=np.asarray([p["command"] for p in points], dtype=int),
        kind=[p["kind"] for p in points],
        target_pose=np.asarray(
            [p["target_pose"] if p["target_pose"] is not None else [np.nan] * 6 for p in points], dtype=np.float64
        ).reshape(-1, 6),
        blocked=[p["blocked"] for p in points],
        objects=data["objects"],
        events=data["events"],
        commands=data["commands"],
        timings=data.get("timings", []),
        machine=data.get("machine", {}),
    )


def capsules(run: Run, role: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All capsules of the objects with one role.

    run: The run. role: "avoid" or "eef_touch".
    Returns: (starts_m (M, 3), ends_m (M, 3), radii_m (M,)); M may be 0.
    """
    chosen = [o for o in run.objects if o["role"] == role]
    if not chosen:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    return tuple(np.concatenate([o[k] for o in chosen]) for k in ("starts_m", "ends_m", "radii_m"))


def dense_path(run: Run) -> tuple[np.ndarray, np.ndarray]:
    """The arm's joint path sampled along every move (straight joint lines between targets).

    run: The run.
    Returns: (samples in degrees (S, 6) - the start, then each move's samples ending at its target;
        index (S,) of the target each sample moves toward, -1 for the start).
    """
    samples, owner = [run.start_deg[None]], [-1]
    previous = run.start_deg
    for i, target in enumerate(run.joints_deg):
        move = sample_joint_path(np.stack([previous, target]), SAMPLE_STEP_DEG)[1:]
        samples.append(move)
        owner.extend([i] * len(move))
        previous = target
    return np.concatenate(samples), np.asarray(owner)


def link_distances(run: Run, joints_deg: np.ndarray, role: str) -> np.ndarray:
    """Distance from every link to the nearest object of a role, at each configuration.

    run: The run (arm model and objects).
    joints_deg: (S, 6) configurations in degrees.
    role: "avoid" or "eef_touch".
    Returns: (S, L) meters, negative inside an object; +inf for static links or when there are no objects.
    """
    starts, ends, radii = capsules(run, role)
    dist = np.full((len(joints_deg), len(run.model.link_names)), np.inf)
    if len(radii) == 0 or len(joints_deg) == 0:
        return dist
    for begin in range(0, len(joints_deg), 2000):  # chunks keep the (S, L, M) array small
        chunk = np.deg2rad(joints_deg[begin : begin + 2000])
        dist[begin : begin + 2000] = link_obstacle_distances(run.model, chunk, starts, ends, radii).min(axis=2)
    return np.where(run.model.movable_links[None, :], dist, np.inf)


def tcp_poses(run: Run, joints_deg: np.ndarray) -> np.ndarray:
    """TCP poses of configurations.

    run: The run (arm model). joints_deg: (S, 6) degrees.
    Returns: (S, 6) [x, y, z mm, rx, ry, rz rad rotation vector], xArm base frame.
    """
    T = link_transforms(run.model, np.deg2rad(joints_deg))[:, run.model.link_names.index("link_tcp")]
    return np.concatenate([T[:, :3, 3] * 1000.0, Rotation.from_matrix(T[:, :3, :3]).as_rotvec()], axis=1)


def eef_mask(run: Run) -> np.ndarray:
    """Which links are the end effector.

    run: The run. Returns: (L,) bool over run.model.link_names.
    """
    return np.array([name in EEF_LINKS for name in run.model.link_names])
