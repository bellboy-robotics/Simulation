"""Reads a plan step's results and measures the arm's path against the world objects (NumPy; the real-mesh
checks through mesh_check also load this machine's URDF meshes with trimesh / yourdfpy)."""

import json
from dataclasses import dataclass, field

import numpy as np
from billie_utils.arm_collision_model import (
    CollisionModel,
    link_obstacle_distances,
    link_transforms,
    sample_joint_path,
)
from scipy.spatial.transform import Rotation

from Simulation.world_sim.mesh_check import link_mesh_distances, mount_differences

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
    # The same commands planned without the avoid objects (plan.run_without_obstacles), or None.
    no_obstacles: dict | None = None
    # The planner's build and what it logged / compiled during the run (plan.run_scenario keys "started_at",
    # "planner_build", "planner_log", "planner_log_dropped", "planner_messages", "planner_compile"); {} for older results.
    planner: dict = field(default_factory=dict)


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
        no_obstacles=data.get("no_obstacles"),
        planner={k: data[k] for k in ("started_at", "planner_build", "planner_log", "planner_log_dropped",
                                      "planner_messages", "planner_compile") if k in data},  # fmt: skip
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


def mesh_distances(run: Run, joints_deg: np.ndarray, role: str) -> np.ndarray:
    """Distance from every link's real collision mesh (this machine's URDF) to the nearest object of a role.

    The meshes are posed with the run model's arm frames (see mesh_check), so a robot run keeps its
    calibration even when its planner modelled the gripper as a box.

    run: The run (arm model and objects).
    joints_deg: (S, 6) configurations in degrees.
    role: "avoid" or "eef_touch".
    Returns: (S, L) meters over run.model.link_names, negative inside; +inf for links without a mesh or
        when there are no objects; values >= mesh_check.EXACT_WITHIN_M are lower bounds.
    """
    return link_mesh_distances(run.model, joints_deg, *capsules(run, role))


def planner_gripper_box(run: Run) -> bool:
    """Whether the run's planner modelled the gripper as its bounding box (gripper mesh missing there).

    run: The run; its machine["gripper_geometry"] decides when present, else the model's gripper mount
        is compared with the URDF's mesh mount (the box fallback mounts it elsewhere).
    Returns: True for the box fallback.
    """
    geometry = run.machine.get("gripper_geometry")
    if geometry is not None:
        return "box" in str(geometry).lower()
    return EEF_LINKS[0] in mount_differences(run.model)


def tcp_poses(run: Run, joints_deg: np.ndarray) -> np.ndarray:
    """TCP poses of configurations.

    run: The run (arm model). joints_deg: (S, 6) degrees.
    Returns: (S, 6) [x, y, z mm, rx, ry, rz rad rotation vector], xArm base frame.
    """
    T = link_transforms(run.model, np.deg2rad(joints_deg))[:, run.model.link_names.index("link_tcp")]
    return np.concatenate([T[:, :3, 3] * 1000.0, Rotation.from_matrix(T[:, :3, :3]).as_rotvec()], axis=1)


# [mm, deg] A target the arm ends farther from than this was not reached (the IK's own accuracy is ~1mm).
TARGET_TOLERANCE_MM, TARGET_TOLERANCE_DEG = 5.0, 2.0
# [mm] replay_policy's default final_position_threshold (commands.FINAL_POSITION_THRESHOLD_M): the robot
# fails a pose replay whose last frame ends farther than this; closer misses pass silently.
REPLAY_FINAL_THRESHOLD_MM = 50.0
# The sim's own note when a pose target is inside the robot base (the robot keeps the joints silently).
BASE_COLLISION_NOTE = "inside the robot base"


def target_errors(run: Run) -> tuple[np.ndarray, np.ndarray]:
    """How far each target's reached TCP is from the pose the planner was asked for.

    run: The run.
    Returns: ((N,) position error mm, (N,) orientation error deg); NaN for targets without a pose.
    """
    n = len(run.joints_deg)
    reached = tcp_poses(run, run.joints_deg) if n else np.zeros((0, 6))
    pos = np.linalg.norm(reached[:, :3] - run.target_pose[:, :3], axis=1)
    has_target = ~np.isnan(run.target_pose[:, 0])
    ori = np.full(n, np.nan)
    if has_target.any():
        delta = Rotation.from_rotvec(run.target_pose[has_target, 3:]).inv() * Rotation.from_rotvec(reached[has_target, 3:])
        ori[has_target] = np.rad2deg(delta.magnitude())
    return pos, ori


def target_checks(run: Run, pos_err_mm: np.ndarray, ori_err_deg: np.ndarray) -> list[dict]:
    """Per command: did the arm reach its targets, and would Billie have said so if it did not.

    On the robot nothing checks a pose's IK result or the replay frames on the way; only a pose replay's
    last frame is checked (REPLAY_FINAL_THRESHOLD_MM). A command that finished with a miss beyond the
    tolerance was therefore reported as a success: a "silent miss".

    run: The run. pos_err_mm, ori_err_deg: (N,) per target, distance of the reached TCP from its target
        (NaN without a target; report.point_metrics' "pos_err_mm" / "ori_err_deg").
    Returns: Per run command {"verdict": "reached" | "failed" | "silent miss" | "stayed put" (pose inside
        the base) | "not checked" (no pose targets: joints, replay in joints mode), "checked" / "missed":
        targets checked / beyond tolerance, "worst_mm", "worst_deg", "worst_target", "final_mm", "final_deg"
        (the command's last checked target), "text": one line for people}.
    """
    out = []
    for c, outcome in enumerate(run.commands):
        spec = outcome["spec"]
        kinds = ("pose",) if spec["cmd"] == "pose" else ("replay",) if spec.get("transform", "poses") == "poses" else ()
        idx = np.flatnonzero((run.command == c) & np.isin(run.kind, kinds) & ~np.isnan(pos_err_mm))
        pos, ori = pos_err_mm[idx], np.nan_to_num(ori_err_deg[idx])
        missed = (pos > TARGET_TOLERANCE_MM) | (ori > TARGET_TOLERANCE_DEG)
        row = {"checked": len(idx), "missed": int(missed.sum()), "worst_mm": None, "worst_deg": None, "worst_target": None,
               "final_mm": None, "final_deg": None}  # fmt: skip
        if len(idx):
            w = int(np.argmax(pos + ori))  # a combined score only to pick the worst one
            row.update(worst_mm=round(float(pos[w]), 1), worst_deg=round(float(ori[w]), 1), worst_target=int(idx[w]),
                       final_mm=round(float(pos[-1]), 1), final_deg=round(float(ori[-1]), 1))  # fmt: skip
        stayed = any(e["command"] == c and BASE_COLLISION_NOTE in e["message"] for e in run.events)
        if not outcome["ok"]:
            row["verdict"], row["text"] = "failed", f"failed (reported): {outcome['error']}"
        elif stayed:
            row["verdict"], row["text"] = "stayed put", "finished, but the target is inside the robot base: the arm stayed where it was (Billie reports success)"
        elif not len(idx):
            row["verdict"], row["text"] = "not checked", "finished (no pose target to check)"
        elif row["missed"]:
            row["verdict"] = "silent miss"
            where = (f"{row['missed']} of {row['checked']} frames off target, worst {row['worst_mm']} mm / {row['worst_deg']}° "
                     f"at target {row['worst_target']}, last frame {row['final_mm']} mm" if spec["cmd"] != "pose"
                     else f"{row['final_mm']} mm / {row['final_deg']}° from its target")  # fmt: skip
            row["text"] = f"finished, but missed: {where} — Billie reports success"
        else:
            row["verdict"], row["text"] = "reached", f"reached (within {TARGET_TOLERANCE_MM:.0f} mm / {TARGET_TOLERANCE_DEG:.0f}°)"
        out.append(row)
    return out


def eef_mask(run: Run) -> np.ndarray:
    """Which links are the end effector.

    run: The run. Returns: (L,) bool over run.model.link_names.
    """
    return np.array([name in EEF_LINKS for name in run.model.link_names])
