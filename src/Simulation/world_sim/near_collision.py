"""Where the arm comes closer to an avoid object than is safe: zones along the path, for marking.

Safe means the planner's own clearance, WORLD_COL_MARGIN_M (the cost starts pushing inside it). A zone
is a stretch of consecutive path samples, within one command, where some link is closer than that,
by the planner's capsule or by the real collision mesh (mesh_check), whichever is closer.
"""

import numpy as np
from billie_utils.messages.pyroki_world import WORLD_COL_MARGIN_M
from billie_utils.arm_collision_model import CollisionModel

from Simulation.world_sim.mesh_check import MeshPoints, capsule_point_distances, mesh_points, pose_points

# [mm] Closer than this to an avoid object is not safe: the planner's own clearance.
SAFE_MM = WORLD_COL_MARGIN_M * 1000.0


def near_zones(
    model: CollisionModel, samples_deg: np.ndarray, owner: np.ndarray, command: np.ndarray,
    capsule_mm: np.ndarray, mesh_mm: np.ndarray, objects: list[tuple], points: MeshPoints | None = None,
) -> list[dict]:  # fmt: skip
    """The zones of a path where the arm is closer than SAFE_MM to an avoid object.

    model: The arm model that posed the path (the run's).
    samples_deg: (S, 6) the path (analysis.dense_path). owner: (S,) target index per sample (-1 = start).
    command: (N,) command index per target.
    capsule_mm: (S, L) planner capsule distance per link (+inf for none).
    mesh_mm: (S, L) real-mesh distance per link (+inf for none).
    objects: Avoid objects as (name, starts_m (M, 3), ends_m (M, 3), radii_m (M,)).
    points: The mesh points; None for this machine's URDF.
    Returns: Per zone, in path order: {"first_sample", "last_sample", "worst_sample", "first_target",
        "last_target", "command" (-1 before the first move), "worst_mm" (negative = inside), "inside",
        "by": "real mesh" or "planner capsule" (which reads closer there), "link", "object",
        "capsule_min_mm", "mesh_min_mm" (over the zone), "arm_point_mm" / "object_point_mm" (closest points
        at the worst sample, xArm base frame; None when the link has no mesh)}.
    """
    closest = np.minimum(capsule_mm, mesh_mm).min(axis=1)
    near = closest < SAFE_MM
    sample_command = np.where(owner >= 0, command[np.maximum(owner, 0)], -1) if len(command) else np.full(len(owner), -1)
    zones, k = [], 0
    while k < len(near):
        if not near[k]:
            k += 1
            continue
        end = k
        while end + 1 < len(near) and near[end + 1] and sample_command[end + 1] == sample_command[k]:
            end += 1
        zones.append(_zone(model, samples_deg, owner, sample_command, capsule_mm, mesh_mm, objects, points, k, end))
        k = end + 1
    return zones


def _zone(model: CollisionModel, samples_deg: np.ndarray, owner: np.ndarray, sample_command: np.ndarray,
          capsule_mm: np.ndarray, mesh_mm: np.ndarray, objects: list[tuple], points: MeshPoints | None,
          first: int, last: int) -> dict:  # fmt: skip
    """Describes one zone (see near_zones).

    model, samples_deg, owner, capsule_mm, mesh_mm, objects, points: See near_zones.
    sample_command: (S,) command index per sample. first, last: The zone's sample range (inclusive).
    Returns: The zone description.
    """
    span = slice(first, last + 1)
    both = np.minimum(capsule_mm[span], mesh_mm[span])  # (Z, L)
    k_rel, link = np.unravel_index(np.argmin(both), both.shape)
    k = first + int(k_rel)
    by_mesh = mesh_mm[k, link] <= capsule_mm[k, link]
    arm_point, object_point, name = _closest_points(model, samples_deg[k], int(link), objects, points)
    return {
        "first_sample": first, "last_sample": last, "worst_sample": k,
        "first_target": int(owner[first]), "last_target": int(owner[last]), "command": int(sample_command[first]),
        "worst_mm": round(float(both[k_rel, link]), 1), "inside": bool(both[k_rel, link] < 0),
        "by": "real mesh" if by_mesh else "planner capsule", "link": model.link_names[link], "object": name,
        "capsule_min_mm": round(float(capsule_mm[span].min()), 1), "mesh_min_mm": round(float(mesh_mm[span].min()), 1),
        "arm_point_mm": arm_point, "object_point_mm": object_point,
    }  # fmt: skip


def _closest_points(model: CollisionModel, joints_deg: np.ndarray, link: int, objects: list[tuple],
                    points: MeshPoints | None) -> tuple[list | None, list | None, str | None]:  # fmt: skip
    """The point of one link's real mesh nearest the avoid objects, and the nearest object point to it.

    model: The arm model. joints_deg: (6,) the configuration. link: Index into model.link_names.
    objects: See near_zones. points: The mesh points; None for this machine's URDF.
    Returns: (arm point [x, y, z] mm, object surface point [x, y, z] mm, object name); Nones when the
        link has no mesh points or there are no objects.
    """
    points = points or mesh_points()
    name = model.link_names[link]
    if name not in points.link_names or not any(len(o[3]) for o in objects):
        return None, None, None
    posed = pose_points(model, joints_deg, points)[points.link_index == points.link_names.index(name)]
    posed = posed[np.isfinite(posed).all(axis=1)]
    if not len(posed):
        return None, None, None
    starts = np.concatenate([np.reshape(o[1], (-1, 3)) for o in objects])
    ends = np.concatenate([np.reshape(o[2], (-1, 3)) for o in objects])
    radii = np.concatenate([np.reshape(o[3], -1) for o in objects])
    owner = np.concatenate([np.full(len(o[3]), i) for i, o in enumerate(objects)])
    distance, capsule = capsule_point_distances(posed, starts, ends, radii, with_index=True)
    i = int(np.argmin(distance))
    p, a, b, r = posed[i], starts[capsule[i]], ends[capsule[i]], radii[capsule[i]]
    t = float(np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0.0, 1.0))
    axis_point = a + t * (b - a)
    away = p - axis_point
    surface = axis_point + r * away / max(np.linalg.norm(away), 1e-9)  # the capsule surface toward the arm point
    return (p * 1000.0).round(1).tolist(), (surface * 1000.0).round(1).tolist(), objects[owner[capsule[i]]][0]
