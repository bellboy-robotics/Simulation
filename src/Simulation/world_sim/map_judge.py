"""The neutral judge of the planner comparison: how close the arm comes to the real 3D map.

Both planners' joint paths are judged the same way, against the office-2 ESDF itself (never against
capsules or a planner's own collision model), with the arm's real shape: surface points of every
moving link's URDF collision mesh (mesh_check.MeshPoints), posed with the run's arm model, looked up
in the map. A path passes when every point keeps CLEARANCE_M (user: 2cm) and touches no unobserved
space. config_clearance (the planner's link capsules instead of meshes) is kept as a quick check.
"""

from dataclasses import dataclass

import numpy as np
from billie_utils.arm_collision_model import CollisionModel, link_transforms, sample_joint_path

from Simulation.world_sim.frames import map_from_arm  # noqa: F401 (re-exported for callers)
from Simulation.world_sim.office_map import OfficeMap

# [m] Clearance a path must keep from the real map everywhere (the user's success rule, both planners).
CLEARANCE_M = 0.02
# [m] Spacing of the sample points along a link's capsule axis; small next to the 5cm map voxels.
JUDGE_STEP_M = 0.02
# Use every n-th mesh surface point (they are ~8mm apart; the map's voxels are 5cm).
MESH_POINT_STRIDE = 3
# [deg] Largest joint step between judged configurations along a path (1 deg moves a point <= ~2cm).
PATH_STEP_DEG = 1.0


@dataclass
class PathVerdict:
    """The judge's result for one joint path."""

    min_clearance_m: float  # smallest (map distance - link radius) over the observed samples, meters
    unseen: bool  # some sample touched space the map never observed
    first_bad_index: int  # first judged configuration below CLEARANCE_M or unseen, -1 if none
    worst_link: str  # link with the smallest clearance

    @property
    def clear(self) -> bool:
        """Whether the path keeps CLEARANCE_M everywhere and stays in observed space."""
        return not self.unseen and self.min_clearance_m >= CLEARANCE_M


def link_samples(model: CollisionModel, joints_deg: np.ndarray, T_map_arm: np.ndarray):
    """Map points along every moving link's capsule axis, for many configurations.

    model: The planner's arm model. joints_deg: (S, 6) configurations, degrees.
    T_map_arm: From map_from_arm.
    Returns: (points (S, L, K, 3) map meters, radii (L,), link names (L,)) for the L moving links.
    """
    moving = np.flatnonzero(model.movable_links & (model.capsule_radius > 0))
    T = (link_transforms(model, np.deg2rad(np.atleast_2d(joints_deg))) @ model.capsule_local)[:, moving]
    k = int(np.ceil(model.capsule_height[moving].max() / JUDGE_STEP_M)) + 1
    t = np.linspace(-0.5, 0.5, k)[None, None, :, None] * model.capsule_height[moving][None, :, None, None]
    arm = T[:, :, None, :3, 3] + t * T[:, :, None, :3, 2]
    points = arm @ T_map_arm[:3, :3].T + T_map_arm[:3, 3]
    return points, model.capsule_radius[moving], [model.link_names[i] for i in moving]


def config_clearance(office: OfficeMap, model: CollisionModel, joints_deg: np.ndarray, T_map_arm: np.ndarray):
    """Clearance of configurations from the real map.

    office: The map. model: The planner's arm model. joints_deg: (S, 6) degrees. T_map_arm: From map_from_arm.
    Returns: (clearance (S, L) meters, NaN where a sample touches unobserved space; link names (L,)).
    """
    points, radii, names = link_samples(model, joints_deg, T_map_arm)
    distance = office.query(points.reshape(-1, 3)).reshape(points.shape[:-1])
    unseen = np.isnan(distance).any(axis=2)
    clearance = np.nanmin(np.where(np.isnan(distance), np.inf, distance), axis=2) - radii[None]
    return np.where(unseen, np.nan, clearance), names


def mesh_clearance(office: OfficeMap, model: CollisionModel, joints_deg: np.ndarray, T_map_arm: np.ndarray, points) -> tuple[np.ndarray, list[str]]:
    """Clearance of the arm's real collision meshes from the real map, for many configurations.

    office: The map. model: Arm model whose link frames pose the points (a run's calibration).
    joints_deg: (S, 6) degrees. T_map_arm: From map_from_arm. points: mesh_check.MeshPoints.
    Returns: (clearance (S, K) meters, NaN where a point touches unobserved space; mesh link names (K,)).
    """
    T = link_transforms(model, np.deg2rad(np.atleast_2d(joints_deg)))
    out = np.full((len(T), len(points.link_names)), np.inf)
    with np.errstate(all="ignore"):  # spurious macOS Accelerate matmul warnings
        for k, ancestor in enumerate(points.ancestors):
            local = points.points[points.link_index == k][::MESH_POINT_STRIDE]
            frame = T_map_arm[None] @ T[:, model.link_names.index(ancestor)]  # (S, 4, 4) map from ancestor
            world = np.einsum("sij,pj->spi", frame[:, :3, :3], local) + frame[:, None, :3, 3]
            distance = office.query(world.reshape(-1, 3)).reshape(world.shape[:2])
            out[:, k] = np.where(np.isnan(distance).any(axis=1), np.nan, np.nanmin(distance, axis=1))
    return out, list(points.link_names)


def judge_path(office: OfficeMap, model: CollisionModel, waypoints_deg: np.ndarray, T_map_arm: np.ndarray, points) -> PathVerdict:
    """Judges a joint path as the controller drives it: straight joint lines between the waypoints.

    office: The map. model: The run's arm model. waypoints_deg: (W, 6) degrees, start first.
    T_map_arm: From map_from_arm. points: mesh_check.MeshPoints (the arm's real link meshes).
    Returns: The verdict.
    """
    path = sample_joint_path(np.atleast_2d(waypoints_deg), PATH_STEP_DEG)
    clearance, names = mesh_clearance(office, model, path, T_map_arm, points)
    unseen = np.isnan(clearance).any(axis=1)
    known = np.where(np.isnan(clearance), np.inf, clearance)
    bad = np.flatnonzero(unseen | (known.min(axis=1) < CLEARANCE_M))
    link = int(np.unravel_index(np.argmin(known), known.shape)[1])
    return PathVerdict(float(known.min()), bool(unseen.any()), int(bad[0]) if len(bad) else -1, names[link])
