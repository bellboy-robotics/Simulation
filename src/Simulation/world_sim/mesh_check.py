"""True-geometry clearance: the URDF's collision meshes sampled into points and measured against capsules.

The planner, its guard and link_distances see each link as one capsule, which can leave parts of the
real link outside it (the camera bracket behind the flange, the flange itself, or the whole gripper
when the planner fell back to its gripper box). This module measures the real meshes instead:

- every movable link's collision mesh in the merged URDF is sampled once into surface points (thinned
  vertices plus area-weighted samples), expressed in the frame of the link's nearest actuated ancestor
  (gripper points in the link6 frame) and cached in output/world_sim/mesh_points_<urdf>.npz;
- link_mesh_distances poses those points with ANY CollisionModel's ancestor frames, so a robot run's
  arm calibration is used even when its gripper link frame is the planner's box fallback.
"""

import os
from dataclasses import dataclass

import numpy as np
import trimesh
import yourdfpy
from billie_utils.world_collision_check_poc import CollisionModel, link_transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "output", "world_sim"))
# Bumped when the sampling below changes, so older cache files are rebuilt.
_CACHE_VERSION = 1
# [m] Target spacing of the surface points. 8mm keeps the deepest point within ~1-2mm of a 30k-sample
# reference on the gripper (pose #51 of the table run: -23.0 vs -23.7mm) at ~3.9k points.
POINT_SPACING_M = 0.008
# Points per link (vertices + samples): small links still get a usable cloud, the gripper stays bounded.
_MIN_SAMPLES, _MAX_SAMPLES = 300, 4000
# Fixed seed: the same URDF always gives the same points, so numbers repeat between sessions.
_SEED = 0
# [m] Distances are exact below this; further away a link reports a lower bound from its bounding sphere
# (skips the per-point work for links far from every obstacle). 0.1m is 5x the planner's 20mm margin.
EXACT_WITHIN_M = 0.1
# Point x capsule pairs per distance chunk (~32MB of float64 per temporary array).
_CHUNK_PAIRS = 4_000_000

_MEMORY: dict[tuple[str, float], "MeshPoints"] = {}  # (urdf path, mtime) -> points, for the live preview


@dataclass
class MeshPoints:
    """Surface points of every movable link's collision mesh, K links, P points."""

    link_names: list[str]  # (K,) movable links that have a collision mesh
    ancestors: list[str]  # (K,) each link's nearest actuated ancestor link (itself for link1..link6)
    link_in_ancestor: np.ndarray  # (K, 4, 4) link frame in its ancestor frame, from the URDF's fixed joints
    points: np.ndarray  # (P, 3) meters, in the ancestor frame of the point's link
    link_index: np.ndarray  # (P,) index into link_names
    is_vertex: np.ndarray  # (P,) True for (thinned) mesh vertices, False for surface samples


def urdf_path() -> str:
    """The planner's merged URDF (base + arm + gripper + TCP), generated if this machine has none yet.

    Returns: Path of /tmp/urdf/<XARM_SN>-with-tcp.urdf; its root link is the xArm base frame.
    """
    path = f"/tmp/urdf/{os.environ['XARM_SN']}-with-tcp.urdf"
    if not os.path.exists(path):
        from pyroki_planner.urdf import load_urdf  # noqa: PLC0415

        load_urdf()
    return path


def _actuated_ancestors(urdf: yourdfpy.URDF) -> dict[str, tuple[str, np.ndarray]]:
    """Each movable link's nearest actuated ancestor, walking up the URDF's fixed joints.

    urdf: The parsed URDF.
    Returns: {link name: (ancestor link name, (4, 4) link frame in the ancestor frame)}; links that reach
        the root through fixed joints only (the robot base) are left out.
    """
    by_child = {j.child: j for j in urdf.robot.joints}
    out = {}
    for link in urdf.robot.links:
        name, T = link.name, np.eye(4)
        while name in by_child and by_child[name].type == "fixed":
            joint = by_child[name]
            T = (joint.origin if joint.origin is not None else np.eye(4)) @ T
            name = joint.parent
        if name in by_child:  # stopped at an actuated joint: `name` is the ancestor
            out[link.name] = (name, T)
    return out


def _link_points(link: yourdfpy.Link, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Surface points of one link's collision meshes, in the link frame.

    link: The URDF link. rng: Draws the surface samples.
    Returns: (points (P, 3) meters, is_vertex (P,) bool); empty when the link has no mesh collision.
    """
    points, is_vertex = [np.zeros((0, 3))], [np.zeros(0, dtype=bool)]
    for collision in link.collisions:
        if collision.geometry.mesh is None:
            continue
        mesh = trimesh.load(collision.geometry.mesh.filename, force="mesh")
        if collision.geometry.mesh.scale is not None:
            mesh.apply_scale(collision.geometry.mesh.scale)
        if collision.origin is not None:
            mesh.apply_transform(collision.origin)
        # One vertex per spacing-sized voxel keeps the corners without thousands of points.
        _, keep = np.unique(np.floor(mesh.vertices / POINT_SPACING_M).astype(np.int64), axis=0, return_index=True)
        vertices = np.asarray(mesh.vertices)[np.sort(keep)]
        n = int(np.clip(mesh.area / POINT_SPACING_M**2, _MIN_SAMPLES, _MAX_SAMPLES)) - len(vertices)
        samples = trimesh.sample.sample_surface(mesh, max(n, _MIN_SAMPLES // 2), seed=int(rng.integers(2**31)))[0]
        points += [vertices, samples]
        is_vertex += [np.ones(len(vertices), dtype=bool), np.zeros(len(samples), dtype=bool)]
    return np.concatenate(points), np.concatenate(is_vertex)


def build_mesh_points(path: str) -> MeshPoints:
    """Samples every movable link's collision mesh of a URDF (a few seconds).

    path: The merged URDF.
    Returns: The points, each link's in its actuated ancestor's frame.
    """
    urdf = yourdfpy.URDF.load(path, load_collision_meshes=False, build_scene_graph=False)
    ancestors = _actuated_ancestors(urdf)
    rng = np.random.default_rng(_SEED)
    names, parents, offsets, points, index, is_vertex = [], [], [], [], [], []
    for link in urdf.robot.links:
        if link.name not in ancestors:
            continue
        local, vertex = _link_points(link, rng)
        if not len(local):
            continue
        parent, T = ancestors[link.name]
        index.append(np.full(len(local), len(names)))
        names.append(link.name)
        parents.append(parent)
        offsets.append(T)
        with np.errstate(all="ignore"):  # spurious macOS Accelerate matmul warnings
            points.append(local @ T[:3, :3].T + T[:3, 3])
        is_vertex.append(vertex)
    return MeshPoints(names, parents, np.asarray(offsets), np.concatenate(points), np.concatenate(index),
                      np.concatenate(is_vertex))  # fmt: skip


def mesh_points(path: str | None = None) -> MeshPoints:
    """The URDF's mesh points: from memory, else from the disk cache, else sampled (and cached).

    path: The merged URDF; None for this machine's (urdf_path()).
    Returns: The points. The caches are keyed by the URDF's modification time.
    """
    path = path or urdf_path()
    key = (path, os.path.getmtime(path))
    if key in _MEMORY:
        return _MEMORY[key]
    cache = os.path.join(_CACHE_DIR, f"mesh_points_{os.path.splitext(os.path.basename(path))[0]}.npz")
    points = None
    if os.path.exists(cache):
        with np.load(cache) as data:
            if float(data["mtime"]) == key[1] and int(data["version"]) == _CACHE_VERSION:
                points = MeshPoints(data["link_names"].tolist(), data["ancestors"].tolist(), data["link_in_ancestor"],
                                    data["points"], data["link_index"], data["is_vertex"])  # fmt: skip
    if points is None:
        points = build_mesh_points(path)
        os.makedirs(_CACHE_DIR, exist_ok=True)
        np.savez(cache, mtime=key[1], version=_CACHE_VERSION, link_names=np.array(points.link_names),
                 ancestors=np.array(points.ancestors), link_in_ancestor=points.link_in_ancestor,
                 points=points.points, link_index=points.link_index, is_vertex=points.is_vertex)  # fmt: skip
    _MEMORY[key] = points
    return points


def capsule_point_distances(points: np.ndarray, starts: np.ndarray, ends: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Signed distance from each point to the nearest capsule (point-to-segment minus radius), chunked.

    points: (N, 3) meters.
    starts, ends: (M, 3) capsule segment ends, meters (M >= 1). radii: (M,) meters.
    Returns: (N,) meters; negative inside a capsule.
    """
    axis = ends - starts
    length2 = np.maximum(np.sum(axis * axis, axis=1), 1e-18)
    start_axis, start2 = np.sum(starts * axis, axis=1), np.sum(starts * starts, axis=1)
    out = np.empty(len(points))
    step = max(1, _CHUNK_PAIRS // len(radii))
    # The matrix products expand |p - a - t d|^2 so BLAS does the heavy part; macOS Accelerate can raise
    # spurious floating point warnings inside matmul, hence errstate.
    with np.errstate(all="ignore"):
        for begin in range(0, len(points), step):
            p = points[begin : begin + step]
            along = p @ axis.T - start_axis  # (p - a) . d
            rel2 = np.sum(p * p, axis=1)[:, None] - 2.0 * (p @ starts.T) + start2  # |p - a|^2
            t = np.clip(along / length2, 0.0, 1.0)
            dist2 = rel2 - t * (2.0 * along - t * length2)
            out[begin : begin + step] = (np.sqrt(np.maximum(dist2, 0.0)) - radii).min(axis=1)
    return out


def link_mesh_distances(
    model: CollisionModel,
    joints_deg: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    points: MeshPoints | None = None,
) -> np.ndarray:
    """Signed distance from every link's real collision mesh to the nearest capsule, per configuration.

    model: Arm model whose ancestor link frames pose the points (a run's model uses its calibration).
    joints_deg: (S, 6) configurations, degrees.
    starts, ends: (M, 3) obstacle capsule segment ends, xArm base frame, meters. radii: (M,) meters.
    points: The mesh points; None for this machine's URDF (mesh_points()).
    Returns: (S, L) meters over model.link_names, negative inside; +inf for links without mesh points
        (static links, link_tcp) or with no capsules. Values >= EXACT_WITHIN_M are lower bounds.
    """
    joints = np.atleast_2d(np.asarray(joints_deg, dtype=np.float64))
    dist = np.full((len(joints), len(model.link_names)), np.inf)
    if len(radii) == 0 or len(joints) == 0:
        return dist
    points = points or mesh_points()
    T = link_transforms(model, np.deg2rad(joints))
    with np.errstate(all="ignore"):  # spurious macOS Accelerate matmul warnings
        for k, (name, ancestor) in enumerate(zip(points.link_names, points.ancestors)):
            if name not in model.link_names or ancestor not in model.link_names:
                continue
            local = points.points[points.link_index == k]
            frame = T[:, model.link_names.index(ancestor)]  # (S, 4, 4)
            center = (local.min(axis=0) + local.max(axis=0)) / 2.0
            reach = np.linalg.norm(local - center, axis=1).max()  # bounding sphere radius
            bound = capsule_point_distances(frame[:, :3, :3] @ center + frame[:, :3, 3], starts, ends, radii) - reach
            near = np.flatnonzero(bound < EXACT_WITHIN_M)
            step = max(1, _CHUNK_PAIRS // (len(local) * len(radii)))
            for begin in range(0, len(near), step):
                chunk = frame[near[begin : begin + step]]
                world = np.einsum("sij,pj->spi", chunk[:, :3, :3], local) + chunk[:, None, :3, 3]
                exact = capsule_point_distances(world.reshape(-1, 3), starts, ends, radii).reshape(len(chunk), -1)
                bound[near[begin : begin + step]] = exact.min(axis=1)
            dist[:, model.link_names.index(name)] = bound
    return dist


def pose_points(model: CollisionModel, joints_deg: np.ndarray, points: MeshPoints | None = None) -> np.ndarray:
    """The mesh points of one configuration in the xArm base frame, for drawing.

    model: Arm model whose ancestor link frames pose the points. joints_deg: (6,) degrees.
    points: The mesh points; None for this machine's URDF.
    Returns: (P, 3) meters, in points.points order (NaN for links whose ancestor the model lacks).
    """
    points = points or mesh_points()
    T = link_transforms(model, np.deg2rad(np.asarray(joints_deg, dtype=np.float64))[None])[0]
    world = np.full(points.points.shape, np.nan)
    with np.errstate(all="ignore"):  # spurious macOS Accelerate matmul warnings
        for k, ancestor in enumerate(points.ancestors):
            if ancestor in model.link_names:
                frame, mine = T[model.link_names.index(ancestor)], points.link_index == k
                world[mine] = points.points[mine] @ frame[:3, :3].T + frame[:3, 3]
    return world


def mount_differences(model: CollisionModel, points: MeshPoints | None = None) -> dict[str, float]:
    """Links that a model mounts on their ancestor differently from the URDF the meshes come from.

    A robot run whose planner had no gripper mesh mounts link_gripper_and_camera at the box offset
    (pyroki_planner/urdf.py CAMERA_GRIPPER_BOX_OFFSET) instead of the mesh mount.

    model: The arm model to compare. points: The mesh points; None for this machine's URDF.
    Returns: {link name: how far the model's link frame is from the URDF's, mm} for links > 1mm apart.
    """
    points = points or mesh_points()
    T = link_transforms(model, np.zeros((1, 6)))[0]
    out = {}
    for name, ancestor, expected in zip(points.link_names, points.ancestors, points.link_in_ancestor):
        if name not in model.link_names or ancestor not in model.link_names:
            continue
        actual = np.linalg.inv(T[model.link_names.index(ancestor)]) @ T[model.link_names.index(name)]
        # A rotation difference counts as the distance it moves a point 100mm from the frame origin.
        offset = np.linalg.norm(actual[:3, 3] - expected[:3, 3]) + 0.1 * np.linalg.norm(actual[:3, :3] - expected[:3, :3])
        if offset > 1e-3:
            out[name] = float(offset * 1000.0)
    return out
