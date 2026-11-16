"""Which space around Billie the active-mapping scan sees, and why the rest stays unknown.

    python -m Simulation.mapping_sim.scan_coverage --robot output/robot_logs/billie-28/scan --out output/scan_sim

Replays the scan's camera views (active_mapping.SCAN_VIEWS_DEG, then where the arm rests) in the room the robot's
mapper mapped. Each view's depth is rendered from the mapped occupied cells, the floor, a ceiling and Billie's base,
through the ZED's intrinsics; the mapper's rules then decide what each cell learns (the gripper mask, the height cut at
MAX_MAP_Z_M, the depth range, the far and robot pixels), as nvblox does: each cell's center projected to one pixel.
Every cell around the arm ends seen (free or occupied) or with why not. The sim is compared with the real map: what
the sim sees but the real map left unknown is what the rules here do not explain.

The robot folder holds the mapper's live files: occupancy.bin.gz, robot.json (the arm at rest after the scan, its
camera pose and calibration) and arm.urdf (the planner's /tmp/urdf/<SN>-with-tcp.urdf).
"""

import argparse
import ast
import gzip
import importlib.util
import json
import os
import struct

import numpy as np
import trimesh
import yourdfpy
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial.transform import Rotation

# The billie-onboard checkout (the scan views and the mapper's gripper mask are read from it, so they stay in sync).
_ONBOARD = os.path.join(os.path.dirname(__file__), "../../../billie-onboard/billie")
_ACTIVE_MAPPING = os.path.join(_ONBOARD, "nodes/brain/brain/behaviours/active_mapping.py")
_GRIPPER_MASK = os.path.join(_ONBOARD, "nodes/nvblox_mapper/nvblox_mapper/gripper_mask.py")

# [px] billie-28's ZED X left camera at HD1200 (its factory calibration, /usr/local/zed/settings); use_camera takes
# a robot's own when its folder has CAMERA_FILE.
ZED_FX, ZED_FY, ZED_CX, ZED_CY, ZED_W, ZED_H = 737.6, 737.2, 965.7, 632.2, 1920, 1200
# The robot folder's left-camera calibration: the [LEFT_CAM_FHD1200] section of its ZED settings file (fx=..., ...).
CAMERA_FILE = "zed_left_fhd1200.txt"
# The rendered image is this much smaller: a 2.5 cm cell still spans ~1.5 px at 3 m.
DOWNSCALE = 4
# [m] The mapper's depth range, height cut and free-ray rules (nvblox_mapper integration_config, occupancy_backend).
MIN_DEPTH_M, MAX_DEPTH_M, MAX_MAP_Z_M = 0.10, 5.0, 2.2
# [m] A far pixel (nothing within the depth) clears its whole ray up to the depth (the user, 2026-11-08; 1 m before).
FAR_FREE_M, SELF_FREE_BEFORE_M, HIGH_FREE_MAX_M = MAX_DEPTH_M, 0.05, 2.5
# [m] Half the band around a measured depth that nvblox marks occupied (OccupancySensorModel).
OCCUPIED_HALF_M = 0.025
# [m] The floor in billie-28's map (where the occupied cells peak near 0), and an assumed ceiling 2.6 m above it:
# nothing above MAX_MAP_Z_M is ever kept, so the map cannot tell.
FLOOR_M, CEILING_M = -0.11, 2.49
# [m] Billie's base (spatial_awareness BASE_HALF_LENGTH_M, BASE_HALF_SIDE_M); its top is the arm base's height.
BASE_HALF_LENGTH_M, BASE_HALF_SIDE_M = 0.365, 0.240
# [m] The studied space: this far around the arm base (about the planner's crop), from just above the floor to 2 m.
AREA_HALF_M, AREA_BOTTOM_M, AREA_TOP_M = 1.4, 0.05, 2.0
# [m] The ray-march step (half a cell).
STEP_M = 0.0125
# Places sampled along each move between stops, to match the frames taken while moving.
MOVE_SAMPLES = 41
# [m] A recorded camera position this near a sampled place is taken as that place.
MATCH_M = 0.015
# Where the URDF's collision meshes are found on this machine (the robot's paths do not exist here): the planner's base
# meshes and a robot's arm and gripper meshes (billie-07's capture; every xArm6 1305 and Rome gripper share them).
MESH_ROOTS = (os.path.join(_ONBOARD, "nodes/pyroki-planner/pyroki_planner/base-urdf/meshes"),
              os.path.join(os.path.dirname(__file__), "../../../output/world_sim/urdf/billie-07/meshes"))  # fmt: skip
# [m] Billie's body is sampled this densely (half a cell) and drawn this much thicker: the mapper masks its own
# spheres with a margin (dynamic_runtime: 0.08 m around spheres that already cover the body loosely).
BODY_SAMPLE_M, BODY_MARGIN_CELLS = 0.0125, 0
# [m] Half the box around the arm base that holds Billie's body at any joints (reach 0.9 m + the base's length).
BODY_BOX_HALF_M = 1.3
# Rays marched at once (memory: rays x steps x 3 floats).
RAY_CHUNK = 4096

# What each cell ends as, in drawing order; the first two are seen.
CLASSES = ("free", "occupied", "above the height cut", "behind the gripper mask", "no stereo depth",
           "hidden behind something", "behind Billie's own body", "past the far-pixel meter", "never in view",
           "over Billie's base")  # fmt: skip
# The classes' colors in the slice pictures.
COLORS = ((255, 255, 255), (200, 30, 30), (150, 60, 200), (240, 150, 30), (40, 170, 160), (90, 120, 170),
          (120, 90, 60), (230, 210, 60), (60, 60, 60), (150, 220, 150))  # fmt: skip
# [m] The ZED X's baseline: its right camera sits this far along the left camera's x. A point the right camera cannot
# see has no stereo depth (the band beside a near object), so its pixel gives the mapper nothing.
ZED_BASELINE_M = 0.12
# render's kind for a pixel whose hit the right camera cannot see.
NO_STEREO = 5
# What the real map's cells draw as: no block, never seen, free, occupied.
REAL_COLORS = {"none": (60, 60, 60), "unknown": (160, 160, 160), "free": (255, 255, 255), "occupied": (200, 30, 30)}


def load_grid(path: str) -> tuple[np.ndarray, np.ndarray]:
    """The mapper's occupancy snapshot as one dense grid.

    path: occupancy.bin.gz (occupancy_stream: 8^3 blocks of int8 log odds x10, 0 never seen).
    Returns: (grid (X, Y, Z) int8, -128 where no block was ever made; its first cell's index (3,) in the map,
    cell i at (i + lo + 0.5) * voxel).
    """
    raw = gzip.decompress(open(path, "rb").read())
    size = struct.unpack("<I", raw[8:12])[0]
    records = np.frombuffer(raw[12 + size :], np.uint8).reshape(-1, 12 + 512)
    index = records[:, :12].copy().view("<i4").reshape(-1, 3)
    values = records[:, 12:].copy().view(np.int8).reshape(-1, 8, 8, 8)
    lo, hi = index.min(0) * 8, (index.max(0) + 1) * 8
    grid = np.full(tuple(hi - lo), -128, np.int8)
    for block, cells in zip(index * 8 - lo, values):
        grid[block[0] : block[0] + 8, block[1] : block[1] + 8, block[2] : block[2] + 8] = cells
    return grid, lo


def scan_views() -> list[tuple[float, ...]]:
    """The scan's views from active_mapping.py. The brain is not installed here, so its SCAN_* settings and the _view
    helper that builds the views are taken from the source and run alone.

    Returns: Each view's six joints, degrees.
    """
    tree = ast.parse(open(_ACTIVE_MAPPING).read())
    wanted = []
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        names = [getattr(target, "id", "") for target in targets]
        if any(name.startswith("SCAN_") for name in names) or getattr(node, "name", "") == "_view":
            wanted.append(node)
    scope: dict = {}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), _ACTIVE_MAPPING, "exec"), scope)  # noqa: S102
    return [tuple(float(joint) for joint in view) for view in scope["SCAN_VIEWS_DEG"]]


def camera_model(urdf_path: str, robot: dict):
    """The camera in the map for any arm joints, as camera_autoxing_bridge chains it: the AutoXing base, the arm base
    on it, the xArm TCP (from the URDF), the hand-eye calibration (taken from the robot's pose at rest).

    urdf_path: The planner's URDF. robot: robot.json (base, joints_deg, arm_to_base, arm_base_z_m, T_map_from_camera).
    Returns: joints (6,) degrees -> (4, 4) the camera in the map, looking along +z.
    """
    urdf = yourdfpy.URDF.load(urdf_path, load_meshes=False, build_collision_scene_graph=False,
                              load_collision_meshes=False)  # fmt: skip

    def tcp(joints: tuple[float, ...]) -> np.ndarray:
        """The TCP in the arm base. joints: Degrees. Returns: (4, 4)."""
        urdf.update_cfg(np.radians(joints))
        return urdf.get_transform("link_tcp", "link_base")

    base = robot["base"]
    map_from_arm = np.eye(4)
    map_from_arm[:3, :3] = Rotation.from_euler("z", base["theta_rad"]).as_matrix()
    map_from_arm[:3, 3] = [base["x_m"], base["y_m"], 0.0]
    arm_on_base = np.eye(4)
    arm_on_base[:3, :3] = Rotation.from_euler("z", robot["arm_to_base"][2], degrees=True).as_matrix()
    arm_on_base[:3, 3] = [robot["arm_to_base"][0] / 1000.0, robot["arm_to_base"][1] / 1000.0, robot["arm_base_z_m"]]
    map_from_arm = map_from_arm @ arm_on_base
    tcp_from_camera = np.linalg.inv(map_from_arm @ tcp(robot["joints_deg"])) @ np.asarray(robot["T_map_from_camera"])
    return lambda joints: map_from_arm @ tcp(joints) @ tcp_from_camera


def move_poses(camera, stops: list[tuple[float, ...]], track: list[dict]) -> tuple[list[np.ndarray], int]:
    """The camera poses of the frames the mapper took while the arm moved between the scan's stops: each frame's
    recorded camera position matched on the straight joint move between the stops it lies between (the planner's
    "direct" moves); positions at a stop are left out (the stops are simulated themselves).

    camera: camera_model's. stops: The joints of the stops in order (the views, then the rest). track: The mapper's
    trajectory.json (each integrated frame's camera position).
    Returns: (one pose per distinct matched place, the frames no move matched within MATCH_M).
    """
    samples = [(i, s) for i in range(len(stops) - 1) for s in np.linspace(0.0, 1.0, MOVE_SAMPLES)]
    poses = [camera(np.asarray(stops[i]) + s * (np.asarray(stops[i + 1]) - np.asarray(stops[i]))) for i, s in samples]
    places = np.array([pose[:3, 3] for pose in poses])
    segment, picked, unmatched = 0, set(), 0
    for frame in track:
        position = np.array([frame["x"], frame["y"], frame["z"]])
        near = [k for k, (i, _) in enumerate(samples) if segment <= i <= segment + 1]
        distances = np.linalg.norm(places[near] - position, axis=1)
        best = near[int(np.argmin(distances))]
        if distances.min() > MATCH_M:
            unmatched += 1
            continue
        segment = samples[best][0]
        if 0.0 < samples[best][1] < 1.0:
            picked.add(best)
    return [poses[k] for k in sorted(picked)], unmatched


def use_camera(robot_dir: str) -> None:
    """Takes the robot's own ZED intrinsics when its folder has them (CAMERA_FILE); else billie-28's stay.

    robot_dir: The robot folder.
    """
    global ZED_FX, ZED_FY, ZED_CX, ZED_CY
    path = os.path.join(robot_dir, CAMERA_FILE)
    if not os.path.exists(path):
        return
    values = dict(line.strip().split("=", 1) for line in open(path) if "=" in line)
    ZED_FX, ZED_FY, ZED_CX, ZED_CY = (float(values[key]) for key in ("fx", "fy", "cx", "cy"))


def gripper_keep() -> np.ndarray:
    """The mapper's gripper mask (billie-gripper-body, the live mapper's since 2026-11-08) at the rendered size, from
    nvblox_mapper.

    Returns: (H, W) bool, True where a pixel is used.
    """
    spec = importlib.util.spec_from_file_location("gripper_mask", _GRIPPER_MASK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    keep = module.apply_gripper_policy(np.ones((ZED_H, ZED_W), np.uint8), "billie-gripper-body", 0.0)
    return keep[DOWNSCALE // 2 :: DOWNSCALE, DOWNSCALE // 2 :: DOWNSCALE].astype(bool)


def _mesh_file(name: str) -> str | None:
    """A URDF mesh path's file on this machine: the same last folders under one of MESH_ROOTS, else the same name.

    name: The URDF's filename (the robot's path). Returns: The local path, or None when it is not here.
    """
    parts = name.replace("\\", "/").split("/")
    for keep in (3, 2, 1):
        for root in MESH_ROOTS:
            path = os.path.join(root, *parts[-keep:])
            if os.path.exists(path):
                return path
    return None


class RobotBody:
    """Billie's own body, base and arm, as surface points per link (the URDF's collision meshes): what the camera sees
    of the robot itself at any joints. The mapper masks those pixels, so the space behind the body stays unseen."""

    def __init__(self, urdf_path: str, robot: dict):
        """urdf_path: The planner's URDF. robot: robot.json (the base pose, the arm on it)."""
        self.urdf = yourdfpy.URDF.load(urdf_path, load_meshes=False, build_collision_scene_graph=False,
                                       load_collision_meshes=False)  # fmt: skip
        base = robot["base"]
        self.map_from_arm = np.eye(4)
        self.map_from_arm[:3, :3] = Rotation.from_euler("z", base["theta_rad"]).as_matrix()
        self.map_from_arm[:3, 3] = [base["x_m"], base["y_m"], 0.0]
        on_base = np.eye(4)
        on_base[:3, :3] = Rotation.from_euler("z", robot["arm_to_base"][2], degrees=True).as_matrix()
        on_base[:3, 3] = [robot["arm_to_base"][0] / 1000.0, robot["arm_to_base"][1] / 1000.0, robot["arm_base_z_m"]]
        self.map_from_arm = self.map_from_arm @ on_base
        self.points: dict[str, np.ndarray] = {}  # link -> (N, 3) surface points, link frame
        self.missing: list[str] = []  # links whose mesh is not on this machine
        for link in self.urdf.robot.links:
            for collision in link.collisions:
                mesh = collision.geometry.mesh
                path = _mesh_file(mesh.filename) if mesh is not None else None
                if path is None:
                    self.missing.append(link.name)
                    continue
                shape = trimesh.load(path, force="mesh")
                count = int(np.clip(shape.area * float(np.prod(mesh.scale if mesh.scale is not None else 1.0)) ** (2 / 3)
                                    / BODY_SAMPLE_M**2, 2000, 400000))  # fmt: skip
                points = np.asarray(trimesh.sample.sample_surface(shape, count, seed=0)[0])
                points = points * (mesh.scale if mesh.scale is not None else 1.0)
                origin = collision.origin if collision.origin is not None else np.eye(4)
                self.points[link.name] = points @ origin[:3, :3].T + origin[:3, 3]

    def at(self, joints_deg, links: tuple[str, ...] | None = None) -> np.ndarray:
        """The body's surface points in the map at arm joints.

        joints_deg: (6,) degrees. links: Only these links (None: all). Returns: (N, 3) map frame.
        """
        self.urdf.update_cfg(np.radians(joints_deg))
        out = []
        for name, points in self.points.items():
            if links is not None and name not in links:
                continue
            frame = self.map_from_arm @ self.urdf.get_transform(name, "link_base")
            out.append(points @ frame[:3, :3].T + frame[:3, 3])
        return np.vstack(out)


class World:
    """What the camera's rays hit: the mapped occupied cells, the floor, the ceiling and Billie's own body (its base's
    box, and with pose_body the arm and base as the URDF has them)."""

    def __init__(self, grid: np.ndarray, lo: np.ndarray, voxel: float, robot: dict):
        """grid, lo: load_grid's. voxel: The cell size, m. robot: robot.json (the base pose, the arm base height)."""
        self.solid, self.lo, self.voxel = grid > 0, lo, voxel
        self.body: np.ndarray | None = None  # cells of Billie's body at the posed joints (a box around the arm base)
        self.body_lo = np.zeros(3, int)
        base = robot["base"]
        self.base_xy = np.array([base["x_m"], base["y_m"]])
        self.base_axes = Rotation.from_euler("z", base["theta_rad"]).as_matrix()[:2, :2]
        self.base_top = robot["arm_base_z_m"]

    def on_base(self, points: np.ndarray) -> np.ndarray:
        """Which map points are over Billie's base footprint. points: (..., 2+) map frame. Returns: (...) bool."""
        local = (points[..., :2] - self.base_xy) @ self.base_axes
        return (np.abs(local[..., 0]) <= BASE_HALF_LENGTH_M) & (np.abs(local[..., 1]) <= BASE_HALF_SIDE_M)

    def pose_body(self, body: "RobotBody | None", joints_deg) -> None:
        """Puts Billie's body at arm joints for the next renders (None: only the base's box).

        body: The robot's body, or None. joints_deg: (6,) degrees.
        """
        if body is None:
            self.body = None
            return
        points = body.at(joints_deg)
        half = int(BODY_BOX_HALF_M / self.voxel)
        self.body_lo = np.floor(body.map_from_arm[:3, 3] / self.voxel).astype(int) - half
        cells = np.floor(points / self.voxel).astype(int) - self.body_lo
        cells = cells[np.all((cells >= 0) & (cells < 2 * half), axis=1)]
        grid = np.zeros((2 * half,) * 3, bool)
        grid[tuple(cells.T)] = True
        self.body = ndimage.binary_dilation(grid, iterations=BODY_MARGIN_CELLS) if BODY_MARGIN_CELLS else grid

    def hit_kind(self, points: np.ndarray) -> np.ndarray:
        """What is at each point: 0 nothing, 1 a mapped surface, 2 the floor, 3 the ceiling, 4 Billie's own body.

        points: (..., 3) map frame. Returns: (...) int8.
        """
        kind = np.zeros(points.shape[:-1], np.int8)
        cells = np.floor(points / self.voxel).astype(int) - self.lo
        inside = np.all((cells >= 0) & (cells < self.solid.shape), axis=-1)
        kind[inside] = np.where(self.solid[tuple(cells[inside].T)], 1, 0)
        kind[points[..., 2] <= FLOOR_M] = 2
        kind[points[..., 2] >= CEILING_M] = 3
        kind[self.on_base(points) & (points[..., 2] <= self.base_top) & (points[..., 2] > FLOOR_M)] = 4
        if self.body is not None:
            own = np.floor(points / self.voxel).astype(int) - self.body_lo
            within = np.all((own >= 0) & (own < self.body.shape), axis=-1)
            kind[within] = np.where(self.body[tuple(own[within].T)], 4, kind[within])
        return kind


def render_at(world: World, body: "RobotBody | None", pose: np.ndarray, joints_deg,
              stereo: bool = False) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """render() with Billie's body at the view's joints (the arm in front of its own camera hides what is behind it).

    world: The room. body: The robot's body, or None (only the base's box). pose: (4, 4) the camera in the map.
    joints_deg: (6,) the view's joints, degrees.
    stereo: Also drop the pixels the right camera cannot see (stereo_seen). Off by default: the ZED's neural depth
        fills most of them (billie-28, 2026-11-08: 88% of such cells were mapped; billie-23, 2026-11-10: 40%).
    Returns: As render, the dropped pixels' kind NO_STEREO.
    """
    world.pose_body(body, joints_deg)
    depth, kind = render(world, pose)
    if stereo:
        kind[~stereo_seen(world, pose, depth)] = NO_STEREO
    return depth, kind


def stereo_seen(world: World, pose: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Which pixels' hits the ZED's right camera sees too (stereo depth needs both): a clear line from the right
    camera to the hit, or to MAX_DEPTH_M along the ray for a pixel that hit nothing.

    world: The room (Billie's body posed as for the render). pose: (4, 4) the left camera in the map.
    depth: render's depth. Returns: (H, W) bool.
    """
    v, u = np.mgrid[DOWNSCALE // 2 : ZED_H : DOWNSCALE, DOWNSCALE // 2 : ZED_W : DOWNSCALE]
    rays = np.stack([(u - ZED_CX) / ZED_FX, (v - ZED_CY) / ZED_FY, np.ones_like(u, float)], -1).reshape(-1, 3)
    reach = np.where(np.isfinite(depth), depth, MAX_DEPTH_M).reshape(-1)
    hits = pose[:3, 3] + (rays * reach[:, None]) @ pose[:3, :3].T
    right = pose[:3, 3] + pose[:3, 0] * ZED_BASELINE_M
    seen = np.ones(len(rays), bool)
    samples = int(MAX_DEPTH_M / STEP_M)
    for start in range(0, len(rays), RAY_CHUNK):
        line = hits[start : start + RAY_CHUNK] - right
        length = np.linalg.norm(line, axis=1)
        # up to 3 cells before the hit: the hit's own surface is not in the way
        last = np.clip(1.0 - 3 * world.voxel / np.maximum(length, 1e-6), 0.0, 1.0)
        part = np.linspace(0.0, 1.0, samples)[None, :] * last[:, None]
        near_enough = part * length[:, None] >= MIN_DEPTH_M
        kinds = world.hit_kind(right + line[:, None, :] * part[..., None])
        seen[start : start + RAY_CHUNK] = ~((kinds > 0) & near_enough).any(axis=1)
    return seen.reshape(depth.shape)


def render(world: World, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One view's depth image: the first thing each pixel's ray hits within MAX_DEPTH_M.

    world: The room. pose: (4, 4) the camera in the map.
    Returns: (depth (H, W) m along the camera's z, inf when nothing is hit; what it hit (H, W), World.hit_kind's codes).
    """
    h, w = ZED_H // DOWNSCALE, ZED_W // DOWNSCALE
    v, u = np.mgrid[DOWNSCALE // 2 : ZED_H : DOWNSCALE, DOWNSCALE // 2 : ZED_W : DOWNSCALE]
    rays = np.stack([(u - ZED_CX) / ZED_FX, (v - ZED_CY) / ZED_FY, np.ones_like(u, float)], -1).reshape(-1, 3)
    directions = rays @ pose[:3, :3].T  # map frame, one meter of camera z each
    depth = np.full(len(rays), np.inf)
    kind = np.zeros(len(rays), np.int8)
    for start in range(0, len(rays), RAY_CHUNK):
        part = directions[start : start + RAY_CHUNK]
        step = STEP_M / np.linalg.norm(part, axis=1)  # camera-z step per ray
        t = MIN_DEPTH_M + np.arange(int(MAX_DEPTH_M / STEP_M) + 1)[None, :] * step[:, None]
        points = pose[:3, 3] + part[:, None, :] * t[..., None]
        kinds = world.hit_kind(points)
        kinds[t > MAX_DEPTH_M] = 0
        first = np.argmax(kinds > 0, axis=1)
        hit = kinds[np.arange(len(part)), first] > 0
        depth[start : start + RAY_CHUNK][hit] = t[hit, first[hit]]
        kind[start : start + RAY_CHUNK][hit] = kinds[hit, first[hit]]
    return depth.reshape(h, w), kind.reshape(h, w)


def pixel_rules(depth: np.ndarray, kind: np.ndarray, pose: np.ndarray, high_rule: bool) -> dict[str, np.ndarray]:
    """What the mapper takes from each pixel (occupancy_backend, nvblox): how far its ray is free, and its hit.

    depth, kind: render's. pose: (4, 4) the camera. high_rule: Whether a pixel hitting above MAX_MAP_Z_M clears its
    ray up to that height (the 2026-11-08 fix); without it the pixel gives nothing.
    Returns: {"free": (H, W) m of free ray, "occupied": (H, W) bool (its hit is marked), "high": (H, W) bool (its hit
    is above the cut), "cross": (H, W) m along z where the ray rises past the cut}.
    """
    v, u = np.mgrid[DOWNSCALE // 2 : ZED_H : DOWNSCALE, DOWNSCALE // 2 : ZED_W : DOWNSCALE]
    rise = ((u - ZED_CX) / ZED_FX) * pose[2, 0] + ((v - ZED_CY) / ZED_FY) * pose[2, 1] + pose[2, 2]
    below = MAX_MAP_Z_M - pose[2, 3]
    cross = np.where(rise > 1e-6, below / np.maximum(rise, 1e-6), np.inf)
    lost = kind == NO_STEREO  # no stereo depth: the pixel gives nothing
    measured = np.isfinite(depth) & ~lost
    high = measured & (depth * rise > below)
    robot = measured & (kind == 4)
    normal = measured & ~high & ~robot
    free = np.where(normal, depth - OCCUPIED_HALF_M, 0.0)
    free = np.where(~np.isfinite(depth) & ~lost, FAR_FREE_M, free)
    free = np.where(robot, depth - SELF_FREE_BEFORE_M, free)
    if high_rule:
        free = np.where(high, np.minimum(np.minimum(cross, depth - SELF_FREE_BEFORE_M), HIGH_FREE_MAX_M), free)
    return {"free": np.maximum(free, 0.0), "occupied": normal, "high": high, "cross": cross, "lost": lost,
            "robot": robot}  # fmt: skip


def view_cells(centers: np.ndarray, pose: np.ndarray, depth: np.ndarray, rules: dict, keep: np.ndarray) -> np.ndarray:
    """What one view does to each cell, nvblox style (the cell's center projected to one pixel).

    centers: (N, 3) cell centers, map frame. pose: (4, 4) the camera. depth: render's. rules: pixel_rules'.
    keep: gripper_keep's.
    Returns: (N,) int8 index into CLASSES (0 free, 1 occupied, 2.. why not seen), never-in-view for cells outside.
    """
    local = (centers - pose[:3, 3]) @ pose[:3, :3]
    z = local[:, 2]
    u = np.round((local[:, 0] / np.maximum(z, 1e-6) * ZED_FX + ZED_CX - DOWNSCALE // 2) / DOWNSCALE).astype(int)
    v = np.round((local[:, 1] / np.maximum(z, 1e-6) * ZED_FY + ZED_CY - DOWNSCALE // 2) / DOWNSCALE).astype(int)
    h, w = depth.shape
    inside = (z > MIN_DEPTH_M) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    out = np.full(len(centers), CLASSES.index("never in view"), np.int8)
    pu, pv, zi = u[inside], v[inside], z[inside]
    hit = depth[pv, pu]
    result = np.full(len(zi), CLASSES.index("hidden behind something"), np.int8)
    result[rules["robot"][pv, pu]] = CLASSES.index("behind Billie's own body")
    result[~np.isfinite(hit) & (zi > rules["free"][pv, pu])] = CLASSES.index("past the far-pixel meter")
    result[rules["high"][pv, pu] & (zi < rules["cross"][pv, pu])] = CLASSES.index("above the height cut")
    result[rules["occupied"][pv, pu] & (np.abs(zi - hit) <= OCCUPIED_HALF_M)] = CLASSES.index("occupied")
    result[zi < rules["free"][pv, pu]] = CLASSES.index("free")
    lost = rules["lost"][pv, pu] & (zi < np.where(np.isfinite(hit), hit, MAX_DEPTH_M))
    result[lost] = CLASSES.index("no stereo depth")
    result[~keep[pv, pu]] = CLASSES.index("behind the gripper mask")
    out[inside] = result
    return out


def combine(per_view: list[np.ndarray]) -> np.ndarray:
    """Each cell's end state over the views: seen if any view saw it (occupied wins, as later views add up), else the
    first reason by CLASSES' order (the reason a better view would remove first).

    per_view: view_cells' answers. Returns: (N,) int8 index into CLASSES.
    """
    stacked = np.stack(per_view)
    seen = (stacked <= 1).any(0)
    occupied = (stacked == 1).any(0)
    reason = stacked.min(0)
    return np.where(seen, np.where(occupied, 1, 0), reason).astype(np.int8)


def area_cells(world: World, robot: dict) -> tuple[np.ndarray, tuple[int, int, int]]:
    """The studied cells: AREA_HALF_M around the arm base, AREA_BOTTOM_M to AREA_TOP_M above the floor, on the
    map's cell grid.

    world: The room (its cell grid). robot: robot.json (the base pose, the arm on it).
    Returns: (centers (N, 3) map frame, the box's shape (X, Y, Z), N = X * Y * Z).
    """
    arm = np.array([robot["base"]["x_m"], robot["base"]["y_m"]])
    arm += Rotation.from_euler("z", robot["base"]["theta_rad"]).as_matrix()[:2, :2] @ (
        np.array(robot["arm_to_base"][:2]) / 1000.0
    )
    v = world.voxel
    first = np.floor((np.array([*arm - AREA_HALF_M, FLOOR_M + AREA_BOTTOM_M])) / v).astype(int)
    last = np.floor((np.array([*arm + AREA_HALF_M, FLOOR_M + AREA_TOP_M])) / v).astype(int)
    axes = [(np.arange(a, b + 1) + 0.5) * v for a, b in zip(first, last)]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    return centers.reshape(-1, 3), centers.shape[:3]


def real_states(grid: np.ndarray, lo: np.ndarray, voxel: float, centers: np.ndarray) -> np.ndarray:
    """The real map at the studied cells. grid, lo: load_grid's. voxel: m. centers: (N, 3).

    Returns: (N,) str: "none" (no block), "unknown" (never seen), "free" or "occupied".
    """
    cells = np.floor(centers / voxel).astype(int) - lo
    inside = np.all((cells >= 0) & (cells < grid.shape), axis=1)
    values = np.full(len(centers), -128, np.int16)
    values[inside] = grid[tuple(cells[inside].T)]
    return np.select([values == -128, values == 0, values < 0], ["none", "unknown", "free"], "occupied")


def report(names: list[str], per_view: list[np.ndarray], final: np.ndarray, real: np.ndarray, base: np.ndarray) -> str:
    """The text summary: what each view adds, why the rest is unseen, and the sim against the real map.

    names: The views' names. per_view, final: view_cells' and combine's (one rule). real: real_states'.
    base: (N,) bool, the cells over Billie's base (always free in spatial_awareness; left out of the shares).
    Returns: The text.
    """
    studied = ~base
    lines, seen_so_far = [f"{studied.sum()} cells studied (+{base.sum()} over Billie's base, always free)"], None
    for name, cells in zip(names, per_view):
        seen = (cells <= 1) & studied
        seen_so_far = seen if seen_so_far is None else seen_so_far | seen
        lines.append(f"  {name:<46} sees {seen.sum() / studied.sum():5.1%}, all views so far {seen_so_far.mean() / studied.mean():5.1%}")
    lines.append("end state:")
    for i, name in enumerate(CLASSES[:-1]):
        lines.append(f"  {name:<26} {np.count_nonzero((final == i) & studied) / studied.sum():6.1%}")
    sim_seen, real_seen = final <= 1, np.isin(real, ("free", "occupied"))
    lines.append("against the real map (studied cells):")
    lines.append(f"  real map knows {real_seen[studied].mean():5.1%}, the sim {sim_seen[studied].mean():5.1%}")
    gap = sim_seen & ~real_seen & studied
    lines.append(f"  sim seen, real unknown: {gap.sum() / studied.sum():5.1%} of the space "
                 f"({np.count_nonzero(gap & (real == 'none'))} with no block at all)")  # fmt: skip
    lines.append("  each sim class, as the real map has it (none / unknown / free / occupied):")
    for i, name in enumerate(CLASSES[:-1]):
        cells = (final == i) & studied
        if cells.any():
            shares = " / ".join(f"{np.count_nonzero(cells & (real == s)) / cells.sum():4.0%}" for s in REAL_COLORS)
            lines.append(f"    {name:<26} {shares}")
    lines.append(f"  real known, sim unseen: {np.count_nonzero(real_seen & ~sim_seen & studied) / studied.sum():5.1%}")
    return "\n".join(lines)


def draw(final: np.ndarray, real: np.ndarray, shape: tuple[int, int, int], heights: list[int], path: str) -> None:
    """Top-down slices: the sim's end state (top row) above the real map (bottom row), at a few heights.

    final: combine's, with over-the-base cells set to its class. real: real_states'. shape: The box (X, Y, Z).
    heights: The slices' z indices. path: The PNG to write.
    """
    scale, pad = 3, 8
    tile = (shape[0] * scale, shape[1] * scale)
    out = Image.new("RGB", ((tile[0] + pad) * len(heights), (tile[1] + pad) * 2 + 18 * len(CLASSES)), "white")
    sim_rgb = np.array(COLORS, np.uint8)[final.reshape(shape)]
    real_rgb = np.array([REAL_COLORS[s] for s in real], np.uint8).reshape(*shape, 3)
    for col, k in enumerate(heights):
        for row, rgb in enumerate((sim_rgb, real_rgb)):
            image = Image.fromarray(np.flipud(rgb[:, :, k].transpose(1, 0, 2))).resize(tile, Image.NEAREST)
            draw_on = ImageDraw.Draw(image)
            draw_on.rectangle([0, 0, 120, 13], fill="white")
            draw_on.text((2, 1), f"{'sim' if row == 0 else 'real'} {(k * 0.025 + AREA_BOTTOM_M):.2f} m up", fill="black")
            out.paste(image, (col * (tile[0] + pad), row * (tile[1] + pad)))
    legend = ImageDraw.Draw(out)
    for i, (name, color) in enumerate(zip(CLASSES, COLORS)):
        y = (tile[1] + pad) * 2 + 18 * i
        legend.rectangle([4, y + 2, 18, y + 14], fill=color, outline="black")
        legend.text((24, y + 2), name, fill="black")
    out.save(path)


def run_rule(cases: list, centers: np.ndarray, keep: np.ndarray, high_rule: bool) -> list[np.ndarray]:
    """Every case's cells under one height rule; a case of several poses (the moves) counts as one view.

    cases: (name, [(pose, (depth, kind)), ...]). centers: area_cells'. keep: gripper_keep's.
    high_rule: As pixel_rules'. Returns: view_cells' answer per case.
    """
    return [combine([view_cells(centers, pose, depth, pixel_rules(depth, kind, pose, high_rule), keep)
                     for pose, (depth, kind) in shots]) for _, shots in cases]  # fmt: skip


def main() -> None:
    """Runs the scan in the robot's mapped room with the old and the new height rule; writes the report and pictures."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--robot", required=True, help="folder with occupancy.bin.gz, robot.json, arm.urdf, trajectory.json")
    parser.add_argument("--out", required=True, help="folder for report.txt and the slice pictures")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    use_camera(args.robot)
    robot = json.load(open(os.path.join(args.robot, "robot.json")))
    grid, lo = load_grid(os.path.join(args.robot, "occupancy.bin.gz"))
    world = World(grid, lo, 0.025, robot)
    camera = camera_model(os.path.join(args.robot, "arm.urdf"), robot)
    views = scan_views()
    track = json.load(open(os.path.join(args.robot, "trajectory.json")))
    moving, unmatched = move_poses(camera, [*views, tuple(robot["joints_deg"])], track)
    stills = [camera(view) for view in views] + [np.asarray(robot["T_map_from_camera"])]
    joints = [*views, tuple(robot["joints_deg"])]
    body = RobotBody(os.path.join(args.robot, "arm.urdf"), robot)
    if body.missing:
        print(f"no mesh here for {body.missing}: those links hide nothing in the sim")
    names = [f"view {i}" for i in range(len(views))] + ["rest", f"moves ({len(moving)} places; {unmatched} frames unmatched)"]
    cases = [(name, [(pose, render_at(world, body, pose, q))]) for name, pose, q in zip(names, stills, joints)]
    world.pose_body(None, None)
    if moving:
        cases.append((names[-1], [(pose, render(world, pose)) for pose in moving]))
    names = [name for name, _ in cases]
    centers, shape = area_cells(world, robot)
    base = world.on_base(centers)
    real = real_states(grid, lo, world.voxel, centers)
    keep, text = gripper_keep(), []
    for label, high_rule in (("before the height fix", False), ("with the height fix", True)):
        per_view = run_rule(cases, centers, keep, high_rule)
        final = combine(per_view)
        text.append(f"== {label} ==\n" + report(names, per_view, final, real, base))
        final[base] = CLASSES.index("over Billie's base")
        heights = [int(round((m - AREA_BOTTOM_M) / world.voxel)) for m in (0.3, 0.7, 1.1, 1.5, 1.9)]
        draw(final, real, shape, heights, os.path.join(args.out, f"slices_{'fix' if high_rule else 'old'}.png"))
    open(os.path.join(args.out, "report.txt"), "w").write("\n\n".join(text) + "\n")
    print("\n\n".join(text))


if __name__ == "__main__":
    main()
