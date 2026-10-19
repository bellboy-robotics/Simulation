"""Shows the arm planner's room field (the ESDF crop around the arm) as slices in the browser.

    python -m Simulation.world_sim.esdf_view --host bellboy@billie-07.bellboy        # the robot's live occupancy
    python -m Simulation.world_sim.esdf_view --file <occupancy.bin.gz> --at -6.0,-6.7
    python -m Simulation.world_sim.esdf_view --esdf output/world_sim/maps/office-2-20261006/esdf_grid.esdfbin --at 4,-1.8

From an occupancy snapshot the field is built with spatial_awareness's own code (occupancy_field.build_field: holes
filled, distances by the box rule, capped as the planner gets them). Billie's base is not cleared (the snapshot has
no robot heading) and AutoXing's walls are not added. From an .esdfbin the crop is cut as the planner cuts it.
With --host, a local page with an Update button and the history of updates (esdf_server.py); --static, --file
and --esdf write two pages next to the snapshot instead. esdf_3d.html (opened) shows the field in 3D: the room's cells as cubes colored by
height, the 5cm and 10cm keep-out shells and the never-observed cells, each switchable in the legend, and Billie
(base, arm and gripper from the planner's URDF at the robot's joints, from the mapper's robot.json), with the map's
floor height around it.
esdf_slices.html shows horizontal slices (a height slider) and vertical x-z slices (a y slider): the distance to the
room in cm (black = occupied), unknown cells shaded. The arm base is marked on both.
"""

import argparse
import glob
import html
import json
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from billie_utils.esdf_grid import load_esdf_grid
from billie_utils.messages.pyroki_world import (
    DEFAULT_ARM_BASE_HEIGHT_M,
    planar_transform,
)

from Simulation.world_sim import mesh_check, occupancy_view, robot

sys.path.insert(0, os.path.join(robot._local_billie_root(), "billie", "nodes", "spatial_awareness"))
from spatial_awareness import occupancy_field

# Crop cells per side: the planner's 2.5cm crop (esdf_world.crop_shape: reach 1.297m + 15cm pad, both sides, + 2).
CROP_CELLS = 118
# Slices drawn: every second cell (5cm apart), so the page stays a few MB.
SLICE_STEP = 2
# The robots' shared mesh files on this machine (a robot env's mesh links point at /billie-env/base/meshes).
_BASE_MESHES = os.path.expanduser("~/releases/env/base/meshes")
# Where per-arm copies of a robot env (its URDF + _BASE_MESHES) are made to generate its whole-Billie URDF.
_ENV_COPIES = "/tmp/billie_env"
# Billie's parts in the 3D view, by URDF link (any other link is the arm; the TCP marker is left out).
_PARTS = {"mobile_base_link": "base", "upper_platform_link": "base", "covering_cylinder_link": "base",
          "backpack_link": "base", "link_gripper_and_camera": "gripper", "link_tcp": None}  # fmt: skip
# [m] Billie's meshes are thinned for drawing by merging vertices this close (the gripper's STL has 235k triangles).
_DRAW_CELL_M = 0.004
_PART_COLORS = {"base": "#9e9e9e", "arm": "#64b5f6", "gripper": "#ff8c00"}
# [m] Floor columns looked at around Billie for the floor height: past the base, within the arm's reach.
_FLOOR_RING_M = (0.5, 1.5)
# Color scale of the distance (cm): black for occupied, then dark to light up to the cap.
_SCALE = [[0.0, "black"], [0.02, "#440154"], [0.35, "#31688e"], [0.7, "#35b779"], [1.0, "#fde725"]]


def field_from_occupancy(path: str, at_xyz: np.ndarray, capped: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """The planner's crop built from an occupancy snapshot around a point (spatial_awareness's build).

    path: The occupancy.bin.gz. at_xyz: (3,) the arm base, map frame. capped: Cap the distances as the planner does.
    Returns: (distances (n, n, n) m, unknown (n, n, n) bool, origin (3,) of cell 0's center, voxel m, the build note).
    """
    header, indices, values = occupancy_view.read_blocks(path)
    voxel = float(header["voxel_size_m"])
    area = occupancy_field.area_around(at_xyz, (CROP_CELLS,) * 3, voxel)
    field = occupancy_field.build_field(area, indices, values, None, None, None, float(at_xyz[2]),
                                        occupancy_field.CAP_M if capped else None)  # fmt: skip
    return field.values, field.unknown, field.origin_m, voxel, field.note


def field_from_esdf(path: str, at_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """The crop of a baked ESDF around a point, on the map's own cells (as the planner's pack_field cuts it).

    path: The .esdfbin. at_xyz: (3,) the arm base, map frame.
    Returns: As field_from_occupancy (distances NaN where unknown).
    """
    grid = load_esdf_grid(Path(path))
    cells = int(np.ceil(CROP_CELLS * 0.025 / grid.voxel_size_m))
    start = np.round((at_xyz - grid.origin_m) / grid.voxel_size_m).astype(int) - cells // 2
    crop = np.full((cells,) * 3, np.nan, np.float32)
    lo, hi = np.maximum(start, 0), np.minimum(start + cells, grid.dimensions)
    if np.all(hi > lo):
        crop[tuple(slice(a - s, b - s) for a, b, s in zip(lo, hi, start))] = grid.distances_m[
            tuple(slice(a, b) for a, b in zip(lo, hi))
        ]
    unknown = np.isnan(crop)
    note = f"{Path(path).parent.name}: {grid.voxel_size_m * 100:g}cm cells, crop {unknown.mean():.0%} unknown"
    return crop, unknown, grid.origin_m + start * grid.voxel_size_m, grid.voxel_size_m, note


def exposed_faces(mask: np.ndarray, origin: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """The outer faces of a set of cells, as a triangle mesh (each cell a cube around its center).

    mask: (n, n, n) bool cells. origin: (3,) cell 0's center, meters. voxel: Cell edge, meters.
    Returns: (vertices (V, 3) meters, triangles (T, 3) vertex indices); empty when no cell is set.
    """
    corners, quads, count = [], [], 0
    padded = np.pad(mask, 1)
    for axis in range(3):
        for side in (0, 1):  # the face toward -axis (0) or +axis (1)
            neighbour = np.roll(padded, 1 if side == 0 else -1, axis=axis)[1:-1, 1:-1, 1:-1]
            cells = np.argwhere(mask & ~neighbour)
            if not len(cells):
                continue
            u, w = [a for a in range(3) if a != axis]
            base = cells.copy()
            base[:, axis] += side
            offsets = np.zeros((4, 3), int)
            offsets[[1, 2], u] = 1
            offsets[[2, 3], w] = 1
            quads.append(count + np.arange(len(cells))[:, None] * 4 + np.arange(4)[None])  # 4 corners per face
            corners.append((base[:, None, :] + offsets[None]).reshape(-1, 3))
            count += 4 * len(cells)
    if not corners:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    lattice, inverse = np.unique(np.concatenate(corners), axis=0, return_inverse=True)
    quad = inverse.reshape(-1)[np.concatenate(quads)]
    triangles = np.concatenate([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]])
    return (origin - voxel / 2.0 + lattice * voxel).astype(np.float32), triangles.astype(np.int32)


def figure_3d(values: np.ndarray, unknown: np.ndarray, origin: np.ndarray, voxel: float, at_xyz: np.ndarray,
              title: str) -> go.Figure:  # fmt: skip
    """The field in 3D: the room's cells colored by height, the 5cm and 10cm shells, the never-observed cells.

    values, unknown: The crop (n, n, n) distances (m; NaN unknown in a baked map) and unknown cells.
    origin: (3,) cell 0's center. voxel: Cell edge. at_xyz: (3,) the arm base. title: The figure title.
    Returns: The figure (the shells and the unknown cells start hidden; the legend shows them).
    """
    distance = np.nan_to_num(values, nan=np.inf)
    layers = [("room (occupied cells)", distance <= 1e-6, None, 1.0, True),
              ("within 5cm", distance <= 0.05, "orange", 0.18, "legendonly"),
              ("within 10cm", distance <= 0.10, "gold", 0.10, "legendonly"),
              ("never observed", unknown, "grey", 0.06, "legendonly")]  # fmt: skip
    fig = go.Figure()
    for name, mask, color, opacity, visible in layers:
        vertices, triangles = exposed_faces(mask, origin, voxel)
        if not len(triangles):
            continue
        style = {"intensity": vertices[:, 2], "colorscale": "Viridis", "colorbar": {"title": "z (m)"}} if color is None \
            else {"color": color}  # fmt: skip
        fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=triangles[:, 0],
                                j=triangles[:, 1], k=triangles[:, 2], name=name, showlegend=True, opacity=opacity,
                                visible=visible, flatshading=True, hoverinfo="skip", **style))  # fmt: skip
    fig.add_trace(go.Scatter3d(x=[at_xyz[0]], y=[at_xyz[1]], z=[at_xyz[2]], mode="markers", name="arm base",
                               marker={"color": "red", "size": 6, "symbol": "x"}))  # fmt: skip
    fig.update_layout(title=title, height=850, scene={"aspectmode": "data", "xaxis_title": "x (m)",
                                                      "yaxis_title": "y (m)", "zaxis_title": "z (m)"},
                      legend={"itemsizing": "constant"}, margin={"l": 0, "r": 0, "t": 40, "b": 0})  # fmt: skip
    return fig


def billie_urdf(xarm_sn: str | None) -> tuple[str, str]:
    """The planner's whole-Billie URDF (base + arm + gripper; root = the arm base) for an arm, made once.

    xarm_sn: The robot's arm (robot.json), or None. Its URDF comes from a local env (~/releases/env/*/urdf); without
        one, the simulation's arm is used (every xArm6 has the same links; only the calibration differs).
    Returns: (the URDF's path, which arm it is).
    """
    found = sorted(glob.glob(os.path.expanduser(f"~/releases/env/*/urdf/{xarm_sn}.urdf"))) if xarm_sn else []
    if not found:
        return mesh_check.urdf_path(), f"arm {os.environ['XARM_SN']} (the simulation's; {xarm_sn or 'the robot'}'s not found)"
    path = f"/tmp/urdf/{xarm_sn}-with-tcp.urdf"
    if not os.path.exists(path):  # the planner's loader reads its env at import: a separate process per arm
        # A robot env's meshes link to /billie-env/base/meshes (inside the container); here that is _BASE_MESHES.
        env_dir = os.path.join(_ENV_COPIES, str(xarm_sn))
        os.makedirs(os.path.join(env_dir, "urdf"), exist_ok=True)
        shutil.copy(found[0], os.path.join(env_dir, "urdf", f"{xarm_sn}.urdf"))
        meshes = os.path.join(env_dir, "urdf", "meshes")
        if not os.path.lexists(meshes):
            os.symlink(_BASE_MESHES, meshes)
        env = {**os.environ, "XARM_SN": str(xarm_sn), "BILLIE_ENVDIR": env_dir}
        code = "import Simulation.world_sim.env_setup as e; e.setup_env(); from pyroki_planner.urdf import load_urdf; load_urdf()"
        subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True)
    return path, f"arm {xarm_sn}"


def _thin(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A mesh with its vertices merged on a _DRAW_CELL_M grid (for drawing only; collapsed triangles dropped).

    vertices: (V, 3) meters. faces: (F, 3) vertex indices. Returns: (vertices, faces) of the thinner mesh.
    """
    keys, inverse = np.unique(np.round(vertices / _DRAW_CELL_M).astype(np.int64), axis=0, return_inverse=True)
    merged = np.zeros((len(keys), 3))
    np.add.at(merged, inverse.reshape(-1), vertices)
    merged /= np.bincount(inverse.reshape(-1))[:, None]
    faces = inverse.reshape(-1)[faces]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return merged, np.unique(np.sort(faces[keep], axis=1), axis=0)


def billie_traces(urdf_path: str, joints_deg: list[float], map_from_arm: np.ndarray) -> list:
    """Billie's visual meshes at its joints, placed in the map: the base grey, the arm blue, the gripper orange.

    urdf_path: The whole-Billie URDF (root = the arm base). joints_deg: (6,) the arm's joints, degrees.
    map_from_arm: (4, 4) the arm base in the map. Returns: Mesh3d traces, one per part.
    """
    import yourdfpy

    urdf = yourdfpy.URDF.load(urdf_path)
    urdf.update_cfg(np.deg2rad(np.asarray(joints_deg, dtype=float))[: len(urdf.actuated_joint_names)])
    parts: dict[str, list] = {}
    graph = urdf.scene.graph
    for node in graph.nodes_geometry:
        transform, name = graph[node]
        part = _PARTS.get(graph.transforms.parents.get(node, ""), "arm")
        if part is None:
            continue
        mesh = urdf.scene.geometry[name]
        with np.errstate(all="ignore"):  # spurious macOS Accelerate matmul warnings
            vertices = np.c_[mesh.vertices, np.ones(len(mesh.vertices))] @ (map_from_arm @ transform).T
        parts.setdefault(part, []).append(_thin(vertices[:, :3], np.asarray(mesh.faces)))
    traces = []
    for part, meshes in parts.items():
        offsets = np.cumsum([0] + [len(v) for v, _ in meshes[:-1]])
        vertices = np.concatenate([v for v, _ in meshes]).astype(np.float32)
        faces = np.concatenate([f + o for (_, f), o in zip(meshes, offsets)]).astype(np.int32)
        traces.append(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=faces[:, 0], j=faces[:, 1],
                                k=faces[:, 2], color=_PART_COLORS[part], opacity=1.0, name=f"Billie ({part})",
                                showlegend=True, flatshading=False, hoverinfo="name"))  # fmt: skip
    return traces


def arm_in_map(state: dict, height_m: float | None) -> tuple[np.ndarray, float]:
    """The arm base in the map from the robot's state, as the planner and the camera bridge place it.

    state: robot.json (base pose, arm_to_base [x_mm, y_mm, yaw_deg], arm_base_z_m).
    height_m: The arm base's height above the floor to use, or None for the robot's own setting.
    Returns: ((4, 4) map_from_arm, the height used).
    """
    base = state["base"]
    x_mm, y_mm, yaw_deg = state.get("arm_to_base") or (0.0, 0.0, 0.0)
    height = float(height_m if height_m is not None else state.get("arm_base_z_m") or DEFAULT_ARM_BASE_HEIGHT_M)
    mount = planar_transform(x_mm / 1000.0, y_mm / 1000.0, np.deg2rad(yaw_deg), height)
    return planar_transform(base["x_m"], base["y_m"], base["theta_rad"]) @ mount, height


def floor_near(values: np.ndarray, origin: np.ndarray, voxel: float, base_xy: np.ndarray) -> float | None:
    """The map's floor around Billie: the median top of the occupied cells near z = 0, in a ring past the base.

    values: The crop's distances (0 = occupied). origin: Cell 0's center. voxel: Cell edge. base_xy: (2,) Billie.
    Returns: The floor's top surface z, meters, or None without floor cells there.
    """
    x = origin[0] + np.arange(values.shape[0]) * voxel
    y = origin[1] + np.arange(values.shape[1]) * voxel
    z = origin[2] + np.arange(values.shape[2]) * voxel
    band = (z > -0.3) & (z < 0.15)
    occupied = np.nan_to_num(values[:, :, band], nan=1.0) <= 1e-6
    r = np.hypot(*np.meshgrid(x - base_xy[0], y - base_xy[1], indexing="ij"))
    ring = (r >= _FLOOR_RING_M[0]) & (r <= _FLOOR_RING_M[1]) & occupied.any(axis=2)
    if not ring.any():
        return None
    top = z[band][occupied.shape[2] - 1 - np.argmax(occupied[:, :, ::-1], axis=2)]
    return float(np.median(top[ring]) + voxel / 2.0)


def _slice_traces(distance_cm: np.ndarray, unknown: np.ndarray, x: np.ndarray, y: np.ndarray, top_cm: float) -> list:
    """One slice: the distance heatmap and the unknown cells over it.

    distance_cm: (len(x), len(y)) cm (NaN shows nothing). unknown: Same shape, bool.
    x, y: The slice's axes, meters. top_cm: The color scale's top, cm.
    Returns: [distance heatmap, unknown overlay].
    """
    shade = np.where(unknown, 1.0, np.nan)
    return [
        go.Heatmap(x=x, y=y, z=np.round(distance_cm, 1).T, zmin=0.0, zmax=top_cm, colorscale=_SCALE,
                   colorbar={"title": "cm"}, hovertemplate="%{x:.3f}, %{y:.3f}: %{z} cm<extra></extra>"),
        go.Heatmap(x=x, y=y, z=shade.T, zmin=0.0, zmax=1.0, showscale=False, hoverinfo="skip",
                   colorscale=[[0.0, "rgba(160,160,160,0.55)"], [1.0, "rgba(160,160,160,0.55)"]]),
    ]  # fmt: skip


def slices_figure(values: np.ndarray, unknown: np.ndarray, axes: list[np.ndarray], at_xyz: np.ndarray, normal: int,
                  title: str, top_cm: float) -> go.Figure:  # fmt: skip
    """Slices across one axis with a slider: horizontal (normal 2: x-y at each height) or vertical (normal 1: x-z).

    values, unknown: The crop (n, n, n) distances (m) and unknown cells. axes: Its cell centers along x, y, z (m).
    at_xyz: (3,) the arm base, marked. normal: The axis the slider moves along. title: The figure title.
    top_cm: The color scale's top, cm.
    Returns: The figure.
    """
    shown = [a for a in range(3) if a != normal]
    levels = list(range(0, values.shape[normal], SLICE_STEP))
    start = min(levels, key=lambda i: abs(axes[normal][i] - at_xyz[normal]))
    frames = []
    for i in levels:
        cut = [slice(None)] * 3
        cut[normal] = i
        traces = _slice_traces(values[tuple(cut)] * 100.0, unknown[tuple(cut)], axes[shown[0]], axes[shown[1]], top_cm)
        frames.append(go.Frame(data=traces, name=f"{axes[normal][i]:.3f}"))
    fig = go.Figure(data=frames[levels.index(start)].data, frames=frames)
    fig.add_trace(go.Scatter(x=[at_xyz[shown[0]]], y=[at_xyz[shown[1]]], mode="markers", name="arm base",
                             marker={"color": "red", "size": 10, "symbol": "x"}))  # fmt: skip
    label = "xyz"[normal]
    steps = [{"method": "animate", "label": f.name, "args": [[f.name], {"mode": "immediate", "frame": {"duration": 0}}]}
             for f in frames]  # fmt: skip
    fig.update_layout(
        title=title, height=750, width=800, xaxis_title=f"{'xyz'[shown[0]]} (m)", yaxis_title=f"{'xyz'[shown[1]]} (m)",
        yaxis={"scaleanchor": "x"},
        sliders=[{"active": levels.index(start), "currentvalue": {"prefix": f"{label} = ", "suffix": " m"}, "steps": steps}],
    )  # fmt: skip
    return fig


def _arm_position(args: argparse.Namespace, folder: str, state: dict | None) -> tuple[np.ndarray, np.ndarray | None, float]:
    """The arm base to center on: --at, else the robot's (robot.json), else its last camera position.

    args: The command line. folder: Where the snapshot's trajectory.json would be. state: robot.json, or None.
    Returns: ((3,) the arm base in the map; (4, 4) map_from_arm when the robot's pose is known, else None;
        the height used). Raises: SystemExit without any of them.
    """
    if state is not None and not args.at:
        map_from_arm, height = arm_in_map(state, args.height)
        return map_from_arm[:3, 3], map_from_arm, height
    height = args.height if args.height is not None else DEFAULT_ARM_BASE_HEIGHT_M
    if args.at:
        xy = [float(v) for v in args.at.split(",")][:2]
    else:
        path = os.path.join(folder, "trajectory.json")
        if not os.path.exists(path):
            raise SystemExit("Give --at x,y: there is no robot.json or trajectory.json next to the snapshot")
        with open(path) as f:
            last = json.load(f)[-1]
        xy = [last["x"], last["y"]]
    return np.array([xy[0], xy[1], height]), None, height


def _robot_note(state: dict, height: float, floor_z: float | None) -> str:
    """A line on how Billie stands in the map. state: robot.json. height: Arm base height used. floor_z: Map floor."""
    note = f"arm base {height:.3f}m above z = 0 (the robot's setting {state.get('arm_base_z_m')})"
    if floor_z is not None:
        note += (f"; the map's floor around Billie is at {floor_z * 100:+.1f}cm, so a flat floor at Billie's wheels "
                 f"would mean an arm base height of {height - floor_z:.3f}m")  # fmt: skip
    return note


def tcp_pose(urdf_path: str, joints_deg: list[float]) -> list[float]:
    """The TCP pose of the planner's model at some joints, as the brain's pose command takes it.

    urdf_path: The whole-Billie URDF (root = the arm base). joints_deg: (6,) degrees.
    Returns: [x, y, z mm, rx, ry, rz rotation vector rad] in the arm base frame.
    """
    import yourdfpy
    from scipy.spatial.transform import Rotation

    urdf = yourdfpy.URDF.load(urdf_path)
    urdf.update_cfg(np.deg2rad(np.asarray(joints_deg, dtype=float))[: len(urdf.actuated_joint_names)])
    transform = urdf.get_transform("link_tcp")
    return [*(transform[:3, 3] * 1000.0).tolist(), *Rotation.from_matrix(transform[:3, :3]).as_rotvec().tolist()]


def vectors(state: dict | None, urdf_path: str | None) -> dict:
    """The robot's pose, joints and base as copyable vectors (the brain's command formats).

    state: robot.json, or None. urdf_path: The whole-Billie URDF for a model pose when the robot's is missing.
    Returns: {"pose": "pose:[...]", "pose_from": "the arm" or the model, "joints": "joints:[...]",
        "base": "base:[x_m, y_m, theta_rad]"}; empty without a robot state.
    """
    if state is None:
        return {}
    pose, source = state.get("pose6"), "the arm"
    if not pose and urdf_path:
        pose, source = tcp_pose(urdf_path, state["joints_deg"]), "the planner's model; this mapper sends no TCP pose yet"
    base = state["base"]
    out = {"joints": "joints:" + json.dumps([round(v, 2) for v in state["joints_deg"]]),
           "base": "base:" + json.dumps([round(base["x_m"], 3), round(base["y_m"], 3), round(base["theta_rad"], 4)])}  # fmt: skip
    if pose:
        out["pose"] = "pose:" + json.dumps([round(v, 1) for v in pose[:3]] + [round(v, 4) for v in pose[3:]])
        out["pose_from"] = source
    return out


def build_entry(path: str, folder: str, args: argparse.Namespace) -> tuple[go.Figure, dict]:
    """The views of one snapshot: the 3D figure, the slices page (written in folder) and what they show.

    path: The occupancy.bin.gz (or .esdfbin with --esdf). folder: Where its robot.json is and the slices go.
    args: The command line (--at, --height, --on-floor, --uncapped, --esdf).
    Returns: (the 3D figure, info: {"note", "robot_line", "frames", and the vectors()}).
    """
    state = None
    state_path = os.path.join(folder, "robot.json")
    if os.path.exists(state_path) and not args.esdf:
        with open(state_path) as f:
            state = json.load(f)
    at, map_from_arm, height = _arm_position(args, folder, state)
    if args.esdf:
        values, unknown, origin, voxel, note = field_from_esdf(path, at)
        frames = None
    else:
        values, unknown, origin, voxel, note = field_from_occupancy(path, at, not args.uncapped)
        frames = occupancy_view.read_blocks(path)[0].get("frames_integrated")
    extra, robot_line, urdf_path = [], "Billie not drawn (no robot.json: the mapper on this robot does not write it yet)", None
    if state is not None and map_from_arm is not None:
        floor_z = floor_near(values, origin, voxel, np.array([state["base"]["x_m"], state["base"]["y_m"]]))
        if args.on_floor and floor_z is not None:
            map_from_arm, height = arm_in_map(state, height + floor_z)
        urdf_path, arm = billie_urdf(state.get("xarm_sn"))
        extra = billie_traces(urdf_path, state["joints_deg"], map_from_arm)
        robot_line = f"Billie ({arm}); " + _robot_note(state, height, floor_z)
    name = os.path.basename(folder)
    figure = figure_3d(values, unknown, origin, voxel, at, f"{name}: the room field in 3D")
    figure.add_traces(extra)
    figure.update_layout(uirevision="keep")  # switching maps keeps the view
    axes = [origin[a] + np.arange(values.shape[a]) * voxel for a in range(3)]
    top_cm = 20.0 if not args.uncapped else 50.0
    with open(os.path.join(folder, "esdf_slices.html"), "w") as f:
        f.write(f"<html><head><meta charset='utf-8'><title>{name} room field</title></head><body>"
                f"<p>{note}. Black: occupied. Grey: never observed (the planner counts it as free).</p>")  # fmt: skip
        for normal, kind in ((2, "horizontal"), (1, "vertical x-z")):
            fig = slices_figure(values, unknown, axes, at, normal, f"{name}: {kind} slices", top_cm)
            f.write(fig.to_html(full_html=False, include_plotlyjs="cdn" if normal == 2 else False))
        f.write("</body></html>")
    info = {"note": note, "robot_line": robot_line, "frames": frames, **vectors(state, urdf_path)}
    print(f"arm base at {np.round(at, 2).tolist()}: {note}\n{robot_line}")
    return figure, info


def info_html(info: dict) -> str:
    """The info block of a map: its notes, and its pose / joints / base vectors with copy buttons. Returns: HTML."""
    rows = []
    for key in ("pose", "joints", "base"):
        if info.get(key):
            source = f" <small>(from {info['pose_from']})</small>" if key == "pose" and info.get("pose_from") else ""
            rows.append(f"<div class='vec'><code>{html.escape(info[key])}</code>{source} "
                        f"<button onclick='copyText({json.dumps(info[key])}, this)'>Copy</button></div>")  # fmt: skip
    return (f"<div class='info'>{''.join(rows)}<div class='note'>{html.escape(info['note'])}</div>"
            f"<div class='note'>{html.escape(info['robot_line'])}</div></div>")  # fmt: skip


# The page script and style shared by the static page and the server's (copy buttons, layout).
PAGE_HEAD = """<meta charset='utf-8'><style>
body { font-family: system-ui, sans-serif; margin: 8px; }
.vec { margin: 3px 0; } .vec code { background: #f2f2f2; padding: 2px 4px; font-size: 13px; }
.note { color: #555; font-size: 12px; margin-top: 3px; } button { cursor: pointer; }
</style><script>
function copyText(text, button) {
  const done = () => { const old = button.textContent; button.textContent = "Copied"; setTimeout(() => button.textContent = old, 1200); };
  if (navigator.clipboard) { navigator.clipboard.writeText(text).then(done); return; }
  const area = document.createElement("textarea"); area.value = text; document.body.appendChild(area);
  area.select(); document.execCommand("copy"); area.remove(); done();
}
</script>"""


def main() -> None:
    """Command line: --host serves the page with Update and the history (esdf_server); otherwise one static page."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--host", help="robot ssh destination: its live occupancy, e.g. bellboy@billie-07.bellboy")
    source.add_argument("--file", help="a local occupancy.bin.gz")
    source.add_argument("--esdf", help="a baked .esdfbin map")
    parser.add_argument("--at", help="x,y of the arm base in the map (default: the robot's last camera position)")
    parser.add_argument("--height", type=float, default=None,
                        help="arm base height above the floor (m); default: the robot's setting (robot.json)")
    parser.add_argument("--on-floor", action="store_true", help="draw Billie standing on the map's measured floor")
    parser.add_argument("--uncapped", action="store_true", help="distances without the planner's 20cm cap")
    parser.add_argument("--static", action="store_true", help="with --host: one static page instead of the server")
    parser.add_argument("--port", type=int, default=8765, help="the server's local port (--host)")
    args = parser.parse_args()
    if args.host and not args.static:
        from Simulation.world_sim import esdf_server

        esdf_server.serve(args)
        return
    if args.host:
        folder = os.path.join(robot._REPO, "output", "world_sim", "occupancy", robot.robot_name(args.host))
        path, _ = occupancy_view.fetch(args.host, folder)
    else:
        path = args.file or args.esdf
        folder = os.path.dirname(os.path.abspath(path))
    figure, info = build_entry(path, folder, args)
    page = os.path.join(folder, "esdf_3d.html")
    with open(page, "w") as f:
        f.write(f"<html><head><title>{os.path.basename(folder)} room field</title>{PAGE_HEAD}</head><body>"
                f"{info_html(info)}{figure.to_html(full_html=False, include_plotlyjs='cdn')}</body></html>")  # fmt: skip
    print(f"Wrote {page} and esdf_slices.html")
    webbrowser.open(f"file://{page}")


if __name__ == "__main__":
    main()
