"""A 3D page of the active-mapping scan: pick a snap (a camera stop) and see the unknown cells its view covers.

    python -m Simulation.mapping_sim.snap_viewer --robot output/robot_logs/billie-28/2026-11-08/scan \
        --out output/scan_sim/billie-28/snap_viewer.html

Runs scan_coverage's simulation (with the height fix) and keeps, for each still-unknown cell of the real map, what each
snap did to it: outside the snap's view, in view with a clear line (the sim sees it; the real map does not), behind
the gripper mask, hidden behind something, or past the far-pixel meter. Everything is shown in Billie's frame (x
forward, y left, z up from the floor) on a 5 cm grid, with the real map's occupied cells and each snap's camera.
"""

import argparse
import base64
import json
import os

import numpy as np

from Simulation.mapping_sim import scan_coverage as sc

# The page the data is put into (its `/*DATA*/null` placeholder).
_TEMPLATE = os.path.join(os.path.dirname(__file__), "snap_viewer.html")
# Shown cells are every this many studied cells along each axis (2.5 cm -> 5 cm; a 2.5 cm page is 8x heavier).
THIN = 2
# [m] How far each snap's view pyramid is drawn from its camera.
FRUSTUM_M = 0.5
# [m] A recorded camera position this near a snap's camera is one of its frames.
SNAP_M = 0.02
# What a snap did to an unknown cell, as stored (index): its scan_coverage class, outside its view first.
SNAP_CLASSES = ("never in view", "free", "behind the gripper mask", "hidden behind something", "past the far-pixel meter")


def _b64(array: np.ndarray) -> str:
    """An array's bytes as base64 (the page decodes them with its known dtype). array: Any. Returns: The text."""
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def billie_frame(robot: dict) -> np.ndarray:
    """Billie's frame in the map: x forward, y left, z up, on the floor. robot: robot.json. Returns: (4, 4)."""
    base = robot["base"]
    c, s = np.cos(base["theta_rad"]), np.sin(base["theta_rad"])
    frame = np.eye(4)
    frame[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    frame[:3, 3] = [base["x_m"], base["y_m"], sc.FLOOR_M]
    return frame


def snap_shapes(poses: list[np.ndarray], to_billie: np.ndarray) -> list[dict]:
    """Each snap's camera and view pyramid in Billie's frame.

    poses: (4, 4) each snap's camera in the map. to_billie: (4, 4) map -> Billie's frame.
    Returns: Per snap {"at": [x, y, z], "corners": 4 x [x, y, z] FRUSTUM_M along the view at the image corners}.
    """
    pixels = [(0, 0), (sc.ZED_W, 0), (sc.ZED_W, sc.ZED_H), (0, sc.ZED_H)]
    rays = np.array([[(u - sc.ZED_CX) / sc.ZED_FX, (v - sc.ZED_CY) / sc.ZED_FY, 1.0] for u, v in pixels])
    shapes = []
    for pose in poses:
        camera = to_billie @ pose
        corners = camera[:3, 3] + (rays * FRUSTUM_M) @ camera[:3, :3].T
        shapes.append({"at": camera[:3, 3].round(3).tolist(), "corners": corners.round(3).tolist()})
    return shapes


def snap_frames(poses: list[np.ndarray], track: list[dict]) -> list[int]:
    """How many mapper frames each snap took (recorded camera positions within SNAP_M of it).

    poses: Each snap's camera in the map. track: trajectory.json. Returns: The count per snap.
    """
    positions = np.array([[frame["x"], frame["y"], frame["z"]] for frame in track])
    return [int(np.count_nonzero(np.linalg.norm(positions - pose[:3, 3], axis=1) < SNAP_M)) for pose in poses]


def _snap_codes(per_view: list[np.ndarray], cells: np.ndarray) -> list[str]:
    """Each snap's SNAP_CLASSES index for the shown cells, base64 uint8.

    per_view: scan_coverage's per snap class (all studied cells). cells: The shown cells' indices into them.
    Returns: One text per snap.
    """
    remap = {sc.CLASSES.index(name): i for i, name in enumerate(SNAP_CLASSES)}
    remap[sc.CLASSES.index("occupied")] = SNAP_CLASSES.index("free")  # seen either way
    remap[sc.CLASSES.index("above the height cut")] = SNAP_CLASSES.index("free")  # cleared by the height fix
    remap[sc.CLASSES.index("behind Billie's own body")] = SNAP_CLASSES.index("hidden behind something")
    remap[sc.CLASSES.index("no stereo depth")] = SNAP_CLASSES.index("hidden behind something")
    table = np.zeros(len(sc.CLASSES), np.uint8)
    for old, new in remap.items():
        table[old] = new
    return [_b64(table[view[cells]]) for view in per_view]


def build_data(robot_dir: str) -> dict:
    """The page's data: the shown cells, the occupied ones, each snap's codes, cameras and frame counts.

    robot_dir: The robot folder (scan_coverage's --robot). Returns: The JSON-ready dict.
    """
    sc.use_camera(robot_dir)
    robot = json.load(open(os.path.join(robot_dir, "robot.json")))
    grid, lo = sc.load_grid(os.path.join(robot_dir, "occupancy.bin.gz"))
    world = sc.World(grid, lo, 0.025, robot)
    camera = sc.camera_model(os.path.join(robot_dir, "arm.urdf"), robot)
    views = sc.scan_views()
    poses = [camera(view) for view in views] + [np.asarray(robot["T_map_from_camera"])]
    names = [f"View {i}" for i in range(len(views))] + ["Rest"]
    centers, shape = sc.area_cells(world, robot)
    body = sc.RobotBody(os.path.join(robot_dir, "arm.urdf"), robot)
    joints = [*views, tuple(robot["joints_deg"])]
    per_view = sc.run_rule([(n, [(p, sc.render_at(world, body, p, q))]) for n, p, q in zip(names, poses, joints)],
                           centers, sc.gripper_keep(), True)  # fmt: skip
    real = sc.real_states(grid, lo, world.voxel, centers)
    thin = np.zeros(shape, bool)
    thin[::THIN, ::THIN, ::THIN] = True
    thin = thin.reshape(-1) & ~world.on_base(centers)
    unknown = np.flatnonzero(thin & np.isin(real, ("none", "unknown")))
    occupied = np.flatnonzero(thin & (real == "occupied"))
    to_billie = np.linalg.inv(billie_frame(robot))

    def local_cm(cells: np.ndarray) -> str:
        """The cells' centers in Billie's frame, cm, base64 int16 (x, y, z per cell)."""
        points = centers[cells] @ to_billie[:3, :3].T + to_billie[:3, 3]
        return _b64(np.rint(points * 100).astype("<i2"))

    track = json.load(open(os.path.join(robot_dir, "trajectory.json")))
    return {
        "robot": os.path.basename(os.path.normpath(os.path.join(robot_dir, "../.."))),
        "cell_cm": round(world.voxel * THIN * 100, 1),
        "base_half_m": [sc.BASE_HALF_LENGTH_M, sc.BASE_HALF_SIDE_M, robot["arm_base_z_m"] - sc.FLOOR_M],
        "classes": list(SNAP_CLASSES),
        "unknown": local_cm(unknown),
        "occupied": local_cm(occupied),
        "snaps": [
            {"name": name, "joints": [round(j, 2) for j in joints], "frames": frames, "codes": codes, **shape_}
            for name, joints, frames, codes, shape_ in zip(
                names, [*views, robot["joints_deg"]], snap_frames(poses, track), _snap_codes(per_view, unknown),
                snap_shapes(poses, to_billie),
            )
        ],
    }


def main() -> None:
    """Builds the data and writes the page."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--robot", required=True, help="folder with occupancy.bin.gz, robot.json, arm.urdf, trajectory.json")
    parser.add_argument("--out", required=True, help="the HTML page to write")
    args = parser.parse_args()
    data = build_data(args.robot)
    page = open(_TEMPLATE).read().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write(page)
    print(f"{args.out}: {len(page) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
