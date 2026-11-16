"""The planner's crop as spatial_awareness holds it before its free zones, rebuilt from a robot folder's files.

The mapper's cells (floating specks dropped, holes filled), what spatial_awareness's export shows occupied beyond
them (lidar returns, AutoXing walls) and the lidar's free band (taken from the export, outside the base's shadow).
The zones (above the base, the base's shadow, near the arm) are returned apart: they are presumed, not seen.
Checked against the export itself: the same unknown share in the arm's reach on billie-23 and billie-28 (2026-11-10).
"""

import gzip
import json

import numpy as np
from billie_utils.esdf_grid import decode_esdf_grid
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from spatial_awareness import occupancy_field as of

from Simulation.mapping_sim import scan_coverage as sc

# [m] The arm's reach from its base: the xArm6's 0.7 m and the gripper.
REACH_M = 0.9
# [m] "In front of Billie" is ahead of its centre, in Billie's frame (the user, 2026-11-11: ahead of the arm base,
# 18 cm behind the centre, took in cells beside the base's rear half).
FRONT_FROM_X_M = 0.0


def crop_states(folder: str) -> dict:
    """The crop's cells and their places for one robot folder.

    folder: The robot folder (occupancy.bin.gz, arm_esdf.bin.gz, robot.json, arm.urdf).
    Returns: {"states": (X, Y, Z) int8 as occupancy_field has them (1 occupied, -1 free, 0 unknown),
        "shape", "pts": (N, 3) cell centres in the map, "loc": (N, 3) in Billie's frame (x forward, y left, z above
        the floor), "zones": name -> (N,) bool, "reach": (N,) bool within REACH_M of the arm base and not over
        Billie's base, "front": (N,) bool the reach ahead of Billie's centre, "robot": robot.json, "grid", "lo": the
        mapper's grid (scan_coverage.load_grid)}.
    """
    robot = json.load(open(f"{folder}/robot.json"))
    base = robot["base"]
    turn = Rotation.from_euler("z", base["theta_rad"]).as_matrix()
    export = decode_esdf_grid(gzip.decompress(open(f"{folder}/arm_esdf.bin.gz", "rb").read()))
    shape = export.distances_m.shape
    pts = np.asarray(export.origin_m) + np.argwhere(np.ones(shape, bool)) * export.voxel_size_m
    grid, lo = sc.load_grid(f"{folder}/occupancy.bin.gz")
    raw = sc.real_states(grid, lo, 0.025, pts)
    states = np.select([raw == "occupied", raw == "free"], [1, -1], 0).astype(np.int8).reshape(shape)
    of.drop_floating(states)
    of.fill_holes(states)
    distance = export.distances_m.reshape(-1)
    loc = np.c_[(pts[:, :2] - [base["x_m"], base["y_m"]]) @ turn[:2, :2], pts[:, 2] - sc.FLOOR_M]
    edge = np.hypot(np.maximum(np.abs(loc[:, 0]) - of.BASE_HALF_LENGTH_M, 0),
                    np.maximum(np.abs(loc[:, 1]) - of.BASE_HALF_SIDE_M, 0))  # fmt: skip
    flat = states.reshape(-1)
    flat[(distance <= 0) & (flat == 0)] = 1  # lidar returns, walls
    lidar_top = of.LIDAR_HEIGHT_M + of.LIDAR_FREE_HALF_BAND_M + 0.0125
    flat[(flat == 0) & (distance > 0) & (edge > of.BASE_SHADOW_M) & (loc[:, 2] <= lidar_top)] = -1  # the lidar's rays
    arm = np.array([robot["arm_to_base"][0] / 1000, robot["arm_to_base"][1] / 1000, robot["arm_base_z_m"] - sc.FLOOR_M])
    body = sc.RobotBody(f"{folder}/arm.urdf", robot)
    arm_links = tuple(name for name in body.points if name.startswith("link") and name != "link_base")
    near_arm = cKDTree(body.at(robot["joints_deg"], arm_links)[::8]).query(pts, distance_upper_bound=of.ARM_SEEN_M)[0]
    zones = {
        "above the base": (edge <= of.CLEAR_MARGIN_M + 1e-9) & (loc[:, 2] <= of.ABOVE_BASE_FREE_TOP_M),
        "base's shadow": (edge <= of.BASE_SHADOW_M) & (loc[:, 2] > 0.0125) & (loc[:, 2] <= of.BASE_SHADOW_TOP_M),
        "near the arm": near_arm < of.ARM_SEEN_M,
    }
    on_base = (np.abs(loc[:, 0]) <= of.BASE_HALF_LENGTH_M) & (np.abs(loc[:, 1]) <= of.BASE_HALF_SIDE_M)
    reach = (np.linalg.norm(loc - arm, axis=1) <= REACH_M) & ~on_base & (loc[:, 2] > 0.05)
    return {"states": states, "shape": shape, "pts": pts, "loc": loc, "zones": zones, "reach": reach,
            "front": reach & (loc[:, 0] > FRONT_FROM_X_M), "robot": robot, "grid": grid, "lo": lo}  # fmt: skip
