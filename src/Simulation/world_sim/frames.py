"""Map <-> xArm base frame for a Billie base pose, with no planner imports (used on both planner branches)."""

import numpy as np

# [m] Arm base height above the floor (pyroki_world.DEFAULT_ARM_BASE_HEIGHT_M, which only one branch has).
DEFAULT_ARM_BASE_HEIGHT_M = 0.4405


def planar_transform(x: float, y: float, yaw_rad: float, z: float = 0.0) -> np.ndarray:
    """4x4 transform for a level frame: a yaw rotation plus a translation (pyroki_world.planar_transform).

    x, y, z: Translation, meters. yaw_rad: Rotation about Z. Returns: 4x4 matrix, child -> parent.
    """
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = [x, y, z]
    return T


def map_from_arm(billie: dict, arm_base_height_m: float = DEFAULT_ARM_BASE_HEIGHT_M) -> np.ndarray:
    """4x4 transform from the xArm base frame to the map (z above the floor), for a scenario's Billie.

    billie: Scenario "billie", the sim's format (world.object_in_arm_frame): {"pose": [x_mm, y_mm, yaw_deg]
        of the robot in the map, "arm_to_base": [x_mm, y_mm, yaw_deg] (ARM_TO_BASE_CALIBRATION)}.
    arm_base_height_m: Arm base height above the floor, meters.
    Returns: 4x4 matrix taking arm-frame points (meters) into the map.
    """
    x, y, yaw = billie["pose"]
    ax, ay, ayaw = billie["arm_to_base"]
    return planar_transform(x / 1000, y / 1000, np.deg2rad(yaw)) @ planar_transform(ax / 1000, ay / 1000, np.deg2rad(ayaw), arm_base_height_m)
