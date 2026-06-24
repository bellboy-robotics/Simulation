import numpy as np
from scipy.spatial.transform import Rotation as R
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from Simulation.ik_sim.dataset_reader import DatasetReader


# def _estimate_pose_offset(dataset: LeRobotDataset, arm, n_samples: int = 3) -> np.ndarray:
#     ...


def get_pos_from_recording(repo_id):
    """Load poses and joint angles from a recording. Joint angles are untouched."""
    reader = DatasetReader(repo_id, episode=0)
    expected_poses, joint_angles = _generate_poses(reader.dataset, reader)
    return expected_poses, joint_angles


def get_random_diff(
    xy_range_mm: float = 200.0,
    angle_range_deg: float = 30.0,
    no_transform: int = 1
):
    """
    Sample random base-position shift parameters.

    xy_range_mm:      max shift in X and Y (mm)
    angle_range_deg:  max rotation (degrees)
    no_transform:     number of random transforms to sample
    returns:          (dx_mm, dy_mm, dtheta_deg) arrays of shape (no_transform,)
    """
    dx_mm      = np.random.uniform(-xy_range_mm,     xy_range_mm,     no_transform)
    dy_mm      = np.random.uniform(-xy_range_mm,     xy_range_mm,     no_transform)
    dtheta_deg = np.random.uniform(-angle_range_deg, angle_range_deg, no_transform)
    return dx_mm, dy_mm, dtheta_deg


def apply_base_transform(
    poses: list,
    dx_mm: float,
    dy_mm: float,
    dtheta_deg: float,
) -> list:
    """
    Transform poses as if the robot base moved by (dx_mm, dy_mm, dtheta_deg) from
    its recorded position.  Equivalent to billie-onboard map-mode transform
    (transformation.py) without the arm-to-base calibration offset.

    poses:      list of pose6d [x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad]
    returns:    list of transformed pose6d arrays (same format, joint_angles unchanged)
    """
    dtheta_rad = np.deg2rad(dtheta_deg)

    # New base position in the map frame (old base = map origin = identity).
    T_new = np.eye(4)
    T_new[:2, 3] = [dx_mm, dy_mm]
    T_new[:2, :2] = R.from_rotvec([0, 0, dtheta_rad]).as_matrix()[:2, :2]

    T_new_from_old = np.linalg.inv(T_new)

    transformed = []
    for pose in poses:
        pose = np.asarray(pose, dtype=float)
        T_in = np.eye(4)
        T_in[:3, :3] = R.from_rotvec(pose[3:6]).as_matrix()
        T_in[:3, 3] = pose[:3]

        T_out = T_new_from_old @ T_in
        new_pose = np.concatenate([T_out[:3, 3], R.from_matrix(T_out[:3, :3]).as_rotvec()])
        transformed.append(new_pose)

    return transformed


def _generate_poses(
    dataset: LeRobotDataset,
    reader: DatasetReader,
) -> list:
    poses = []
    joint_angles = []
    for frame in dataset:
        poses.append(frame["observation.xarm_pose"].numpy())
        joint_angles.append(frame["observation.xarm_joints"].numpy())
    return poses, joint_angles


def gripper_reader(frame, robot_type):
    if robot_type == robot_type._BILLIE_V2:
        return frame["observation.state"][6]


if __name__ == "__main__":
    repo_id = 'bellboy-robotics/B-unknown-20260301-180408-BILLIE-12'
    expected_poses, joint_angles = get_pos_from_recording(repo_id)

    # random transform
    dx, dy, dtheta = get_random_diff(xy_range_mm=200, angle_range_deg=30)
    transformed_poses = apply_base_transform(expected_poses, dx, dy, dtheta)

    # explicit transform
    transformed_poses2 = apply_base_transform(expected_poses, dx_mm=50, dy_mm=-30, dtheta_deg=15)
