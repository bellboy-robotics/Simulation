from lerobot.datasets.lerobot_dataset import LeRobotDataset
from Simulation.dataset_reader import DatasetReader


# def _estimate_pose_offset(dataset: LeRobotDataset, arm, n_samples: int = 3) -> np.ndarray:
#     """
#     Estimate the systematic XYZ offset between arm-reported TCP position and dataset poses.
#     Sets recorded joint angles, reads back position, and averages the diff over n_samples frames.
#     """
#     frames = list(dataset)
#     indices = np.random.choice(len(frames), min(n_samples, len(frames)), replace=False)
#     offsets = []
#     for i in indices:
#         frame = frames[i]
#         joints = frame["observation.xarm_joints"].numpy()
#         recorded_pos = frame["observation.xarm_pose"].numpy()[:3]
#         xarm_code, reported = uf_sim.set_joint_angles(arm, joints)
#         if xarm_code == 0 and reported is not None:
#             offsets.append(recorded_pos - np.array(reported[:3]))
#     offset = np.mean(offsets, axis=0) if offsets else np.zeros(3)
#     print(f"Estimated pose offset (mm): {' '.join(f'{v:.3f}' for v in offset)}")
#     return offset


def get_pos_from_recording(repo_id):
    'read dataset'
    reader = DatasetReader(repo_id, episode=0)
    expected_poses, joint_angles = _generate_poses(
        reader.dataset,
        reader
    )
    return expected_poses, joint_angles


def _generate_poses(
    dataset: LeRobotDataset,
    reader: DatasetReader,
    ) -> list:
    poses = []
    joint_angles = []
    for f_i, frame in enumerate(dataset):
        'debug: print frame keys'
        # if f_i>=100 and f_i<110:
            # width = max(len(k) for k in frame.keys())
            # for key in frame.keys():
            #     print(f"{key:{width}}  {frame[key]}")

        pose = frame["observation.xarm_pose"].numpy()
        poses.append(pose)
        joint_angles.append(frame["observation.xarm_joints"].numpy())

    return poses, joint_angles

def gripper_reader(frame, robot_type):
    if robot_type == robot_type._BILLIE_V2:
        return frame["observation.state"][6]

if __name__ == "__main__":
    repo_id = 'bellboy-robotics/B-unknown-20260301-180408-BILLIE-12'
    expected_poses = get_pos_from_recording(repo_id)
