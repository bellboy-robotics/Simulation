
import pybullet as p
import numpy as np

# Headless mode — no GUI, just computation
client = p.connect(p.DIRECT)
urdf_path = "/Users/ronit/Work/config/urdf_with_gripper_and_wheels"
robot = p.loadURDF(urdf_path, useFixedBase=True)

# Find your end-effector link index (usually the last link)
num_joints = p.getNumJoints(robot)
end_effector_index = num_joints - 1  # adjust as needed

def check_point(joint_angles_rad):
    # Set joints (no simulation step, just set state)
    for i, angle in enumerate(joint_angles_rad):
        p.resetJointState(robot, i, angle)
    
    # Forward kinematics — get end-effector pose
    state = p.getLinkState(robot, end_effector_index, computeForwardKinematics=True)
    pos = state[4]        # (x, y, z)
    quat = state[5]       # quaternion (x, y, z, w)
    euler = p.getEulerFromQuaternion(quat)  # (rx, ry, rz)
    
    # Self-collision detection
    p.performCollisionDetection()
    contacts = p.getContactPoints(bodyA=robot, bodyB=robot)
    has_collision = len(contacts) > 0
    
    return {
        "position": pos,
        "orientation_euler": euler,
        "self_collision": has_collision
    }

# Run over a trajectory
trajectory = [
    [0, 0, 0, 0, 0, 0],
    [0.1, 0.2, -0.3, 0.1, 0.5, 0],
    # ... more points
]

for joints in trajectory:
    result = check_point(joints)
    print(result)