from xarm.wrapper import XArmAPI
import time

TCP_OFFSET = [0, 0, 200, 0, 0, 0]  # [x, y, z mm, roll, pitch, yaw deg] — gripper/camera offset

def set_xarm():
    arm = XArmAPI('127.0.0.1', baud_checkset=False, check_joint_limit=False)
    clear_errors(arm)

    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    arm.set_tcp_offset(TCP_OFFSET, is_radian=False)
    time.sleep(0.5)  # give the arm time to settle
    return arm

def clear_errors(arm):
    arm.clean_error()   # clear any existing controller error
    arm.clean_warn()    # clear any warnings

def set_joint_angles(arm, angles):
    """
    set the joint angles of the simulated robot and return the resulting end-effector pose
    angles_deg: list of 6 joint angles in degrees
    returns: (error_code, [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg])
    """
    xarm_code =arm.set_servo_angle(angle=angles, wait=True, is_radian=False)
    if xarm_code != 0:
        print(f"Error setting joint angles: {xarm_code}")
        new_cart_pos = None
    else:
        new_cart_pos = arm.get_position_aa()[1]
    return xarm_code, new_cart_pos

def get_joint_angles(arm, is_radian=False):
    joints = arm.get_servo_angle(is_radian=is_radian)[1]
    return joints[:-1] # exclude the gripper joint

if __name__ == "__main__":
    
    arm = set_xarm()
    j = get_joint_angles(arm, is_radian=False)
    print(j)
    # xarm_code, cart_pos = set_joint_angles(arm, [0, 0, 0, 0, 0, 0])
    # if xarm_code > 0:
    #     clear_errors(arm)