import os
import json
import numpy as np
from dotenv import load_dotenv
from Simulation.ufactory_sim import TCP_OFFSET

load_dotenv("/Users/ronit/releases/env/base/.env")
load_dotenv("/Users/ronit/releases/env/RONIT-DEV/.env", override=True)
if "BILLIE_ENVDIR" in os.environ:
    os.environ["BILLIE_ENVDIR"] = os.path.expanduser(os.environ["BILLIE_ENVDIR"])
# Required env vars
os.environ["XARM_SN"]           = "XI130506F56A19"
os.environ["XARM_TCP_OFFSET"]   = json.dumps(TCP_OFFSET)
os.environ["ENABLE_PYROKI_GPU"] = "false"
os.environ["ENABLE_PYROKI_SELF_COLLISION"]  = "false"
os.environ["PYROKI_TERMINATION_THRESHOLD"]  = "1e-6"
os.environ["ENABLE_PYROKI_DEFAULT_INIT"]    = "false"
os.environ["PYROKI_COL_WEIGHT"]             = "100.0"
os.environ["PYROKI_COL_MARGIN"]             = "0.01" #"0.01"
os.environ["PYROKI_REG_WEIGHT"]             = "0.000001"

from scipy.spatial.transform import Rotation
import jax.numpy as jnp


from pyroki_planner.resolver import _build_solver
from billie_utils.matrix import xarm_pose_to_pose7


def set_pyroki():
    robot, robot_coll, target_link_index, solve_fn, solve_fn_batch, base_collision_checker, _ = _build_solver()
    return robot, robot_coll, target_link_index, solve_fn, solve_fn_batch, base_collision_checker
# ^ this blocks for ~20-30 seconds on first run (JAX JIT compilation)

def solve_ik_pyroki(solve_fn, robot, target_link_index, curr_joints, target_pose6,
                    reg_weights=(0.01, 0.0001), threshold_pos_mm=1.0, threshold_ori_deg=1.0,
                    n_random_starts=3, random_range_deg=4.0):
    """
    solve_fn:                pyroki IK solver function
    robot:                   pyroki robot (used for FK to evaluate solution quality)
    target_link_index:       index of the TCP link in the robot
    curr_joints:             current joint angles in radians (length 6)
    target_pose6:            target pose [x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad]
    reg_weights:             regularization weights tried for each starting point
    threshold_pos_mm:        per-dimension position error threshold (mm)
    threshold_ori_deg:       per-dimension orientation error threshold (deg)
    n_random_starts:         number of random perturbations of curr_joints to try
    random_start_range_deg:  max perturbation per joint (deg)
    returns:                 joint angles in degrees
    """



    if isinstance(reg_weights, (int, float)):
        reg_weights = [reg_weights]

    'if rand_free - chose random start in [-random_range_deg, random_range_deg]'
    'else chose random start in [-random_range_deg, -1] U [1, random_range_deg]'
    rand_free = True
    if rand_free:
        max_shift = np.deg2rad(random_range_deg)
        shifts = [np.zeros(len(curr_joints))] + (
            [
            np.random.uniform(-max_shift, max_shift, len(curr_joints))
            for _ in range(n_random_starts)
            ])
    else:
        shifts = [np.zeros(len(curr_joints))] + (
            [
            np.random.choice([-1, 1], size=len(curr_joints))
            * (np.deg2rad(1) + np.random.uniform(0, np.deg2rad(random_range_deg), len(curr_joints)))
            for _ in range(n_random_starts)
            ])

    if n_random_starts<0:
        calc_no, best_joints, best_shift  = solve_ik_flex(solve_fn, robot, target_pose6, target_link_index, curr_joints, threshold_pos_mm, threshold_ori_deg, n_random_starts, reg_weights, max_shift)
    else:
        best_err, best_joints, best_shift, calc_no = solve_ik_shifts(solve_fn, robot, target_pose6, target_link_index, curr_joints, threshold_pos_mm, threshold_ori_deg, shifts, reg_weights)
    

    return np.rad2deg(best_joints), np.rad2deg(best_shift), calc_no # return the best joints, best shift from current and number of calculation done

def solve_ik_flex(solve_fn, robot, target_pose6, target_link_index, curr_joints, threshold_pos_mm, threshold_ori_deg, n_random_starts, reg_weights, max_shift):

    max_shift = 0
    temp_max_shift = 1*np.pi/180 # 1[deg] in [rad]

    target_pose7 = jnp.array(xarm_pose_to_pose7(target_pose6), dtype=jnp.float32)
    target_pos_m = np.array(target_pose7[4:])                    # [x, y, z] meters
    target_rotvec_deg = np.rad2deg(target_pose6[3:])             # [rx, ry, rz] degrees

    best_joints = None
    best_shift  = None
    best_err    = float('inf')

    max_starts = -1* n_random_starts
    'look for the best start for max max_starts times - if we find a start with good results we stop - return the best result and the number of tries we used'
    found = False
    for start_ind in range(max_starts):
        shift = np.random.uniform(-max_shift-start_ind*temp_max_shift, max_shift+start_ind*temp_max_shift, len(curr_joints))
        cfg_init = jnp.array(np.array(curr_joints) + shift, dtype=jnp.float32)
        for w in reg_weights:
            joints_rad = np.array(solve_fn(cfg_init, target_pose7, cfg_init, jnp.array(w, dtype=jnp.float32)))

            fk = np.array(robot.forward_kinematics(jnp.array(joints_rad, dtype=jnp.float32)))
            fk_wxyz_xyz = fk[target_link_index]                  # [qw, qx, qy, qz, x, y, z]
            fk_pos_m = fk_wxyz_xyz[4:]
            fk_rotvec_deg = np.rad2deg(
                Rotation.from_quat([fk_wxyz_xyz[1], fk_wxyz_xyz[2], fk_wxyz_xyz[3], fk_wxyz_xyz[0]]).as_rotvec()
            )

            err_pos = np.abs(fk_pos_m - target_pos_m) * 1000.0  # mm, per dim
            err_ori = np.abs(fk_rotvec_deg - target_rotvec_deg)  # deg, per dim

            score = float(np.max(err_pos / threshold_pos_mm) + np.max(err_ori / threshold_ori_deg))
            if score < best_err:
                best_err    = score
                best_joints = joints_rad
                best_shift  = shift
            if best_err < 0.1:
                found = True
                break
        if found:
            break
    return start_ind, best_joints, best_shift                  

def solve_ik_shifts(solve_fn, robot, target_pose6, target_link_index, curr_joints, threshold_pos_mm, threshold_ori_deg, shifts, reg_weights):
    target_pose7 = jnp.array(xarm_pose_to_pose7(target_pose6), dtype=jnp.float32)
    target_pos_m = np.array(target_pose7[4:])                    # [x, y, z] meters
    target_rotvec_deg = np.rad2deg(target_pose6[3:])             # [rx, ry, rz] degrees

    best_joints = None
    best_shift  = shifts[0]
    best_err    = float('inf')

    for shift in shifts:
        cfg_init = jnp.array(np.array(curr_joints) + shift, dtype=jnp.float32)
        for w in reg_weights:
            joints_rad = np.array(solve_fn(cfg_init, target_pose7, cfg_init, jnp.array(w, dtype=jnp.float32)))

            fk = np.array(robot.forward_kinematics(jnp.array(joints_rad, dtype=jnp.float32)))
            fk_wxyz_xyz = fk[target_link_index]                  # [qw, qx, qy, qz, x, y, z]
            fk_pos_m = fk_wxyz_xyz[4:]
            fk_rotvec_deg = np.rad2deg(
                Rotation.from_quat([fk_wxyz_xyz[1], fk_wxyz_xyz[2], fk_wxyz_xyz[3], fk_wxyz_xyz[0]]).as_rotvec()
            )

            err_pos = np.abs(fk_pos_m - target_pos_m) * 1000.0  # mm, per dim
            err_ori = np.abs(fk_rotvec_deg - target_rotvec_deg)  # deg, per dim

            score = float(np.max(err_pos / threshold_pos_mm) + np.max(err_ori / threshold_ori_deg))
            if score < best_err:
                best_err    = score
                best_joints = joints_rad
                best_shift  = shift

    return best_err, best_joints, best_shift, shifts.shape[0]

def solve_ik_pyroki_old(solve_fn, robot, target_link_index, curr_joints, target_pose6,
                    reg_weights=(0.01, 0.0001), threshold_pos_mm=1.0, threshold_ori_deg=1.0):
    """
    solve_fn:           pyroki IK solver function
    robot:              pyroki robot (used for FK to evaluate solution quality)
    target_link_index:  index of the TCP link in the robot
    curr_joints:        current joint angles in radians (length 6)
    target_pose6:       target pose [x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad]
    reg_weights:        regularization weights tried in order; length defines max attempts
    threshold_pos_mm:   exit early if per-dimension position error is below this (mm)
    threshold_ori_deg:  exit early if per-dimension orientation error is below this (deg)
    returns:            joint angles in degrees
    """
    pose7 = jnp.array(xarm_pose_to_pose7(target_pose6), dtype=jnp.float32)
    cfg_init = jnp.array(curr_joints, dtype=jnp.float32)
    target_pos_m = np.array(pose7[4:])                           # [x, y, z] meters
    target_rotvec_deg = np.rad2deg(target_pose6[3:])             # [rx, ry, rz] degrees

    best_joints = None
    best_err = float('inf')

    if isinstance(reg_weights, (int, float)):
        reg_weights = [reg_weights]

    for w in reg_weights:
        joints_rad = np.array(solve_fn(cfg_init, pose7, cfg_init, jnp.array(w, dtype=jnp.float32)))

        fk = np.array(robot.forward_kinematics(jnp.array(joints_rad, dtype=jnp.float32)))
        fk_wxyz_xyz = fk[target_link_index]                      # [qw, qx, qy, qz, x, y, z]
        fk_pos_m = fk_wxyz_xyz[4:]
        fk_rotvec_deg = np.rad2deg(
            Rotation.from_quat([fk_wxyz_xyz[1], fk_wxyz_xyz[2], fk_wxyz_xyz[3], fk_wxyz_xyz[0]]).as_rotvec()
        )

        err_pos = np.abs(fk_pos_m - target_pos_m) * 1000.0      # mm, per dim
        err_ori = np.abs(fk_rotvec_deg - target_rotvec_deg)      # deg, per dim

        # combined score: max position error weighted by thresholds
        score = float(np.max(err_pos / threshold_pos_mm) + np.max(err_ori / threshold_ori_deg))
        if score < best_err:
            best_err = score
            best_joints = joints_rad

    return np.rad2deg(best_joints)



