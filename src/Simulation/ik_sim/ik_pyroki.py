import os
import json
import numpy as np
from dotenv import load_dotenv
from Simulation.ik_sim.ufactory_sim import TCP_OFFSET

load_dotenv("/Users/ronit/releases/env/base/.env")
load_dotenv("/Users/ronit/releases/env/RONIT-DEV/.env", override=True)
if "BILLIE_ENVDIR" in os.environ:
    os.environ["BILLIE_ENVDIR"] = os.path.expanduser(os.environ["BILLIE_ENVDIR"])
# Required env vars
os.environ["XARM_SN"]           = "XI130506F56A19"
os.environ["XARM_TCP_OFFSET"]   = json.dumps(TCP_OFFSET)
os.environ["ENABLE_PYROKI_GPU"] = "true"
os.environ["ENABLE_PYROKI_SELF_COLLISION"]  = "true"
os.environ["PYROKI_TERMINATION_THRESHOLD"]  = "1e-6"
os.environ["ENABLE_PYROKI_DEFAULT_INIT"]    = "false"
os.environ["PYROKI_COL_WEIGHT"]             = "100.0"
os.environ["PYROKI_COL_MARGIN"]             = "0.01" #"0.01" #  
os.environ["PYROKI_REG_WEIGHT"]             = "0.000001"

from scipy.spatial.transform import Rotation
import jax.numpy as jnp

from pyroki_planner.resolver import _build_solver
from billie_utils.matrix import xarm_pose_to_pose7

IGNORE_COLLISION_PAIRS: tuple[tuple[str, str], ...] = (
    ("link_gripper_and_camera", "link5"),
)

def set_pyroki():
    robot, robot_coll, target_link_index, solve_fn, solve_fn_batch, base_collision_checker, _ = _build_solver(
        user_ignore_pairs=IGNORE_COLLISION_PAIRS
    )
    return robot, robot_coll, target_link_index, solve_fn, solve_fn_batch, base_collision_checker
# ^ this blocks for ~20-30 seconds on first run (JAX JIT compilation)

def check_self_collision_pyroki(robot, robot_coll, joints_rad: np.ndarray):
    """
    Check pyroki self-collision for a single joint configuration.

    joints_rad: joint angles in radians, shape (num_joints,)

    Returns (is_collision, colliding_pairs) where colliding_pairs is a list of
    (link_name_a, link_name_b, penetration_depth) tuples for penetrating pairs.
    """
    cfg = jnp.array(joints_rad, dtype=jnp.float32)
    distances = np.array(robot_coll.compute_self_collision_distance(robot, cfg))  # (num_active_pairs,)
    min_distance = distances.min()
    if min_distance < 0.001:
        print('proximity alert')
    colliding_mask = distances < 0
    if not np.any(colliding_mask):
        return False, [], min_distance
    colliding_pairs = [
        (robot_coll.link_names[int(i)], robot_coll.link_names[int(j)], float(-d))
        for i, j, d, is_col in zip(
            robot_coll.active_idx_i, robot_coll.active_idx_j, distances, colliding_mask
        )
        if is_col
    ]
    return True, colliding_pairs, min_distance

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
    # if n_random_starts<0:
    #     calc_no, best_joints, best_shift  = solve_ik_flex(solve_fn, robot, target_pose6, target_link_index, curr_joints, threshold_pos_mm, threshold_ori_deg, n_random_starts, reg_weights, max_shift)
    # else:
    if not isinstance(reg_weights, list):
        reg_weights = [reg_weights]
    best_joints = solve_ik_multi(solve_fn, curr_joints, target_pose6, reg_weight = reg_weights[0], n_seeds = n_random_starts)
    # best_joints, error, best_i = best_joints
    
    # best_err, best_joints, best_shift, calc_no = solve_ik_shifts(solve_fn, robot, target_pose6, target_link_index, curr_joints, threshold_pos_mm, threshold_ori_deg, shifts, reg_weights)
    best_shift  = 0
    calc_no     = 0

    return np.rad2deg(best_joints), np.rad2deg(best_shift), calc_no # return the best joints, best shift from current and number of calculation done

def solve_ik_multi(
    solve_fn,
    curr_joints: np.ndarray,
    target_pose6: np.ndarray,
    reg_weight: float = 0.01,
    n_seeds: int = 4,
) -> np.ndarray:
    """
    solve_fn:         pyroki IK solver (from set_pyroki)
    curr_joints:      current joint angles in radians
    target_pose6:     target pose [x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad]
    reg_weight:       regularization weight towards curr_joints
    n_seeds:          number of perturbed starts solved in parallel via vmap;
                      perturbation magnitude is controlled by MULTI_START_PERTURB in resolver.py
    returns:          best joint angles in radians (selected by lowest FK pose error)
    """
    target_pose7 = jnp.array(xarm_pose_to_pose7(target_pose6), dtype=jnp.float32)
    cfg_init = jnp.array(curr_joints, dtype=jnp.float32)
    best_solution = solve_fn(cfg_init, target_pose7, cfg_init, jnp.array(reg_weight, dtype=jnp.float32), n_seeds)
    if n_seeds <= 1:
        return np.array(best_solution)
    best_joints, error, best_i, close_ind, all_sol = best_solution
    return np.array(best_joints)

def solve_ik_batch_multi(
    solve_batch_fn,
    chunk_size: int,
    batch_size: int,
    target_pose6: np.ndarray,
    target_joints: np.ndarray,
    reg_weight: float = 0.01,
    n_seeds: int = 4,
) -> np.ndarray:
    """
    solve_fn:         pyroki IK solver (from set_pyroki)
    curr_joints:      current joint angles in radians
    target_pose6:     target pose [x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad]
    reg_weight:       regularization weight towards curr_joints
    n_seeds:          number of perturbed starts solved in parallel via vmap;
                      perturbation magnitude is controlled by MULTI_START_PERTURB in resolver.py
    returns:          best joint angles in radians (selected by lowest FK pose error)
    """
    cfg_init        = np.deg2rad(target_joints[0]).astype(np.float32)
    target_poses7   = np.array([xarm_pose_to_pose7(p) for p in target_pose6], dtype=np.float32)
    reg_refs        = np.deg2rad(target_pose6).astype(np.float32)
    reg_weight      = np.float32(reg_weight)

    if chunk_size < batch_size:
        pad             = batch_size - chunk_size
        target_poses7   = np.concatenate([target_poses7,    np.tile(target_poses7[-1:],   (pad, 1))], axis=0)
        reg_refs        = np.concatenate([reg_refs,  np.tile(reg_refs[-1:], (pad, 1))], axis=0)

    cfg_sol_batch = np.array(
        solve_batch_fn(cfg_init, target_poses7, reg_refs, reg_weight, n_seeds)
    )[:chunk_size]  # (chunk, num_joints) radians

    return np.array(cfg_sol_batch)





