import time
import json
import os
import webbrowser
import Simulation.ik_sim.get_pos_from_recording as path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from Simulation.ik_sim.script_sim import conect_to_sim, plot_results, plot_loaded_results
from Simulation.ik_sim.ik_pyroki import solve_ik_batch_multi, set_pyroki, check_self_collision_pyroki

from billie_utils.matrix import xarm_pose_to_pose7
from billie_utils.messages.pyroki_node import BATCH_SIZE as DEFAULT_BATCH_SIZE

    

def get_bacth_ik(ik_type):
    if ik_type == 'pyroki':
        robot, robot_coll, target_link_index, solve_fn, solve_fn_batch, base_collision_checker = set_pyroki()
    else:
        raise ValueError(f"Unknown IK solver type: {ik_type}")
    return solve_fn, solve_fn_batch, base_collision_checker, robot, robot_coll, target_link_index

def calc_ik_batch(ik_solver, n_random_starts, expected_poses, joint_angles,
                  sim, arm, pos_th=2.0, ori_th=3.0):
    """Batch version of calc_ik: solves DEFAULT_BATCH_SIZE poses at once with recorded joints as regularization."""

    if ik_solver == 'pyroki':
        _, solve_fn_batch, base_collision_checker, robot, robot_coll, _ = get_bacth_ik('pyroki')
    else:
        raise ValueError(f"Unknown IK solver: {ik_solver}")

    batch_size = 50
    new_pos            = []
    new_joints         = []
    problematic_frames = []
    n_poses            = len(expected_poses)
    expected_poses     = np.array(expected_poses)
    joint_angles       = np.array(joint_angles)  # (N, 6) degrees
    all_min_dist = np.inf
    '''loop batchs'''
    for start in range(0, n_poses, batch_size):
        'gather positions for batchs'
        end        = min(start + batch_size, n_poses)
        chunk_size = end - start
        batch_target_pose6  = expected_poses[start:end]   # (chunk, 6)
        batch_joints = joint_angles[start:end]     # (chunk, 6) degrees

        # cfg_init   = np.deg2rad(batch_joints[0]).astype(np.float32)
        # poses7     = np.array([xarm_pose_to_pose7(p) for p in batch_poses], dtype=np.float32)
        # reg_refs   = np.deg2rad(batch_joints).astype(np.float32)
        # reg_weight = np.float32(REC_WEIGHT)

        # if chunk_size < DEFAULT_BATCH_SIZE:
        #     pad      = DEFAULT_BATCH_SIZE - chunk_size
        #     poses7   = np.concatenate([poses7,    np.tile(poses7[-1:],   (pad, 1))], axis=0)
        #     reg_refs = np.concatenate([reg_refs,  np.tile(reg_refs[-1:], (pad, 1))], axis=0)

        # cfg_sol_batch = np.array(
        #     solve_fn_batch(cfg_init, poses7, reg_refs, reg_weight, n_seeds)
        # )[:chunk_size]  # (chunk, num_joints) radians

        cfg_sol_batch = solve_ik_batch_multi(solve_fn_batch, chunk_size, batch_size, batch_target_pose6,
                            batch_joints, reg_weight=0.001, n_seeds=n_random_starts) #0.1

        'henadl each position in the solution batch'
        for i, (ik_joints_rad, rec_pose) in enumerate(zip(cfg_sol_batch, batch_target_pose6)):
            frame_idx    = start + i
            ik_joints_deg = np.rad2deg(ik_joints_rad)

            if base_collision_checker.check_collision(rec_pose):
                print(f"Frame {frame_idx}: base collision")
                problematic_frames.append({frame_idx: 'base collision', 'pos': rec_pose[:3].tolist()})

            is_self_col, col_pairs, min_dist = check_self_collision_pyroki(robot, robot_coll, ik_joints_rad)
            all_min_dist = min(all_min_dist, min_dist)
            if is_self_col:
                print(f"Frame {frame_idx}: self collision — {col_pairs}")
                problematic_frames.append({frame_idx: 'self collision', 'pairs': col_pairs})

            xarm_code, new_cart_pos = sim.set_joint_angles(arm, ik_joints_deg)
            try_count = 0
            while xarm_code == 9 and try_count < 3:
                time.sleep(0.5)
                sim.clear_errors(arm)
                xarm_code, new_cart_pos = sim.set_joint_angles(arm, ik_joints_deg)
                try_count += 1

            if xarm_code != 0:
                print(f"Error setting joint angles: {xarm_code}")
                sim.clear_errors(arm)
                new_pos.append(new_pos[-1] if new_pos else np.zeros(6))
                new_joints.append(new_joints[-1] if new_joints else ik_joints_deg)
                problematic_frames.append({frame_idx: 'Error setting joint angles', 'xarm_code': xarm_code})
            else:
                new_joints.append(sim.get_joint_angles(arm, is_radian=False))
                target_deg = np.concatenate([rec_pose[:3], np.rad2deg(rec_pose[3:])])
                res_pos    = np.linalg.norm(np.array(new_cart_pos[:3]) - target_deg[:3])
                ori_diff_wrapped = ((np.array(new_cart_pos[3:]) - target_deg[3:]) + 180) % 360 - 180
                res_ori    = np.linalg.norm(ori_diff_wrapped)
                if res_pos > pos_th or res_ori > ori_th:
                    problematic_frames.append({frame_idx: f'high error pos={res_pos:.2f}mm ori={res_ori:.2f}deg'})
                new_pos.append(new_cart_pos)

            curr_joints = np.array(sim.get_joint_angles(arm, is_radian=False))
            if not np.allclose(curr_joints, ik_joints_deg):
                problematic_frames.append({frame_idx: 'mismatch', 'set_joints': ik_joints_deg.tolist(), 'read_joints': curr_joints.tolist()})

    print(f"min dist: {all_min_dist}")
    new_pos           = np.array(new_pos)
    expected_poses_out = expected_poses[:len(new_pos)].copy()
    expected_poses_out[:, 3:] = np.rad2deg(expected_poses_out[:, 3:])
    return expected_poses_out, new_pos, problematic_frames

def test_transformation(repo_id, no_transforms, arm_sim_type='xarm', ik_solver='pyroki',
                        use_curr_joints='actual', n_random_starts=5,
                        xy_range_mm=100, angle_range_deg=15, select_transform=None, gui=False):
    """ read recordings according to repo_id and create no_transorms transformations and run calc_ik per transformation
        repo_id:          location of the recording in hf
        no_trasforms:     number of trasformation to randomly create - if 0 - use original path
        select_transform: 0-based index of a single transformation to run (None = run all)
    """
    'get expected poses from dataset'
    expected_poses, joint_angles = path.get_pos_from_recording(repo_id)

    if no_transforms == 0:
        dxs     = [0]
        dys     = [0]
        dthetas = [0]
        poses   = [expected_poses]
    else:
        if select_transform is not None:
            dxs, dys, dthetas = select_transform
            dxs = [dxs]
            dys = [dys]
            dthetas = [dthetas]
        else:
            dxs, dys, dthetas = path.get_random_diff(xy_range_mm=xy_range_mm,
                                                    angle_range_deg=angle_range_deg,
                                                    no_transform=no_transforms)
        poses = [path.apply_base_transform(expected_poses, dx, dy, dtheta)
                 for dx, dy, dtheta in zip(dxs, dys, dthetas)]

    items = list(zip(poses, dxs, dys, dthetas))

    sim, arm = conect_to_sim(arm_sim_type, gui=False)

    results = []
    for i, (pose_set, dx, dy, dtheta) in enumerate(items):
        # print(f"\n--- Transform {i+1}/{len(poses)} ---")
        exp_out, new_pos, problematic_frames = calc_ik_batch(
            ik_solver, n_random_starts,
            pose_set, joint_angles, sim=sim, arm=arm
        )

        results.append((exp_out, new_pos, problematic_frames, dx, dy, dtheta))
        pos_diff = new_pos[:, :3] - exp_out[:, :3]
        ori_diff = ((new_pos[:, 3:] - exp_out[:, 3:]) + 180) % 360 - 180
        pos_err = np.linalg.norm(pos_diff, axis=1)
        ori_err = np.linalg.norm(ori_diff, axis=1)
        print(f"-transform({dx},{dy},{dtheta}):  pos error(mean/max): {pos_err.mean():.2f}/{pos_err.max():.2f}[mm]  "
              f"ori error(mean/max): {ori_err.mean():.2f}/{ori_err.max():.2f}[deg]  "
              f"problematic frames: {len(problematic_frames)}")

    return results

'''-------------------------------------------------------------------------------------------------------'''
if __name__ == "__main__":

    np.random.seed(42)

    ' sim parameters: '
    load_results    = False
    # arm_sim_type    = 'xarm'
    # extra_name      = 'col_w_100' #'col_w_10'

    arm_sim_type    = 'pybullet'
    extra_name      = 'batch_try1' #'init_3deg' #'curr_rec'
    ik_solver       = 'pyroki'
    use_curr_joints = 'actual'     # 'rec': recorded joints as IK init | 'actual': read from sim
    n_random_starts = 0 #16 #-1 = dynamic
    repo_id = 'bellboy-robotics/B-unknown-20260301-180408-BILLIE-12'
    no_transforms   = 10
    select_transform = None #(39.46339367881464,-115.06435572868953,-8.018289402378498)  # 0-based index to run a single transform; None = run all

    if load_results:
        plot_loaded_results(arm_sim_type, extra_name)
        exit()

    'run inverce kinematics'

    results = test_transformation(repo_id, no_transforms=no_transforms, arm_sim_type=arm_sim_type, ik_solver=ik_solver,
                                  use_curr_joints=use_curr_joints, n_random_starts=n_random_starts,
                                  select_transform=select_transform, gui=False)


    # plot_idx = 0 #if select_transform is not None else 3
    # expected_poses, new_pos, problematic_frames, dx, dy, dtheta = results[plot_idx]
    # out_path = os.path.join(os.path.dirname(__file__), 'sim_results.html')
    # the_title = f'transform dx={dx:.1f} dy={dy:.1f} dθ={dtheta:.1f}'

    # plot_results(expected_poses, new_pos, problematic_frames,
    #              title=the_title,
    #              output_path=out_path)
    # webbrowser.open(f'file://{out_path}')

    # for res in results:
    #     expected_poses, new_pos, problematic_frames, dx, dy, dtheta = res
    #     'save results and plot'
    #     if extra_name is None:
    #         extra_name = ''
    #     output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output', arm_sim_type+'_'+extra_name)
    #     os.makedirs(output_dir, exist_ok=True)
    #     np.save(os.path.join(output_dir, 'sim_expected_poses.npy'), expected_poses)
    #     np.save(os.path.join(output_dir, 'sim_new_pos.npy'), new_pos)
    #     with open(os.path.join(output_dir, 'sim_problematic_frames.json'), 'w') as f:
    #         json.dump([{str(k): str(v) for k, v in frame.items()} for frame in problematic_frames], f, indent=2)
    #     print(f"Saved results to {output_dir}")

    #     diff = new_pos - expected_poses
    #     print(f"Mean error per axis (mm): X={np.mean(diff[:, 0]):.2f}±{np.std(diff[:, 0]):.2f}  Y={np.mean(diff[:, 1]):.2f}±{np.std(diff[:, 1]):.2f}  Z={np.mean(diff[:, 2]):.2f}±{np.std(diff[:, 2]):.2f}")
    #     print(f"Mean error per angle:     rx={np.mean(diff[:, 3]):.2f}±{np.std(diff[:, 3]):.2f}  ry={np.mean(diff[:, 4]):.2f}±{np.std(diff[:, 4]):.2f}  rz={np.mean(diff[:, 5]):.2f}±{np.std(diff[:, 5]):.2f}")
    #     plot_results(expected_poses, new_pos, problematic_frames,
    #                 title=f"{arm_sim_type} | ik={ik_solver} | joints={use_curr_joints} | tranform={}")