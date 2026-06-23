import time
import json
import os
import webbrowser
from Simulation.ik_sim.get_pos_from_recording import get_pos_from_recording
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from Simulation.ik_sim.ik_pyroki import set_pyroki, solve_ik_pyroki

def conect_to_sim(sim_type):
    if sim_type == 'xarm':
        import Simulation.ik_sim.ufactory_sim as m
        arm = m.set_xarm()
    elif sim_type == 'pybullet':
        import Simulation.ik_sim.pybullet_sim as m
        arm = m.set_pybullet()
    else:
        raise ValueError(f"Unknown simulation type: {sim_type}")
    return m, arm
    
def plot_results(expected_poses, new_pos, problematic_frames, title='plot path', output_path=None):
    prob_indices = [int(list(f.keys())[0]) for f in problematic_frames]
    frames = np.arange(len(new_pos))
    diff = new_pos - expected_poses
    pos_labels = ['X (mm)', 'Y (mm)', 'Z (mm)']
    ori_labels = ['rx (deg)', 'ry (deg)', 'rz (deg)']
    all_labels = pos_labels + ori_labels
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    def vlines(fig, rows):
        for idx in prob_indices:
            if idx < len(frames):
                for r in rows:
                    fig.add_vline(x=int(idx), line_color='red', line_width=1, opacity=0.35, row=r, col=1)

    # ── Figure 1: actual vs expected, per dimension ──────────────────────────
    fig1 = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                          subplot_titles=all_labels)
    for i, label in enumerate(all_labels):
        fig1.add_trace(go.Scatter(x=frames, y=expected_poses[:, i], name='expected',
                                   line=dict(color=colors[i]), legendgroup=label,
                                   showlegend=(i == 0)), row=i+1, col=1)
        fig1.add_trace(go.Scatter(x=frames, y=new_pos[:, i], name='actual',
                                   line=dict(color=colors[i], dash='dash'), legendgroup=label,
                                   showlegend=(i == 0)), row=i+1, col=1)
    vlines(fig1, range(1, 7))
    fig1.update_layout(height=1200, title='Actual vs Expected — Per Dimension')

    # ── Figure 2: error per dimension ────────────────────────────────────────
    fig2 = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                          subplot_titles=all_labels)
    for i, label in enumerate(all_labels):
        fig2.add_trace(go.Scatter(x=frames, y=diff[:, i], name=label, fill='tozeroy',
                                   line=dict(color=colors[i])), row=i+1, col=1)
        fig2.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.5, row=i+1, col=1)
    vlines(fig2, range(1, 7))
    fig2.update_layout(height=1200, title='Error — Per Dimension')

    # ── Figure 3: L2 norm of position and orientation, expected vs actual ────────
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                          subplot_titles=['Position norm ||[x,y,z]|| (mm)', 'Orientation norm ||[rx,ry,rz]|| (deg)'])
    fig3.add_trace(go.Scatter(x=frames, y=np.linalg.norm(expected_poses[:, :3], axis=1),
                               name='expected', line=dict(color=colors[0])), row=1, col=1)
    fig3.add_trace(go.Scatter(x=frames, y=np.linalg.norm(new_pos[:, :3], axis=1),
                               name='actual', line=dict(color=colors[0], dash='dash')), row=1, col=1)
    fig3.add_trace(go.Scatter(x=frames, y=np.linalg.norm(expected_poses[:, 3:], axis=1),
                               name='expected', line=dict(color=colors[3]), showlegend=False), row=2, col=1)
    fig3.add_trace(go.Scatter(x=frames, y=np.linalg.norm(new_pos[:, 3:], axis=1),
                               name='actual', line=dict(color=colors[3], dash='dash'), showlegend=False), row=2, col=1)
    vlines(fig3, [1, 2])
    fig3.update_layout(height=700, title='Actual vs Expected — Combined Norm')

    # ── Figure 4: error norm, position + orientation combined ─────────────────
    fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                          subplot_titles=['Position error norm (mm)', 'Orientation error norm (deg)'])
    fig4.add_trace(go.Scatter(x=frames, y=np.linalg.norm(diff[:, :3], axis=1),
                               name='pos error', fill='tozeroy', line=dict(color=colors[0])), row=1, col=1)
    fig4.add_trace(go.Scatter(x=frames, y=np.linalg.norm(diff[:, 3:], axis=1),
                               name='ori error', fill='tozeroy', line=dict(color=colors[3])), row=2, col=1)
    vlines(fig4, [1, 2])
    fig4.update_layout(height=700, title='Error Norm — Position & Orientation')

    # ── Figure 5: IK residual (how close IK got to the target) ───────────────
    views = [
        ('actual_per_dim',   'Actual — Per Dim',      fig1),
        ('error_per_dim',    'Error — Per Dim',        fig2),
        ('actual_combined',  'Actual — Combined',      fig3),
        ('error_combined',   'Error — Combined',       fig4),
    ]
    # ── Assemble HTML with toggle buttons ────────────────────────────────────
    div_parts = []
    for i, (vid, _, fig) in enumerate(views):
        include_js = 'inline' if i == 0 else False
        display = 'block' if i == 0 else 'none'
        div_parts.append(
            f'<div id="{vid}" class="view" style="display:{display}">{pio.to_html(fig, full_html=False, include_plotlyjs=include_js)}</div>'
        )
    divs = '\n'.join(div_parts)
    first_vid = views[0][0]
    buttons = ' '.join(
        f'<button id="btn_{vid}" onclick="show(\'{vid}\')" style="margin:4px;padding:6px 14px;font-size:14px"{"" if vid != first_vid else " class=active"}>{label}</button>'
        for vid, label, _ in views
    )
    html = f"""<!DOCTYPE html><html><head>
<style>body{{font-family:sans-serif;padding:16px}} button.active{{background:#1f77b4;color:#fff}} h2{{margin:0 0 10px 0;font-size:18px;color:#333}}</style>
</head><body>
<h2>{title}</h2>
<div style="margin-bottom:12px">{buttons}</div>
{divs}
<script>
function show(id) {{
  document.querySelectorAll('div.view').forEach(d => d.style.display='none');
  document.getElementById(id).style.display='block';
  document.querySelectorAll('button').forEach(b => b.classList.remove('active'));
  document.getElementById('btn_'+id).classList.add('active');
  document.getElementById(id).querySelectorAll('.plotly-graph-div').forEach(function(gd){{ Plotly.Plots.resize(gd); }});
}}
</script></body></html>"""

    out = output_path if output_path is not None else os.path.join(os.path.dirname(__file__), 'sim_results.html')
    with open(out, 'w') as f:
        f.write(html)
    if output_path is None:
        webbrowser.open(f'file://{out}')
    print(f"Results saved to {out}")

def get_ik(ik_type):
    if ik_type == 'pyroki':
        import Simulation.ik_sim.ik_pyroki as ik_pyroki
        robot, robot_coll, target_link_index, solve_fn, solve_fn_batch, base_collision_checker = ik_pyroki.set_pyroki()
    else:
        raise ValueError(f"Unknown IK solver type: {ik_type}")
    return solve_fn, base_collision_checker, robot, target_link_index

def plot_loaded_results(arm_sim_type, extra_name):
    if extra_name is None:
        extra_name = ''
    _out = os.path.join(os.path.dirname(__file__), '..', '..', 'output', arm_sim_type+'_'+extra_name)
    expected_poses = np.load(os.path.join(_out, 'sim_expected_poses.npy'))
    new_pos = np.load(os.path.join(_out, 'sim_new_pos.npy'))
    with open(os.path.join(_out, 'sim_problematic_frames.json'), 'r') as f:
        problematic_frames = json.load(f)
    plot_results(expected_poses, new_pos, problematic_frames, title=arm_sim_type+'_'+extra_name)

def calc_ik(arm_sim_type, ik_solver, use_curr_joints, n_random_starts, repo_id,
            pos_th=1.0, ori_th=3.0, error_recovery='random_starts'):
    # error_recovery: 'random_starts' → set n_random_starts=1 on next frame
    #                 'shift'         → add random 2-3 deg shift to curr_joints_red on next frame
    """
    """
    'connect to IK solver (must run first — generates /tmp/urdf/XI130506F56A19-with-tcp.urdf used by pybullet)'
    solve_fn, base_collision_checker, ik_robot, ik_target_link_index = get_ik('pyroki')
    'connect to arm simulation'
    sim, arm = conect_to_sim(arm_sim_type)

    'get expected poses from dataset'
    expected_poses, joint_angles = get_pos_from_recording(repo_id)
    new_pos             = []
    new_joints          = []
    xarm_response       = []
    problematic_frames  = []
    ik_residuals        = []  # per-frame IK residual: how close IK got to the target pose
    shifts_chosen           = []
    current_n_random_starts = n_random_starts
    error_shift             = np.zeros(6)
    recovering              = False
    'run IK for each pose and check for collisions'
    for i, (rec_pose, rec_joint_angle) in enumerate(zip(expected_poses, joint_angles)):
        if use_curr_joints == 'rec':
            curr_joints_red = np.deg2rad(rec_joint_angle)
        else:
            curr_joints_red = sim.get_joint_angles(arm, is_radian=True)
        if error_recovery == 'shift':
            curr_joints_red = curr_joints_red + error_shift
        # print(f"Processing pose {i}:\t{' '.join(f'{v:.2f}[mm]' for v in rec_pose[:3])}\t:\t{' '.join(f'{v:.2f}[deg]' for v in rec_pose[3:]*180/np.pi)}, joint angles: {' '.join(f'{v:.2f}' for v in rec_joint_angle)} ")
        if base_collision_checker.check_collision(rec_pose):
            print(f"Frame {i}: target position is inside robot base (base collision)")
            problematic_frames.append({i: 'base collision', 'pos': rec_pose[:3].tolist()})
        if ik_solver == 'pyroki':
            ik_joints_deg, shift_chosen, calc_done = solve_ik_pyroki(solve_fn, ik_robot, ik_target_link_index, curr_joints_red, rec_pose, reg_weights=(0.0005), n_random_starts=current_n_random_starts)
            shifts_chosen.append(shift_chosen)
            if recovering:
                if np.abs(shift_chosen).sum() != 0:
                    print(f"  recovery shift chosen (deg): {' '.join(f'j{j+1}={s:+.2f}' for j, s in enumerate(shift_chosen))} - no calc done: {calc_done}")
                recovering = False
        else:
            raise ValueError(f"Unknown IK solver: {ik_solver}")
        xarm_code, new_cart_pos = sim.set_joint_angles(arm, ik_joints_deg)
        try_count = 0
        'handle errors in simulator'
        while xarm_code == 9 and try_count < 3:  # 9 is the error code for "collision detected"
            time.sleep(0.5)  # give the arm time to settle
            sim.clear_errors(arm)
            xarm_code, new_cart_pos = sim.set_joint_angles(arm, ik_joints_deg)
            try_count += 1
        if xarm_code != 0:
            print(f"Error setting joint angles: {xarm_code}")
            sim.clear_errors(arm)
            new_pos.append(new_pos[-1] if new_pos else np.zeros(6))
            new_joints.append(new_joints[-1] if new_joints else curr_joints_red*180/np.pi)
            problematic_frames.append({i: 'Error setting joint angles', 'xarm_code': xarm_code})
            current_n_random_starts = n_random_starts
            error_shift             = np.zeros(6)
        else:
            'handle new joints'
            new_joints.append(sim.get_joint_angles(arm, is_radian=False))

            target_deg = np.concatenate([rec_pose[:3], np.rad2deg(rec_pose[3:])])
            res_pos = np.linalg.norm(np.array(new_cart_pos[:3]) - target_deg[:3])
            ori_diff_wrapped = ((np.array(new_cart_pos[3:]) - target_deg[3:]) + 180) % 360 - 180
            res_ori = np.linalg.norm(ori_diff_wrapped)
            ik_residuals.append([res_pos, res_ori])
            if res_pos > pos_th or res_ori > ori_th:
                print(f"Frame {i}: high error (pos={res_pos:.2f}mm ori={res_ori:.2f}deg) → recovery={error_recovery}, n_rand={current_n_random_starts}")
                if error_recovery == 'random_starts':
                    current_n_random_starts = min(current_n_random_starts+1, 5)
                    error_shift             = np.zeros(6)
                    recovering              = True
                elif error_recovery == 'shift':
                    current_n_random_starts = n_random_starts
                    magnitudes  = np.random.uniform(np.deg2rad(2.0), np.deg2rad(3.0), 6)
                    error_shift = np.random.choice([-1, 1], size=6) * magnitudes
                    print(f"  shift (deg): {' '.join(f'j{j+1}={np.rad2deg(s):+.2f}' for j, s in enumerate(error_shift))}")
                else:
                    current_n_random_starts = n_random_starts
                    error_shift             = np.zeros(6)
                    recovering              = True
            else:
                current_n_random_starts = n_random_starts
                error_shift             = np.zeros(6)

            # print(f"new pose        {i}:\t{' '.join(f'{v:.2f}[mm]' for v in new_cart_pos[:3])}\t:\t{' '.join(f'{v:.2f}[deg]' for v in new_cart_pos[3:])}\tres: {res_pos:.2f}mm {res_ori:.2f}deg")
            new_pos.append(new_cart_pos)
            xarm_response.append((xarm_code))
        curr_joints = np.array(sim.get_joint_angles(arm, is_radian=False))
        if not np.allclose(curr_joints, ik_joints_deg):
            problematic_frames.append({i: 'mismatch between set joints and read joints', 'set_joints': ik_joints_deg, 'read_joints': curr_joints})
    
    new_pos = np.array(new_pos)
    ik_residuals = np.array(ik_residuals)
    expected_poses = np.array(expected_poses[:len(new_pos)])
    expected_poses[:, 3:] = np.rad2deg(expected_poses[:, 3:])  # convert expected rotation to degrees for comparison

    if shifts_chosen:
        shifts_arr = np.array(shifts_chosen)
        non_zero_mask = np.any(shifts_arr != 0, axis=1)
        n_non_zero = int(non_zero_mask.sum())
        print(f"Random shift chosen: {n_non_zero}/{len(shifts_chosen)} frames "
              f"({100*n_non_zero/len(shifts_chosen):.1f}%)")
        if n_non_zero > 0:
            mean_shift = np.abs(shifts_arr[non_zero_mask]).mean(axis=0)
            print(f"Mean |shift| when non-zero (deg): {' '.join(f'j{j+1}={v:.2f}' for j, v in enumerate(mean_shift))}")

    return expected_poses, new_pos, problematic_frames

'''-------------------------------------------------------------------------------------------------------'''
if __name__ == "__main__":
    'set seed'
    seed = 8
    print(f"[ik_pyroki] random seed: {seed}")
    np.random.seed(seed)
    ' sim parameters: '
    load_results    = False
    # arm_sim_type    = 'xarm'
    # extra_name      = 'col_w_100' #'col_w_10'

    arm_sim_type    = 'pybullet'
    extra_name      = 'big_box_w_col_mash_2st' #'init_3deg' #'curr_rec'
    ik_solver       = 'pyroki'
    use_curr_joints = 'actual'     # 'rec': recorded joints as IK init | 'actual': read from sim
    n_random_starts = -20 #-1 = dynamic
    error_recovery = None #'random_starts' #'shift' #None
    repo_id = 'bellboy-robotics/B-unknown-20260301-180408-BILLIE-12'

    if load_results:
        plot_loaded_results(arm_sim_type, extra_name)
        exit()

    'run inverce kinematics'
    expected_poses, new_pos, problematic_frames = calc_ik(arm_sim_type, ik_solver, use_curr_joints, n_random_starts, repo_id, error_recovery=error_recovery)

    'save results and plot'
    if extra_name is None:
        extra_name = ''
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output', arm_sim_type+'_'+extra_name)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'sim_expected_poses.npy'), expected_poses)
    np.save(os.path.join(output_dir, 'sim_new_pos.npy'), new_pos)
    with open(os.path.join(output_dir, 'sim_problematic_frames.json'), 'w') as f:
        json.dump([{str(k): str(v) for k, v in frame.items()} for frame in problematic_frames], f, indent=2)
    print(f"Saved results to {output_dir}")

    diff = new_pos - expected_poses
    print(f"Mean error per axis (mm): X={np.mean(diff[:, 0]):.2f}±{np.std(diff[:, 0]):.2f}  Y={np.mean(diff[:, 1]):.2f}±{np.std(diff[:, 1]):.2f}  Z={np.mean(diff[:, 2]):.2f}±{np.std(diff[:, 2]):.2f}")
    print(f"Mean error per angle:     rx={np.mean(diff[:, 3]):.2f}±{np.std(diff[:, 3]):.2f}  ry={np.mean(diff[:, 4]):.2f}±{np.std(diff[:, 4]):.2f}  rz={np.mean(diff[:, 5]):.2f}±{np.std(diff[:, 5]):.2f}")
    plot_results(expected_poses, new_pos, problematic_frames,
                 title=f"{arm_sim_type} | ik={ik_solver} | joints={use_curr_joints}")