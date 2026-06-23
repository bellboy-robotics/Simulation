import os
import webbrowser
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

_OUTPUT_BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'output')

xarm_col_w10  = os.path.join(_OUTPUT_BASE, 'xarm_col_w_10')
xarm_col_w100 = os.path.join(_OUTPUT_BASE, 'xarm_col_w_100')
pybullet      = os.path.join(_OUTPUT_BASE, 'pybullet')
pybullet_v1   = os.path.join(_OUTPUT_BASE, 'pybullet_v1')
xarm          = os.path.join(_OUTPUT_BASE, 'xarm')
xarm_v1       = os.path.join(_OUTPUT_BASE, 'xarm_pyroki_v1')


def _load(directory):
    new_pos        = np.load(os.path.join(directory, 'sim_new_pos.npy'))
    expected_poses = np.load(os.path.join(directory, 'sim_expected_poses.npy'))
    ik_res_path    = os.path.join(directory, 'sim_ik_residuals.npy')
    ik_residuals   = np.load(ik_res_path) if os.path.exists(ik_res_path) else None
    return new_pos, expected_poses, ik_residuals


def compare_results(dirs_passes: list[str]):
    labels  = [os.path.basename(d) for d in dirs_passes]
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    colors  = [palette[i % len(palette)] for i in range(len(dirs_passes))]

    runs = []
    for d in dirs_passes:
        new_pos, expected, ik_res = _load(d)
        runs.append({'new_pos': new_pos, 'expected': expected, 'ik_res': ik_res})

    # use the expected poses from the first run as the common reference
    reference = runs[0]['expected']
    frames    = np.arange(len(reference))

    pos_labels = ['X (mm)', 'Y (mm)', 'Z (mm)']
    ori_labels = ['rx (deg)', 'ry (deg)', 'rz (deg)']
    all_labels = pos_labels + ori_labels

    # ── Figure 1: actual positions per dimension, all runs overlaid ──────────
    fig1 = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                          subplot_titles=all_labels)
    for dim in range(6):
        # reference (expected) — one grey line
        fig1.add_trace(go.Scatter(x=frames, y=reference[:, dim], name='expected',
                                   line=dict(color='gray', dash='dot'),
                                   legendgroup='expected', showlegend=(dim == 0)),
                       row=dim+1, col=1)
        for run, label, color in zip(runs, labels, colors):
            n = min(len(frames), len(run['new_pos']))
            fig1.add_trace(go.Scatter(x=frames[:n], y=run['new_pos'][:n, dim],
                                       name=label, line=dict(color=color),
                                       legendgroup=label, showlegend=(dim == 0)),
                           row=dim+1, col=1)
    fig1.update_layout(height=1300, title='Actual positions — all runs')

    # ── Figure 2: error norm (pos + ori) per run ─────────────────────────────
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                          subplot_titles=['Position error norm (mm)', 'Orientation error norm (deg)'])
    for run, label, color in zip(runs, labels, colors):
        n   = min(len(frames), len(run['new_pos']), len(run['expected']))
        ref = run['expected'][:n]
        act = run['new_pos'][:n]
        diff = act - ref
        fig2.add_trace(go.Scatter(x=frames[:n], y=np.linalg.norm(diff[:, :3], axis=1),
                                   name=label, line=dict(color=color),
                                   legendgroup=label, showlegend=True),
                       row=1, col=1)
        fig2.add_trace(go.Scatter(x=frames[:n], y=np.linalg.norm(diff[:, 3:], axis=1),
                                   name=label, line=dict(color=color),
                                   legendgroup=label, showlegend=False),
                       row=2, col=1)
    fig2.update_layout(height=700, title='Error norm — all runs')

    # ── Figure 3: mean absolute error per dim, bar chart ─────────────────────
    mae_pos = []
    mae_ori = []
    for run in runs:
        n    = min(len(run['new_pos']), len(run['expected']))
        diff = np.abs(run['new_pos'][:n] - run['expected'][:n])
        mae_pos.append(diff[:, :3].mean(axis=0))
        mae_ori.append(diff[:, 3:].mean(axis=0))

    fig3 = make_subplots(rows=1, cols=2,
                          subplot_titles=['Mean absolute position error (mm)',
                                          'Mean absolute orientation error (deg)'])
    for dim, axis in enumerate(['X', 'Y', 'Z']):
        fig3.add_trace(go.Bar(name=axis, x=labels,
                               y=[mae_pos[ri][dim] for ri in range(len(runs))],
                               marker_color=palette[dim]),
                       row=1, col=1)
    for dim, axis in enumerate(['rx', 'ry', 'rz']):
        fig3.add_trace(go.Bar(name=axis, x=labels,
                               y=[mae_ori[ri][dim] for ri in range(len(runs))],
                               marker_color=palette[dim+3]),
                       row=1, col=2)
    fig3.update_layout(height=500, barmode='group', title='Mean absolute error per axis — all runs')

    # ── assemble HTML ─────────────────────────────────────────────────────────
    views = [
        ('actual_per_dim', 'Actual — Per Dim',  fig1),
        ('error_norm',     'Error Norm',         fig2),
        ('mae_bar',        'MAE Bar',            fig3),
    ]
    div_parts = []
    for i, (vid, _, fig) in enumerate(views):
        include_js = 'inline' if i == 0 else False
        display    = 'block'  if i == 0 else 'none'
        div_parts.append(
            f'<div id="{vid}" class="view" style="display:{display}">'
            f'{pio.to_html(fig, full_html=False, include_plotlyjs=include_js)}</div>'
        )

    divs    = '\n'.join(div_parts)
    first   = views[0][0]
    buttons = ' '.join(
        f'<button id="btn_{vid}" onclick="show(\'{vid}\')" '
        f'style="margin:4px;padding:6px 14px;font-size:14px"'
        f'{"" if vid != first else " class=active"}>{label}</button>'
        for vid, label, _ in views
    )
    html = f"""<!DOCTYPE html><html><head>
<style>body{{font-family:sans-serif;padding:16px}} button.active{{background:#1f77b4;color:#fff}}</style>
</head><body>
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

    out = os.path.join(os.path.dirname(__file__), 'compare_results.html')
    with open(out, 'w') as f:
        f.write(html)
    webbrowser.open(f'file://{out}')
    print(f"Comparison saved to {out}")


if __name__ == "__main__":
    # dirs_passes = [xarm, xarm_col_w100, pybullet]
    dirs_passes = [pybullet_v1, pybullet]
    compare_results(dirs_passes)
