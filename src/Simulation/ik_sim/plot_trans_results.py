"""
Parse trans_1000_no_col.log and plot IK error vs transformation parameters.

4 scatter plots:
  pos error (mm)  vs xy_norm (mm)
  ori error (deg) vs xy_norm (mm)
  pos error (mm)  vs |dtheta| (deg)
  ori error (deg) vs |dtheta| (deg)
"""

import re
import math
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "trans_1000_no_col.log")

# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------
pattern = re.compile(
    r"-transform\(([^,]+),([^,]+),([^)]+)\):\s*"
    r"pos error:\s*([\d.]+)mm\s*"
    r"ori error:\s*([\d.]+)deg"
)

dx_list, dy_list, dtheta_list, pos_list, ori_list = [], [], [], [], []

with open(LOG_PATH) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            dx, dy, dtheta, pos, ori = (float(x) for x in m.groups())
            dx_list.append(dx)
            dy_list.append(dy)
            dtheta_list.append(dtheta)
            pos_list.append(pos)
            ori_list.append(ori)

dx    = np.array(dx_list)
dy    = np.array(dy_list)
dtheta = np.array(dtheta_list)
pos   = np.array(pos_list)
ori   = np.array(ori_list)

xy_norm = np.sqrt(dx**2 + dy**2)
abs_dtheta = np.abs(dtheta)

print(f"Parsed {len(pos)} transforms")
print(f"  xy_norm : {xy_norm.min():.1f} – {xy_norm.max():.1f} mm")
print(f"  |dtheta|: {abs_dtheta.min():.2f} – {abs_dtheta.max():.2f} deg")
print(f"  pos err : {pos.min():.2f} – {pos.max():.2f} mm   mean={pos.mean():.2f}")
print(f"  ori err : {ori.min():.2f} – {ori.max():.2f} deg  mean={ori.mean():.2f}")

# ---------------------------------------------------------------------------
# Linear regression helper
# ---------------------------------------------------------------------------
def linreg(x, y):
    mx, my = x.mean(), y.mean()
    num = ((x - mx) * (y - my)).sum()
    den_x = ((x - mx) ** 2).sum()
    den_y = ((y - my) ** 2).sum()
    m = num / den_x if den_x else 0.0
    b = my - m * mx
    r = num / math.sqrt(den_x * den_y) if (den_x and den_y) else 0.0
    return m, b, r

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
BLUE = "#2B7FE1"
TEAL = "#009E86"
ZERO_COLOR = "rgba(180,200,220,0.5)"

configs = [
    dict(x=xy_norm,    y=pos, xlabel="XY displacement (mm)", ylabel="Position error (mm)",    color=BLUE, title="Position Error vs XY Displacement",  row=1, col=1),
    dict(x=xy_norm,    y=ori, xlabel="XY displacement (mm)", ylabel="Orientation error (°)",  color=TEAL, title="Orientation Error vs XY Displacement", row=1, col=2),
    dict(x=abs_dtheta, y=pos, xlabel="|Δθ| rotation (°)",   ylabel="Position error (mm)",    color=BLUE, title="Position Error vs Base Rotation",      row=2, col=1),
    dict(x=abs_dtheta, y=ori, xlabel="|Δθ| rotation (°)",   ylabel="Orientation error (°)",  color=TEAL, title="Orientation Error vs Base Rotation",    row=2, col=2),
]

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[c["title"] for c in configs],
    horizontal_spacing=0.10,
    vertical_spacing=0.14,
)

for cfg in configs:
    x, y = cfg["x"], cfg["y"]
    color = cfg["color"]
    row, col = cfg["row"], cfg["col"]

    # split zero / non-zero
    zero_mask = y == 0
    nonzero_mask = ~zero_mask

    # zero-error dots (dim)
    fig.add_trace(go.Scatter(
        x=x[zero_mask], y=y[zero_mask],
        mode="markers",
        marker=dict(size=5, color=ZERO_COLOR, line=dict(color="rgba(150,180,210,0.6)", width=0.5)),
        name="no error",
        showlegend=False,
        hovertemplate="xy=%.1f<br>err=0<extra></extra>" if cfg["xlabel"].startswith("XY") else "|Δθ|=%.2f<br>err=0<extra></extra>",
    ), row=row, col=col)

    # non-zero dots, sized by error magnitude
    if nonzero_mask.any():
        y_nz = y[nonzero_mask]
        x_nz = x[nonzero_mask]
        sizes = 5 + 9 * (y_nz / y_nz.max())
        fig.add_trace(go.Scatter(
            x=x_nz, y=y_nz,
            mode="markers",
            marker=dict(
                size=sizes,
                color=y_nz,
                colorscale=[[0, "rgba(180,210,240,0.4)"], [1, color]],
                showscale=False,
                line=dict(color=color, width=0.6),
            ),
            name=cfg["ylabel"],
            showlegend=False,
            hovertemplate=f"{cfg['xlabel'].split('(')[0].strip()}=%{{x:.1f}}<br>{cfg['ylabel'].split('(')[0].strip()}=%{{y:.2f}}<extra></extra>",
        ), row=row, col=col)

    # trend line
    m, b, r = linreg(x, y)
    x_line = np.array([x.min(), x.max()])
    y_line = np.clip(m * x_line + b, 0, None)
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        line=dict(color=color, width=1.5, dash="dash"),
        showlegend=False,
        hoverinfo="skip",
    ), row=row, col=col)

    # annotation: r value
    fig.add_annotation(
        xref="x domain", yref="y domain",
        x=0.98, y=0.97,
        text=f"r = {r:.3f}",
        showarrow=False,
        font=dict(size=11, color="#666"),
        xanchor="right", yanchor="top",
        row=row, col=col,
    )

    # axis labels
    axis_idx = (row - 1) * 2 + col
    xaxis_key = f"xaxis{axis_idx if axis_idx > 1 else ''}"
    yaxis_key = f"yaxis{axis_idx if axis_idx > 1 else ''}"
    fig.layout[xaxis_key].title.text = cfg["xlabel"]
    fig.layout[yaxis_key].title.text = cfg["ylabel"]

fig.update_layout(
    paper_bgcolor="#F4F6F9",
    plot_bgcolor="#FFFFFF",
    font=dict(family="system-ui, Arial, sans-serif", size=12, color="#1A2332"),
    title=dict(
        text=f"IK Simulation — Error Analysis  (n={len(pos)})",
        font=dict(size=16),
        x=0.5,
    ),
    margin=dict(l=60, r=30, t=80, b=60),
    height=700,
)

fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False, linecolor="#D0D8E8")
fig.update_yaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False, linecolor="#D0D8E8", rangemode="tozero")

out_path = os.path.join(os.path.dirname(__file__), "trans_results.html")
fig.write_html(out_path)
print(f"\nSaved to {out_path}")

import webbrowser
webbrowser.open(f"file://{os.path.abspath(out_path)}")
