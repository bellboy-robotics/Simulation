"""HTML report of a simulated run: the 3D scene, clearance to the objects, tracking and joint steps."""

import html
import webbrowser

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from billie_utils.arm_motion_guard_poc import MAX_TRAJECTORY_STEP_DEG
from billie_utils.messages.pyroki_world_poc import WORLD_COL_MARGIN_M
from billie_utils.world_collision_check_poc import link_transforms
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation

from Simulation.world_sim.analysis import Run, dense_path, eef_mask, link_distances, tcp_poses
from Simulation.world_sim.recording import BRAIN_TICK_RATE

ROLE_COLORS = {"avoid": "#d62728", "eef_touch": "#2ca02c"}
COMMAND_COLORS = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]


def point_metrics(run: Run) -> dict:
    """Per arm target: clearance along the move into it, tracking error and joint step.

    run: The run.
    Returns: dict of (N,) arrays: "avoid_mm" (all links to avoid objects), "touch_eef_mm" / "touch_arm_mm"
        (EEF / other links to eef_touch objects), "pos_err_mm", "ori_err_deg" (NaN without a target),
        "step_deg" (largest joint change from the previous target); plus "samples" (S, 6), "owner" (S,),
        "avoid_samples_mm" (S,) for the 3D view.
    """
    samples, owner = dense_path(run)
    avoid = link_distances(run, samples, "avoid").min(axis=1) * 1000.0
    touch = link_distances(run, samples, "eef_touch") * 1000.0
    eef = eef_mask(run)
    n, moved = len(run.joints_deg), owner >= 0
    per_point = {}
    for key, values in (("avoid_mm", avoid), ("touch_eef_mm", touch[:, eef].min(axis=1)),
                        ("touch_arm_mm", touch[:, ~eef].min(axis=1))):  # fmt: skip
        out = np.full(n, np.inf)
        np.minimum.at(out, owner[moved], values[moved])
        per_point[key] = out
    reached = tcp_poses(run, run.joints_deg) if n else np.zeros((0, 6))
    per_point["pos_err_mm"] = np.linalg.norm(reached[:, :3] - run.target_pose[:, :3], axis=1)
    has_target = ~np.isnan(run.target_pose[:, 0])
    ori = np.full(n, np.nan)
    if has_target.any():
        delta = Rotation.from_rotvec(run.target_pose[has_target, 3:]).inv() * Rotation.from_rotvec(reached[has_target, 3:])
        ori[has_target] = np.rad2deg(delta.magnitude())
    per_point["ori_err_deg"] = ori
    per_point["step_deg"] = np.abs(np.diff(np.vstack([run.start_deg[None], run.joints_deg]), axis=0)).max(axis=1) \
        if n else np.zeros(0)  # fmt: skip
    per_point.update(samples=samples, owner=owner, avoid_samples_mm=avoid)
    return per_point


def capsule_surface(start: np.ndarray, end: np.ndarray, radius: float, n: int = 12) -> tuple[np.ndarray, ...]:
    """Surface grid of a capsule, for go.Surface.

    start, end: (3,) segment ends, meters. radius: meters. n: grid resolution per half.
    Returns: (x, y, z) arrays of shape (2n, 2n) in millimeters.
    """
    axis = end - start
    length = float(np.linalg.norm(axis))
    z_dir = axis / length if length > 1e-9 else np.array([0.0, 0.0, 1.0])
    helper = np.array([1.0, 0.0, 0.0]) if abs(z_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_dir = np.cross(helper, z_dir)
    x_dir /= np.linalg.norm(x_dir)
    y_dir = np.cross(z_dir, x_dir)
    theta = np.concatenate([np.linspace(0, np.pi / 2, n), np.linspace(np.pi / 2, np.pi, n)])[:, None]
    shift = np.concatenate([np.full(n, length / 2), np.full(n, -length / 2)])[:, None]
    phi = np.linspace(0, 2 * np.pi, 2 * n)[None, :]
    local = [radius * np.sin(theta) * np.cos(phi), radius * np.sin(theta) * np.sin(phi), radius * np.cos(theta) + shift]
    center = (start + end) / 2
    world = [(center[k] + local[0] * x_dir[k] + local[1] * y_dir[k] + local[2] * z_dir[k]) * 1000.0 for k in range(3)]
    return tuple(world)


def scene_figure(run: Run, metrics: dict) -> go.Figure:
    """3D view: objects, the TCP path per command, IK targets, refused moves and the arm at its closest approach.

    run: The run. metrics: From point_metrics.
    Returns: The figure.
    """
    fig = go.Figure()
    for obj in run.objects:
        color = ROLE_COLORS[obj["role"]]
        for i, (s, e, r) in enumerate(zip(obj["starts_m"], obj["ends_m"], obj["radii_m"])):
            x, y, z = capsule_surface(s, e, r)
            fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, color], [1, color]], showscale=False,
                                     opacity=0.45, name=f"{obj['name']} ({obj['role']})", legendgroup=obj["name"],
                                     showlegend=(i == 0)))  # fmt: skip
    samples, owner = metrics["samples"], metrics["owner"]
    tcp = tcp_poses(run, samples)[:, :3]
    # With no arm targets (every command refused) the path is only the start sample.
    sample_command = np.where(owner >= 0, run.command[np.maximum(owner, 0)], -1) if len(run.command) else np.full(len(owner), -1)
    for c, spec in enumerate(run.commands):
        sel = (sample_command == c) | np.r_[sample_command[1:] == c, False]  # include each move's start
        fig.add_trace(go.Scatter3d(x=tcp[sel, 0], y=tcp[sel, 1], z=tcp[sel, 2], mode="lines",
                                   line=dict(width=5, color=COMMAND_COLORS[c % len(COMMAND_COLORS)]),
                                   name=f"{c}: {spec['spec']['cmd']}"))  # fmt: skip
    targets = run.target_pose[~np.isnan(run.target_pose[:, 0])]
    fig.add_trace(go.Scatter3d(x=targets[:, 0], y=targets[:, 1], z=targets[:, 2], mode="markers",
                               marker=dict(size=2, color="black"), name="IK targets", visible="legendonly"))  # fmt: skip
    refused = [i for i, b in enumerate(run.blocked) if b]
    if refused:
        p = tcp_poses(run, run.joints_deg[refused])
        fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                                   marker=dict(size=5, symbol="x", color="red"), name="guard refused"))  # fmt: skip
    worst = int(np.argmin(metrics["avoid_samples_mm"]))
    _add_arm(fig, run, samples[worst], f"arm at closest approach ({metrics['avoid_samples_mm'][worst]:.0f}mm)")
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(size=6, color="gray"), name="arm base"))
    fig.update_layout(height=850, title="Scene (xArm base frame, mm)", scene=dict(aspectmode="data"))
    return fig


def _add_arm(fig: go.Figure, run: Run, joints_deg: np.ndarray, name: str) -> None:
    """Draws the moving links' collision capsules as thick segments.

    fig: Figure to add to. run: The run (arm model). joints_deg: (6,) configuration. name: Legend entry.
    """
    model = run.model
    T = link_transforms(model, np.deg2rad(joints_deg)[None])[0] @ model.capsule_local
    half = (model.capsule_height / 2.0)[:, None] * T[:, :3, 2]
    a, b = (T[:, :3, 3] - half) * 1000.0, (T[:, :3, 3] + half) * 1000.0
    xs, ys, zs = [], [], []
    for i in np.flatnonzero(model.movable_links):
        xs += [a[i, 0], b[i, 0], None]
        ys += [a[i, 1], b[i, 1], None]
        zs += [a[i, 2], b[i, 2], None]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines+markers", line=dict(width=12, color="#444"),
                               marker=dict(size=3), name=name))  # fmt: skip


def timeline_figure(run: Run, metrics: dict) -> go.Figure:
    """Clearance, tracking error and joint step per arm target, with each command's span shaded.

    run: The run. metrics: From point_metrics.
    Returns: The figure.
    """
    x = np.arange(len(run.joints_deg))
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=[
        "Clearance to avoid objects (mm, min over the move, all links)", "eef_touch objects: EEF vs rest of arm (mm)",
        "Tracking error vs IK target", "Largest joint step (deg)"])  # fmt: skip
    fig.add_trace(go.Scatter(x=x, y=metrics["avoid_mm"], name="avoid clearance", line=dict(color="#d62728")), 1, 1)
    fig.add_hline(y=WORLD_COL_MARGIN_M * 1000, line_dash="dot", line_color="orange", row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=metrics["touch_eef_mm"], name="EEF", line=dict(color="#2ca02c")), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=metrics["touch_arm_mm"], name="rest of arm", line=dict(color="#8c564b")), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=metrics["pos_err_mm"], name="position (mm)"), 3, 1)
    fig.add_trace(go.Scatter(x=x, y=metrics["ori_err_deg"], name="orientation (deg)"), 3, 1)
    fig.add_trace(go.Scatter(x=x, y=metrics["step_deg"], name="joint step", line=dict(color="#7f7f7f")), 4, 1)
    fig.add_hline(y=MAX_TRAJECTORY_STEP_DEG, line_dash="dot", line_color="red", row=4, col=1)
    for row in (1, 2):
        fig.add_hline(y=0, line_color="black", line_width=1, row=row, col=1)
    for c in range(len(run.commands)):
        idx = np.flatnonzero(run.command == c)
        if len(idx):
            fig.add_vrect(x0=idx[0] - 0.5, x1=idx[-1] + 0.5, fillcolor=COMMAND_COLORS[c % len(COMMAND_COLORS)],
                          opacity=0.08, line_width=0, annotation_text=f"cmd {c}", annotation_position="top left")  # fmt: skip
    blocked = [i for i, b in enumerate(run.blocked) if b]
    if blocked:
        fig.add_trace(go.Scatter(x=blocked, y=metrics["avoid_mm"][blocked], mode="markers", name="guard refused",
                                 marker=dict(symbol="x", size=9, color="red")), 1, 1)  # fmt: skip
    fig.update_layout(height=1100, title="Per arm target (x = target index)")
    return fig


def summary_html(run: Run, metrics: dict) -> str:
    """Commands table (outcome, time, closest approach per command) and the operator messages.

    run: The run. metrics: From point_metrics.
    Returns: HTML fragment.
    """
    rows = []
    for c, outcome in enumerate(run.commands):
        sel = run.command == c
        closest = f"{metrics['avoid_mm'][sel].min():.0f}" if sel.any() and np.isfinite(metrics['avoid_mm'][sel]).any() else "—"
        kinds = ", ".join(f"{k}×{run.kind.count(k)}" for k in dict.fromkeys(np.array(run.kind)[sel]))
        status = "ok" if outcome["ok"] else f"<b style='color:#d62728'>FAILED</b>: {html.escape(outcome['error'])}"
        spec = html.escape(", ".join(f"{k}={v}" for k, v in outcome["spec"].items()))
        rows.append(f"<tr><td>{c}</td><td>{spec}</td><td>{status}</td><td>{outcome['seconds']:.1f}</td>"
                    f"<td>{sel.sum()}</td><td>{kinds}</td><td>{closest}</td></tr>")  # fmt: skip
    events = "".join(
        f"<tr><td>{e['command']}</td><td>{e['level']}</td><td>{html.escape(e['message'])}</td></tr>"
        for e in run.events if e["level"] != "DEBUG"
    )
    return (
        "<table><tr><th>#</th><th>command</th><th>result</th><th>plan s</th><th>targets</th><th>moves</th>"
        f"<th>closest to avoid (mm)</th></tr>{''.join(rows)}</table>"
        f"<h3>Messages</h3><table><tr><th>cmd</th><th>level</th><th>message</th></tr>{events}</table>"
    )


def playback_estimate(batch_seconds: list[float], batch_frames: list[int]) -> tuple[float, float]:
    """Wall time of a pose replay on the robot's BufferingPlayer, from each batch's solve time.

    Batches are solved one after another; the brain plays one frame per tick once the first batch is
    in (it holds more than the 20-frame minimum buffer) and stalls whenever the next batch is not ready.

    batch_seconds: Solve time of each replay batch, in order.
    batch_frames: Frames in each batch.
    Returns: (seconds from the first batch request to the last played frame, seconds of playback alone).
    """
    ready = np.cumsum(batch_seconds)
    finished = 0.0
    for k, frames in enumerate(batch_frames):
        finished = max(ready[k], finished) + frames / BRAIN_TICK_RATE
    return float(finished), sum(batch_frames) / BRAIN_TICK_RATE


def timing_html(run: Run) -> str:
    """Where the planner ran, its build time, and per command the planner time and replay wall-time estimate.

    run: The run.
    Returns: HTML fragment ("" for results without timings).
    """
    if not run.timings:
        return ""
    t = run.timings
    build = ", ".join(f"{x['call'][6:]} {x['seconds']:.1f}s" for x in t if x["call"].startswith("build_"))
    rows = []
    for c, outcome in enumerate(run.commands):
        calls = [x for x in t if x["command"] == c]
        cells = []
        for name in ("solve", "detour", "transit", "batch_solve"):
            s = [x["seconds"] for x in calls if x["call"] == name]
            cells.append(f"{len(s)} × {np.mean(s):.2f}s (max {max(s):.2f}, total {sum(s):.1f})" if s else "—")
        batches = [x for x in calls if x["call"] == "batch_solve"]
        estimate = "—"
        if batches:
            wall, playback = playback_estimate([x["seconds"] for x in batches], [x["n"] for x in batches])
            before = sum(x["seconds"] for x in calls if x["call"] in ("detour", "transit"))
            estimate = f"{before + wall:.1f}s ({before:.1f}s planning before motion + {wall:.1f}s replay, of which {playback:.1f}s playback)"
        rows.append(f"<tr><td>{c}</td><td>{outcome['spec']['cmd']}</td>{''.join(f'<td>{x}</td>' for x in cells)}"
                    f"<td>{estimate}</td></tr>")  # fmt: skip
    machine = html.escape(f"{run.machine.get('host', '?')} {run.machine.get('jax_devices', '')}")
    return (
        f"<h3>Planner timing</h3><p>Ran on <b>{machine}</b>. Build: {build}.</p>"
        "<table><tr><th>#</th><th>command</th><th>single IK</th><th>detour batch</th><th>transit</th>"
        f"<th>replay batches</th><th>robot time estimate</th></tr>{''.join(rows)}</table>"
    )


def write_report(run: Run, output_path: str, open_browser: bool = True) -> None:
    """Writes the HTML report: summary, then the 3D scene and the timeline behind toggle buttons.

    run: The run. output_path: HTML file to write. open_browser: Open it when done.
    """
    metrics = point_metrics(run)
    views = [("scene", "3D scene", scene_figure(run, metrics)), ("timeline", "Timeline", timeline_figure(run, metrics))]
    divs = "\n".join(
        f'<div id="{vid}" class="view" style="display:{"block" if i == 0 else "none"}">'
        f"{pio.to_html(fig, full_html=False, include_plotlyjs='inline' if i == 0 else False)}</div>"
        for i, (vid, _, fig) in enumerate(views)
    )
    buttons = " ".join(f"<button onclick=\"show('{vid}')\">{label}</button>" for vid, label, _ in views)
    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{html.escape(run.name)}</title>
<style>body{{font-family:sans-serif;padding:16px}} table{{border-collapse:collapse;margin-bottom:12px;font-size:13px}}
td,th{{border:1px solid #ccc;padding:3px 8px;text-align:left;vertical-align:top}} button{{margin:4px;padding:6px 14px}}</style>
</head><body><h2>{html.escape(run.name)}</h2>{summary_html(run, metrics)}{timing_html(run)}<div>{buttons}</div>{divs}
<script>function show(id){{document.querySelectorAll('div.view').forEach(d=>d.style.display='none');
var v=document.getElementById(id);v.style.display='block';v.querySelectorAll('.plotly-graph-div').forEach(g=>Plotly.Plots.resize(g));}}</script>
</body></html>"""
    with open(output_path, "w") as f:
        f.write(page)
    if open_browser:
        webbrowser.open(f"file://{output_path}")
