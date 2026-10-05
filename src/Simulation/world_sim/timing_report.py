"""Timing sections of the HTML report: planner build per phase, a Gantt of the run, call statistics, the log.

Reads the fields plan.run_scenario records (planner_build, planner_log, planner_messages, start_s, compile_s,
cache hits/misses); for results written before them, each section shows what is there or nothing.
"""

import html

import numpy as np
import plotly.graph_objects as go

from Simulation.world_sim.analysis import Run
from Simulation.world_sim.planner_capture import NOTABLE

# Planner calls in the order the tables list them (build_* entries are the build, shown separately).
CALLS = ("solve", "detour", "transit", "batch_solve")
# What each build phase does, for the build table.
PHASE_TEXT = {
    "ik_setup": "URDF, robot and collision models, IK problem setup (resolver thread)",
    "ik_warmup_single": "single-pose IK solver: first call compiles or loads it",
    "ik_warmup_batch": "batch IK solver (in parallel with the single one)",
    "ik_total": "IK solvers ready (what a solve waits for: 'waiting for background compilation')",
    "export_model": "arm model for the brain (NumPy copy)",
    "transit_build": "transit planner setup",
    "transit_warmup": "transit planner: first call compiles or loads it",
}
# Background of the log lines about compiles, the JAX cache, waits or fallbacks (planner_capture.NOTABLE).
_HIGHLIGHT = "background:#fff3cd"
# Style of the notes above the build table (build reused / built long before the run): amber, like a warning.
_BANNER = "border:2px solid #e0a800;background:#fff8e1;padding:8px 12px;margin:8px 0;font-size:14px"
# [s] Gap between the build's end and the run's start above which the build is noted as done before the run
# (an editor builds at start-up); smaller gaps are the plan step's own setup between the two.
_BUILT_BEFORE_S = 5


def _verdict(row: dict) -> str:
    """What JAX's events say a build phase or call did.

    row: A phase or timing with "cache_hits", "cache_misses", "compile_s" (absent in old results).
    Returns: "loaded from cache", "compiled (N not in cache)", "compiled (cache not used)", "nothing compiled"
        or "—" without the event counts.
    """
    if "cache_misses" not in row:
        return "—"
    if row["cache_misses"]:
        return f"<b>compiled</b> ({row['cache_misses']} program(s) not in the cache)"
    if row["cache_hits"]:
        return "loaded from cache"
    return "compiled (cache not used)" if row.get("compile_s", 0) > 0.05 else "nothing compiled"


def build_html(run: Run) -> str:
    """The planner build: reuse banner, gripper geometry, JAX cache and a table per build phase.

    run: The run.
    Returns: HTML fragment ("" parts for results without the build record).
    """
    build, machine = run.planner.get("planner_build"), run.machine
    out = ""
    if "gripper_geometry" in machine:
        geo = machine["gripper_geometry"]
        style = "color:#d62728;font-weight:bold" if geo != "mesh" else ""
        out += (f"<p>Gripper collision geometry: <span style='{style}'>{html.escape(geo)}</span> "
                f"<small>({html.escape(str(machine.get('gripper_mesh')))})</small></p>")  # fmt: skip
    if not build:
        return out
    if build.get("reused"):
        out += (f"<div style='{_BANNER}'><b>Planner built before this run</b> and reused (built "
                f"{build['before_run_s']:.0f}s before this run started, by an earlier run of this process): "
                "this run did not wait for the build below.</div>")  # fmt: skip
    elif build.get("before_run_s", 0) - build.get("total_s", 0) > _BUILT_BEFORE_S:
        out += (f"<div style='{_BANNER}'>Planner built {build['before_run_s'] - build['total_s']:.0f}s before this "
                "run started (e.g. at editor start); this is its first run.</div>")  # fmt: skip
    rows = []
    for p in build.get("phases", []):
        total = p["name"] == "ik_total"
        cells = [p["name"], PHASE_TEXT.get(p["name"], ""), f"{p['start_s']:.1f}–{p['end_s']:.1f}", f"{p['seconds']:.2f}",
                 f"{p['trace_s'] + p['lower_s']:.2f}", f"{p['backend_s']:.2f}", f"{p['cache_read_s']:.2f}",
                 f"{p['cache_hits']} / {p['cache_misses']}", _verdict(p), html.escape(p.get("resolver_label") or "")]  # fmt: skip
        tds = "".join(f"<td>{c}</td>" for c in cells)
        rows.append(f"<tr style='{'font-weight:bold' if total else ''}'>{tds}</tr>")
    cache = (f"JAX {html.escape(str(build.get('jax_version')))}, cache {html.escape(str(build.get('jax_cache_dir')))}: "
             f"{build.get('cache_files_before')} → {build.get('cache_files_after')} files")  # fmt: skip
    env = ", ".join(f"{k}={html.escape(str(v))}" for k, v in build.get("env", {}).items())
    return out + (
        f"<h4>Planner build: {build.get('total_s', 0):.1f}s</h4><p><small>{cache}. {env}</small></p>"
        "<table><tr><th>phase</th><th>what</th><th>from–to s</th><th>seconds</th><th>trace + lower s<br>"
        "<small>(Python, never cached)</small></th><th>XLA compile / cache load s</th><th>cache read s</th>"
        "<th>cache hits / misses</th><th>verdict (JAX events)</th><th>resolver says<br><small>(guess: &lt;5s = hit)"
        f"</small></th></tr>{''.join(rows)}</table>"
        "<p><small>Warm-ups run in parallel threads, so their compile seconds overlap in wall time; on a warm cache "
        "most of a phase is Python tracing and lowering, which JAX does not cache.</small></p>"
    )


def _call_stats_rows(timings: list[dict], label: str) -> list[str]:
    """Table rows of per call type statistics for one run.

    timings: The run's planner calls (build_* entries are skipped). label: Which run ("with obstacles", ...).
    Returns: One <tr> per call type that ran.
    """
    rows = []
    for call in CALLS:
        calls = [t for t in timings if t["call"] == call]
        if not calls:
            continue
        s = np.array([t["seconds"] for t in calls])
        poses = sum(t["n"] for t in calls)
        penalty = f"{s[0] - np.median(s[1:]):+.2f}" if len(s) > 1 else "—"
        compile_s = f"{sum(t['compile_s'] for t in calls):.2f}" if "compile_s" in calls[0] else "—"
        per_pose = f"{s.sum() / poses * 1000:.0f}" if poses > len(calls) else "—"
        cells = [label, call, len(s), poses, f"{s.mean():.3f}", f"{np.median(s):.3f}", f"{np.percentile(s, 95):.3f}",
                 f"{s.max():.3f}", f"{s.sum():.2f}", penalty, compile_s, per_pose]  # fmt: skip
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return rows


def call_stats_html(run: Run) -> str:
    """Per call type: count, poses, mean/median/p95/max/total seconds, first-call penalty, compile, ms per pose.

    run: The run.
    Returns: HTML fragment ("" without planner calls).
    """
    rows = _call_stats_rows(run.timings, "with obstacles")
    if run.no_obstacles is not None:
        rows += _call_stats_rows(run.no_obstacles["timings"], "without obstacles")
    if not rows:
        return ""
    return (
        "<h4>Planner calls by type</h4><table><tr><th>run</th><th>call</th><th>count</th><th>poses</th><th>mean s</th>"
        "<th>median s</th><th>p95 s</th><th>max s</th><th>total s</th><th>first call − median of the rest s</th>"
        f"<th>JAX compile inside s</th><th>ms / pose</th></tr>{''.join(rows)}</table>"
    )


def _split_cell(timings: list[dict], commands: list[dict], c: int) -> str:
    """One command's planner time split into JAX compile and compute, for the per-command table.

    timings: A run's planner calls. commands: Its command outcomes. c: The command index.
    Returns: "<td>" cells: planner s, compile s, compute s, outcome ("—" cells when the command did not run).
    """
    if c >= len(commands):
        return "<td>—</td>" * 4
    calls = [t for t in timings if t["command"] == c]
    planner_s = sum(t["seconds"] for t in calls)
    has_compile = all("compile_s" in t for t in calls)
    compile_s = sum(t["compile_s"] for t in calls) if has_compile else None
    cells = [f"{planner_s:.2f}", "—" if compile_s is None else f"{compile_s:.2f}",
             "—" if compile_s is None else f"{planner_s - compile_s:.2f}", "ok" if commands[c]["ok"] else "✗"]  # fmt: skip
    return "".join(f"<td>{x}</td>" for x in cells)


def command_split_html(run: Run) -> str:
    """Per command: planner time, JAX compile inside it and the rest (compute), with and without obstacles.

    run: The run.
    Returns: HTML fragment ("" without planner calls or their compile times).
    """
    if not any("compile_s" in t for t in run.timings):  # results from before compile times were recorded
        return ""
    free = run.no_obstacles
    head = "<th>planner s</th><th>compile s</th><th>compute s</th><th>result</th>"
    rows = []
    for c, outcome in enumerate(run.commands):
        free_cells = _split_cell(free["timings"], free["commands"], c) if free is not None else ""
        rows.append(f"<tr><td>{c}</td><td>{html.escape(outcome['spec']['cmd'])}</td>"
                    f"{_split_cell(run.timings, run.commands, c)}{free_cells}</tr>")  # fmt: skip
    top = "<th colspan='4'>with obstacles</th>" + ("<th colspan='4'>without obstacles</th>" if free is not None else "")
    return (f"<h4>Planner time per command: compile vs compute</h4><table><tr><th rowspan='2'>#</th>"
            f"<th rowspan='2'>command</th>{top}</tr><tr>{head}{head if free is not None else ''}</tr>"
            f"{''.join(rows)}</table>")  # fmt: skip


def _gantt_run(fig: go.Figure, results: dict, offset: float, label: str, colors: list[str], rows: list[str]) -> None:
    """Adds one run's commands and planner calls to the Gantt, a trace per command.

    fig: The figure. results: The run's {"commands", "timings"} (with start_s). offset: Its start, seconds after
    the main run's start. label: Row prefix ("with obstacles", ...). colors: Per command colors.
    rows: Row names in display order, extended in place.
    """
    for c, outcome in enumerate(results["commands"]):
        calls = [t for t in results["timings"] if t["command"] == c and "start_s" in t]
        names = [f"{label}: commands"] + [f"{label}: {t['call']}" for t in calls]
        rows += [n for n in names if n not in rows]
        starts = [outcome["start_s"]] + [t["start_s"] for t in calls]
        widths = [outcome["end_s"] - outcome["start_s"]] + [t["seconds"] for t in calls]
        hover = [f"cmd {c} {outcome['spec']['cmd']}: {widths[0]:.2f}s{'' if outcome['ok'] else ' FAILED'}"] + [
            f"cmd {c} {t['call']} ({t['n']} poses): {t['seconds']:.3f}s, compile {t.get('compile_s', 0):.2f}s, "
            f"cache {t.get('cache_hits', 0)} hits / {t.get('cache_misses', 0)} misses" for t in calls]  # fmt: skip
        fig.add_trace(go.Bar(y=names, x=widths, base=np.asarray(starts) + offset, orientation="h",
                             marker_color=colors[c % len(colors)], hovertext=hover, hoverinfo="text",
                             name=f"{label} cmd {c}: {outcome['spec']['cmd']}", legendgroup=label))  # fmt: skip


def gantt_figure(run: Run, colors: list[str]) -> go.Figure | None:
    """Timeline of the planner build phases (when this run waited for them) and every planner call by command.

    run: The run. colors: Per command colors (report.COMMAND_COLORS).
    Returns: The figure (x = seconds since the run start), or None for results without start times.
    """
    if not run.commands or "start_s" not in run.commands[0]:
        return None
    fig, rows = go.Figure(), []
    build = run.planner.get("planner_build")
    if build and not build.get("reused"):
        phases = build.get("phases", [])
        names = [f"build: {p['name']}" for p in phases]
        rows += names
        offset = -build["before_run_s"]
        fig.add_trace(go.Bar(y=names, x=[p["seconds"] for p in phases],
                             base=[p["start_s"] + offset for p in phases], orientation="h", marker_color="#999",
                             hovertext=[f"{p['name']}: {p['seconds']:.2f}s, compile {p['compile_s']:.2f}s, "
                                        f"cache {p['cache_hits']} hits / {p['cache_misses']} misses" for p in phases],
                             hoverinfo="text", name="planner build"))  # fmt: skip
    _gantt_run(fig, {"commands": run.commands, "timings": [t for t in run.timings if t["call"] in CALLS]}, 0.0,
               "with obstacles", colors, rows)  # fmt: skip
    free = run.no_obstacles
    if free is not None and "started_at" in free and "started_at" in run.planner:
        _gantt_run(fig, free, free["started_at"] - run.planner["started_at"], "without obstacles", colors, rows)
    fig.update_layout(height=180 + 28 * len(rows), barmode="overlay", title="Planner timeline (s since the run start)",
                      xaxis_title="seconds", yaxis=dict(categoryorder="array", categoryarray=rows, autorange="reversed"))  # fmt: skip
    return fig


def _log_rows(lines: list[dict], phase: str) -> list[str]:
    """Table rows of captured planner log lines; compile/cache/wait/fallback lines and warnings highlighted.

    lines: planner_log entries {"t", "thread", "src", "level", "message", "command"}. phase: "build" or a run name.
    Returns: One <tr> per line.
    """
    rows = []
    for e in lines:
        notable = e["level"] in ("WARNING", "ERROR", "CRITICAL") or NOTABLE.search(e["message"])
        cmd = "" if e.get("command", -1) < 0 else e["command"]
        rows.append(f"<tr style='{_HIGHLIGHT if notable else ''}'><td>{phase}</td><td>{e['t']:.2f}</td><td>{cmd}</td>"
                    f"<td>{html.escape(e['thread'])}</td><td>{html.escape(e.get('src', ''))}</td><td>{e['level']}</td>"
                    f"<td>{html.escape(e['message'])}</td></tr>")  # fmt: skip
    return rows


def log_panel_html(run: Run) -> str:
    """Collapsible planner log: the build's and this run's log lines and cloud messages.

    run: The run.
    Returns: HTML fragment ("" for results without a captured log).
    """
    build, p = run.planner.get("planner_build") or {}, run.planner
    if "planner_log" not in p:
        return ""
    sources = [("build", build.get("log", []), build.get("messages", []), build.get("log_dropped", 0)),
               ("run", p["planner_log"], p.get("planner_messages", []), p.get("planner_log_dropped", 0))]  # fmt: skip
    if run.no_obstacles is not None and "planner_log" in run.no_obstacles:
        free = run.no_obstacles
        sources.append(("no obstacles", free["planner_log"], free["planner_messages"], free["planner_log_dropped"]))
    rows, messages, dropped = [], [], 0
    for phase, lines, msgs, n_dropped in sources:
        rows += _log_rows(lines, phase)
        messages += [f"<tr><td>{phase}</td><td>{m['t']:.2f}</td><td>{m['level']}</td><td>{html.escape(m['message'])}"
                     "</td></tr>" for m in msgs]  # fmt: skip
        dropped += n_dropped
    more = f" ({dropped} more lines not kept)" if dropped else ""
    msg_table = (f"<table><tr><th>when</th><th>t s</th><th>level</th><th>message</th></tr>{''.join(messages)}</table>"
                 if messages else "<p>No cloud messages. The resolver sends 'IK solver not ready yet — waiting for "
                 "background compilation' only when a solve arrives during its build; the simulation waits for the "
                 "build first, so that wait is the ik_total build phase above.</p>")  # fmt: skip
    return (
        f"<details><summary>Planner log: {len(rows)} lines{more}, {len(messages)} cloud messages "
        "(t = seconds since the build / run start; compile, cache, wait and fallback lines highlighted)</summary>"
        f"<h4>Cloud messages to the operator</h4>{msg_table}<table><tr><th>when</th><th>t s</th><th>cmd</th>"
        f"<th>thread</th><th>source</th><th>level</th><th>message</th></tr>{''.join(rows)}</table></details>"
    )
