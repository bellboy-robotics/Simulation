"""Summary of a finished sweep: what failed and what was slow, with a repro JSON and a report for each.

Writes <sweep>/index.html and copies every flagged repro into the editor's scenarios folder as
<sweep id>-<scenario>-c<command>.json.
"""

import html
import json
import os
import shutil

import numpy as np

from Simulation.world_sim.analysis import load_run
from Simulation.world_sim.report import write_report

SCENARIOS_DIR = os.path.join(os.path.dirname(__file__), "scenarios")
# Problem names by problem kind (see sweep_check.py), in the order the page lists them.
PROBLEMS = {
    "1. planner misses a solution": ("solver_miss", "ik_miss"),
    "1. planner leaves the arm in collision": ("end_state_invalid",),
    "2. too slow on the robot": ("replay_stall", "slow_single_move", "slow_replay_start", "recompile"),
    "probably no solution (oracle found no path either)": ("no_path_found",),
    "not judged (invalid goal on this arm, or invalid start)": ("goal_invalid", "untestable_start"),
}


def _time_stats(rows: list[dict], cmd: str, key: str) -> str:
    """Median / 90th percentile / max of one timing value over one command type.

    rows: sweep.json rows. cmd: Command name. key: Timing key, e.g. "planner_s".
    Returns: Text like "n=120 median 0.21s p90 1.8s max 6.4s", or "—".
    """
    values = np.array([r["timing"][key] for r in rows if r["cmd"] == cmd and r["timing"].get(key) is not None])
    if not len(values):
        return "—"
    return f"n={len(values)} median {np.median(values):.2f}s p90 {np.percentile(values, 90):.2f}s max {values.max():.2f}s"


def _row_html(row: dict, sweep_id: str) -> str:
    """One flagged command as a table row with links to its repro scenario and its scenario's report.

    row: A sweep.json row. sweep_id: The sweep's folder name.
    Returns: HTML table row.
    """
    stem = row["scenario"][:-5]
    repro = f"{sweep_id}-{stem}-c{row['command']}.json"
    t = row["timing"]
    timing = f"planner {t['planner_s']:.2f}s" + (f", max batch {t['max_batch_s']:.2f}s" if t.get("max_batch_s") else "") \
        + (f", before motion {t['before_motion_s']:.2f}s" if row["cmd"] == "replay_policy" else "") \
        + (f", compile {t['compile_s']:.2f}s" if t["compile_s"] > 0.05 else "")  # fmt: skip
    pose = f"<br>pose off by {row['pose_error'][0]:.0f}mm / {row['pose_error'][1]:.1f}°" if row.get("pose_error") else ""
    pose += f"<br>arm ends in: {', '.join(row['end_broken'])}" if row.get("end_broken") else ""
    pose += f"<br>ends on the limit of J{', J'.join(map(str, row['at_limits']))}" if row.get("at_limits") else ""
    pose += f"<br>goal breaks: {', '.join(row['goal_broken'])}" if row.get("goal_broken") else ""
    return (
        f"<tr><td>{stem} c{row['command']}</td><td>{row['cmd']}</td><td>{', '.join(row['flags'])}</td>"
        f"<td>{html.escape(row['error'] or 'ok')}{pose}</td><td>{timing}</td><td>{row.get('oracle') or '—'}</td>"
        f"<td>{'yes' if row.get('reproduced') else '<b>no</b>'}</td>"
        f"<td><a href='flagged/{stem}-c{row['command']}.json'>{repro}</a> · <a href='results/{stem}.html'>report</a></td></tr>"
    )


def summarize(folder: str) -> None:
    """Writes the sweep's index.html, the reports of flagged scenarios, and copies the repros for the editor.

    folder: Finished sweep folder (sweep.json, results/, flagged/).
    """
    sweep_id = os.path.basename(folder)
    with open(os.path.join(folder, "sweep.json")) as f:
        summary = json.load(f)
    rows = summary["rows"]
    flagged = [r for r in rows if r["flags"]]
    for name in sorted({r["scenario"] for r in flagged}):
        write_report(load_run(os.path.join(folder, "results", name)), os.path.join(folder, "results", name[:-5] + ".html"), open_browser=False)
    for path in sorted(os.listdir(os.path.join(folder, "flagged"))):
        shutil.copy(os.path.join(folder, "flagged", path), os.path.join(SCENARIOS_DIR, f"{sweep_id}-{path}"))

    sections, counts = [], []
    head = "<tr><th>case</th><th>command</th><th>flags</th><th>result</th><th>timing</th><th>oracle</th><th>reproduces</th><th>files</th></tr>"
    for title, names in PROBLEMS.items():
        chosen = [r for r in flagged if set(r["flags"]) & set(names)]
        counts.append(f"{title}: {len(chosen)}")
        body = "".join(_row_html(r, sweep_id) for r in chosen) or "<tr><td colspan=8>none</td></tr>"
        sections.append(f"<h3>{html.escape(title)} ({len(chosen)})</h3><table>{head}{body}</table>")
    build = ", ".join(f"{t['call'][6:]} {t['seconds']:.1f}s" for t in summary.get("build", []))
    timing = "".join(
        f"<tr><td>{cmd}</td><td>{_time_stats(rows, cmd, 'planner_s')}</td><td>{_time_stats(rows, cmd, 'max_batch_s')}</td>"
        f"<td>{_time_stats(rows, cmd, 'before_motion_s')}</td></tr>"
        for cmd in ("joints", "pose", "replay_policy")
    )
    machine = summary.get("machine") or {}
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{sweep_id}</title><style>
body {{ font: 13px -apple-system, sans-serif; margin: 20px; }} table {{ border-collapse: collapse; margin-bottom: 8px; }}
td, th {{ border: 1px solid #ddd; padding: 3px 8px; vertical-align: top; text-align: left; }}</style></head><body>
<h2>{sweep_id}</h2><p>{len({r['scenario'] for r in rows})} scenarios, {len(rows)} commands, {len(flagged)} flagged ·
ran on <b>{html.escape(str(machine.get('host', '?')))}</b> {html.escape(str(machine.get('jax_devices', '')))} ·
planner build {build} · sweep {summary.get('elapsed_s', 0) / 60:.1f} min{'' if summary.get('finished') else ' · <b>not finished</b>'}</p>
<h3>Planner time per command (all commands)</h3><table><tr><th>command</th><th>planner time</th><th>slowest replay batch</th>
<th>planning before a replay moves</th></tr>{timing}</table>{''.join(sections)}
<p>Repro JSONs are also in the editor's scenarios list as {sweep_id}-…; each has a "flag" block with the problem,
the goal joints, and for solver misses the oracle's collision-free path.</p></body></html>"""
    with open(os.path.join(folder, "index.html"), "w") as f:
        f.write(page)
    print("\n".join(counts))
    print(f"Summary: {os.path.join(folder, 'index.html')}")
