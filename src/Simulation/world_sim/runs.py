"""Editor runs: a scenario planned on this machine and/or on robots over ssh, every run kept with its reports.

Each Run is archived in output/world_sim/runs/<run id>/ (run id = <time>_<scenario file>):
    run.json           what ran: scenario name and file, note, flags, and per machine its status,
                       log tail and timing summary
    <run id>.json      the scenario exactly as it was run (also the file synced to the robots)
    <machine>.json     the plan step's results on that machine ("local" or the robot name)
    <machine>.html     its report
runs/index.html lists every run with its reports and a per-command timing comparison; it is
rewritten after each machine finishes and opens from disk or through the editor at /runs/index.html.
"""

import html
import json
import os
import shutil
import threading
import time
import traceback
from typing import Callable

from Simulation.world_sim import robot
from Simulation.world_sim.analysis import Run, load_run
from Simulation.world_sim.report import playback_estimate, write_report

RUNS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "output", "world_sim", "runs"))
LOCAL = "local"  # target name of this machine
# Log lines kept per machine (the page shows the tail; the full output is in the console).
_LOG_LINES = 200

_runs: dict[str, dict] = {}  # run id -> run.json content, for runs started by this process
_lock = threading.Lock()  # guards _runs and the run.json / index.html writes


def start_run(
    scenario: dict, file: str, note: str, flags: dict, targets: list[str], robot_checkout: str,
    plan_locally: Callable[[dict, dict], dict],
) -> dict:  # fmt: skip
    """Archives the scenario and plans it on every target at once, each in its own thread.

    scenario: The scenario as edited on the page.
    file: Its file name (names the run); "" for an unsaved scenario.
    note: Free text saying what this run tests.
    flags: {"keep_going": bool, "continue_on_block": bool}, the plan.py flags.
    targets: LOCAL and/or ssh destinations like bellboy@billie-29.bellboy.
    robot_checkout: billie-onboard checkout on the robots for the planner code ("" = local code).
    plan_locally: (scenario, flags) -> plan results dict, run with this machine's planner.
    Returns: The run's meta (see the module docstring), with "id".
    """
    stem = os.path.splitext(os.path.basename(file or "editor.json"))[0]
    with _lock:
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{stem}"
        while os.path.exists(os.path.join(RUNS_DIR, run_id)):
            run_id += "+"
        os.makedirs(os.path.join(RUNS_DIR, run_id))
    with open(os.path.join(RUNS_DIR, run_id, f"{run_id}.json"), "w") as f:
        json.dump(scenario, f, indent=1)
    meta = {
        "id": run_id, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "name": scenario.get("name", stem),
        "file": file, "note": note, "flags": flags, "robot_checkout": robot_checkout,
        "targets": {_target_name(t): {"host": None if t == LOCAL else t, "status": "queued", "stage": "",
                                      "error": None, "summary": None, "log": []}
                    for t in dict.fromkeys(targets)},
    }  # fmt: skip
    with _lock:
        _runs[run_id] = meta
        _save_meta(meta)
    for name in meta["targets"]:
        threading.Thread(target=_run_target, args=(meta, name, plan_locally), daemon=True).start()
    return meta


def _target_name(target: str) -> str:
    """File-friendly name of a target. target: LOCAL or an ssh destination. Returns: "local" or the robot name."""
    return LOCAL if target == LOCAL else robot.robot_name(target)


def _run_target(meta: dict, name: str, plan_locally: Callable[[dict, dict], dict]) -> None:
    """Plans the run's scenario on one machine, then writes its results, report and timing summary.

    meta: The run's meta (updated in place and saved as the machine progresses).
    name: The machine's target name in meta["targets"].
    plan_locally: See start_run.
    """
    target, folder = meta["targets"][name], os.path.join(RUNS_DIR, meta["id"])
    scenario_file = os.path.join(folder, f"{meta['id']}.json")
    results = os.path.join(folder, f"{name}.json")

    def log(line: str) -> None:
        """Keeps one output line of this machine. line: The text."""
        target["log"] = (target["log"] + [line])[-_LOG_LINES:]

    def stage(text: str) -> None:
        """Marks what this machine is doing now. text: The stage."""
        target.update(status="running", stage=text)
        log(f"--- {text}")
        with _lock:
            _save_meta(meta)

    try:
        if target["host"] is None:
            stage("planning on this machine")
            with open(scenario_file) as f:
                planned = plan_locally(json.load(f), meta["flags"])
            from Simulation.world_sim.plan import save_results  # noqa: PLC0415 (imports JAX)

            save_results(planned, results)
        else:
            flags = [f"--{k.replace('_', '-')}" for k, on in meta["flags"].items() if on]
            stage("syncing code and scenario")
            robot.sync(target["host"], [scenario_file], meta["robot_checkout"], log=log)
            stage("planning on the robot")
            robot.run(target["host"], [scenario_file], flags, log=log)
            stage("fetching results")
            fetched = robot.fetch(target["host"], [scenario_file], folder=os.path.join(folder, name))
            shutil.move(fetched[0], results)
            shutil.rmtree(os.path.join(folder, name), ignore_errors=True)
        stage("writing the report")
        run = load_run(results)
        write_report(run, os.path.join(folder, f"{name}.html"), open_browser=False)
        target.update(status="done", stage="", summary=timing_summary(run))
    except Exception as e:  # the machine's failure is shown on the page; other machines go on
        log(traceback.format_exc())
        target.update(status="failed", stage="", error=f"{type(e).__name__}: {e}")
    with _lock:
        _save_meta(meta)
        write_index()


def timing_summary(run: Run) -> dict:
    """Outcome and planner time of a run, per command and in total, for comparing machines.

    run: The planned run.
    Returns: {"host", "devices", "build_s" (planner build or cache load), "planner_s" (all planner calls),
        "estimate_s" (robot time estimate), "failed", "commands": per command {"cmd", "ok", "planner_s",
        "estimate_s": replay wall time with the BufferingPlayer for replays, else the planning time}}.
    """
    commands = []
    for c, outcome in enumerate(run.commands):
        calls = [t for t in run.timings if t["command"] == c]
        planner_s = sum(t["seconds"] for t in calls)
        batches = [t for t in calls if t["call"] == "batch_solve"]
        estimate_s = planner_s
        if batches:
            wall, _ = playback_estimate([t["seconds"] for t in batches], [t["n"] for t in batches])
            estimate_s = sum(t["seconds"] for t in calls if t["call"] in ("detour", "transit")) + wall
        commands.append({"cmd": outcome["spec"]["cmd"], "ok": outcome["ok"], "planner_s": round(planner_s, 3),
                         "estimate_s": round(estimate_s, 3)})  # fmt: skip
    return {
        "host": run.machine.get("host", "?"),
        "devices": ", ".join(run.machine.get("jax_devices", [])),
        "build_s": round(sum(t["seconds"] for t in run.timings if t["call"].startswith("build_")), 2),
        "planner_s": round(sum(c["planner_s"] for c in commands), 3),
        "estimate_s": round(sum(c["estimate_s"] for c in commands), 3),
        "failed": sum(not c["ok"] for c in commands),
        "commands": commands,
    }


def _save_meta(meta: dict) -> None:
    """Writes a run's run.json (call with _lock held). meta: The run's meta."""
    with open(os.path.join(RUNS_DIR, meta["id"], "run.json"), "w") as f:
        json.dump(meta, f, indent=1)


def get_run(run_id: str) -> dict:
    """A run's meta, live for runs of this process.

    run_id: The run id. Returns: Its meta. Raises: FileNotFoundError for an unknown run.
    """
    if run_id in _runs:
        return _runs[run_id]
    with open(os.path.join(RUNS_DIR, os.path.basename(run_id), "run.json")) as f:
        return json.load(f)


def list_runs() -> list[dict]:
    """Every archived run, newest first.

    Returns: Run metas (live ones from memory).
    """
    if not os.path.isdir(RUNS_DIR):
        return []
    ids = sorted((d for d in os.listdir(RUNS_DIR) if os.path.exists(os.path.join(RUNS_DIR, d, "run.json"))), reverse=True)
    return [get_run(i) for i in ids]


def delete_run(run_id: str) -> None:
    """Removes a finished run's folder and rewrites the index.

    run_id: The run id. Raises: ValueError while one of its machines is still running.
    """
    meta = get_run(run_id)
    if any(t["status"] in ("queued", "running") for t in meta["targets"].values()):
        raise ValueError("The run is still running")
    with _lock:
        _runs.pop(run_id, None)
        shutil.rmtree(os.path.join(RUNS_DIR, os.path.basename(run_id)))
        write_index()


def results_path(run_id: str, name: str) -> str:
    """Results file of one machine of a run.

    run_id: The run id. name: The machine's target name. Returns: The path (may not exist yet).
    """
    return os.path.join(RUNS_DIR, os.path.basename(run_id), f"{os.path.basename(name)}.json")


def write_index() -> None:
    """Rewrites runs/index.html: every run with its machines, reports and per-command timing (call with _lock held)."""
    rows = []
    for meta in list_runs():
        cells, names = [], list(meta["targets"])
        for name, t in meta["targets"].items():
            s = t["summary"]
            if t["status"] == "done":
                link = f'<a href="{meta["id"]}/{name}.html">report</a>'
                state = "✓" if not s["failed"] else f'<span class="bad">{s["failed"]} failed</span>'
                cells.append(f"<b>{name}</b> {state} · planner {s['planner_s']:.1f}s · estimate {s['estimate_s']:.1f}s · {link}")
            else:
                cells.append(f"<b>{name}</b> {html.escape(t['status'])} {html.escape(t['error'] or '')}")
        done = [n for n in names if meta["targets"][n]["status"] == "done"]
        detail = ""
        if done:
            n_commands = max(len(meta["targets"][n]["summary"]["commands"]) for n in done)
            head = "".join(f"<th>{n}<br>planner / estimate</th>" for n in done)
            lines = []
            for c in range(n_commands):
                per = [meta["targets"][n]["summary"]["commands"] for n in done]
                cmd = next(p[c]["cmd"] for p in per if c < len(p))
                tds = "".join(f"<td>{p[c]['planner_s']:.2f}s / {p[c]['estimate_s']:.2f}s{'' if p[c]['ok'] else ' ✗'}</td>"
                              if c < len(p) else "<td>—</td>" for p in per)  # fmt: skip
                lines.append(f"<tr><td>{c + 1}. {cmd}</td>{tds}</tr>")
            builds = "".join(f"<td>{meta['targets'][n]['summary']['build_s']:.1f}s</td>" for n in done)
            detail = (f"<details><summary>timing per command</summary><table><tr><th>command</th>{head}</tr>"
                      f"{''.join(lines)}<tr><td>planner build</td>{builds}</tr></table></details>")  # fmt: skip
        rows.append(
            f"<tr><td>{meta['created']}</td><td>{html.escape(meta['name'])}<br>"
            f'<a href="{meta["id"]}/{meta["id"]}.json">scenario</a></td><td>{html.escape(meta["note"] or "")}</td>'
            f"<td>{'<br>'.join(cells)}{detail}</td></tr>"
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>World sim runs</title><style>
body {{ font: 13px -apple-system, sans-serif; margin: 20px; }} table {{ border-collapse: collapse; }}
td, th {{ border: 1px solid #ddd; padding: 4px 8px; vertical-align: top; text-align: left; }}
.bad {{ color: #c62f2f; }} details table {{ margin-top: 4px; }}
</style></head><body><h2>World sim runs</h2><table><tr><th>when</th><th>scenario</th><th>note</th>
<th>machines</th></tr>{''.join(rows)}</table></body></html>"""
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(os.path.join(RUNS_DIR, "index.html"), "w") as f:
        f.write(page)

