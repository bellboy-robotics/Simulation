"""Judges and compares the two planners' kitchen runs (compare_run.py output).

For every scenario of every run: the planned / executed joint path is judged against the real
office-2 map with the arm's real link meshes (map_judge), and the command gets one outcome:
- "success": the planner returned a path, it reached the target, and the judge found it clear;
- "collision": the path comes closer than CLEARANCE_M to the observed map (checked before "unseen");
- "unseen": the path touches space the map never observed (and is otherwise clear);
- "not_reached": the path ends away from the target (pose: > POSE_TOL_MM or > POSE_TOL_DEG, joints:
  > JOINT_TOL_DEG on a joint, replay: TCP > REPLAY_TOL_MM from the last recorded pose);
- "refused": the planner returned an error (no motion).
Then per run: success rate per command type, planning time, GPU memory and warm-up time.

    python -m Simulation.world_sim.compare_report <sweep folder>
"""

import argparse
import glob
import json
import os

import numpy as np
from billie_utils.arm_collision_model import CollisionModel, link_transforms
from scipy.spatial.transform import Rotation

from Simulation.world_sim.compare_run import scenario_map
from Simulation.world_sim.map_judge import CLEARANCE_M, judge_path, map_from_arm
from Simulation.world_sim.mesh_check import mesh_points
from Simulation.world_sim.office_map import load_office_map
from Simulation.world_sim.recording import load_recording

# [mm], [deg] A pose command reached its target within these (the ESDF branch's own default tolerance).
POSE_TOL_MM, POSE_TOL_DEG = 5.0, 1.0
# [deg] A joints command reached its target when every joint is this close.
JOINT_TOL_DEG = 0.5
# [mm] A replay reached its end when the TCP is this close to the last recorded pose (the brain's check).
REPLAY_TOL_MM = 50.0
OUTCOMES = ("success", "collision", "unseen", "not_reached", "refused")


def _tcp(model: CollisionModel, joints_deg: np.ndarray) -> np.ndarray:
    """TCP pose of one configuration.

    model: Arm model. joints_deg: (6,) degrees.
    Returns: (4, 4) TCP frame in the xArm base frame, meters.
    """
    return link_transforms(model, np.deg2rad(np.asarray(joints_deg, float))[None])[0, model.link_names.index("link_tcp")]


def _reached(model: CollisionModel, scenario: dict, final_deg: np.ndarray) -> bool:
    """Whether the path's last configuration reaches the scenario command's target.

    model: Arm model. scenario: The scenario (one command). final_deg: (6,) last configuration, degrees.
    Returns: True when within the tolerances of the module docstring.
    """
    spec = scenario["commands"][0]
    if spec["cmd"] == "joints":
        return bool(np.abs(np.asarray(spec["joints"]) - final_deg).max() <= JOINT_TOL_DEG)
    T = _tcp(model, final_deg)
    if spec["cmd"] == "pose":
        target = np.asarray(spec["pose"], float)
        error_mm = np.linalg.norm(T[:3, 3] * 1000 - target[:3])
        rot = Rotation.from_matrix(T[:3, :3]).inv() * Rotation.from_rotvec(target[3:])
        return bool(error_mm <= POSE_TOL_MM and np.degrees(rot.magnitude()) <= POSE_TOL_DEG)
    last = load_recording(spec["repo_id"])[0][-1].position
    return bool(np.linalg.norm(T[:3, 3] * 1000 - last[:3]) <= REPLAY_TOL_MM)


def _path_and_status(result: dict) -> tuple[np.ndarray, bool, str | None, float]:
    """The joint path, planner ok flag, error and planning time of a run's scenario result (either side).

    result: One results/*.json (sweep_check for "capsules", esdf_runner for "esdf").
    Returns: ((W, 6) waypoints in degrees, start first; ok; error message; seconds).
    """
    start = np.asarray(result["start_joints_deg"], float)
    rows = result.get("path") if result.get("planner") == "esdf" else result.get("points", [])
    waypoints = [np.asarray(p["joints_deg"], float) for p in rows if p.get("command", 0) == 0]
    command = result["commands"][0]
    return np.vstack([start, *waypoints]), bool(command["ok"]), command.get("error"), float(command["seconds"])


def judge_run(run_folder: str, model: CollisionModel, office, points) -> list[dict]:
    """Judges every scenario of one run.

    run_folder: <sweep>/runs/<side>-<label>. model: The robot's arm model. office: The map.
    points: The arm's real link meshes (mesh_check.MeshPoints).
    Returns: One row per scenario: name, kind, outcome, seconds, clearance, error.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(run_folder, "results", "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(run_folder, "scenarios", f"{name}.json")) as f:
            scenario = json.load(f)
        with open(path) as f:
            result = json.load(f)
        waypoints, ok, error, seconds = _path_and_status(result)
        verdict = judge_path(office, model, waypoints, map_from_arm(scenario["billie"]), points)
        if not ok:
            outcome = "refused"
        elif verdict.min_clearance_m < CLEARANCE_M:
            outcome = "collision"
        elif verdict.unseen:
            outcome = "unseen"
        elif not _reached(model, scenario, waypoints[-1]):
            outcome = "not_reached"
        else:
            outcome = "success"
        reached = ok and _reached(model, scenario, waypoints[-1])
        rows.append({"name": name, "kind": scenario["kitchen"]["kind"], "outcome": outcome, "seconds": seconds,
                     "ok_if_unseen_allowed": outcome == "success" or (outcome == "unseen" and reached),
                     "min_clearance_mm": round(verdict.min_clearance_m * 1000, 1), "worst_link": verdict.worst_link,
                     "error": (error or "")[:300], "memory": result.get("memory")})  # fmt: skip
    return rows


def warmup(run_folder: str) -> dict:
    """The planner's start-up times and memory of one run (either side).

    run_folder: <sweep>/runs/<side>-<label>.
    Returns: {"total_s", "stages": {name: seconds}, "memory"}.
    """
    finished = os.path.join(run_folder, "finished.json")
    if os.path.exists(finished):  # esdf_runner
        with open(finished) as f:
            build = json.load(f)["build"]
        stages = {"ik_ready": build["ik_ready_s"], **{f"{k} ready": v for k, v in build["prepared_s"].items()}}
        return {"total_s": build["total_s"], "stages": stages, "memory": build.get("memory")}
    with open(os.path.join(run_folder, "sweep.json")) as f:  # sweep_check
        summary = json.load(f)
    stages = {b["call"]: b["seconds"] for b in summary.get("build", [])}
    return {"total_s": sum(stages.values()), "stages": stages, "memory": summary.get("build_memory")}


# A refusal caused by this harness, not the planner: the fake mapper's snapshot went stale (esdf_runner).
_HARNESS_ERRORS = ("stale or invalid timestamp",)


def drop_harness_errors(report: dict) -> list[str]:
    """Removes, from every run, the scenarios any run lost to a harness error (so both sides stay equal).

    report: {run: {"rows": [...]}} from judge_run. Returns: The dropped scenario names.
    """
    bad = sorted({r["name"] for run in report.values() for r in run["rows"]
                  if any(h in r["error"] for h in _HARNESS_ERRORS)})  # fmt: skip
    for run in report.values():
        run["rows"] = [r for r in run["rows"] if r["name"] not in bad]
    return bad


def summarize(rows: list[dict]) -> dict:
    """Per command type: outcome counts, success rate and planning-time percentiles.

    rows: From judge_run.
    Returns: {kind: {"n", outcome counts, "success_pct", "seconds_p50", "seconds_p95", "seconds_max"}}.
    """
    out = {}
    for kind in ("joints", "pose", "replay", "all"):
        sel = [r for r in rows if kind == "all" or r["kind"] == kind]
        if not sel:
            continue
        seconds = np.array([r["seconds"] for r in sel])
        counts = {o: sum(r["outcome"] == o for r in sel) for o in OUTCOMES}
        out[kind] = {"n": len(sel), **counts, "success_pct": 100.0 * counts["success"] / len(sel),
                     "success_if_unseen_ok_pct": 100.0 * sum(r["ok_if_unseen_allowed"] for r in sel) / len(sel),
                     "seconds_p50": float(np.percentile(seconds, 50)), "seconds_p95": float(np.percentile(seconds, 95)),
                     "seconds_max": float(seconds.max())}  # fmt: skip
    peaks = [r["memory"]["peak_bytes_in_use"] for r in rows if r.get("memory") and "peak_bytes_in_use" in r["memory"]]
    out["peak_gpu_mb"] = max(peaks) / 2**20 if peaks else None
    return out


def main() -> None:
    """Command line: judge every run of a sweep, print the comparison and save compare.json."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep", help="kitchen sweep folder with runs/")
    parser.add_argument("--model", help="arm model JSON; default: from the first capsules run's results")
    parser.add_argument("--runs", default="*", help="which runs/ folders, a glob (default all)")
    args = parser.parse_args()
    runs = sorted(glob.glob(os.path.join(args.sweep, "runs", args.runs)))
    if args.model:
        model_json = open(args.model).read()
    else:
        first = sorted(glob.glob(os.path.join(args.sweep, "runs", "capsules-*", "results", "*.json")))[0]  # same robot arm for both sides
        model_json = json.load(open(first))["model_json"]
    model, points, maps = CollisionModel.from_json(model_json), mesh_points(), {}
    report = {}
    for run in runs:
        path = scenario_map(run)  # each map version is judged on its own map
        office = maps.setdefault(str(path), load_office_map(path))
        rows = judge_run(run, model, office, points)
        report[os.path.basename(run)] = {"warmup": warmup(run), "rows": rows}
    dropped = drop_harness_errors(report)
    print(f"Left out of every run (harness errors, not planner results): {dropped or 'none'}")
    for run in report:
        report[run]["summary"] = summarize(report[run]["rows"])
        s = report[run]["summary"]
        print(f"\n== {run}  (warm-up {report[run]['warmup']['total_s']:.0f}s, "
              f"peak GPU {s['peak_gpu_mb'] or float('nan'):.0f} MB)")  # fmt: skip
        for kind in ("joints", "pose", "replay", "all"):
            if kind in s:
                k = s[kind]
                print(f"  {kind:7s} n={k['n']:3d} success {k['success_pct']:5.1f}% (unseen allowed {k['success_if_unseen_ok_pct']:5.1f}%)  " +
                      " ".join(f"{o}={k[o]}" for o in OUTCOMES[1:]) +
                      f"  time p50 {k['seconds_p50']:.2f}s p95 {k['seconds_p95']:.2f}s max {k['seconds_max']:.1f}s")  # fmt: skip
    with open(os.path.join(args.sweep, f"compare-{args.runs.replace('*', 'all')}.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nsaved compare-{args.runs.replace('*', 'all')}.json")


if __name__ == "__main__":
    main()
