"""Runs the plan step on a robot, to measure planning times on its Jetson GPU.

    python -m Simulation.world_sim.robot src/Simulation/world_sim/scenarios/*.json [--host bellboy@billie-29.bellboy]

1. sync: copies the billie-onboard planner code into a folder the robot's `billie` container sees.
   The code comes from the robot's own billie-onboard checkout (--robot-checkout, default
   ~/users/ronit/billie-onboard), or with --robot-checkout "" from the local billie-onboard submodule.
   The world_sim code, the scenarios and their recordings always come from this machine. Every run
   syncs, so the robot runs the checkout's current code.
2. run: the plan step inside the `billie` container with the live planner's environment
   (robot_run_plan.sh). The robot's /app code, planner process and JAX cache are not touched.
3. fetch: copies the results to output/world_sim/<robot>/ and writes their reports.
"""

import argparse
import glob
import json
import os
import subprocess
from typing import Callable

from Simulation.world_sim.recording import RECORDINGS_DIR, export_recording

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# On the robot host; mounted at the same path in the billie container.
ROBOT_DIR = "/home/bellboy/billie/world_sim"
# The robot the timings are measured on (Jetson Orin GPU).
DEFAULT_HOST = "bellboy@billie-29.bellboy"
# billie-onboard checkout on the robot to take the planner code from ("" = the local billie-onboard).
DEFAULT_ROBOT_CHECKOUT = "~/users/ronit/billie-onboard"
# (path inside a billie-onboard checkout, path under ROBOT_DIR) of the planner code the plan step imports.
_BILLIE_CODE = [
    ("billie/nodes/pyroki-planner/pyroki_planner", "billie/nodes/pyroki-planner/pyroki_planner"),
    ("billie/billie-utils/billie_utils", "billie/billie-utils/billie_utils"),
    ("billie/node_ticks.py", "billie/node_ticks.py"),
    ("billie/vendor/pyroki/src/pyroki", "pyroki/src/pyroki"),
]
_RSYNC = ["rsync", "-a", "--delete", "--exclude", "__pycache__", "--exclude", "*.html"]


def _local_billie_path(path: str) -> str:
    """Where a billie-onboard path is in this repo (its vendored pyroki submodule is empty here).

    path: Path inside a billie-onboard checkout.
    Returns: The local path (pyroki from the Simulation repo's own pyroki checkout, same commit).
    """
    if path.startswith("billie/vendor/pyroki/"):
        return os.path.join(_REPO, path.replace("billie/vendor/", "", 1))
    return os.path.join(_REPO, "billie-onboard", path)


def _dir_slash(path: str, is_dir: bool) -> str:
    """rsync source spelling: a trailing slash copies a folder's contents into the target folder.

    path: Source path. is_dir: Whether it is a folder.
    Returns: The path with exactly one trailing slash for folders, unchanged for files.
    """
    return path.rstrip("/") + ("/" if is_dir else "")


def _sh(args: list[str], capture: bool = False, log: Callable[[str], None] | None = None) -> str:
    """Runs a local command, failing loudly.

    args: Command and arguments. capture: Return its stdout instead of streaming it.
    log: Receives each output line (stdout and stderr) when not capturing; None streams to this console.
    Returns: stdout when captured, else "".
    """
    if capture or log is None:
        result = subprocess.run(args, check=True, text=True, capture_output=capture)
        return result.stdout if capture else ""
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = []  # the last lines, for the error message
    for line in process.stdout:
        log(line.rstrip())
        tail = (tail + [line])[-20:]
    if process.wait():
        raise subprocess.CalledProcessError(process.returncode, args, "".join(tail))
    return ""


def sync(host: str, scenarios: list[str], robot_checkout: str, log: Callable[[str], None] | None = None) -> None:
    """Copies the planner code, world_sim, the scenarios and their recordings to ROBOT_DIR.

    host: ssh destination, e.g. bellboy@billie-29.bellboy.
    scenarios: Local scenario files.
    robot_checkout: billie-onboard checkout on the robot to take the planner code from, or "" for
        the local billie-onboard submodule.
    log: Receives progress and command output lines; None prints them.
    """
    say = log or print
    for path in scenarios:
        with open(path) as f:
            for command in json.load(f).get("commands", []):
                if command.get("repo_id"):
                    export_recording(command["repo_id"], RECORDINGS_DIR)
    local_commit = _sh(["git", "-C", os.path.join(_REPO, "billie-onboard"), "log", "-1", "--format=%h %s"],
                       capture=True).strip()  # fmt: skip
    local_items = [(os.path.join(_REPO, "src", "Simulation"), "src/Simulation"), (RECORDINGS_DIR, "recordings")]
    local_items += [(p, f"scenarios/{os.path.basename(p)}") for p in scenarios]
    remotes = [r for _, r in _BILLIE_CODE] + [r for _, r in local_items]
    _sh(["ssh", host, "mkdir", "-p", *sorted({os.path.dirname(f"{ROBOT_DIR}/{r}") for r in remotes})])

    if robot_checkout:
        commit = _sh(["ssh", host, f"git -C {robot_checkout} log -1 --format='%h %s'"], capture=True).strip()
        say(f"Planner code: robot checkout {robot_checkout} at {commit} (local billie-onboard: {local_commit})")
        copies = [" ".join([*_RSYNC, _dir_slash(f"{robot_checkout}/{src}", "." not in os.path.basename(src)),
                            f"{ROBOT_DIR}/{dst}"]) for src, dst in _BILLIE_CODE]  # fmt: skip
        _sh(["ssh", host, " && ".join(copies)], log=log)
    else:
        say(f"Planner code: local billie-onboard at {local_commit}")
        local_items = [(_local_billie_path(src), dst) for src, dst in _BILLIE_CODE] + local_items
    for local, remote in local_items:
        _sh([*_RSYNC, _dir_slash(local, os.path.isdir(local)), f"{host}:{ROBOT_DIR}/{remote}"], log=log)
    _sync_arm_urdf(host, say)


def _sync_arm_urdf(host: str, say: Callable[[str], None] = print) -> None:
    """Copies the robot arm's URDF (and meshes) from a local developer env, for robots with no physical arm (no URDF, e.g. billie-29).

    robot_run_plan.sh uses the copy only when the robot's own $BILLIE_ENVDIR/urdf has no URDF for its arm.

    host: ssh destination. say: Receives the progress messages.
    """
    environ = r"""docker exec billie sh -c 'tr "\0" "\n" < /proc/$(pgrep -f bin/pyroki-planner | head -1)/environ'"""
    env = dict(line.split("=", 1) for line in _sh(["ssh", host, environ], capture=True).splitlines() if "=" in line)
    sn = env.get("XARM_SN", "")
    found = sorted(glob.glob(os.path.expanduser(f"~/releases/env/*/urdf/{sn}.urdf"))) if sn else []
    if not found:
        say(f"No local URDF for arm {sn or '?'}; the robot's own one must exist")
        return
    urdf_dir = os.path.dirname(found[0])
    _sh(["ssh", host, "mkdir", "-p", f"{ROBOT_DIR}/env/urdf/meshes"])
    _sh(["rsync", "-a", found[0], f"{host}:{ROBOT_DIR}/env/urdf/"])
    # Symlinks are copied as links: a gripper mesh linked into a missing base env stays missing, so the
    # planner uses its gripper box, as the robots without that mesh do.
    _sh([*_RSYNC, _dir_slash(os.path.join(urdf_dir, "meshes"), True), f"{host}:{ROBOT_DIR}/env/urdf/meshes/"])
    say(f"Arm URDF {sn}: synced a fallback copy from {urdf_dir}")


def run(host: str, scenarios: list[str], plan_args: list[str], log: Callable[[str], None] | None = None) -> None:
    """Runs the plan step in the robot's billie container; its output streams here.

    host: ssh destination. scenarios: Local scenario files (already synced).
    plan_args: Extra plan.py flags, e.g. ["--keep-going"].
    log: Receives the plan step's output lines; None streams them to this console.
    """
    remote = [f"scenarios/{os.path.basename(p)}" for p in scenarios]
    _sh(["ssh", host, "docker", "exec", "billie", "bash", f"{ROBOT_DIR}/src/Simulation/world_sim/robot_run_plan.sh",
         ROBOT_DIR, *remote, "--out", f"{ROBOT_DIR}/results", *plan_args], log=log)  # fmt: skip


def robot_name(host: str) -> str:
    """Short robot name of an ssh destination.

    host: e.g. bellboy@billie-29.bellboy. Returns: e.g. billie-29.
    """
    return host.split("@")[-1].split(".")[0]


def fetch(host: str, scenarios: list[str], folder: str | None = None) -> list[str]:
    """Copies the scenarios' results back.

    host: ssh destination. scenarios: Local scenario files.
    folder: Local folder for the results; default output/world_sim/<robot name>/.
    Returns: Local results files, named like the scenarios.
    """
    folder = folder or os.path.join(_REPO, "output", "world_sim", robot_name(host))
    os.makedirs(folder, exist_ok=True)
    out = []
    for path in scenarios:
        name = os.path.splitext(os.path.basename(path))[0] + ".json"
        _sh(["rsync", "-a", f"{host}:{ROBOT_DIR}/results/{name}", os.path.join(folder, name)])
        out.append(os.path.join(folder, name))
    return out


def main() -> None:
    """Command line: sync, run and fetch, then write a report per scenario."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenarios", nargs="+", help="scenario JSON files")
    parser.add_argument("--host", default=DEFAULT_HOST, help="robot ssh destination")
    parser.add_argument("--robot-checkout", default=DEFAULT_ROBOT_CHECKOUT,
                        help='billie-onboard checkout on the robot to take the planner code from ("" = local code)')
    parser.add_argument("--keep-going", action="store_true", help="run the remaining commands after a failure")
    parser.add_argument("--continue-on-block", action="store_true", help="play moves the guard refuses, flagged")
    args = parser.parse_args()
    plan_args = [f for f, on in (("--keep-going", args.keep_going), ("--continue-on-block", args.continue_on_block)) if on]

    sync(args.host, args.scenarios, args.robot_checkout)
    run(args.host, args.scenarios, plan_args)
    from Simulation.world_sim.analysis import load_run
    from Simulation.world_sim.report import write_report

    for path in fetch(args.host, args.scenarios):
        report = os.path.splitext(path)[0] + ".html"
        write_report(load_run(path), report, open_browser=False)
        print(f"Report: {report}")


if __name__ == "__main__":
    main()
