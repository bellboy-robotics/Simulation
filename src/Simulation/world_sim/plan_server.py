"""Planner worker for a connected robot: builds the planner once, then plans every scenario sent to it.

Started by robot_link.py over ssh inside the robot's billie container (robot.worker_command), so a Run
on that robot skips the sync and the planner build. One JSON object per line:
- in (stdin):  {"id": "...", "scenario": {...}, "flags": {...}}, or {"cmd": "quit"}
- out (stdout): lines starting with PREFIX, then {"event": "ready", "build_s", "host", "devices"} once the
  planner is built, {"event": "result", "id", "results"} or {"event": "error", "id", "error"} per job.
Everything else (log lines, on stderr) is the worker's log. The worker ends on "quit", when its stdin
closes (the editor or the ssh connection went away), or after IDLE_TIMEOUT_S without a job, so it does
not keep the robot's GPU memory for nothing.
"""

import json
import logging
import os
import select
import socket
import sys
import traceback

from Simulation.world_sim.plan import _to_jsonable, plan_scenario
from Simulation.world_sim.planner import SimPlanner
from Simulation.world_sim.robot import WORKER_PREFIX as PREFIX

# [s] The worker exits after this long without a job (WORLD_SIM_IDLE_S overrides): it holds GPU memory.
IDLE_TIMEOUT_S = float(os.environ.get("WORLD_SIM_IDLE_S", 3600))


def _emit(message: dict) -> None:
    """Writes one protocol line to stdout.

    message: The event (NumPy values allowed).
    """
    sys.stdout.write(PREFIX + json.dumps(_to_jsonable(message)) + "\n")
    sys.stdout.flush()


def _serve(planner: SimPlanner) -> None:
    """Plans jobs from stdin until quit, end of input or the idle timeout.

    planner: The built planner, reused for every job.
    """
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], IDLE_TIMEOUT_S)
        if not ready:
            logging.info(f"No job for {IDLE_TIMEOUT_S:.0f}s; the worker exits to free the robot's GPU memory.")
            return
        line = sys.stdin.readline()
        if not line:
            return  # the connection closed
        job = json.loads(line)
        if job.get("cmd") == "quit":
            return
        try:
            results = plan_scenario(planner, job["scenario"], job.get("flags", {}), "connected run")
            _emit({"event": "result", "id": job["id"], "results": results})
        except Exception as e:  # the job fails, the worker stays up for the next one
            logging.error(traceback.format_exc())
            _emit({"event": "error", "id": job.get("id"), "error": f"{type(e).__name__}: {e}"})


def main() -> None:
    """Builds the planner, says it is ready, then serves jobs."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")
    planner = SimPlanner()
    build_s = sum(t["seconds"] for t in planner.timings if t["call"].startswith("build_"))
    from jax import devices  # noqa: PLC0415 (after the planner picked the JAX platform)

    _emit({"event": "ready", "build_s": round(build_s, 1), "host": socket.gethostname(), "devices": [str(d) for d in devices()]})
    _serve(planner)


if __name__ == "__main__":
    main()
