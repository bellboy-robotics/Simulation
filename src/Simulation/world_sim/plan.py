"""Plan step: runs a scenario's commands through the pyroki planner and saves what the arm did.

Headless and needs only the planner, so it can run on a GPU machine; view.py shows the result.

    python -m Simulation.world_sim.plan scenarios/table_block.json --out output/world_sim/table_block.json
"""

import argparse
import inspect
import json
import logging
import os
import time
import socket
import traceback

import numpy as np
from billie_utils import arm_motion_guard_poc

from Simulation.world_sim.commands import SimBrain, run_command
from Simulation.world_sim.planner import SimPlanner
from Simulation.world_sim.sim_arm import SimArm
from Simulation.world_sim.world import World, load_scenario

# Imported after the planner modules: the resolver picks the JAX platform (cpu/cuda) at its import.
from jax import devices as jax_devices  # noqa: E402


def run_scenario(
    planner: SimPlanner,
    world: World,
    start_joints_deg: np.ndarray,
    commands: list[dict],
    continue_on_block: bool = False,
    stop_on_error: bool = True,
) -> dict:
    """Loads the world's obstacles into the planner and runs the commands in order.

    planner: The simulated planner (built once, reusable across scenarios).
    world: Objects around the arm; only the "avoid" ones are sent to the planner.
    start_joints_deg: (6,) joints the arm starts at, degrees.
    commands: Command specs, see commands.run_command.
    continue_on_block: Keep playing moves the arm-move guard refuses (flagged), instead of failing.
    stop_on_error: Stop at the first failed command, like a brain script; else run the rest anyway.
    Returns: Results dict (see save_results) with the arm's trajectory, operator messages and command outcomes.
    """
    planner.set_obstacles(*world.capsules("avoid"))
    first_timing = len(planner.timings)
    arm = SimArm(start_joints_deg, continue_on_block=continue_on_block)
    brain = SimBrain(planner, arm)
    outcomes = []
    for index, spec in enumerate(commands):
        arm.command = planner.command = index
        t0 = time.time()
        error = None
        try:
            run_command(brain, spec)
        except (arm_motion_guard_poc.ArmMoveBlockedError, RuntimeError, ValueError) as e:
            error = str(e)
            brain.log(f"{spec['cmd']} failed: {e}", "ERROR")
            logging.debug(traceback.format_exc())
        outcomes.append({"spec": spec, "ok": error is None, "error": error, "seconds": time.time() - t0})
        if error is not None and stop_on_error:
            break
    return {
        "model_json": planner.model.to_json(),  # the arm model, so viewing needs no planner
        "start_joints_deg": start_joints_deg,
        "objects": [{"name": o.name, "role": o.role, "spec": o.spec, "starts_m": o.starts_m,
                     "ends_m": o.ends_m, "radii_m": o.radii_m} for o in world.objects],  # fmt: skip
        "points": [vars(p) for p in arm.points],
        "events": brain.events,
        "commands": outcomes,
        # Planner build (compile or cache load) and every solve of this scenario, for the timing report.
        "timings": [t for t in planner.timings if t["call"].startswith("build_")] + planner.timings[first_timing:],
        "machine": {"host": socket.gethostname(), "jax_devices": [str(d) for d in jax_devices()],
                    "planner_code": inspect.getfile(type(planner.resolver))},  # fmt: skip
    }


def _to_jsonable(value):
    """Recursively turns NumPy arrays and scalars into plain lists and numbers for JSON.

    value: Any mix of dicts, lists, tuples, NumPy arrays/scalars and plain values.
    Returns: The same structure with only JSON-serializable types.
    """
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_results(results: dict, path: str) -> None:
    """Writes a run's results as JSON (plain lists; view.py reads them back).

    results: From run_scenario, plus "name".
    path: Output file; its folder is created if missing.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_jsonable(results), f)


def main() -> None:
    """Command line: plan one or more scenario files and save each one's results."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenarios", nargs="+", help="scenario JSON files")
    parser.add_argument("--out", help="results .json file (one scenario) or folder; default output/world_sim/")
    parser.add_argument("--continue-on-block", action="store_true", help="play moves the guard refuses, flagged")
    parser.add_argument("--keep-going", action="store_true", help="run the remaining commands after a failure")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    planner = SimPlanner()
    for path in args.scenarios:
        scenario = load_scenario(path)
        results = run_scenario(
            planner, scenario["world"], scenario["start_joints_deg"], scenario["commands"],
            continue_on_block=args.continue_on_block, stop_on_error=not args.keep_going,
        )  # fmt: skip
        results["name"] = scenario["name"]
        stem = os.path.splitext(os.path.basename(path))[0]
        to_file = args.out is not None and args.out.endswith(".json") and len(args.scenarios) == 1
        out = args.out if to_file else os.path.join(args.out or "output/world_sim", f"{stem}.json")
        save_results(results, out)
        failed = [c for c in results["commands"] if not c["ok"]]
        print(f"{scenario['name']}: {len(results['points'])} arm targets, {len(failed)} failed command(s) -> {out}")


if __name__ == "__main__":
    main()
