"""Checks arm joints with the planner's own self-collision model and joint limits. No arm moves, no solver is built.

    python pose_check.py poses.json --urdf /tmp/urdf/<SN>-with-tcp.urdf

poses.json: {"name": [J2, J3, J4, J5, J6], ...} degrees. Each pose is checked at J1 all the way round (Billie's
fixed parts do not turn with J1, so a pose can clear them at one J1 and not at another).

The URDF is the one the planner itself wrote (urdf.load_urdf's output), read as it is: load_urdf is not called, since
it also uploads the URDF to S3 under the robot's name. The planner's modules read their settings when imported, so
harmless values are set here for those this check does not use. Runs on the CPU with its own JAX cache folder.
"""

import argparse
import json
import os
import tempfile

# Settings the planner's modules read at import; only the URDF given on the command line is used here.
_SETTINGS = {
    "BILLIE_SYSID": "pose-check", "BILLIE_DIR": tempfile.gettempdir(), "BILLIE_ENVDIR": tempfile.gettempdir(),
    "XARM_SN": "pose-check", "BASE_URDF": "none", "XARM_TCP_OFFSET": "[0, 0, 200, 0, 0, 0]", "ENABLE_CAMERA2": "false",
    "DEBUG_MODE": "false", "DEBUG_PYROKI_PLANNER": "false", "ENABLE_PYROKI_GPU": "false",
    "ENABLE_PYROKI_SELF_COLLISION": "true", "PYROKI_TERMINATION_THRESHOLD": "1e-6",
    "ENABLE_PYROKI_DEFAULT_INIT": "false", "ENABLE_PYROKI_WORLD_COLLISION": "true",
    "PYROKI_JOINTS_AI_WEIGHTS": "0.01", "PYROKI_JOINTS_REC_WEIGHTS": "[0.15, 0.1, 0.1, 0.1, 0.1, 0.1]",
    "PYROKI_JAX_CACHE_DIR": os.path.join(tempfile.gettempdir(), "pose-check-jax-cache"),
}  # fmt: skip
for _name, _value in _SETTINGS.items():
    os.environ.setdefault(_name, _value)

import numpy as np  # noqa: E402
import pyroki as pk  # noqa: E402
import yourdfpy  # noqa: E402
from pyroki_planner import self_check, world_collision  # noqa: E402
from pyroki_planner.base_collision import BaseCollisionChecker  # noqa: E402
from pyroki_planner.resolver import IGNORE_COLLISION_PAIRS  # noqa: E402

# [deg] J1 is checked every this much, all the way round.
J1_STEP_DEG = 15


def load(urdf_path: str, mesh_file) -> yourdfpy.URDF:
    """The planner's URDF with its collision meshes.

    urdf_path: The with-tcp URDF. mesh_file: mesh path in the URDF -> its file here (None: as written).
    Returns: The URDF.
    """
    def handler(fname: str) -> str:
        """A mesh's file on this machine. fname: The URDF's path (yourdfpy's keyword). Returns: The path to open."""
        return (mesh_file(fname) if mesh_file is not None else None) or fname

    return yourdfpy.URDF.load(urdf_path, filename_handler=handler)


def main() -> None:
    """Loads the arm, sets up the planner's self check and prints each pose's verdict."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("poses", help="JSON: name -> [J2..J6] degrees")
    parser.add_argument("--urdf", required=True, help="the planner's with-tcp URDF")
    parser.add_argument("--local-meshes", action="store_true", help="find the meshes on this machine (scan_coverage)")
    args = parser.parse_args()
    mesh_file = None
    if args.local_meshes:
        from Simulation.mapping_sim.scan_coverage import _mesh_file as mesh_file  # noqa: PLC0415
    urdf = load(args.urdf, mesh_file)
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf, min_capsule=True, enable_aabb_for_static_links=True,
                                                       user_ignore_pairs=IGNORE_COLLISION_PAIRS)  # fmt: skip
    world_collision.set_movable_links(robot, robot_coll)
    model = world_collision.export_collision_model(robot, robot_coll)
    self_check.configure(robot, robot_coll, model, BaseCollisionChecker(urdf).boxes)
    turns = np.arange(-180, 181, J1_STEP_DEG)
    results = {}
    for name, rest in json.load(open(args.poses)).items():
        samples = np.array([[j1, *rest] for j1 in turns], dtype=float)
        breach = self_check.limit_breach(samples[0])
        checked = self_check.path_past(samples)
        assert checked is not None, "the self check is not configured"
        past, pair = checked
        worst = int(np.argmin(past))
        bad = turns[past < 0]
        results[name] = {"limits": breach, "min_clearance_m": round(float(past[worst]), 4),
                         "binding_pair": self_check.pair_name(int(pair[worst])), "at_j1": int(turns[worst]),
                         "refused_j1": [int(v) for v in bad]}  # fmt: skip
        verdict = "OK" if breach is None and not len(bad) else "REFUSED"
        print(f"{verdict:8} {name}: J2-J6 {rest}; limits: {breach or 'inside'}; closest pair "
              f"{results[name]['binding_pair']} {past[worst] * 1000:+.0f} mm past the rule at J1 {turns[worst]}"
              + (f"; refused at J1 {bad.tolist()}" if len(bad) else ""))  # fmt: skip
    print("RESULTS " + json.dumps(results))


if __name__ == "__main__":
    main()
