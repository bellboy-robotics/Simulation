"""Environment for running the pyroki planner offline (no dora, no robot).

Import this module before anything from pyroki_planner: the planner reads its configuration from
environment variables at import time. Values already set in the shell win over the defaults here.
"""

import json
import os
import sys
import types

try:
    from dotenv import load_dotenv

    # The developer env of this machine: BILLIE_SYSID, BILLIE_ENVDIR (holds urdf/<XARM_SN>.urdf), BASE_URDF.
    load_dotenv(os.path.expanduser("~/releases/env/base/.env"))
    load_dotenv(os.path.expanduser("~/releases/env/RONIT-DEV/.env"))
except ImportError:
    pass  # on a robot the planner's own env is passed in (see robot.py) and dotenv is not installed

# Arm serial number whose URDF lives in $BILLIE_ENVDIR/urdf (the RONIT-DEV env ships this one).
XARM_SN = "XI130506F56A19"
# [x, y, z mm, roll, pitch, yaw rad] tool offset from link6 to the TCP; same as the Simulation ik_sim.
TCP_OFFSET = [0, 0, 200, 0, 0, 0]

_DEFAULTS = {
    "BILLIE_SYSID": "RONIT-DEV",
    "BILLIE_DIR": os.path.expanduser("~/billie"),
    "BILLIE_ENVDIR": "~/releases/env/RONIT-DEV",
    "BASE_URDF": "base_v0.7.16.urdf",
    "XARM_SN": XARM_SN,
    "XARM_TCP_OFFSET": json.dumps(TCP_OFFSET),
    "ENABLE_CAMERA2": "false",  # read by billie/node_ticks.py, imported through billie_utils.node
    "DEBUG_MODE": "false",
    "DEBUG_PYROKI_PLANNER": "false",
    "ENABLE_PYROKI_GPU": "false",
    "ENABLE_PYROKI_SELF_COLLISION": "true",
    "PYROKI_TERMINATION_THRESHOLD": "1e-6",
    "ENABLE_PYROKI_DEFAULT_INIT": "false",
    # World obstacles are the point of this simulation, so the planner's world cost is compiled in.
    "ENABLE_PYROKI_WORLD_COLLISION": "true",
    # The heightmap cost makes the batch solver's first compile ~10x slower; capsules only for now.
    "ENABLE_PYROKI_WORLD_HEIGHTMAP": "false",
}


def setup_env() -> None:
    """Sets every env var the planner needs (shell values win) and disables the URDF S3 upload."""
    for key, value in _DEFAULTS.items():
        os.environ.setdefault(key, value)
    # The planner's Path(...).resolve() does not expand "~".
    os.environ["BILLIE_ENVDIR"] = os.path.expanduser(os.environ["BILLIE_ENVDIR"])

    # load_urdf() imports urdf_publisher and uploads the generated URDF to S3 in the background; an
    # offline run must not (and need not install its defusedxml/boto3 deps). A no-op stand-in replaces it.
    stub = types.ModuleType("pyroki_planner.urdf_publisher")
    stub.publish_in_background = lambda *args, **kwargs: None
    sys.modules["pyroki_planner.urdf_publisher"] = stub


setup_env()
