"""Simulated world for the pyroki planner: brain arm commands against virtual obstacles.

The planner and billie_utils read env vars at import time, so the environment is set up here,
before any module of this package imports them.
"""

from Simulation.world_sim import env_setup  # noqa: F401
