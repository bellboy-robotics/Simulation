"""Routes a pose replay around the obstacles before it plays, like the brain's play/replay_reroute.py.

The decision is the planner's: plan_replay (pyroki_planner.replay_planner, branch arm_planner) checks the
recording and replaces every stretch that runs into the obstacles with a checked route. Here, as in the
brain, its frames become replay states again (route frames get source_frame -1).
"""

from typing import Callable

import numpy as np
from billie_utils import arm_motion_guard

from Simulation.world_sim.planner import SimPlanner
from Simulation.world_sim.recording import ReplayState


def reroute_around_obstacles(
    planner: SimPlanner, states: list[ReplayState], log: Callable[[str, str], None]
) -> tuple[list[ReplayState], int]:
    """The frames to play: the recording, with each stretch that runs into an obstacle replaced by a route.

    Does nothing while no obstacles are set or when the frames have no recorded joints.

    planner: The simulated planner.
    states: The replay frames, with recorded joints matching their positions.
    log: Receives (message, level) like node.send_cloud_message.
    Returns: (the frames to play, how many of them the planner routed).
    Raises: ArmMoveBlockedError if the planner finds no way to replay the recording around the obstacles.
    """
    if arm_motion_guard.active_obstacles() is None or any(s.recorded_joints is None for s in states):
        return states, 0
    joints = np.array([s.recorded_joints for s in states], dtype=np.float64)
    positions = np.array([s.position for s in states], dtype=np.float64)
    plan = planner.plan_replay(joints, positions)
    if plan.status == "blocked":
        raise arm_motion_guard.ArmMoveBlockedError(f"Replay blocked: {plan.message}")
    if plan.status == "unchanged":
        return states, 0
    log(plan.message, "INFO")
    out = [
        ReplayState(position=p, recorded_joints=q, source_frame=-1) if planned else states[int(source)]
        for q, p, source, planned in zip(plan.joints_deg, plan.positions, plan.source_frames, plan.planned)
    ]
    return out, int(np.sum(plan.planned))
