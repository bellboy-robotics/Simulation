"""Reroutes the stretches of a pose replay that run into the obstacles around them.

Port of nodes/brain/brain/behaviours/play/replay_reroute_poc.py (billie-onboard, arm_awareness):
same steps and constants, with the planner request replaced by SimPlanner.plan_transit. Keep the
two in sync when the brain's version changes.
"""

from typing import Callable

import numpy as np
from billie_utils import arm_motion_guard_poc
from billie_utils.world_collision_check_poc import CollisionModel

from Simulation.world_sim.planner import SimPlanner
from Simulation.world_sim.recording import ReplayState

# [m] Clearance the unchanged frames on each side of a rerouted stretch should keep (as on the robot).
_END_CLEARANCE_M = 0.03
# [mm] Slowest tool speed per played frame assumed for the route (2mm per frame = 10cm/s at 50Hz).
_MIN_TOOL_STEP_MM = 2.0


def _clearances(
    obstacles: arm_motion_guard_poc.ActiveObstacles, model: CollisionModel, joints_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame clearance of a recording, and which frames run into an obstacle.

    obstacles: The loaded obstacles.
    model: The arm model.
    joints_deg: (N, 6) recorded joints in degrees.
    Returns: (clearance per frame in meters (N,), blocked mask (N,) - frame inside an obstacle, or the
        move from the previous frame passes through one).
    """
    clearance = np.array([obstacles.check(model, q[None]).min_distance_m for q in joints_deg])
    blocked = clearance < 0.0
    for i in range(1, len(joints_deg)):
        if not blocked[i] and not obstacles.check(model, joints_deg[i - 1 : i + 1]).clear:
            blocked[i] = True
    return clearance, blocked


def _stretches(blocked: np.ndarray, clearance: np.ndarray) -> list[tuple[int, int]]:
    """Frame ranges to reroute: each blocked run widened to frames with _END_CLEARANCE_M, overlaps merged.

    blocked: (N,) frames running into an obstacle.
    clearance: (N,) per-frame clearance, meters.
    Returns: Sorted (first, last) frame index pairs; the route replaces frames first..last.
    Raises: ArmMoveBlockedError if the recording starts or ends inside an obstacle.
    """
    n = len(blocked)
    ranges: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not blocked[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and blocked[j + 1]:
            j += 1
        a, b = max(i - 1, 0), min(j + 1, n - 1)
        while a > 0 and clearance[a] < _END_CLEARANCE_M:
            a -= 1
        while b < n - 1 and clearance[b] < _END_CLEARANCE_M:
            b += 1
        if clearance[a] < 0.0 or clearance[b] < 0.0:
            where = "starts" if clearance[a] < 0.0 else "ends"
            raise arm_motion_guard_poc.ArmMoveBlockedError(
                f"The recording {where} inside an obstacle, so no route around it can reach its "
                f"{'first' if where == 'starts' else 'last'} frame."
            )
        if ranges and a <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(b, ranges[-1][1]))
        else:
            ranges.append((a, b))
        i = j + 1
    return ranges


def _resample(route_deg: np.ndarray, planner: SimPlanner, recorded: list[ReplayState]) -> np.ndarray:
    """Spreads the planned route over enough frames that the tool moves about as fast as in the recording.

    route_deg: (T, 6) planned route in degrees.
    planner: For the TCP forward kinematics.
    recorded: The recorded frames the route replaces (their positions give the recorded tool speed).
    Returns: (K, 6) route frames in degrees, K >= T, first and last frames unchanged.
    """
    tcp = planner.tcp_poses(route_deg)[:, :3]
    route_length_mm = float(np.sum(np.linalg.norm(np.diff(tcp, axis=0), axis=1)))
    positions = np.array([s.position[:3] for s in recorded])
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1) if len(positions) > 1 else np.zeros(1)
    step_mm = max(float(np.median(steps)), _MIN_TOOL_STEP_MM)
    k = max(len(route_deg), int(np.ceil(route_length_mm / step_mm)) + 1)
    fraction = np.linspace(0.0, len(route_deg) - 1, k)
    lower = np.floor(fraction).astype(int).clip(0, len(route_deg) - 2)
    t = (fraction - lower)[:, None]
    return route_deg[lower] * (1.0 - t) + route_deg[lower + 1] * t


def reroute_around_obstacles(
    planner: SimPlanner, states: list[ReplayState], log: Callable[[str, str], None]
) -> tuple[list[ReplayState], list[tuple[int, int]]]:
    """Replaces every stretch of a pose replay that runs into the obstacles with a planned route.

    Does nothing when no obstacles are loaded or the recording has no joints.

    planner: The simulated planner.
    states: The replay frames, with recorded joints matching their positions.
    log: Receives (message, level) like node.send_cloud_message.
    Returns: (the frames to play - planned ones have source_frame -1, the replaced (first, last) ranges).
    Raises: ArmMoveBlockedError if the recording starts or ends inside an obstacle, or no clear route is found.
    """
    obstacles = arm_motion_guard_poc.active_obstacles()
    if obstacles is None or any(s.recorded_joints is None for s in states):
        return states, []
    joints = np.array([s.recorded_joints for s in states], dtype=np.float64)
    clearance, blocked = _clearances(obstacles, planner.model, joints)
    stretches = _stretches(blocked, clearance)

    out: list[ReplayState] = []
    previous_end = -1
    for first, last in stretches:
        log(f"Frames {first}-{last} of the replay run into an obstacle; planning a route around it...", "INFO")
        route = planner.plan_transit(joints[first], joints[last])
        result = obstacles.check(planner.model, route)
        if not result.clear:
            raise arm_motion_guard_poc.ArmMoveBlockedError(
                f"No route found around the obstacle for frames {first}-{last} ({result.worst_link} would "
                f"still go {-result.min_distance_m * 1000:.0f}mm in)."
            )
        route = _resample(route, planner, states[first : last + 1])
        out.extend(states[previous_end + 1 : first])
        poses = planner.tcp_poses(route)
        out.extend(ReplayState(position=p, recorded_joints=q, source_frame=-1) for p, q in zip(poses, route))
        previous_end = last
        log(
            f"Route planned for frames {first}-{last}: {len(route)} frames, "
            f"{result.min_distance_m * 1000:.0f}mm from the obstacle at its closest.",
            "INFO",
        )
    out.extend(states[previous_end + 1 :])
    return out, stretches
