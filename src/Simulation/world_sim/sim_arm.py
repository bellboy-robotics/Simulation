"""Simulated arm: xarm_writer's joint queue plus the brain's arm-move guard, recording every move.

The real arm drives a straight joint-space line from where it is to each queued target. Here a
move completes instantly: the arm is always at the last queued target, which is also where the
robot's arm ends once its queue empties.
"""

from dataclasses import dataclass

import numpy as np
from billie_utils import arm_motion_guard_poc


@dataclass
class TrajectoryPoint:
    """One joint target the arm was sent to."""

    joints_deg: np.ndarray  # (6,) target joints, degrees
    command: int  # index of the scenario command that sent it
    kind: str  # what produced it, e.g. "joints", "pose", "detour", "replay", "reroute"
    target_pose: np.ndarray | None  # (6,) TCP pose the planner was asked for [mm, rad], if any
    blocked: str | None  # the guard's refusal message, if the guard refused it (continue_on_block only)


class SimArm:
    """The arm's joint state and the record of everything sent to it."""

    def __init__(self, start_joints_deg: np.ndarray, continue_on_block: bool = False):
        """Places the arm and arms the guard.

        start_joints_deg: (6,) joints the arm starts at, degrees.
        continue_on_block: If True, a move the guard refuses is still recorded (flagged) and played,
            so the whole planned path can be inspected; if False it raises, like on the robot.
        """
        self.joints_deg = np.asarray(start_joints_deg, dtype=np.float64)
        self.start_joints_deg = self.joints_deg.copy()
        self.continue_on_block = continue_on_block
        self.points: list[TrajectoryPoint] = []
        self.command = -1  # set by the runner before each command
        arm_motion_guard_poc.set_current_joints_provider(lambda: self.joints_deg)
        arm_motion_guard_poc._last_target_deg = None
        arm_motion_guard_poc._queued_targets.clear()

    def enqueue(
        self, joints_deg: np.ndarray, clear_queue: bool, kind: str, target_pose: np.ndarray | None = None
    ) -> None:
        """Sends one joint target through the guard, like xarm_writer.enqueue_joints.

        joints_deg: (6,) target joints, degrees.
        clear_queue: True for a single move (starts from the current joints), False to continue a
            trajectory (starts from the previous target and is also checked for joint spins).
        kind: Label of what produced the move, for the viewer and the report.
        target_pose: (6,) TCP pose the planner was asked for, if this target came from IK.
        Raises: ArmMoveBlockedError if the guard refuses the move and continue_on_block is False.
        """
        target = np.asarray(joints_deg, dtype=np.float64)[:6]
        blocked = None
        try:
            arm_motion_guard_poc.check_move(target, clear_queue)
        except arm_motion_guard_poc.ArmMoveBlockedError as e:
            if not self.continue_on_block:
                raise
            blocked = str(e)
            arm_motion_guard_poc._last_target_deg = target
        self.points.append(TrajectoryPoint(target, self.command, kind, target_pose, blocked))
        self.joints_deg = target
