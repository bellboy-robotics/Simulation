"""The brain's arm commands (`joints`, `pose`, `replay_policy`), run against the simulated planner and arm.

Ports of billie-onboard (arm_awareness) brain code, keeping its decisions and order:
- single moves: behaviours/state.py set_state_with_arm_joints + behaviours/arm_detour_poc.py, except
  that a blocked move is routed by the transit planner (DETOUR_PLANNER); the robot still uses batch IK
- `pose`: behaviours/state.py set_state_with_arm_pose
- `joints`: behaviours/arm.py joints / set_smooth_joints
- `replay_policy`: behaviours/play/replay_policy.py, play/buffering_player.py (pose batches),
  arm.py replay_joints_batch, play/replay_reroute_poc.py (see reroute.py)
"""

import logging
import os

import numpy as np
from billie_utils import arm_motion_guard_poc
from billie_utils.messages.pyroki_node import BATCH_SIZE

from Simulation.world_sim.planner import SimPlanner
from Simulation.world_sim.recording import ReplayState, load_recording, select_poses
from Simulation.world_sim.reroute import reroute_around_obstacles
from Simulation.world_sim.sim_arm import SimArm

# [m] replay_policy's default final_position_threshold: the replay fails if the arm ends farther away.
FINAL_POSITION_THRESHOLD_M = 0.05
# What routes a blocked single move (joints, pose, a replay's move to its first frame):
# "transit": the transit planner, a joint-space path with fixed start and goal (the fix);
# "batch_ik": the robot's arm_detour_poc today, batch IK along the straight line's TCP poses.
DETOUR_PLANNER = os.environ.get("WORLD_SIM_DETOUR", "transit")


class SimBrain:
    """Runs brain arm commands; collects the operator messages the robot would send to the cloud."""

    def __init__(self, planner: SimPlanner, arm: SimArm):
        """planner: The simulated planner. arm: The simulated arm the moves are sent to."""
        self.planner = planner
        self.arm = arm
        self.events: list[dict] = []  # {"command", "level", "message"} in order

    def log(self, message: str, level: str = "INFO") -> None:
        """Records an operator message, like node.send_cloud_message.

        message: The text. level: "DEBUG", "INFO", "WARNING" or "ERROR".
        """
        self.events.append({"command": self.arm.command, "level": level, "message": message})
        logging.log(getattr(logging, level, logging.INFO), f"[cmd {self.arm.command}] {message}")

    # ----- single moves ------------------------------------------------------------------------

    def move_joints(self, goal_deg: np.ndarray, kind: str, target_pose: np.ndarray | None = None) -> None:
        """A single move (clear_queue=True): a detour if the direct line is blocked, then the move itself.

        goal_deg: (6,) joints to reach, degrees.
        kind: Label of the move for the report.
        target_pose: (6,) TCP pose this move is meant to reach, if it came from IK.
        Raises: ArmMoveBlockedError if blocked and no clear detour exists (or the guard refuses).
        """
        goal = np.asarray(goal_deg, dtype=np.float64)[:6]
        clear_queue = not self._enqueue_detour_if_blocked(goal)
        self.arm.enqueue(goal, clear_queue=clear_queue, kind=kind, target_pose=target_pose)

    def _enqueue_detour_if_blocked(self, goal_deg: np.ndarray) -> bool:
        """Queues a detour when the direct move would enter an obstacle (arm_detour_poc.enqueue_detour_if_blocked).

        goal_deg: (6,) joints the move must reach, degrees.
        Returns: True if a detour was queued (the caller then continues it), False if the direct move is clear.
        Raises: ArmMoveBlockedError if the move is blocked and no clear detour was found.
        """
        obstacles = arm_motion_guard_poc.active_obstacles()
        if obstacles is None:
            return False
        start = self.arm.joints_deg.copy()
        direct = obstacles.check(self.planner.model, np.stack([start, goal_deg]))
        if direct.clear:
            return False
        self.log(
            f"The direct arm move would put {direct.worst_link} {-direct.min_distance_m * 1000:.0f}mm into the "
            "obstacle; planning a detour around it...",
        )
        frames = self._detour_frames(start, goal_deg)
        result = obstacles.check(self.planner.model, np.vstack([start[None], frames, goal_deg[None]]))
        if not result.clear:
            raise arm_motion_guard_poc.ArmMoveBlockedError(
                f"Arm move blocked: no detour found around the obstacle ({result.worst_link} would still "
                f"go {-result.min_distance_m * 1000:.0f}mm in)."
            )
        for i, frame in enumerate(frames):
            self.arm.enqueue(frame, clear_queue=(i == 0), kind="detour")
        step = float(np.max(np.abs(np.diff(np.vstack([start[None], frames, goal_deg[None]]), axis=0))))
        self.log(
            f"Detour planned and queued ({len(frames)} frames by {DETOUR_PLANNER}, {result.min_distance_m * 1000:.0f}mm "
            f"from the obstacles at its closest, largest joint step {step:.1f}deg)."
        )
        return True

    def _detour_frames(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """The frames of a detour between two configurations, from the planner DETOUR_PLANNER names.

        start, goal: (6,) joints, degrees.
        Returns: (K, 6) frames in degrees after the start; the caller adds the exact goal after them.
        """
        if DETOUR_PLANNER == "transit":
            route = self.planner.plan_transit(start, goal, call="detour")
            return route[1:-1]  # its ends are pinned to start and goal; the exact ones are driven instead
        # The straight joint line, split into one planner batch, solved in REC mode toward its own TCP poses.
        line = start[None] + np.arange(1, BATCH_SIZE + 1)[:, None] / BATCH_SIZE * (goal - start)[None]
        detour, _ = self.planner.batch_solve(start, self.planner.tcp_poses(line), line, call="detour")
        return detour

    # ----- commands ----------------------------------------------------------------------------

    def joints(self, joints: list[float], relative: bool = False) -> None:
        """The `joints` command: a single smooth move to the given joints.

        joints: 6 joint angles, degrees. relative: Add them to the current joints instead.
        """
        goal = np.asarray(joints, dtype=np.float64) + (self.arm.joints_deg if relative else 0.0)
        self.move_joints(goal, kind="joints")

    def pose(self, pose: list[float]) -> None:
        """The `pose` command: AI-mode IK for one TCP pose, then a single move to the solution.

        pose: [x, y, z mm, rx, ry, rz rad rotation vector] in the xArm base frame.
        """
        target = np.asarray(pose, dtype=np.float64)
        solution, is_base_collision = self.planner.solve(self.arm.joints_deg, target)
        if is_base_collision:
            self.log("The target pose is inside the robot base; the planner kept the current joints.", "WARNING")
        reached = self.planner.tcp_poses(solution[None])[0]
        error_mm = float(np.linalg.norm(reached[:3] - target[:3]))
        self.log(f"pose: IK solution reaches the target within {error_mm:.1f}mm.", "DEBUG")
        self.move_joints(solution, kind="pose", target_pose=target)

    def replay_policy(
        self,
        repo_id: str | None = None,
        speed: float = 1.0,
        transform: str = "poses",
        reverse: bool = False,
        states: list[ReplayState] | None = None,
        fps: float | None = None,
    ) -> None:
        """The `replay_policy` command in `poses` or `joints` mode.

        repo_id: Recording to replay (or pass states and fps directly).
        speed: Playback speed, 1.0 = as recorded.
        transform: "poses" (IK on the recorded poses, rerouted around obstacles) or "joints" (recorded joints).
        reverse: Play the recording backwards.
        states: Recorded frames, instead of repo_id (e.g. a synthetic path).
        fps: Frame rate of states.
        Raises: ArmMoveBlockedError / RuntimeError when the robot's replay would fail.
        """
        if transform not in ("poses", "joints"):
            raise ValueError(f"transform={transform!r}: the simulation supports 'poses' and 'joints'")
        if states is None:
            states, fps = load_recording(repo_id)
        assert fps is not None
        states = select_poses(states, fps, speed)
        if reverse:
            states = list(reversed(states))
        self.log(f"Replaying {len(states)} frames ({transform}, speed {speed}).", "DEBUG")

        # _set_initial_state: a single move to the first recorded joints (detour and guard apply).
        self.move_joints(states[0].recorded_joints, kind="replay_start", target_pose=states[0].position)
        if transform == "joints":
            for s in states:  # replay_joints_batch: every frame is its own single move
                self.move_joints(s.recorded_joints, kind="replay", target_pose=s.position)
            return
        states, _ = reroute_around_obstacles(self.planner, states, self.log)
        self._play_poses(states)
        final = self.planner.tcp_poses(self.arm.joints_deg[None])[0]
        distance_m = float(np.linalg.norm(final[:3] - states[-1].position[:3])) / 1000.0
        self.log(f"Distance from desired final position: {distance_m:.3f}m", "DEBUG")
        if distance_m > FINAL_POSITION_THRESHOLD_M:
            raise RuntimeError(
                f"Final arm position is {distance_m:.3f}m from the desired one (threshold {FINAL_POSITION_THRESHOLD_M}m)."
            )

    def _play_poses(self, states: list[ReplayState]) -> None:
        """BufferingPlayer: solve the frames in batches of BATCH_SIZE, seeding each batch with the last solution.

        states: Frames to play (positions required; recorded joints switch the solver to REC mode).
        """
        previous = states[0].recorded_joints if states[0].recorded_joints is not None else self.arm.joints_deg
        for begin in range(0, len(states), BATCH_SIZE):
            batch = states[begin : begin + BATCH_SIZE]
            poses = np.stack([s.position for s in batch])
            recorded = None
            if all(s.recorded_joints is not None for s in batch):
                recorded = np.stack([s.recorded_joints for s in batch])
            solved, large_motion = self.planner.batch_solve(previous, poses, recorded)
            for i in np.flatnonzero(large_motion):
                self.log(f"Large motion at replay frame {begin + i} (a joint moved > 20deg).", "WARNING")
            for s, joints in zip(batch, solved):
                kind = "replay" if s.source_frame >= 0 else "reroute"
                self.arm.enqueue(joints, clear_queue=False, kind=kind, target_pose=s.position)
            previous = solved[-1]


def run_command(brain: SimBrain, spec: dict) -> None:
    """Runs one scenario command.

    brain: The simulated brain.
    spec: {"cmd": "joints" | "pose" | "replay_policy", ...the command's arguments as on the robot}, e.g.
        {"cmd": "joints", "joints": [0, -30, -60, 0, 90, 0]},
        {"cmd": "pose", "pose": [400, 0, 200, 3.14, 0, 0]},
        {"cmd": "replay_policy", "repo_id": "bellboy-robotics/...", "speed": 1.0, "transform": "poses"}.
    """
    args = {k: v for k, v in spec.items() if k != "cmd"}
    handlers = {"joints": brain.joints, "pose": brain.pose, "replay_policy": brain.replay_policy}
    if spec["cmd"] not in handlers:
        raise ValueError(f"Unknown command {spec['cmd']!r}; use one of {list(handlers)}")
    handlers[spec["cmd"]](**args)
