"""The robot's pyroki planner, driven offline: IK solvers, transit planner and world obstacles.

Replaces the dora requests the brain sends (pyroki_node.solve / batch_solve, pyroki_world_poc.
set_world_obstacles / plan_transit / get_collision_model) with direct calls into the same planner
code, and keeps the brain-side module state (obstacles, arm model) the arm-move guard reads.
"""

import logging
import time
from typing import Callable

# First: the resolver picks the JAX platform (cpu, or cuda with ENABLE_PYROKI_GPU) before jax is imported.
from pyroki_planner.resolver import IKResolver  # isort: skip

import jax.numpy as jnp
import numpy as np
from billie_utils.messages import pyroki_world_poc
from billie_utils.messages.pyroki_node import BATCH_SIZE, MAX_N_ST_PERTURB
from billie_utils.world_collision_check_poc import CollisionModel, link_transforms
from pyroki_planner import world_collision_poc
from pyroki_planner.transit_planner_poc import build_transit_solver
from scipy.spatial.transform import Rotation

# The planner's target link (resolver.TARGET_LINK_NAME); pose targets are its poses.
TCP_LINK = "link_tcp"


class _NoNode:
    """Stands in for the planner's dora node; it only ever sends 'still compiling' cloud messages."""

    def send_output(self, *args, **kwargs) -> None:
        """Drops the message."""


class SimPlanner:
    """The planner node and the brain's planner state, in one process. All joints in degrees."""

    def __init__(self):
        """Builds (or loads from the JAX cache) the IK solvers and the transit planner."""
        logging.info("Building the pyroki planner (first run compiles, later runs load the cache)...")
        self.timings: list[dict] = []  # {"call", "command", "n", "seconds"} per planner call, in order
        self.command = -1  # scenario command being run, set by the runner for the timings
        t0 = time.time()
        self.resolver = IKResolver(_NoNode())
        self.resolver._ready.wait()
        self.timings.append({"call": "build_ik_solvers", "command": -1, "n": 0, "seconds": time.time() - t0})
        assert self.resolver.robot is not None, "IK solver build failed, see the log above"
        self.model: CollisionModel = world_collision_poc.export_collision_model(
            self.resolver.robot, self.resolver.robot_coll
        )
        self._tcp_index = self.model.link_names.index(TCP_LINK)
        self._transit = build_transit_solver(self.resolver.robot, self.resolver.robot_coll)
        # The brain's cached arm model, read by the arm-move guard, detours and reroutes.
        pyroki_world_poc._model_cache = self.model
        self.set_obstacles(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
        # The robot compiles the transit planner at startup (warm_up_in_background), not on the first replay.
        self._timed("build_transit_planner", 0, lambda: self.plan_transit(np.zeros(6), np.full(6, 10.0)))
        self.timings = [t for t in self.timings if t["call"] != "transit"]

    def _timed(self, call: str, n: int, fn: Callable):
        """Runs one planner call and records how long it took.

        call: Name of the call for the timings. n: Poses (or frames) it handled.
        fn: The call, without arguments.
        Returns: What fn returns.
        """
        t0 = time.time()
        result = fn()
        self.timings.append({"call": call, "command": self.command, "n": n, "seconds": time.time() - t0})
        return result

    def set_obstacles(self, starts_m: np.ndarray, ends_m: np.ndarray, radii_m: np.ndarray) -> None:
        """Replaces the world obstacles on both sides, like poc_set_world_obstacles on the robot.

        starts_m, ends_m: (M, 3) capsule segment endpoints, xArm base frame, meters (M may be 0).
        radii_m: (M,) capsule radii, meters.
        """
        world_collision_poc._current_capsule_rows = world_collision_poc._capsule_rows(starts_m, ends_m, radii_m)
        world_collision_poc._current_heightmap = world_collision_poc._inactive_heightmap()
        pyroki_world_poc._last_sent = (np.asarray(starts_m), np.asarray(ends_m), np.asarray(radii_m))
        pyroki_world_poc._last_sent_heightmap = None

    def solve(self, current_deg: np.ndarray, pose6: np.ndarray) -> tuple[np.ndarray, bool]:
        """Single-pose IK in AI mode (regularized toward the rest pose), like pyroki_node.solve for `pose`.

        current_deg: (6,) joints the arm is at, degrees.
        pose6: Target TCP pose [x, y, z mm, rx, ry, rz rad rotation vector], xArm base frame.
        Returns: (solved joints in degrees (6,), True if the target is inside the robot base and was skipped).
        """
        joints_rad, is_base_collision, _, _ = self._timed("solve", 1, lambda: self.resolver.resolve(
            np.deg2rad(current_deg), np.asarray(pose6, dtype=np.float64), n_st_perturb=MAX_N_ST_PERTURB
        ))  # fmt: skip
        return np.rad2deg(np.asarray(joints_rad, dtype=np.float64)), bool(is_base_collision)

    def batch_solve(
        self, current_deg: np.ndarray, poses6: np.ndarray, recorded_deg: np.ndarray | None, call: str = "batch_solve"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch IK over up to BATCH_SIZE poses, like pyroki_node.batch_solve (REC mode when recorded joints are given).

        current_deg: (6,) joints the batch starts from, degrees (the seed and the first move's start).
        poses6: (N, 6) target TCP poses, N <= BATCH_SIZE, [mm, rad rotation vector].
        recorded_deg: (N, 6) recorded joints to regularize toward, degrees, or None for AI mode.
        call: Name of this use in the timings ("batch_solve" for replay batches, "detour" for detours).
        Returns: (solved joints (N, 6) degrees, large-motion mask (N,) - a joint jumped > 20 deg).
        """
        assert len(poses6) <= BATCH_SIZE, f"{len(poses6)} poses; the batch solver takes {BATCH_SIZE}"
        joints_rad, _, large_motion, _, _ = self._timed(call, len(poses6), lambda: self.resolver.resolve_batch(
            np.deg2rad(current_deg),
            np.asarray(poses6, dtype=np.float64),
            None if recorded_deg is None else np.deg2rad(recorded_deg),
            n_st_perturb=MAX_N_ST_PERTURB,
        ))  # fmt: skip
        return np.rad2deg(np.asarray(joints_rad, dtype=np.float64)), np.asarray(large_motion)

    def plan_transit(self, start_deg: np.ndarray, goal_deg: np.ndarray) -> np.ndarray:
        """Plans a joint path around the obstacles, like pyroki_world_poc.plan_transit.

        start_deg, goal_deg: (6,) end joints of the path, degrees.
        Returns: (TRANSIT_STEPS, 6) path in degrees; not guaranteed clear, the caller checks it.
        """
        path = self._timed("transit", 1, lambda: np.asarray(self._transit(
            jnp.asarray(np.deg2rad(start_deg), dtype=jnp.float32),
            jnp.asarray(np.deg2rad(goal_deg), dtype=jnp.float32),
            world_collision_poc.current_world_capsules(),
        )))  # fmt: skip
        return np.rad2deg(path.astype(np.float64))

    def tcp_poses(self, joints_deg: np.ndarray) -> np.ndarray:
        """TCP poses along joint configurations, by the NumPy copy of the planner's FK.

        joints_deg: (N, 6) joints in degrees.
        Returns: (N, 6) poses [x, y, z mm, rx, ry, rz rad rotation vector] in the xArm base frame.
        """
        T = link_transforms(self.model, np.deg2rad(np.atleast_2d(joints_deg)))[:, self._tcp_index]
        return np.concatenate([T[:, :3, 3] * 1000.0, Rotation.from_matrix(T[:, :3, :3]).as_rotvec()], axis=1)
