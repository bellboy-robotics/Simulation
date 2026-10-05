"""The robot's pyroki planner, driven offline: IK solvers, transit planner and world obstacles.

Replaces the dora requests the brain sends (pyroki_node.solve / batch_solve, pyroki_world_poc.
set_world_obstacles / plan_transit / get_collision_model) with direct calls into the same planner
code, and keeps the brain-side module state (obstacles, arm model) the arm-move guard reads.
Also records how the build went (per phase: time, JAX compile and cache use, the planner's own log
lines) and, through planner_capture, what each run logs and compiles, for the timing report.
"""

import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator

from Simulation.world_sim.planner_capture import ZERO_STATS, CaptureWindow, capture, with_compile_s  # isort: skip

# urdf.py logs its gripper mesh fallback while the resolver imports it: those lines go in the build log.
with capture.import_lines() as _import_window:
    # First: the resolver picks the JAX platform (cpu, or cuda with ENABLE_PYROKI_GPU) before jax is imported.
    from pyroki_planner.resolver import IKResolver  # isort: skip

import jax
import jax.numpy as jnp
import numpy as np
from billie_utils.messages import pyroki_world_poc
from billie_utils.messages.pyroki_node import BATCH_SIZE, MAX_N_ST_PERTURB
from billie_utils.world_collision_check_poc import CollisionModel, link_transforms
from pyroki_planner import resolver as planner_resolver
from pyroki_planner import urdf as planner_urdf
from pyroki_planner import world_collision_poc
from pyroki_planner.transit_planner_poc import build_transit_solver
from scipy.spatial.transform import Rotation

# The planner's target link (resolver.TARGET_LINK_NAME); pose targets are its poses.
TCP_LINK = "link_tcp"
# Env vars that change what the planner compiles or where it runs, recorded with the build.
_BUILD_ENV = ("ENABLE_PYROKI_GPU", "JAX_PLATFORM_NAME", "DEBUG_PYROKI_PLANNER", "DEBUG_MODE",
              "ENABLE_PYROKI_WORLD_COLLISION", "ENABLE_PYROKI_WORLD_HEIGHTMAP", "ENABLE_PYROKI_SELF_COLLISION",
              "ENABLE_PYROKI_DEFAULT_INIT", "PYROKI_TERMINATION_THRESHOLD", "PYROKI_JAX_CACHE_DIR", "XLA_FLAGS")  # fmt: skip
# Resolver build threads -> the build phase each runs (pyroki-compile: URDF, models, jit setup).
_IK_THREADS = {"pyroki-compile": "ik_setup", "pyroki-warmup-single": "ik_warmup_single",
               "pyroki-warmup-batch": "ik_warmup_batch"}  # fmt: skip
# The resolver's own verdict per warm-up: "[compile] IK solver (n=16): 29.7s (cache hit)" (a guess: < 5s).
_RESOLVER_LABEL = re.compile(r"\[compile\] (batch IK solver|IK solver) \(.*?\): ([\d.]+)s \((.*)\)")
# urdf.py's gripper STL (relative to its URDF_DIR); it sets its mesh constant to None when it falls back.
_GRIPPER_STL = "meshes/ee-rome-v0-lod.stl"
# [s] How long to wait for the resolver's build thread to log its last line after it reports ready.
_BUILD_THREAD_JOIN_S = 5.0


class _NoNode:
    """Stands in for the planner's dora node; keeps the cloud messages it sends (e.g. 'still compiling')."""

    def send_output(self, output_id: str, data=None, metadata=None) -> None:
        """Records an "upstream" cloud message in the open capture windows.

        output_id: The dora output ("upstream"). data: JSON bytes {"message": {"message", "level", ...}}.
        metadata: Unused.
        """
        try:
            message = json.loads(bytes(data))["message"]
            capture.add_message(message.get("level", "INFO"), message.get("message", ""))
        except (TypeError, ValueError, KeyError, AttributeError):
            capture.add_message("INFO", f"{output_id}: {data!r}"[:300])


def gripper_geometry() -> dict:
    """What the planner's gripper/camera link is: urdf.py's STL mesh, or its box when the STL is missing.

    Returns: {"gripper_geometry": "mesh" or "box fallback", "gripper_mesh": the STL used (or looked for)}.
    """
    mesh = planner_urdf.CAMERA_GRIPPER_COLLISION_MESH
    return {"gripper_geometry": "mesh" if mesh else "box fallback",
            "gripper_mesh": mesh or str(planner_urdf.URDF_DIR / _GRIPPER_STL)}  # fmt: skip


def _count_files(folder: str | None) -> int | None:
    """Files in a folder tree (the JAX cache: one per compiled program).

    folder: The folder, or None. Returns: The count, or None if the folder does not exist.
    """
    if not folder or not os.path.isdir(folder):
        return None
    return sum(len(files) for _, _, files in os.walk(folder))


def _phase_row(name: str, thread: str, start: float, end: float, stats: dict, origin: float, label=None) -> dict:
    """One build phase for planner_build["phases"].

    name: Phase name. thread: Thread it ran on. start, end: Epoch seconds. stats: Its compile stats
    (see planner_capture.ZERO_STATS). origin: Build start, epoch s. label: The resolver's own verdict or None.
    Returns: {"name", "thread", "start_s", "end_s" (since the build start), "seconds", compile stats with
        "compile_s", "resolver_label"}.
    """
    return {"name": name, "thread": thread, "start_s": round(start - origin, 4), "end_s": round(end - origin, 4),
            "seconds": round(end - start, 4), **with_compile_s(stats), "resolver_label": label}  # fmt: skip


class SimPlanner:
    """The planner node and the brain's planner state, in one process. All joints in degrees."""

    def __init__(self):
        """Builds (or loads from the JAX cache) the IK solvers and the transit planner, timing every phase."""
        logging.info("Building the pyroki planner (first run compiles, later runs load the cache)...")
        # {"call", "command", "n", "seconds", "start" (epoch s), "compile_s", "cache_hits", "cache_misses"}
        # per planner call, in order; build_* entries first.
        self.timings: list[dict] = []
        self.command = -1  # scenario command being run, set by the runner for the timings
        self.runs_started = 0  # runs that used this planner (see capture_run)
        capture.install(lambda: self.command)
        # How the build went (see _finish_build); "phases" grows as they run.
        self.build: dict = {"built_at": time.time(), "phases": []}
        window = capture.open_window(self.build["built_at"])
        window.log[:0] = _import_window.log  # the lines logged while the planner was imported
        cache_files_before = _count_files(self._cache_dir())
        try:  # a failed build (e.g. the editor's, which keeps running) must not leave its window collecting
            self._build_ik(window)
            self.model: CollisionModel = self._phase("export_model", lambda: world_collision_poc.export_collision_model(
                self.resolver.robot, self.resolver.robot_coll
            ))  # fmt: skip
            self._tcp_index = self.model.link_names.index(TCP_LINK)
            self._transit = self._phase("transit_build", lambda: build_transit_solver(
                self.resolver.robot, self.resolver.robot_coll
            ))  # fmt: skip
            # The brain's cached arm model, read by the arm-move guard, detours and reroutes.
            pyroki_world_poc._model_cache = self.model
            self.set_obstacles(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
            # The robot compiles the transit planner at startup (warm_up_in_background), not on the first replay.
            self._phase("transit_warmup", lambda: self._timed(
                "build_transit_planner", 0, lambda: self.plan_transit(np.zeros(6), np.full(6, 10.0))
            ))  # fmt: skip
        finally:
            capture.close_window(window)
        self.timings = [t for t in self.timings if t["call"] != "transit"]
        self._finish_build(window, cache_files_before)

    @staticmethod
    def _cache_dir() -> str | None:
        """The JAX persistent compilation cache folder the resolver configured. Returns: Its path, or None."""
        return getattr(planner_resolver, "_JAX_CACHE_DIR", None) or getattr(jax.config, "jax_compilation_cache_dir", None)

    def _build_ik(self, window: CaptureWindow) -> None:
        """Runs the resolver's background build and waits for it; adds build_ik_solvers and the ik_* phases.

        window: The build's capture window (its lines hold the resolver's own cache verdicts).
        """
        capture.thread_stats.clear()
        capture.thread_span.clear()
        capture.tracking = True
        try:
            t0 = time.time()
            self.resolver = IKResolver(_NoNode())
            self.resolver._ready.wait()
            ready = time.time()
            # The build thread logs "compilation complete" right after setting _ready: keep that line too.
            for thread in threading.enumerate():
                if thread.name == "pyroki-compile":
                    thread.join(timeout=_BUILD_THREAD_JOIN_S)
        finally:
            capture.tracking = False
        phases = self._ik_phases(t0, ready, window.log)
        total = phases[-1]
        self.timings.append({"call": "build_ik_solvers", "command": -1, "n": 0, "seconds": ready - t0, "start": t0,
                             "compile_s": total["compile_s"], "cache_hits": total["cache_hits"],
                             "cache_misses": total["cache_misses"]})  # fmt: skip
        self.build["phases"] += phases
        assert self.resolver.robot is not None, "IK solver build failed, see the log above"

    def _ik_phases(self, t0: float, ready: float, log: list[dict]) -> list[dict]:
        """The resolver build's phases, from what its threads logged and compiled.

        ik_setup runs from the start until the first warm-up thread starts; each warm-up spans its thread's
        first to last line or JAX event; ik_total is the whole wait. Compile stats are each thread's events:
        ik_setup's are all of the pyroki-compile thread's, including what it compiles after the warm-ups
        (with DEBUG_PYROKI_PLANNER, the batch near-collision check), outside its from-to span.

        t0: When the resolver was created, epoch s. ready: When it reported ready, epoch s.
        log: The build's log lines so far (see planner_capture.CaptureWindow.log).
        Returns: Phase rows (see _phase_row), ik_total last.
        """
        labels = {}
        for line in log:
            match = _RESOLVER_LABEL.search(line["message"])
            if match:
                labels["ik_warmup_batch" if match[1].startswith("batch") else "ik_warmup_single"] = \
                    f"{match[3]} ({match[2]}s)"  # fmt: skip
        spans = capture.thread_span
        warm_starts = [spans[t][0] for t in ("pyroki-warmup-single", "pyroki-warmup-batch") if t in spans]
        phases, total = [], dict(ZERO_STATS)
        for thread, name in _IK_THREADS.items():
            if name == "ik_setup":
                start, end = t0, min(warm_starts, default=ready)
            elif thread in spans:
                start, end = spans[thread]
            else:
                continue
            stats = capture.thread_stats.get(thread, ZERO_STATS)
            total = {k: total[k] + stats[k] for k in total}
            phases.append(_phase_row(name, thread, start, end, stats, self.build["built_at"], labels.get(name)))
        phases.append(_phase_row("ik_total", "pyroki-compile", t0, ready, total, self.build["built_at"]))
        return phases

    def _phase(self, name: str, fn: Callable):
        """Runs one build step on this thread and adds it to the build phases with its compile stats.

        name: Phase name. fn: The step, without arguments.
        Returns: What fn returns.
        """
        t0 = time.time()
        with capture.slot() as stats:
            result = fn()
        self.build["phases"].append(_phase_row(name, threading.current_thread().name, t0, time.time(), stats,
                                               self.build["built_at"]))  # fmt: skip
        return result

    def _finish_build(self, window: CaptureWindow, cache_files_before: int | None) -> None:
        """Completes self.build with the totals, setup and the build's log lines.

        self.build: {"built_at" (epoch s), "total_s", "jax_version", "jax_cache_dir", "cache_files_before" /
        "cache_files_after" (programs in the JAX cache; None without a cache folder), "env" (planner env vars),
        "phases" (see _phase_row), "log" / "messages" / "log_dropped" / "compile" (see CaptureWindow.result)}.

        window: The build's capture window (closed). cache_files_before: Cache files before the build.
        """
        self.build.update(
            total_s=round(time.time() - self.build["built_at"], 3), jax_version=getattr(jax, "__version__", "?"),
            jax_cache_dir=self._cache_dir(), cache_files_before=cache_files_before,
            cache_files_after=_count_files(self._cache_dir()),
            env={k: os.environ[k] for k in _BUILD_ENV if k in os.environ}, **window.result(),
        )  # fmt: skip

    @contextmanager
    def capture_run(self) -> Iterator[CaptureWindow]:
        """Collects the planner's log lines, cloud messages and compile events while one run plans.

        Yields: The run's window (window.result() has them, times relative to the run start).
        """
        self.runs_started += 1
        window = capture.open_window(time.time())
        try:
            yield window
        finally:
            capture.close_window(window)

    def _timed(self, call: str, n: int, fn: Callable):
        """Runs one planner call and records how long it took and what JAX compiled during it.

        call: Name of the call for the timings. n: Poses (or frames) it handled.
        fn: The call, without arguments.
        Returns: What fn returns.
        """
        t0 = time.time()
        with capture.slot() as stats:
            result = fn()
        seconds, compiled = time.time() - t0, with_compile_s(stats)
        self.timings.append({"call": call, "command": self.command, "n": n, "seconds": seconds, "start": t0,
                             "compile_s": compiled["compile_s"], "cache_hits": stats["cache_hits"],
                             "cache_misses": stats["cache_misses"]})  # fmt: skip
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

    def plan_transit(self, start_deg: np.ndarray, goal_deg: np.ndarray, call: str = "transit") -> np.ndarray:
        """Plans a joint path around the obstacles, like pyroki_world_poc.plan_transit.

        start_deg, goal_deg: (6,) end joints of the path, degrees.
        call: Name of this use in the timings ("transit" for replay reroutes, "detour" for single moves).
        Returns: (TRANSIT_STEPS, 6) path in degrees; not guaranteed clear, the caller checks it.
        """
        path = self._timed(call, 1, lambda: np.asarray(self._transit(
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
