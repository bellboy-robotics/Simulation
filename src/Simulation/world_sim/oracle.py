"""Independent check that a collision-free joint path exists: a slow, thorough sampling planner (RRT-Connect).

Used to tell the planner's misses (a path exists, the planner did not find it) from cases with no
solution. It knows the same constraints the robot enforces, and a bit more:
- joint limits;
- every moving link at least CLEARANCE_M from every avoid capsule (the arm-move guard asks for >= 0);
- no self-collision and no contact with Billie's body beyond SELF_MARGIN_M (the planner's own model);
- no link below the floor (the planner does not model the floor; a path through it is no solution).
"""

import jax
import jax.numpy as jnp
import numpy as np
from billie_utils.world_collision_check_poc import link_obstacle_distances, link_transforms, sample_joint_path

from Simulation.world_sim.planner import SimPlanner

# [m] Clearance to obstacles: the arm-move guard's 0 plus 1mm, so an oracle path is one the robot accepts.
CLEARANCE_M = 0.001
# [m] Smallest self-collision distance accepted: the planner's soft self-collision cost lets links come
# ~1-2mm into each other, so the arm's real states would otherwise count as invalid.
SELF_MARGIN_M = -0.002
# [deg] Joint-limit tolerance: the planner's soft limit cost leaves solutions up to ~0.005deg past a limit.
LIMIT_TOL_DEG = 0.01
# [deg] Largest joint step between the configurations an oracle path segment is checked at.
CHECK_STEP_DEG = 2.0
# [deg] Largest step one RRT extension takes toward a sample.
EXTEND_STEP_DEG = 10.0
# RRT iterations before giving up: enough for a cluttered 6-joint case, ~10-30s on a laptop CPU.
MAX_ITERATIONS = 4000
# [deg] Range sampled around the start for joints that turn > 360deg (J1, J4, J6), so paths stay sane.
_TURN_RANGE_DEG = 180.0
_SELF_MIN: dict = {}  # id(robot collision model) -> its compiled batched self-collision distance


class Oracle:
    """Collision checks and RRT-Connect for one world."""

    def __init__(
        self, planner: SimPlanner, avoid: tuple[np.ndarray, np.ndarray, np.ndarray], floor_z_m: float,
        clearance_m: float = CLEARANCE_M, self_margin_m: float = SELF_MARGIN_M,
    ):  # fmt: skip
        """planner: For the arm model and its self-collision model. avoid: (starts_m, ends_m, radii_m)
        avoid capsules, xArm base frame. floor_z_m: Arm-frame z of the floor, meters.
        clearance_m, self_margin_m: Obstacle clearance and self-collision distance required, meters
        (stricter values make sure generated targets are clearly valid)."""
        self.clearance_m, self.self_margin_m = clearance_m, self_margin_m
        self.model = planner.model
        self.avoid = avoid
        self.floor_z_m = floor_z_m
        robot, coll = planner.resolver.robot, planner.resolver.robot_coll
        self.lower = np.rad2deg(np.asarray(robot.joints.lower_limits, dtype=np.float64))
        self.upper = np.rad2deg(np.asarray(robot.joints.upper_limits, dtype=np.float64))
        # Smallest self-collision distance per configuration (pairs, ignore list and base AABBs as the planner);
        # one compiled function per robot, shared by every world's oracle.
        if id(coll) not in _SELF_MIN:
            _SELF_MIN[id(coll)] = jax.jit(jax.vmap(lambda q: jnp.min(coll.compute_self_collision_distance(robot, q))))
        self._self_min = _SELF_MIN[id(coll)]
        self._movable = self.model.movable_links & (self.model.capsule_radius > 0)

    def checks(self, joints_deg: np.ndarray) -> dict[str, np.ndarray]:
        """Each constraint of the module docstring, per configuration.

        joints_deg: (S, 6) configurations, degrees.
        Returns: {"limits", "obstacles", "floor", "self_collision"}: (S,) bool arrays, True = satisfied.
        """
        q = np.atleast_2d(joints_deg)
        rad = np.deg2rad(q)
        in_limits = (q >= self.lower - LIMIT_TOL_DEG) & (q <= self.upper + LIMIT_TOL_DEG)
        out = {"limits": np.all(in_limits, axis=1), "obstacles": np.ones(len(q), bool)}
        if len(self.avoid[2]):
            d = link_obstacle_distances(self.model, rad, *self.avoid).min(axis=2)[:, self._movable]
            out["obstacles"] = d.min(axis=1) >= self.clearance_m
        T = link_transforms(self.model, rad) @ self.model.capsule_local
        half = (self.model.capsule_height / 2.0)[None, :, None] * T[..., :3, 2]
        lowest = np.minimum((T[..., :3, 3] - half)[..., 2], (T[..., :3, 3] + half)[..., 2]) - self.model.capsule_radius
        out["floor"] = lowest[:, self._movable].min(axis=1) > self.floor_z_m
        padded = np.vstack([rad, np.zeros((-len(rad) % 64, 6))])  # fixed batch sizes: no recompiles
        self_min = np.concatenate([np.asarray(self._self_min(jnp.asarray(b, jnp.float32))) for b in padded.reshape(-1, 64, 6)])
        out["self_collision"] = self_min[: len(q)] >= self.self_margin_m
        return out

    def valid(self, joints_deg: np.ndarray) -> np.ndarray:
        """Which configurations satisfy every constraint. joints_deg: (S, 6) degrees. Returns: (S,) bool."""
        return np.logical_and.reduce(list(self.checks(joints_deg).values()))

    def at_limits(self, joints_deg: np.ndarray, within_deg: float = 0.05) -> list[int]:
        """Joints sitting on a limit (an IK pinned there usually missed its target).

        joints_deg: (6,) degrees. within_deg: How close counts as on the limit.
        Returns: 1-based joint numbers.
        """
        q = np.asarray(joints_deg, dtype=np.float64)
        return [j + 1 for j in range(6) if min(abs(q[j] - self.lower[j]), abs(q[j] - self.upper[j])) <= within_deg]

    def broken(self, joints_deg: np.ndarray) -> list[str]:
        """The constraints one configuration breaks. joints_deg: (6,) degrees. Returns: Their names ([] if valid)."""
        return [name for name, ok in self.checks(np.asarray(joints_deg)[None]).items() if not ok[0]]

    def segment_ok(self, a_deg: np.ndarray, b_deg: np.ndarray) -> bool:
        """Whether the straight joint line a -> b stays valid (checked every CHECK_STEP_DEG).

        a_deg, b_deg: (6,) end configurations, degrees. Returns: True if every check point is valid.
        """
        return bool(self.valid(sample_joint_path(np.stack([a_deg, b_deg]), CHECK_STEP_DEG)).all())

    def path_ok(self, path_deg: np.ndarray) -> bool:
        """Whether a whole joint polyline is valid. path_deg: (W, 6) waypoints. Returns: True if valid."""
        return bool(self.valid(sample_joint_path(path_deg, CHECK_STEP_DEG)).all())

    def plan(self, start_deg: np.ndarray, goal_deg: np.ndarray, seed: int = 0) -> tuple[np.ndarray | None, str]:
        """Searches a valid joint path with RRT-Connect, then shortens it.

        start_deg, goal_deg: (6,) end configurations, degrees.
        seed: Random seed (the search is reproducible).
        Returns: (path (W, 6) degrees from start to goal, or None; "found", "direct", "start_invalid",
            "goal_invalid" or "not_found").
        """
        start, goal = np.asarray(start_deg, dtype=np.float64), np.asarray(goal_deg, dtype=np.float64)
        ends_ok = self.valid(np.stack([start, goal]))
        if not ends_ok[0] or not ends_ok[1]:
            return None, "start_invalid" if not ends_ok[0] else "goal_invalid"
        if self.segment_ok(start, goal):
            return np.stack([start, goal]), "direct"
        rng = np.random.default_rng(seed)
        low = np.maximum(self.lower, np.minimum(start, goal) - _TURN_RANGE_DEG)
        high = np.minimum(self.upper, np.maximum(start, goal) + _TURN_RANGE_DEG)
        trees = [([start], [-1]), ([goal], [-1])]  # (nodes, parent index) from the start and from the goal
        for i in range(MAX_ITERATIONS):
            sample = goal if i % 10 == 0 else rng.uniform(low, high)  # 10% goal bias keeps it heading home
            grown = self._extend(trees[0], sample)
            if grown is not None and self._connect(trees[1], grown):
                path = self._join(trees, i % 2 == 1)
                return self._shorten(path, rng), "found"
            trees.reverse()
        return None, "not_found"

    def _extend(self, tree: tuple[list, list], target: np.ndarray) -> np.ndarray | None:
        """Grows a tree one step from its nearest node toward a target.

        tree: (nodes, parents), extended in place. target: (6,) degrees.
        Returns: The new node, or None if the step is blocked.
        """
        nodes, parents = tree
        nearest = int(np.argmin(np.abs(np.asarray(nodes) - target).max(axis=1)))
        step = target - nodes[nearest]
        reach = np.abs(step).max()
        new = target if reach <= EXTEND_STEP_DEG else nodes[nearest] + step * (EXTEND_STEP_DEG / reach)
        if not self.segment_ok(nodes[nearest], new):
            return None
        nodes.append(new)
        parents.append(nearest)
        return new

    def _connect(self, tree: tuple[list, list], target: np.ndarray) -> bool:
        """Grows a tree toward a target until it reaches it or is blocked.

        tree: (nodes, parents). target: (6,) degrees. Returns: True if the target was reached.
        """
        while True:
            new = self._extend(tree, target)
            if new is None:
                return False
            if np.allclose(new, target):
                return True

    @staticmethod
    def _join(trees: list, swapped: bool) -> np.ndarray:
        """The start-to-goal path through the two connected trees.

        trees: [tree that grew last, the other], each (nodes, parents); their last nodes meet.
        swapped: True if trees[0] is the goal tree.
        Returns: (W, 6) path, degrees.
        """
        def branch(tree: tuple[list, list]) -> list:
            """Nodes from a tree's root to its last node."""
            nodes, parents = tree
            out, i = [], len(nodes) - 1
            while i != -1:
                out.append(nodes[i])
                i = parents[i]
            return out[::-1]

        a, b = branch(trees[0]), branch(trees[1])
        path = a + b[::-1][1:]  # both branches end at the meeting node
        return np.asarray(path[::-1] if swapped else path)

    def _shorten(self, path: np.ndarray, rng: np.random.Generator, tries: int = 150) -> np.ndarray:
        """Removes detours by replacing random sub-paths with straight lines when those are valid.

        path: (W, 6) valid path. rng: Random generator. tries: Shortcut attempts.
        Returns: The shorter valid path, same ends.
        """
        path = list(path)
        for _ in range(tries):
            if len(path) <= 2:
                break
            i, j = sorted(rng.choice(len(path), 2, replace=False))
            if j - i > 1 and self.segment_ok(path[i], path[j]):
                path = path[: i + 1] + path[j:]
        return np.asarray(path)
