"""Where arm poses put the camera, how far the arm sticks out of Billie's outline, and how close it comes to Billie's
body and to itself, on the planner's own model. For choosing a fold, a look pose or an entry pose. No arm moves.

    PYTHONPATH=src:billie-onboard/billie/nodes/pyroki-planner:billie-onboard/billie/billie-utils \
    billie-onboard/billie/nodes/pyroki-planner/.venv/bin/python -m Simulation.flow_fix.pose_report \
        --urdf output/robot_logs/billie-28/2026-11-15/now/with-tcp.urdf \
        --pose "look back=-189.12,-55.33,-153.42,3.13,118.14,-178.74" --from "fold=0.96,-13.94,-17.8,-2.59,33.74,-177.23"

Positions are in the arm base frame (x forward, z up, meters). "looks" is the gripper's approach axis, which the
camera on it follows. "reaches x" is the arm's farthest point behind the arm base (Billie's backpack ends at about
-0.23 m): a pose that reaches further back sticks out behind Billie. Clearances are the planner's self check (distance
minus its allowance): negative is a refused overlap. --from also checks the straight joints move from each start to
each pose (joints goals check only Billie's body, so that is the one that counts for them).
"""

import argparse

import numpy as np

from Simulation.mapping_sim import pose_check  # sets the planner's import-time settings first
from Simulation.mapping_sim.scan_coverage import _mesh_file

# Billie's fixed parts: their pairs stay checked for a joints goal; the others are the arm against itself.
BODY = ("backpack_link", "covering_cylinder_link", "upper_platform_link", "mobile_base_link", "link_base")


def parse_pose(text: str) -> tuple[str, np.ndarray]:
    """A named pose from "name=j1,j2,j3,j4,j5,j6". text: The argument. Returns: (name, (6,) degrees)."""
    name, _, joints = text.partition("=")
    values = np.array([float(v) for v in joints.split(",")])
    if len(values) != 6:
        raise ValueError(f"{name}: 6 joints needed, got {len(values)}")
    return name.strip(), values


def build(urdf_path: str):
    """The planner's arm model and self check from a with-tcp URDF. urdf_path: The file. Returns: (model, self_check)."""
    import pyroki as pk  # noqa: PLC0415 - the planner's packages, after pose_check's settings
    from pyroki_planner import self_check, world_collision  # noqa: PLC0415
    from pyroki_planner.base_collision import BaseCollisionChecker  # noqa: PLC0415
    from pyroki_planner.resolver import IGNORE_COLLISION_PAIRS  # noqa: PLC0415

    urdf = pose_check.load(urdf_path, _mesh_file)
    robot = pk.Robot.from_urdf(urdf)
    collision = pk.collision.RobotCollision.from_urdf(urdf, min_capsule=True, enable_aabb_for_static_links=True,
                                                      user_ignore_pairs=IGNORE_COLLISION_PAIRS)  # fmt: skip
    world_collision.set_movable_links(robot, collision)
    model = world_collision.export_collision_model(robot, collision)
    self_check.configure(robot, collision, model, BaseCollisionChecker(urdf).boxes)
    return model, self_check


def worst_pairs(check, joints: np.ndarray) -> tuple[tuple[float, str], tuple[float, str]]:
    """The tightest body pair and arm pair at a pose. check: self_check. joints: (6,) deg.
    Returns: ((meters past the rule, pair) for Billie's body, the same for the arm against itself)."""
    d = np.asarray(check._distances(joints[None]))[0] + check._allowance
    kept = [i for i in range(len(d)) if check._keep[i]]
    body = [i for i in kept if any(b in check._pairs[i] for b in BODY)]
    arm = [i for i in kept if i not in body]
    b, a = min(body, key=lambda i: d[i]), min(arm, key=lambda i: d[i])
    return (float(d[b]), check._pairs[b]), (float(d[a]), check._pairs[a])


def pose_line(model, check, name: str, joints: np.ndarray) -> str:
    """One pose's report line. model: The arm model. check: self_check. name, joints: The pose. Returns: The line."""
    from billie_utils.arm_collision_model import link_transforms  # noqa: PLC0415
    from pyroki_planner.look import TCP_LINK  # noqa: PLC0415

    frames = link_transforms(model, np.deg2rad(joints[None]))[0]
    ends = []
    for link in np.flatnonzero(model.movable_links):
        capsule = frames[link] @ model.capsule_local[link]
        for s in (-model.capsule_height[link] / 2, model.capsule_height[link] / 2):
            ends.append(capsule[:3, 3] + s * capsule[:3, 2])
    tcp = frames[model.link_names.index(TCP_LINK)]
    (body, body_pair), (arm, arm_pair) = worst_pairs(check, joints)
    return (f"{name:24} tcp {np.round(tcp[:3, 3], 2).tolist()} looks {np.round(tcp[:3, 2], 2).tolist()} "
            f"reaches x {np.min(np.array(ends)[:, 0]):+.2f} m | body {body * 1000:+.0f} mm ({body_pair}) "
            f"| arm {arm * 1000:+.0f} mm ({arm_pair})")  # fmt: skip


def path_line(check, start: tuple[str, np.ndarray], goal: tuple[str, np.ndarray]) -> str:
    """The straight joints move's tightest point against Billie's body (a joints goal's check).

    check: self_check. start, goal: Named poses. Returns: The report line.
    """
    from billie_utils.arm_collision_model import sample_joint_path  # noqa: PLC0415

    with check.suspended():  # as for a joints goal: Billie's body only
        past, pair = check.path_past(sample_joint_path(np.stack([start[1], goal[1]])))
    k = int(np.argmin(past))
    verdict = "ok" if past[k] >= 0 else "REFUSED"
    return f"{start[0]} -> {goal[0]}: {verdict}, tightest {past[k] * 1000:+.0f} mm ({check.pair_name(int(pair[k]))})"


def main() -> None:
    """Parses the arguments and prints each pose, then each straight move from the --from poses."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--urdf", required=True, help="the robot's <SN>-with-tcp.urdf (never call urdf.load_urdf)")
    parser.add_argument("--pose", action="append", required=True, help='"name=j1,...,j6" degrees; repeat for more')
    parser.add_argument("--from", dest="starts", action="append", default=[], help='"name=j1,...,j6": check moves')
    args = parser.parse_args()
    model, check = build(args.urdf)
    poses = [parse_pose(p) for p in args.pose]
    for name, joints in poses:
        print(pose_line(model, check, name, joints))
    for start in (parse_pose(s) for s in args.starts):
        for goal in poses:
            print(path_line(check, start, goal))


if __name__ == "__main__":
    main()
