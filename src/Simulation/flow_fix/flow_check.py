"""Before the first mapped run: walks a flow as the camera, and lists the arm moves that go where the camera has not
looked and the flow spots where the guide's fixes (F1-F6, FIX_FLOW_FOR_ARM_AWARENESS.md) apply. No robot, no map.

    PYTHONPATH=src:billie-onboard/billie/nodes/pyroki-planner:billie-onboard/billie/billie-utils \
    billie-onboard/billie/nodes/pyroki-planner/.venv/bin/python -m Simulation.flow_fix.flow_check \
        flow.billie.yaml --urdf <SN>-with-tcp.urdf --fold "0.96,-13.94,-17.8,-2.59,33.74,-177.23"

How "seen" is judged. It is optimistic: a warning is worth acting on, silence is no promise.
- The camera sees along the gripper's approach axis, in a cone of CONE_HALF_DEG, out to RANGE_M, through everything:
  walls and furniture hide nothing here (under a bed is caught only by the low-pose rule).
- Frames come from the scan's views, from every pose held still (a photo, sleep or say after it), and from the arm's
  pose where the base stops (the first arm move after a drive waits for the room build, still).
- Never-seen space the planner counts as free (spatial_awareness occupancy_field): above Billie's base, its shadow
  beside the base up to 0.8 m, 0.2 m around where the arm has been.
- A pose near the floor needs a close look down at that floor (the user: the occupancy grid needs it to place the
  floor right).
Billie's pose follows the flow: navigate_poi coordinates, slides along the heading, twists by TWIST_STEP_DEG a step. A
named POI or a map change leaves it unknown; the scan's place then comes from --start-pose, else from the room's first
navigate_poi moved back by the slides before the scan (Billie at the door, facing in).
"""

import argparse
from dataclasses import dataclass, field

import numpy as np
import yaml

from Simulation.mapping_sim.scan_coverage import scan_views

from .pose_report import build  # imports pose_check first: the planner's import-time settings

# [deg] Half the camera's view cone: the ZED X sees about 110 x 80 degrees; the narrower half is taken.
CONE_HALF_DEG = 40.0
# [m] How far the camera's depth is trusted for the map (the mapper keeps 5 m; walls and stereo noise cut it short).
RANGE_M = 3.0
# [m] Where the arm base sits on the AutoXing base: 0.18 m behind its center, 0.4405 m above the floor (robot.json).
ARM_BASE_X_M, ARM_BASE_Z_M = -0.18, 0.4405
# [m] Billie's base footprint half sizes (spatial_awareness BASE_HALF_LENGTH_M, BASE_HALF_SIDE_M).
BASE_HALF_LENGTH_M, BASE_HALF_SIDE_M = 0.365, 0.240
# [m] occupancy_field: free above the base up to 2 m, in its shadow (0.5 m out) up to 0.8 m, and 0.2 m around the arm.
ABOVE_BASE_FREE_TOP_M, BASE_SHADOW_M, BASE_SHADOW_TOP_M, ARM_SEEN_M = 2.0, 0.50, 0.80, 0.20
# [m] An arm point this low above the floor needs a look down at that floor, from no further than FLOOR_LOOK_M
# (occupancy_field measures the floor 0.45-1.2 m from Billie's center).
LOW_M, FLOOR_LOOK_M = 0.20, 1.2
# [deg] One twist step: 7 steps turned Billie about half a turn in room 0159 (2026-11-16).
TWIST_STEP_DEG = 26.0
# [deg] The largest joint step between sampled arm configurations on a move.
SAMPLE_STEP_DEG = 5.0
# A move is reported when this many of its sampled arm points are neither seen nor counted free: one never-seen 5 cm
# spot blocks a move (billie-26's real block, "Hanging chair", 2026-11-16: 6 of 374 points here).
MIN_UNSEEN_POINTS = 3
# [m] A drive this long, ending in arm work, is worth a stop on the way (F4) when it is not already a travel pose.
LONG_DRIVE_M = 2.0
# Actions that hold the arm still long enough for the mapper's frames (arm still 0.75 s, then ~3 frames).
HOLDS = ("photo", "sleep", "say")
# Words in a replay's repo or task that mean it touches something (F2).
CONTACT_WORDS = ("door", "handle", "cabinet", "drawer", "lid", "curtain", "open", "close", "knob", "pull", "push")


@dataclass
class Step:
    """One flow action."""

    label: str  # "area > item #index"
    name: str  # the command, e.g. "joints"
    args: dict  # its arguments


@dataclass
class Walk:
    """What the walk knows at a point of the flow."""

    arm: np.ndarray | None  # the arm's joints, degrees (None after a replay or AI task)
    base: np.ndarray | None = None  # Billie's (x, y, heading rad) in the room map, None when unknown
    mapping: bool = False  # after start_active_mapping
    views: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)  # camera (position, axis), map
    been: list[np.ndarray] = field(default_factory=list)  # (N, 3) arm points in the map, where the arm has been
    drove_m: float = 0.0  # meters driven since the arm last worked


class Arm:
    """The arm's points and camera from the planner's model, in the arm base frame (x ahead, z up, meters)."""

    def __init__(self, urdf: str):
        """urdf: The robot's with-tcp URDF."""
        from billie_utils.arm_collision_model import link_transforms  # noqa: PLC0415
        from pyroki_planner.look import TCP_LINK  # noqa: PLC0415

        self.model, _ = build(urdf)
        self._fk = link_transforms
        self._tcp = self.model.link_names.index(TCP_LINK)
        self._links = np.flatnonzero(self.model.movable_links)

    def points(self, path_deg: np.ndarray) -> np.ndarray:
        """Each sampled configuration's capsule ends and TCP. path_deg: (S, 6). Returns: (S * K, 3)."""
        frames = self._fk(self.model, np.deg2rad(path_deg))
        out = [frames[:, self._tcp, :3, 3]]
        for link in self._links:
            capsule = frames[:, link] @ self.model.capsule_local[link]
            for s in (-0.5, 0.5):
                out.append(capsule[:, :3, 3] + s * self.model.capsule_height[link] * capsule[:, :3, 2])
        return np.concatenate(out)

    def camera(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The camera at a pose. joints: (6,) degrees. Returns: (position, unit view axis), arm base frame."""
        tcp = self._fk(self.model, np.deg2rad(joints[None]))[0, self._tcp]
        return tcp[:3, 3], tcp[:3, 2]


def to_map(base: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Arm base frame points in the room map. base: (x, y, heading). points: (N, 3). Returns: (N, 3), z from floor."""
    c, s = np.cos(base[2]), np.sin(base[2])
    x, y = points[:, 0] + ARM_BASE_X_M, points[:, 1]
    return np.stack([base[0] + c * x - s * y, base[1] + s * x + c * y, points[:, 2] + ARM_BASE_Z_M], axis=1)


def in_base_frame(base: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map points relative to Billie's center. base: (x, y, heading). points: (N, 3). Returns: (N, 3), x ahead."""
    c, s = np.cos(base[2]), np.sin(base[2])
    dx, dy = points[:, 0] - base[0], points[:, 1] - base[1]
    return np.stack([c * dx + s * dy, -s * dx + c * dy, points[:, 2]], axis=1)


def counted_free(walk: Walk, points: np.ndarray) -> np.ndarray:
    """Which never-seen points the planner counts as free. points: (N, 3) map. Returns: (N,) bool."""
    local = in_base_frame(walk.base, points)
    out_x = np.maximum(np.abs(local[:, 0]) - BASE_HALF_LENGTH_M, 0)
    out_y = np.maximum(np.abs(local[:, 1]) - BASE_HALF_SIDE_M, 0)
    off_base = np.hypot(out_x, out_y)
    free = ((off_base == 0) & (local[:, 2] <= ABOVE_BASE_FREE_TOP_M)) | (
        (off_base <= BASE_SHADOW_M) & (local[:, 2] <= BASE_SHADOW_TOP_M)
    )
    if walk.been:
        been = np.concatenate(walk.been)
        for chunk in range(0, len(points), 512):  # bounded memory for long walks
            part = points[chunk : chunk + 512]
            near = np.min(np.linalg.norm(part[:, None] - been[None], axis=2), axis=1) <= ARM_SEEN_M
            free[chunk : chunk + 512] |= near
    return free


def seen(walk: Walk, points: np.ndarray, reach: float = RANGE_M, down_only: bool = False) -> np.ndarray:
    """Which points lie in some earlier view's cone.

    points: (N, 3) map. reach: The view's range, meters. down_only: Count only views looking down (a floor look).
    Returns: (N,) bool.
    """
    out = np.zeros(len(points), bool)
    cos_half = np.cos(np.deg2rad(CONE_HALF_DEG))
    for position, axis in walk.views:
        if down_only and axis[2] > -0.5:
            continue
        ray = points - position
        dist = np.linalg.norm(ray, axis=1)
        out |= (dist <= reach) & (np.einsum("ij,j->i", ray, axis) >= cos_half * np.maximum(dist, 1e-9))
    return out


def where(walk: Walk, points: np.ndarray) -> str:
    """Where points lie around Billie, in words. points: (N, 3) map. Returns: e.g. "behind, left, low"."""
    local = in_base_frame(walk.base, points)
    words = []
    for test, word in ((local[:, 0] > BASE_HALF_LENGTH_M, "ahead"), (local[:, 0] < -BASE_HALF_LENGTH_M, "behind"),
                       (local[:, 1] > BASE_HALF_SIDE_M, "left"), (local[:, 1] < -BASE_HALF_SIDE_M, "right"),
                       (local[:, 2] < LOW_M + 0.2, "low")):  # fmt: skip
        if np.mean(test) > 0.3:
            words.append(word)
    return ", ".join(words) or "around the arm"


def check_move(walk: Walk, arm: Arm, goal: np.ndarray, label: str, out: list[str]) -> None:
    """Reports an arm move into never-seen space or near an unseen floor, then marks its places as been.

    walk: The walk so far. arm: The model. goal: (6,) degrees. label: The action. out: The report lines.
    """
    from billie_utils.arm_collision_model import sample_joint_path  # noqa: PLC0415

    start = goal if walk.arm is None else walk.arm
    points = to_map(walk.base, arm.points(sample_joint_path(np.stack([start, goal]), SAMPLE_STEP_DEG)))
    if walk.mapping:
        missing = ~(seen(walk, points) | counted_free(walk, points))
        if missing.sum() >= MIN_UNSEEN_POINTS:
            side = where(walk, points[missing])
            out.append(f"{label}: {missing.sum()} of {len(points)} arm points on the move unseen ({side}) -> F4")
        low = points[:, 2] < LOW_M
        if low.any() and not seen(walk, points[low], FLOOR_LOOK_M, down_only=True).all():
            out.append(f"{label}: reaches {points[low, 2].min():.2f} m above the floor with no close look down at it "
                       "-> F4 floor look (under furniture, look from beside it)")  # fmt: skip
    walk.been.append(points)


def add_view(walk: Walk, arm: Arm) -> None:
    """Adds the camera's view at the arm's pose where Billie stands. walk: Changed in place. arm: The model."""
    if walk.base is None or walk.arm is None or not walk.mapping:
        return
    position, axis = arm.camera(walk.arm)
    walk.views.append((to_map(walk.base, position[None])[0], to_map(walk.base, (position + axis)[None])[0]
                       - to_map(walk.base, position[None])[0]))  # fmt: skip


def flatten(flow: dict) -> list[Step]:
    """The flow's actions in order. flow: The loaded YAML. Returns: The steps."""
    steps = []
    for area in flow["areas"]:
        for item in area["items"]:
            for i, action in enumerate(item["actions"]):
                ((name, args),) = action.items()
                steps.append(Step(f"{area['name'][:30]} > {item['name']} #{i + 1}", name, args or {}))
    return steps


def numeric_poi(step: Step) -> np.ndarray | None:
    """A navigate_poi's coordinates. step: The action. Returns: (x, y, heading), or None for a named POI."""
    poi = step.args.get("poi") if step.name == "navigate_poi" else None
    return np.array(poi, float) if isinstance(poi, list) and len(poi) == 3 else None


def scan_place(steps: list[Step], i: int, start_pose: np.ndarray | None) -> np.ndarray | None:
    """Where Billie stands for a scan when the walk does not know: --start-pose, else the next coordinates POI moved
    back by the slides between the last map change and the scan.

    steps: The flow. i: The scan's index. start_pose: The user's (x, y, heading) or None. Returns: The pose, or None.
    """
    if start_pose is not None:
        return start_pose
    target = next((p for s in steps[i:] if (p := numeric_poi(s)) is not None), None)
    if target is None:
        return None
    k = i
    while k > 0 and steps[k - 1].name not in ("map", "navigate_poi"):
        k -= 1
    moved = sum(s.args.get("meters", 0) * (1 if s.args.get("direction") == "forward" else -1)
                for s in steps[k:i] if s.name == "slide")  # fmt: skip
    return target + moved * np.array([np.cos(target[2]), np.sin(target[2]), 0.0])


def move_base(walk: Walk, step: Step) -> None:
    """Billie's pose after a base action. walk: Changed in place. step: navigate_poi, slide, twist or map."""
    before = walk.base
    if step.name == "map" or (step.name == "navigate_poi" and numeric_poi(step) is None):
        walk.base = None
    elif step.name == "navigate_poi":
        walk.base = numeric_poi(step)
    elif walk.base is not None and step.name == "slide":
        sign = 1 if step.args.get("direction") == "forward" else -1
        heading = walk.base[2]
        walk.base = walk.base + sign * step.args.get("meters", 0) * np.array([np.cos(heading), np.sin(heading), 0])
    elif walk.base is not None and step.name == "twist" and step.args.get("direction") in ("left", "right"):
        sign = 1 if step.args["direction"] == "left" else -1
        walk.base = walk.base + [0, 0, sign * np.deg2rad(TWIST_STEP_DEG) * step.args.get("steps", 1)]
    if before is not None and walk.base is not None:
        walk.drove_m += float(np.hypot(*(walk.base[:2] - before[:2])))


def is_contact(steps: list[Step], i: int) -> bool:
    """Whether a replay touches something: its repo or task names it, or a close_gripper comes right before it.

    steps: The flow. i: The replay's index. Returns: True for a contact replay.
    """
    s = steps[i]
    text = f"{s.args.get('repo_id', '')} {s.args.get('options', {}).get('task', '')}".lower()
    return any(word in text for word in CONTACT_WORDS) or (i > 0 and steps[i - 1].name == "close_gripper")


def fix_notes(walk: Walk, arm: Arm, steps: list[Step], i: int, fold: np.ndarray) -> list[str]:
    """The guide's fixes that apply at an action, from the flow alone.

    walk: The walk before the action. arm: The model. steps: The flow. i: The action's index. fold: The fold joints.
    Returns: Report lines.
    """
    s, notes = steps[i], []
    if s.name == "start_active_mapping":
        before = steps[max(0, i - 4) : i]
        if not any(b.name == "light_on" for b in before):
            notes.append(f"{s.label}: no light_on just before the scan (the scan needs light) -> F1")
        if not any(b.name == "slide" and b.args.get("direction") == "backward" for b in before):
            notes.append(f"{s.label}: the scan runs where the work starts; back off first -> F1")
    elif s.name == "replay_policy" and not s.args.get("options", {}).get("direct_action"):
        repo = str(s.args.get("repo_id", "")).split("/")[-1]
        if is_contact(steps, i):
            notes.append(f"{s.label}: contact replay {repo} without direct_action -> F2")
        else:
            notes.append(f"{s.label}: replay {repo}: if it only reaches a pose, a joints action -> F3 "
                         "(check with recording_joints)")  # fmt: skip
    elif walk.mapping and s.name == "navigate_poi" and s.args.get("fold_arm", True) is not False:
        target = numeric_poi(s)
        drive = float(np.hypot(*(target[:2] - walk.base[:2]))) if target is not None and walk.base is not None else 0
        works = next((k for k in range(i + 1, len(steps)) if steps[k].name in ("joints", "fold", "replay_policy",
                      "ai_task", "navigate_poi", "slide", "twist")), None)  # fmt: skip
        # arm work next, unless a contact replay (F2: it skips the room, so the travel pose does not matter)
        next_name = steps[works].name if works is not None else ""
        contact = next_name == "replay_policy" and is_contact(steps, works)
        if next_name in ("joints", "replay_policy", "ai_task") and not contact:
            far = f" after a {drive:.1f} m drive (a stop on the way: F4)" if drive > LONG_DRIVE_M else ""
            notes.append(f"{s.label}: folds the camera to the floor before arm work{far} -> F5 "
                         "(room travel pose + fold_arm: false)")  # fmt: skip
    elif walk.mapping and s.name == "slide" and s.args.get("direction") == "backward" and s.args.get("meters", 0) >= 0.3:
        joints = fold if walk.arm is None else walk.arm
        if walk.base is not None and arm.camera(joints)[1][0] > -0.5:  # the camera does not look backwards
            notes.append(f"{s.label}: backs {s.args['meters']} m with the camera not looking back -> F6")
    return notes


def start_mapping(walk: Walk, arm: Arm, steps: list[Step], i: int, start_pose, out: list[str]) -> None:
    """The scan: resets the map, then adds its views and the places the arm went.

    walk: Changed in place. arm: The model. steps, i: The flow and the scan's index. start_pose: --start-pose or None.
    out: The report lines.
    """
    if walk.base is None:
        walk.base = scan_place(steps, i, start_pose)
        where_from = "--start-pose" if start_pose is not None else "the next navigate_poi moved back by the slides"
        out.append(f"{steps[i].label}: scan place {np.round(walk.base, 3).tolist() if walk.base is not None else '?'}"
                   f" assumed ({where_from})")  # fmt: skip
    walk.mapping, walk.views, walk.been = True, [], []
    if walk.base is None:
        return
    for view in scan_views():
        joints = np.array(view)
        position, axis = arm.camera(joints)
        walk.views.append((to_map(walk.base, position[None])[0], to_map(walk.base, (position + axis)[None])[0]
                           - to_map(walk.base, position[None])[0]))  # fmt: skip
        walk.been.append(to_map(walk.base, arm.points(joints[None])))
    add_view(walk, arm)


def step(walk: Walk, arm: Arm, steps: list[Step], i: int, args: argparse.Namespace, out: list[str]) -> None:
    """Walks one action: its fix notes, the arm move check, the base move and the views it adds.

    walk: Changed in place. arm: The model. steps, i: The flow and the action's index. args: The options.
    out: The report lines.
    """
    s = steps[i]
    following = steps[i + 1].name if i + 1 < len(steps) else ""
    out.extend(fix_notes(walk, arm, steps, i, args.fold))
    if s.name == "start_active_mapping":
        start_mapping(walk, arm, steps, i, args.start_pose, out)
    elif s.name == "delete_mapping":
        walk.mapping, walk.views, walk.been = False, [], []
    elif s.name in ("navigate_poi", "slide", "twist", "map"):
        if s.name == "navigate_poi" and s.args.get("fold_arm", True) is not False:
            if walk.base is not None:
                check_move(walk, arm, args.fold, f"{s.label} (the fold before driving)", out)
            walk.arm = args.fold
        move_base(walk, s)
        add_view(walk, arm)  # the base stands still while the room is built for the next arm move
    elif s.name in ("joints", "fold"):
        goal = np.array(s.args["joints"], float) if s.name == "joints" else args.fold
        if walk.base is not None:
            check_move(walk, arm, goal, f"{s.label} {s.name}", out)
        walk.arm, walk.drove_m = goal, 0.0
        if following in HOLDS:
            add_view(walk, arm)
    elif s.name in HOLDS:
        add_view(walk, arm)
    elif s.name in ("replay_policy", "ai_task"):
        walk.arm, walk.drove_m = None, 0.0  # where it ends is not in the flow


def parse_joints(text: str | None) -> np.ndarray | None:
    """Comma separated numbers. text: e.g. "0.96,-13.94,...". Returns: The array, or None."""
    return None if text is None else np.array([float(v) for v in text.split(",")])


def main() -> None:
    """Parses the arguments, walks the flow and prints what to fix, in flow order."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("flow", help="the .billie.yaml")
    parser.add_argument("--urdf", required=True, help="the robot's <SN>-with-tcp.urdf (never call urdf.load_urdf)")
    parser.add_argument("--fold", required=True, help="the robot's fold joints (ARM_FOLD), degrees, comma separated")
    parser.add_argument("--start-pose", help="x,y,heading(rad) of Billie at the scan, in the room map")
    args = parser.parse_args()
    args.fold, args.start_pose = parse_joints(args.fold), parse_joints(args.start_pose)
    steps = flatten(yaml.safe_load(open(args.flow, encoding="utf-8")))
    arm = Arm(args.urdf)
    walk, out = Walk(arm=args.fold), []
    for i in range(len(steps)):
        step(walk, arm, steps, i, args, out)
    if not any(s.name == "start_active_mapping" for s in steps):
        out.insert(0, "no start_active_mapping: add one per room, after light_on, from further back -> F1")
    print("\n".join(out) or "nothing found (the check is optimistic: see its notes)")
    print(f"\n{len(steps)} actions walked. Optimistic: walls and furniture hide nothing here; each new pose needs "
          "pose_report and a try on the robot.")  # fmt: skip


if __name__ == "__main__":
    main()
