"""Pushed objects: doors and handles that turn about a hinge when Billie's arm pushes them.

Kinematic model, run on the planned arm path (the planner does not know these objects):
- Each object starts at 0 deg. Along the path, whenever an arm link would enter it, it turns just
  enough to stay clear: to the free angle nearest its current one, within [min_deg, max_deg].
- It stays where it was left (no spring, no inertia, no friction).
- At a limit with the arm still inside, it is blocked: the arm goes through it there, which is a
  collision. The depth and the pushing link are recorded.
- A handle with mounted_on turns with its door: the door's angle moves the handle's hinge first.
The path is sampled 4x finer than the playback (0.5 deg joint steps, the TCP moves < ~8mm), so a
link cannot jump through a door between two samples.
"""

from dataclasses import dataclass

import numpy as np
from billie_utils.world_collision_check_poc import CollisionModel, link_obstacle_distances
from scipy.spatial.transform import Rotation

from Simulation.world_sim.analysis import SAMPLE_STEP_DEG
from Simulation.world_sim.world import hinge_of

# Fine path samples per playback sample: 2 deg / 4 = 0.5 deg joint steps.
FINE_PER_SAMPLE = 4
# [deg] Angle grid an object is turned on while pushed: 7mm at the edge of a 0.8m door.
ANGLE_STEP_DEG = 0.5
# [m] Overlap still counted as touching rather than pushing: below the arm's tracking accuracy.
TOUCH_TOLERANCE_M = 0.001


@dataclass
class Pushable:
    """A door or handle of a run, ready to be turned."""

    name: str  # the object's name
    starts_m: np.ndarray  # (M, 3) capsule segment starts at 0 deg (its door at 0 deg too), xArm base frame
    ends_m: np.ndarray  # (M, 3) capsule segment ends at 0 deg
    radii_m: np.ndarray  # (M,) capsule radii
    hinge_m: np.ndarray  # (3,) a point on the hinge axis
    axis: np.ndarray  # (3,) unit hinge axis; a positive angle turns the configured direction
    min_deg: float  # turn limits from the initial pose
    max_deg: float
    parent: int | None  # index of the door a handle is mounted on, or None
    reach_m: float  # farthest capsule surface point from the hinge point (for the quick no-contact test)


def find_pushables(objects: list[dict]) -> list[Pushable]:
    """The run's doors and handles, doors first (a mounted handle needs its door's angle).

    objects: The run's objects (analysis.Run.objects: name, role, spec, starts_m, ends_m, radii_m).
    Returns: The pushables; parent indices point into this list.
    Raises: ValueError if a handle is mounted on something that is not a door of the run.
    """
    pushed = sorted((o for o in objects if o["role"] == "push"), key=lambda o: o["spec"]["type"] != "door")
    names = [o["name"] for o in pushed]
    out = []
    for o in pushed:
        # The floor height only places a door's hinge point along its vertical axis, which does not matter.
        hinge, axis, min_deg, max_deg = hinge_of(o["spec"], 0.0)
        parent = o["spec"].get("mounted_on")
        if parent is not None and (parent not in names or pushed[names.index(parent)]["spec"]["type"] != "door"):
            raise ValueError(f"Handle {o['name']!r} is mounted on {parent!r}, which is not a door of this scenario")
        ends = np.concatenate([o["starts_m"], o["ends_m"]])
        reach = float(np.max(np.linalg.norm(ends - hinge, axis=1) + np.concatenate([o["radii_m"], o["radii_m"]])))
        out.append(Pushable(o["name"], o["starts_m"], o["ends_m"], o["radii_m"], hinge, axis, min_deg, max_deg,
                            None if parent is None else names.index(parent), reach))  # fmt: skip
    return out


def _turn(points: np.ndarray, hinge: np.ndarray, axis: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Points turned about a hinge axis by each of several angles.

    points: (M, 3) points. hinge: (3,) point on the axis. axis: (3,) unit axis. angles_deg: (A,) angles.
    Returns: (A, M, 3) turned points.
    """
    R = Rotation.from_rotvec(np.deg2rad(angles_deg)[:, None] * axis[None]).as_matrix()  # (A, 3, 3)
    return np.einsum("aij,mj->ami", R, points - hinge) + hinge


def _posed(p: Pushable, parent: tuple | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """An object's capsules and hinge after its door turned (unchanged without a door).

    p: The object. parent: (door hinge (3,), door axis (3,), door angle deg) or None.
    Returns: (starts (M, 3), ends (M, 3), hinge point (3,), unit axis (3,)), all at the object's own 0 deg.
    """
    if parent is None:
        return p.starts_m, p.ends_m, p.hinge_m, p.axis
    hinge, axis, angle = parent
    turned = _turn(np.vstack([p.starts_m, p.ends_m, p.hinge_m[None], p.hinge_m[None] + p.axis[None]]),
                   hinge, axis, np.array([angle]))[0]  # fmt: skip
    m = len(p.radii_m)
    return turned[:m], turned[m : 2 * m], turned[2 * m], turned[2 * m + 1] - turned[2 * m]


def _push_step(model: CollisionModel, q_rad: np.ndarray, p: Pushable, posed: tuple, angle: float) -> tuple:
    """Turns one object out of the arm at one configuration, if the arm is inside it.

    A push can only turn the object the way that takes it out of the arm step by step: turning it
    into the arm (or through it to a free angle behind it) is not possible for a rigid object.

    model: The arm model. q_rad: (1, 6) joints, radians. p: The object.
    posed: _posed(p, ...) at this moment. angle: Its current angle, degrees.
    Returns: (new angle deg, blocked depth m (0 if it got free), pushing link index or None).
    """
    starts, ends, hinge, axis = posed
    movable = model.movable_links & (model.capsule_radius > 0)
    reach = link_obstacle_distances(model, q_rad, hinge[None], hinge[None], np.array([p.reach_m]))[0, movable]
    if reach.min() > 0:
        return angle, 0.0, None  # no link comes near the object at any angle
    # The angles it could turn to, each way from where it is, nearest first; each way ends at its limit.
    up = np.append(np.arange(angle + ANGLE_STEP_DEG, p.max_deg, ANGLE_STEP_DEG), p.max_deg) if angle < p.max_deg else np.zeros(0)
    down = np.append(np.arange(angle - ANGLE_STEP_DEG, p.min_deg, -ANGLE_STEP_DEG), p.min_deg) if angle > p.min_deg else np.zeros(0)
    ways = [up, down]
    candidates = np.concatenate([[angle], *ways])
    s, e = _turn(starts, hinge, axis, candidates), _turn(ends, hinge, axis, candidates)
    d = link_obstacle_distances(model, q_rad, s.reshape(-1, 3), e.reshape(-1, 3), np.tile(p.radii_m, len(candidates)))
    per_link = d[0].reshape(len(model.link_names), len(candidates), len(p.radii_m)).min(axis=2)  # (L, A)
    per_link[~movable] = np.inf
    clearance = per_link.min(axis=0)  # (A,)
    if clearance[0] >= -TOUCH_TOLERANCE_M:
        return angle, 0.0, None
    pusher = int(np.argmin(per_link[:, 0]))
    best = (angle, float(-clearance[0]))  # (angle, depth) if it cannot move: blocked where it is
    first = 1
    for way in ways:
        c = clearance[first : first + len(way)]
        first += len(way)
        worse = np.flatnonzero(c < clearance[0] - TOUCH_TOLERANCE_M)
        reachable = c[: worse[0]] if len(worse) else c  # turning further would push it into the arm
        free = np.flatnonzero(reachable >= -TOUCH_TOLERANCE_M)
        if len(free):
            option = (float(way[free[0]]), 0.0)
        elif len(reachable) == len(way) and len(way):
            option = (float(way[-1]), float(-reachable[-1]))  # out of the way only up to its limit
        else:
            continue
        if option[1] < best[1] or (option[1] == best[1] == 0.0 and abs(option[0] - angle) < abs(best[0] - angle)):
            best = option
    return best[0], best[1], pusher


def _fine_path(start_deg: np.ndarray, targets_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The arm path sampled FINE_PER_SAMPLE times finer than analysis.dense_path, move by move.

    start_deg: (6,) joints before the first move. targets_deg: (N, 6) the run's targets.
    Returns: (fine samples (F, 6) degrees, index (S,) of the fine sample at each dense_path sample).
    """
    fine, at_dense, previous = [np.asarray(start_deg, dtype=np.float64)[None]], [0], np.asarray(start_deg, dtype=np.float64)
    count = 1
    for target in targets_deg:
        n_dense = max(1, int(np.ceil(np.max(np.abs(target - previous)) / SAMPLE_STEP_DEG)))
        n_fine = n_dense * FINE_PER_SAMPLE
        fine.append(previous + np.linspace(0.0, 1.0, n_fine + 1)[1:, None] * (target - previous))
        at_dense.extend(count + FINE_PER_SAMPLE * np.arange(1, n_dense + 1) - 1)
        count += n_fine
        previous = target
    return np.concatenate(fine), np.asarray(at_dense)


def simulate_pushes(model: CollisionModel, objects: list[dict], start_deg: np.ndarray, targets_deg: np.ndarray) -> dict | None:
    """Turns the run's doors and handles along its arm path.

    model: The arm model the run planned with.
    objects: The run's objects (see find_pushables). start_deg: (6,) start joints. targets_deg: (N, 6) targets.
    Returns: None without doors or handles; else {"names": (P,), "angles_deg": (S, P), "blocked_mm": (S, P),
        "pusher": (S, P) pushing link name or None}, at every analysis.dense_path sample (S of them).
    """
    pushables = find_pushables(objects)
    if not pushables:
        return None
    fine, at_dense = _fine_path(start_deg, np.asarray(targets_deg, dtype=np.float64).reshape(-1, 6))
    angles = np.zeros(len(pushables))
    out_angles = np.zeros((len(fine), len(pushables)))
    out_blocked = np.zeros((len(fine), len(pushables)))
    out_pusher = np.full((len(fine), len(pushables)), -1)
    for k, q in enumerate(np.deg2rad(fine)):
        for i, p in enumerate(pushables):
            parent = None if p.parent is None else (pushables[p.parent].hinge_m, pushables[p.parent].axis, angles[p.parent])
            angles[i], out_blocked[k, i], pusher = _push_step(model, q[None], p, _posed(p, parent), angles[i])
            out_pusher[k, i] = -1 if pusher is None else pusher
        out_angles[k] = angles
    # The worst blocking and the last pusher within each playback sample, so short contacts are not lost.
    bounds = np.concatenate([[0], at_dense[:-1] + 1])
    blocked = np.stack([out_blocked[a : b + 1].max(axis=0) for a, b in zip(bounds, at_dense)])
    pushers = [[next((model.link_names[x] for x in out_pusher[a : b + 1, i][::-1] if x >= 0), None)
                for i in range(len(pushables))] for a, b in zip(bounds, at_dense)]  # fmt: skip
    return {"names": [p.name for p in pushables], "angles_deg": out_angles[at_dense].round(2),
            "blocked_mm": (blocked * 1000.0).round(1), "pusher": pushers}  # fmt: skip


def push_events(pushes: dict, owner: np.ndarray, command: np.ndarray) -> list[dict]:
    """Every stretch of the path during which an object was pushed or blocked, for the report.

    pushes: simulate_pushes' result. owner: (S,) target index per dense sample (-1 = start).
    command: (N,) command index per target.
    Returns: Per stretch {"object", "first_target", "last_target", "command", "from_deg", "to_deg",
        "links": pushing links, "blocked_mm": deepest blocked overlap (0 if never blocked)}.
    """
    events = []
    angles, blocked = pushes["angles_deg"], pushes["blocked_mm"]
    for i, name in enumerate(pushes["names"]):
        active = np.array([row[i] is not None for row in pushes["pusher"]]) | (blocked[:, i] > 0)
        k = 0
        while k < len(active):
            if not active[k]:
                k += 1
                continue
            end = k
            while end + 1 < len(active) and active[end + 1]:
                end += 1
            first, last = int(owner[k]), int(owner[end])
            events.append({
                "object": name, "first_target": first, "last_target": last,
                "command": int(command[first]) if first >= 0 else -1,
                "from_deg": float(angles[k - 1, i] if k else 0.0), "to_deg": float(angles[end, i]),
                "links": sorted({row[i] for row in pushes["pusher"][k : end + 1] if row[i] is not None}),
                "blocked_mm": float(blocked[k : end + 1, i].max()),
            })  # fmt: skip
            k = end + 1
    return events
