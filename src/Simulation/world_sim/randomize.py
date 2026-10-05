"""Random test content for the editor: objects around Billie's arm and arm motions (joints / poses).

Objects are placed in the map, within the arm's reach in front of Billie, and never touch Billie at
the scenario's start joints (otherwise every command would fail at once). Motions are drawn only
from valid configurations: inside the joint limits, clear of the avoid objects and the floor, and not
running the arm into Billie's body. A pose motion is the TCP pose of such a configuration, so it is
reachable. Every draw takes a seed, so a scenario can be drawn again.
"""

import numpy as np
from billie_utils.messages.pyroki_world_poc import MAX_WORLD_CAPSULES, WORLD_COL_MARGIN_M, planar_transform
from billie_utils.world_collision_check_poc import CollisionModel, link_obstacle_distances, link_transforms
from scipy.spatial.transform import Rotation

from Simulation.world_sim.world import object_capsules, object_in_arm_frame, scenario_from_data

OBJECT_TYPES = ("table", "wall", "capsule", "sphere")
# [mm] Distance from the arm base to a random object: inside the xArm6's ~700mm reach, outside Billie's body.
_REACH_MM = (350.0, 750.0)
# [deg] Directions from the arm's +X a random object may lie in; behind the arm is Billie's backpack.
_SPREAD_DEG = 110.0
# [m] Floor clearance of a random configuration: the gripper must not scrape the floor.
_FLOOR_MARGIN_M = 0.03
# [m] Clearance between the arm and Billie's own body (base, column, first arm links).
_BODY_MARGIN_M = 0.01
# [deg] Range for joints that turn more than a full turn (J1, J4, J6): one turn, so moves stay readable.
_TURN_DEG = 180.0
# Links that may hit Billie's body: (link, the links it is checked against besides the static base links).
_BODY_CHECKS = {"link3": (), "link4": (), "link5": ("link1", "link2"), "link6": ("link1", "link2"),
                "link_gripper_and_camera": ("link1", "link2")}  # fmt: skip
# Draws tried before giving up: random objects one at a time, configurations in batches of _BATCH.
_ATTEMPTS = 60
_BATCH = 256


def random_objects(data: dict, model: CollisionModel, rng: np.random.Generator, count: int, kind: str | None) -> list[dict]:
    """New avoid objects in the map frame, around the arm, clear of Billie and within the capsule budget.

    data: The edited scenario (Billie's pose, start joints, existing objects).
    model: The arm model.
    rng: Random generator.
    count: Objects to add.
    kind: "table", "wall", "capsule" or "sphere"; None picks one at random each time.
    Returns: The new objects (fewer than count if no place was found).
    """
    floor_z_m = data.get("floor_z_mm", -440.5) / 1000.0
    T_map_from_arm = planar_transform(*np.asarray(data["billie"]["pose"][:2]) / 1000.0, np.deg2rad(data["billie"]["pose"][2])) \
        @ planar_transform(*np.asarray(data["billie"]["arm_to_base"][:2]) / 1000.0, np.deg2rad(data["billie"]["arm_to_base"][2]))  # fmt: skip
    yaw0 = np.rad2deg(np.arctan2(T_map_from_arm[1, 0], T_map_from_arm[0, 0]))
    start = np.deg2rad(np.asarray(data["start_joints_deg"], dtype=np.float64))[None]
    objects = list(data.get("objects", []))
    added = []
    for _ in range(count * _ATTEMPTS):
        if len(added) == count:
            break
        distance, angle = rng.uniform(*_REACH_MM), rng.uniform(-_SPREAD_DEG, _SPREAD_DEG)
        center = T_map_from_arm @ np.array([distance * np.cos(np.deg2rad(angle)) / 1000.0,
                                            distance * np.sin(np.deg2rad(angle)) / 1000.0, 0.0, 1.0])  # fmt: skip
        spec = _object_at(kind or str(rng.choice(OBJECT_TYPES)), center[:2] * 1000.0, yaw0 + angle, rng)
        spec["name"] = f"random {spec['type']} {sum(o['type'] == spec['type'] for o in objects) + 1}"
        arm_spec = object_in_arm_frame(spec, data["billie"], -floor_z_m)
        starts, ends, radii = object_capsules(arm_spec, floor_z_m)
        used = scenario_from_data({**data, "objects": objects}, "random")["world"].capsules("avoid")[2]
        if len(used) + len(radii) > MAX_WORLD_CAPSULES:
            continue
        if link_obstacle_distances(model, start, starts, ends, radii).min() < WORLD_COL_MARGIN_M:
            continue  # it would touch Billie at the start joints
        objects.append(spec)
        added.append(spec)
    return added


def _object_at(kind: str, xy_mm: np.ndarray, facing_deg: float, rng: np.random.Generator) -> dict:
    """One random object of a type at a map point, sized like things found around a robot.

    kind: Object type. xy_mm: (2,) map point, mm. facing_deg: Map direction from the arm to that point.
    rng: Random generator.
    Returns: The object description in the map frame (role "avoid"), values rounded to 1mm / 1deg.
    """
    r = lambda lo, hi: round(float(rng.uniform(lo, hi)))  # noqa: E731
    x, y = (round(float(v)) for v in xy_mm)
    spec = {"role": "avoid", "frame": "map", "type": kind}
    across = np.deg2rad(facing_deg + 90.0 + rng.uniform(-25.0, 25.0))  # boards face the arm, roughly
    if kind == "table":
        spec.update(center_xy_mm=[x, y], top_height_mm=r(450, 900), size_mm=[r(300, 700), r(200, 450), r(25, 50)],
                    yaw_deg=round(float(np.rad2deg(across))))  # fmt: skip
    elif kind == "wall":
        half = rng.uniform(150.0, 400.0)
        dx, dy = half * np.cos(across), half * np.sin(across)
        spec.update(start_xy_mm=[round(x - dx), round(y - dy)], end_xy_mm=[round(x + dx), round(y + dy)],
                    height_mm=r(700, 1300), thickness_mm=r(30, 80))  # fmt: skip
    elif kind == "capsule" and rng.random() < 0.7:  # an upright post
        spec.update(start_mm=[x, y, 0], end_mm=[x, y, r(600, 1300)], radius_mm=r(25, 60))
    elif kind == "capsule":  # a horizontal bar
        half, z, direction = rng.uniform(150.0, 300.0), r(500, 1000), rng.uniform(0.0, 2 * np.pi)
        dx, dy = half * np.cos(direction), half * np.sin(direction)
        spec.update(start_mm=[round(x - dx), round(y - dy), z], end_mm=[round(x + dx), round(y + dy), z], radius_mm=r(20, 45))
    else:
        spec.update(center_mm=[x, y, r(450, 1000)], radius_mm=r(40, 100))
    return spec


def random_motions(
    data: dict, model: CollisionModel, limits_deg: list[list[float]], rng: np.random.Generator, count: int, kind: str
) -> list[dict]:
    """Random `joints` or `pose` commands to valid configurations (see valid_configurations).

    data: The edited scenario (its avoid objects are avoided).
    model: The arm model.
    limits_deg: 6 [lower, upper] joint limits, degrees.
    rng: Random generator.
    count: Commands to make.
    kind: "joints" (a joints command) or "pose" (a pose command at the configuration's TCP).
    Returns: The commands (fewer than count if valid configurations are too rare).
    """
    limits = np.clip(np.asarray(limits_deg, dtype=np.float64), -_TURN_DEG, _TURN_DEG)
    world = scenario_from_data(data, "random")["world"]
    found = np.zeros((0, 6))
    for _ in range(_ATTEMPTS):
        if len(found) >= count:
            break
        q = rng.uniform(limits[:, 0], limits[:, 1], size=(_BATCH, 6))
        found = np.vstack([found, q[valid_configurations(model, q, world.capsules("avoid"), world.floor_z_m)]])
    found = found[:count].round(1)
    if kind == "joints":
        return [{"cmd": "joints", "joints": q.tolist()} for q in found]
    T = link_transforms(model, np.deg2rad(found))[:, model.link_names.index("link_tcp")]
    poses = np.concatenate([T[:, :3, 3] * 1000.0, Rotation.from_matrix(T[:, :3, :3]).as_rotvec()], axis=1)
    return [{"cmd": "pose", "pose": [*np.round(p[:3], 1).tolist(), *np.round(p[3:], 4).tolist()]} for p in poses]


def valid_configurations(
    model: CollisionModel, joints_deg: np.ndarray, avoid: tuple[np.ndarray, np.ndarray, np.ndarray], floor_z_m: float
) -> np.ndarray:
    """Which configurations keep the arm clear of the avoid objects, the floor and Billie's own body.

    model: The arm model.
    joints_deg: (S, 6) configurations, degrees.
    avoid: (starts_m, ends_m, radii_m) avoid capsules, xArm base frame.
    floor_z_m: Arm-frame z of the floor, meters.
    Returns: (S,) bool.
    """
    q = np.deg2rad(joints_deg)
    T = link_transforms(model, q) @ model.capsule_local
    half = (model.capsule_height / 2.0)[None, :, None] * T[..., :3, 2]
    lowest = np.minimum((T[..., :3, 3] - half)[..., 2], (T[..., :3, 3] + half)[..., 2]) - model.capsule_radius[None]
    movable = model.movable_links & (model.capsule_radius > 0)
    ok = (lowest[:, movable] > floor_z_m + _FLOOR_MARGIN_M).all(axis=1)
    if len(avoid[2]):
        ok &= (link_obstacle_distances(model, q, *avoid).min(axis=2)[:, movable] > WORLD_COL_MARGIN_M).all(axis=1)
    static = np.flatnonzero(~model.movable_links & (model.capsule_radius > 0))
    for s in np.flatnonzero(ok):  # Billie's body: the static base links and, for the wrist, the first arm links
        ok[s] = _clear_of_body(model, q[s], T[s], half[s], static)
    return ok


def _clear_of_body(model: CollisionModel, q: np.ndarray, T: np.ndarray, half: np.ndarray, static: np.ndarray) -> bool:
    """Whether one configuration keeps the _BODY_CHECKS links clear of Billie's body.

    model: The arm model. q: (6,) joints, radians. T: (L, 4, 4) capsule frames at q.
    half: (L, 3) half segments of the capsules at q. static: Indices of the static links with a capsule.
    Returns: True if every checked link stays _BODY_MARGIN_M away.
    """
    for link, extra in _BODY_CHECKS.items():
        if link not in model.link_names:
            continue
        body = np.concatenate([static, [model.link_names.index(n) for n in extra if n in model.link_names]]).astype(int)
        p = T[body, :3, 3]
        distance = link_obstacle_distances(model, q[None], p - half[body], p + half[body], model.capsule_radius[body])
        if distance[0, model.link_names.index(link)].min() < _BODY_MARGIN_M:
            return False
    return True
