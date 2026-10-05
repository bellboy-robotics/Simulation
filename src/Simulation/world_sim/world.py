"""Simulated world: objects around the arm, each turned into the planner's capsule obstacles.

Every object has a role:
- "avoid": no arm link (EEF included) may touch it. Sent to the planner, the arm-move guard,
  detours and reroutes, exactly like the robot's virtual obstacles.
- "eef_touch": the EEF may touch it, the rest of the arm may not (WORLD_COLLISION_DISCUSSION.md §3,
  Set B). Not implemented in the planner yet, so these objects are only drawn and measured.
- "push": a door or a handle, turning about a hinge when the arm pushes it (see pushables.py). Not
  sent to the planner (the arm is meant to touch it); the simulation turns it out of the arm's way.

Coordinates are in the xArm base frame (the planner's frame), in millimeters, like the arm's
`position_aa_rad` status. Heights "above the floor" use the arm base height above the floor.

An object with "frame": "map" is placed in the map instead (z = height above the floor) and is
converted to the arm frame with the scenario's "billie" pose, like the robot's map -> arm chain.
"""

import json
from dataclasses import dataclass, field

import numpy as np
from billie_utils.messages.pyroki_world_poc import (
    DEFAULT_ARM_BASE_HEIGHT_M,
    MAX_WORLD_CAPSULES,
    tabletop_capsules,
    wall_capsules,
)

ROLES = ("avoid", "eef_touch", "push")
# Object types that turn about a hinge; their role is always "push".
PUSHABLE_TYPES = ("door", "handle")
# Spacing of a door's capsules, in radii, when it sets none: the robot's wall builder's overlap.
_DOOR_SPACING_RADII = 1.2


@dataclass
class WorldObject:
    """One object of the world, already converted into capsules (meters, xArm base frame)."""

    name: str  # label used in logs, the viewer and the report
    role: str  # "avoid" or "eef_touch", see the module docstring
    spec: dict  # the object as written in the scenario, kept for the report
    starts_m: np.ndarray  # (M, 3) capsule segment starts
    ends_m: np.ndarray  # (M, 3) capsule segment ends
    radii_m: np.ndarray  # (M,) capsule radii


@dataclass
class World:
    """All objects around the arm, and the floor height they are placed on."""

    objects: list[WorldObject] = field(default_factory=list)
    # [m] Arm-frame z of the floor (minus the arm base height above it).
    floor_z_m: float = -DEFAULT_ARM_BASE_HEIGHT_M

    def add(self, spec: dict) -> WorldObject:
        """Converts an object description into capsules and adds it to the world.

        spec: Object description, see object_capsules for the types and their fields.
        Returns: The added object.
        """
        pushable = spec["type"] in PUSHABLE_TYPES
        role = spec.get("role", "push" if pushable else "avoid")
        if role not in ROLES:
            raise ValueError(f"Object {spec.get('name')!r}: role must be one of {ROLES}, got {role!r}")
        if pushable != (role == "push"):
            raise ValueError(f"Object {spec.get('name')!r}: doors and handles (only) have role 'push', got {role!r}")
        starts, ends, radii = object_capsules(spec, self.floor_z_m)
        obj = WorldObject(spec.get("name", spec["type"]), role, spec, starts, ends, radii)
        self.objects.append(obj)
        return obj

    def capsules(self, role: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All capsules of the objects with one role, stacked.

        role: "avoid" or "eef_touch".
        Returns: (starts_m (M, 3), ends_m (M, 3), radii_m (M,)); M may be 0.
        Raises: ValueError if the avoid set needs more capsules than the planner has slots.
        """
        chosen = [o for o in self.objects if o.role == role]
        if not chosen:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        starts = np.concatenate([o.starts_m for o in chosen])
        ends = np.concatenate([o.ends_m for o in chosen])
        radii = np.concatenate([o.radii_m for o in chosen])
        if role == "avoid" and len(radii) > MAX_WORLD_CAPSULES:
            raise ValueError(
                f"The avoid objects need {len(radii)} capsules; the planner has {MAX_WORLD_CAPSULES} slots. "
                "Use fewer or thicker objects."
            )
        return starts, ends, radii

    @classmethod
    def from_specs(cls, specs: list[dict], floor_z_m: float = -DEFAULT_ARM_BASE_HEIGHT_M) -> "World":
        """Builds a world from a list of object descriptions.

        specs: Object descriptions, see object_capsules.
        floor_z_m: Arm-frame z of the floor, meters.
        Returns: The world.
        """
        world = cls(floor_z_m=floor_z_m)
        for spec in specs:
            world.add(spec)
        return world


def object_capsules(spec: dict, floor_z_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turns one object description into capsules, with the same builders the robot's commands use.

    spec: Object description; all lengths in mm, xArm base frame. By "type":
        - "table": a flat board. center_xy_mm [x, y], top_height_mm (above the floor),
          size_mm [length, width, thickness], yaw_deg (direction of the length, from +X).
        - "wall": an upright board standing on the floor. start_xy_mm, end_xy_mm, height_mm
          (above the floor), thickness_mm.
          Tables and walls may set spacing_mm: the distance between neighbouring capsule centers
          (radius = thickness / 2). Without it the robot's builders choose it.
        - "door": a board turning about a vertical hinge at one edge. hinge_xy_mm, yaw_deg (direction
          from the hinge to the free edge at 0 deg), width_mm, height_mm, bottom_mm (gap above the
          floor, default 0), thickness_mm, optional spacing_mm; plus the hinge fields below.
        - "handle": a short tube turning about a hinge through its start. start_mm (on the hinge),
          end_mm (free end at 0 deg), radius_mm, axis (hinge axis direction [x, y, z]), optional
          mounted_on (name of a door it turns with); plus the hinge fields below.
          Hinge fields: direction "ccw" or "cw" (positive turn, looking down the hinge axis from its
          tip; a door's axis points up), min_deg and max_deg (limits from the initial pose, 0 deg).
        Returned capsules are at 0 deg.
        - "capsule": start_mm [x, y, z], end_mm [x, y, z], radius_mm.
        - "sphere": center_mm [x, y, z], radius_mm.
    floor_z_m: Arm-frame z of the floor, meters.
    Returns: (starts_m (M, 3), ends_m (M, 3), radii_m (M,)) in the xArm base frame.
    """
    kind = spec["type"]
    if kind == "table":
        length, width, thickness = spec["size_mm"]
        center_z_mm = floor_z_m * 1000.0 + spec["top_height_mm"] - thickness / 2.0
        center_mm = (*spec["center_xy_mm"], center_z_mm)
        if spec.get("spacing_mm") is None:
            return tabletop_capsules(center_mm, (length, width, thickness), spec.get("yaw_deg", 0.0))
        # Capsules along the length, side by side across the width, the outer ones inside the edges.
        yaw, radius = np.deg2rad(spec.get("yaw_deg", 0.0)), thickness / 2.0
        length_axis, width_axis = np.array([np.cos(yaw), np.sin(yaw), 0.0]), np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        across = max(width / 2.0 - radius, 0.0) * width_axis
        half = max(length / 2.0 - radius, 0.0) * length_axis
        return capsule_row(np.asarray(center_mm) - across, np.asarray(center_mm) + across, half, radius, spec["spacing_mm"])
    if kind == "wall" and spec.get("spacing_mm") is not None:
        # Upright capsules along the base line, the domes ending at the floor and the wall's top.
        radius, height = spec["thickness_mm"] / 2.0, spec["height_mm"]
        floor_mm = floor_z_m * 1000.0
        half_mm = max(height - radius, 0.0) / 2.0  # the bottom dome may go below the floor, like wall_capsules
        middle = floor_mm + half_mm
        first, last = (np.array([*spec[k], middle]) for k in ("start_xy_mm", "end_xy_mm"))
        return capsule_row(first, last, np.array([0.0, 0.0, half_mm]), radius, spec["spacing_mm"])
    if kind == "wall":
        return wall_capsules(
            np.asarray(spec["start_xy_mm"]) / 1000.0,
            np.asarray(spec["end_xy_mm"]) / 1000.0,
            bottom_z_m=floor_z_m,
            top_z_m=floor_z_m + spec["height_mm"] / 1000.0,
            thickness_m=spec["thickness_mm"] / 1000.0,
        )
    if kind == "capsule":
        start, end = np.asarray(spec["start_mm"]) / 1000.0, np.asarray(spec["end_mm"]) / 1000.0
        return start[None], end[None], np.array([spec["radius_mm"] / 1000.0])
    if kind == "sphere":
        center = np.asarray(spec["center_mm"]) / 1000.0
        return center[None], center[None], np.array([spec["radius_mm"] / 1000.0])
    if kind == "door":
        radius = spec["thickness_mm"] / 2.0
        yaw = np.deg2rad(spec.get("yaw_deg", 0.0))
        along = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        bottom = floor_z_m * 1000.0 + spec.get("bottom_mm", 0.0)
        middle, half = bottom + spec["height_mm"] / 2.0, max(spec["height_mm"] / 2.0 - radius, 0.0)
        hinge = np.array([*spec["hinge_xy_mm"], middle])
        first, last = hinge + radius * along, hinge + (spec["width_mm"] - radius) * along  # edges stay at the board's
        spacing = spec.get("spacing_mm") or _DOOR_SPACING_RADII * radius
        return capsule_row(first, last, np.array([0.0, 0.0, half]), radius, spacing)
    if kind == "handle":
        start, end = np.asarray(spec["start_mm"]) / 1000.0, np.asarray(spec["end_mm"]) / 1000.0
        return start[None], end[None], np.array([spec["radius_mm"] / 1000.0])
    raise ValueError(f"Unknown object type {kind!r}; use table, wall, capsule, sphere, door or handle")


def hinge_of(spec: dict, floor_z_m: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """The hinge a door or handle turns about, with positive angles in its configured direction.

    spec: A door or handle description (see object_capsules), in the frame its capsules are in.
    floor_z_m: z of the floor in that frame, meters (a door's hinge point sits on it).
    Returns: (point on the hinge axis (3,) m; unit axis (3,), flipped for "cw" so a positive angle
        turns the configured way (right-hand rule); min_deg; max_deg).
    """
    sign = {"ccw": 1.0, "cw": -1.0}[spec.get("direction", "ccw")]
    if spec["type"] == "door":
        point, axis = np.array([*np.asarray(spec["hinge_xy_mm"]) / 1000.0, floor_z_m]), np.array([0.0, 0.0, 1.0])
    else:
        point, axis = np.asarray(spec["start_mm"]) / 1000.0, np.asarray(spec["axis"], dtype=np.float64)
        if np.linalg.norm(axis) < 1e-9:
            raise ValueError(f"Handle {spec.get('name')!r}: its axis must not be zero")
        axis = axis / np.linalg.norm(axis)
    min_deg, max_deg = float(spec.get("min_deg", 0.0)), float(spec.get("max_deg", 90.0))
    if not min_deg <= 0.0 <= max_deg:
        raise ValueError(f"Object {spec.get('name')!r}: need min_deg <= 0 <= max_deg, got {min_deg}..{max_deg}")
    return point, sign * axis, min_deg, max_deg


def capsule_row(
    first_mm: np.ndarray, last_mm: np.ndarray, half_axis_mm: np.ndarray, radius_mm: float, spacing_mm: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A row of equal, parallel capsules whose centers are evenly spread from one point to another.

    first_mm, last_mm: (3,) centers of the first and last capsule, mm.
    half_axis_mm: (3,) from a capsule's center to one end of its segment, mm.
    radius_mm: Capsule radius, mm.
    spacing_mm: Largest distance between neighbouring centers, mm; the row uses the fewest capsules that keep it.
    Returns: (starts_m (M, 3), ends_m (M, 3), radii_m (M,)).
    """
    if spacing_mm <= 0:
        raise ValueError(f"spacing_mm must be positive, got {spacing_mm}")
    n = int(np.ceil(np.linalg.norm(last_mm - first_mm) / spacing_mm)) + 1
    centers = first_mm[None] + np.linspace(0.0, 1.0, n)[:, None] * (last_mm - first_mm)[None]
    return (centers - half_axis_mm) / 1000.0, (centers + half_axis_mm) / 1000.0, np.full(n, radius_mm / 1000.0)


def object_in_arm_frame(spec: dict, billie: dict | None, arm_base_height_m: float) -> dict:
    """Converts a map-frame object into the xArm base frame; arm-frame objects are returned unchanged.

    spec: Object description (see object_capsules) with "frame": "map" or "arm" (default "arm").
        In the map frame, points are [x, y mm, z mm above the floor] and yaw_deg is from the map +X.
    billie: The scenario's {"pose": [x_mm, y_mm, yaw_deg] of the robot in the map,
        "arm_to_base": [x_mm, y_mm, yaw_deg] (ARM_TO_BASE_CALIBRATION)}; needed for map objects.
    arm_base_height_m: Height of the arm base above the floor, meters.
    Returns: A copy of the object with "frame": "arm" and arm-frame coordinates.
    """
    frame = spec.get("frame", "arm")
    if frame == "arm":
        return spec
    if frame != "map":
        raise ValueError(f"Object {spec.get('name')!r}: frame must be 'arm' or 'map', got {frame!r}")
    if billie is None:
        raise ValueError(f"Object {spec.get('name')!r} is in the map frame; the scenario needs a 'billie' pose")
    # Imported here so arm-frame scenarios still run on a robot checkout older than the map helpers.
    from billie_utils.messages.pyroki_world_poc import map_to_arm_frame

    x_mm, y_mm, yaw_deg = billie["pose"]
    robot_pose = (x_mm / 1000.0, y_mm / 1000.0, np.deg2rad(yaw_deg))

    def to_arm(point_mm: list[float], yaw: float = 0.0) -> tuple[list[float], float]:
        """Converts one map point and direction into the arm frame.

        point_mm: [x, y] on the floor or [x, y, z above the floor], mm. yaw: Map direction, degrees.
        Returns: (the point in the arm frame with as many values as given, mm; the direction, degrees).
        """
        xyz_m = np.asarray([*point_mm, 0.0][:3], dtype=np.float64) / 1000.0
        point, yaw_arm = map_to_arm_frame(xyz_m, yaw, robot_pose, billie["arm_to_base"], arm_base_height_m)
        return point[: len(point_mm)].tolist(), yaw_arm

    out = dict(spec, frame="arm")
    kind = spec["type"]
    if kind == "table":
        out["center_xy_mm"], out["yaw_deg"] = to_arm(spec["center_xy_mm"], spec.get("yaw_deg", 0.0))
    elif kind == "wall":
        out["start_xy_mm"], out["end_xy_mm"] = to_arm(spec["start_xy_mm"])[0], to_arm(spec["end_xy_mm"])[0]
    elif kind == "capsule":
        out["start_mm"], out["end_mm"] = to_arm(spec["start_mm"])[0], to_arm(spec["end_mm"])[0]
    elif kind == "sphere":
        out["center_mm"] = to_arm(spec["center_mm"])[0]
    elif kind == "door":
        out["hinge_xy_mm"], out["yaw_deg"] = to_arm(spec["hinge_xy_mm"], spec.get("yaw_deg", 0.0))
    elif kind == "handle":
        out["start_mm"], out["end_mm"] = to_arm(spec["start_mm"])[0], to_arm(spec["end_mm"])[0]
        turn = np.deg2rad(to_arm([0.0, 0.0])[1])  # how much the arm frame is turned from the map
        ax, ay, az = spec["axis"]
        out["axis"] = [float(ax * np.cos(turn) - ay * np.sin(turn)), float(ax * np.sin(turn) + ay * np.cos(turn)), float(az)]
    else:
        raise ValueError(f"Unknown object type {kind!r}; use table, wall, capsule, sphere, door or handle")
    return out


def scenario_from_data(data: dict, name: str) -> dict:
    """Builds a scenario from its parsed JSON: the world in the arm frame, start joints and commands.

    data: Parsed scenario: "objects" (see object_capsules, arm or map frame), "start_joints_deg"
        (6 values), "commands" (see Simulation.world_sim.commands.run_command), optionally
        "floor_z_mm" and "billie" (see object_in_arm_frame).
    name: Fallback name when data has none.
    Returns: {"world": World, "start_joints_deg": (6,) array, "commands": list of dicts, "name": str}.
    """
    floor_z_m = data.get("floor_z_mm", -DEFAULT_ARM_BASE_HEIGHT_M * 1000.0) / 1000.0
    specs = [object_in_arm_frame(spec, data.get("billie"), -floor_z_m) for spec in data.get("objects", [])]
    return {
        "name": data.get("name", name),
        "world": World.from_specs(specs, floor_z_m),
        "start_joints_deg": np.asarray(data["start_joints_deg"], dtype=np.float64),
        "commands": data.get("commands", []),
    }


def load_scenario(path: str) -> dict:
    """Reads a scenario file: the world, the arm's start joints and the commands to run.

    path: JSON scenario file, see scenario_from_data.
    Returns: {"world": World, "start_joints_deg": (6,) array, "commands": list of dicts, "name": str}.
    """
    with open(path) as f:
        data = json.load(f)
    return scenario_from_data(data, path)
