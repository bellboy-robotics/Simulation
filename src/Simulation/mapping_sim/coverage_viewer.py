"""A 3D page of what each scan case leaves unknown in front of Billie, in a robot's mapped room.

    PYTHONPATH=src:billie-onboard/billie/nodes/spatial_awareness:billie-onboard/billie/billie-utils \\
        python -m Simulation.mapping_sim.coverage_viewer --robot <folder> [--robot <folder> ...] --out <page.html>

Each case is a set of scan views (joints). A fresh map is simulated from them (scan_coverage: the mapper's rules,
Billie's own body in view), then spatial_awareness's own rules are applied (occupancy_field: missed rays, the
probably-free rim beside seen free space) and its zones (above the base, the base's shadow). Every cell around the arm
ends seen, probably free, or unknown with the reason. The lidar is left out: on a robot its low band adds to this.
"""

import argparse
import base64
import json
import os

import numpy as np
from spatial_awareness import occupancy_field as of

from Simulation.mapping_sim import crop_rules
from Simulation.mapping_sim import scan_coverage as sc
from Simulation.mapping_sim.snap_viewer import billie_frame, snap_shapes

# The page the data is put into (its `/*DATA*/null` placeholder).
_TEMPLATE = os.path.join(os.path.dirname(__file__), "coverage_viewer.html")
# The scan pose (brain arm.MAP_ARM_JOINTS), degrees: every case turns J1 (the camera's yaw) and J5 (its tilt) from it.
SCAN_POSE = (0.0, -36.88, -94.14, -179.10, -66.44, -1.32)
# A lower look pose found by search (2026-11-11): J2-J6 that keep the arm inside the column above Billie's base at
# almost every J1 and see 66-86% of the working wedge in J1's direction (the scan pose: 26%). Checked with the planner's
# self-collision model on billie-39; never run on a robot yet.
LOOK_POSE = (-9.0, -179.0, -18.0, 133.0, 88.0)
# Shown cells are every this many cells along each axis (2.5 cm -> 5 cm), within this far of the arm base.
THIN, SHOWN_M = 2, 1.45
# What a cell ends as, as stored (index).
KINDS = ("seen by the camera", "probably free: beside seen free space", "probably free: above the base, its shadow",
         "unknown: never in view", "unknown: behind Billie's own body", "unknown: hidden behind an object",
         "unknown: behind the gripper mask")  # fmt: skip


# The scan's views until 2026-11-11: six views all round (active_mapping.SCAN_VIEWS_DEG then), kept to compare with.
EARLIER_VIEWS = (
    (0.0, -36.88, -94.14, -179.10, -66.44, -1.32),
    (30.0, -36.88, -94.14, -179.10, -66.44, 88.68),
    (45.0, -36.88, -94.14, -89.10, -91.44, 88.68),
    (15.0, -36.88, -94.14, -119.10, -91.44, 88.68),
    (-15.0, -36.88, -94.14, -209.10, -91.44, -1.32),
    (-45.0, -36.88, -94.14, -239.10, -91.44, 88.68),
)


def front_views(yaw_deg: float, tilt_deg: float, raise_deg: float = 0.0) -> list[tuple[float, ...]]:
    """The front scan: right level, right down, centre down, left down, left level, centre level (J1 - is right, J5 -
    is down); without a tilt, the three level views.

    yaw_deg: J1's turn each side. tilt_deg: J5's turn down (0: level views only).
    raise_deg: J5's turn up for the level views. Returns: The views' joints.
    """
    up = -raise_deg
    steps = ((-yaw_deg, up), (-yaw_deg, tilt_deg), (0, tilt_deg), (yaw_deg, tilt_deg), (yaw_deg, up), (0, up))
    if not tilt_deg:
        steps = ((-yaw_deg, up), (yaw_deg, up), (0, up))
    return [(SCAN_POSE[0] + yaw, *SCAN_POSE[1:4], SCAN_POSE[4] - tilt, SCAN_POSE[5]) for yaw, tilt in steps]


def cases() -> list[tuple[str, str, list[tuple[float, ...]]]]:
    """The scan cases compared. Returns: (name, a short note, the views' joints) each."""
    return [
        ("Earlier scan", "6 views all round", list(EARLIER_VIEWS)),
        ("The scan now", "±30, down 28, up 10", [tuple(view) for view in sc.scan_views()]),
        ("Front ±30", "down 28, level", front_views(30, 28)),
        ("Front ±30", "down 28, up 20", front_views(30, 28, 20)),
        ("Front ±40", "down 28, up 10", front_views(40, 28, 10)),
        ("Front ±20", "down 28, up 10", front_views(20, 28, 10)),
        ("Level ±30", "no low views", front_views(30, 0)),
        ("Scan pose", "1 view", [SCAN_POSE]),
        ("Look pose ±30", "3 views", [(yaw, *LOOK_POSE) for yaw in (-30.0, 30.0, 0.0)]),
        ("Look ±30 + scan", "4 views", [*[(yaw, *LOOK_POSE) for yaw in (-30.0, 30.0, 0.0)], SCAN_POSE]),
    ]


def end_states(crop: dict, per_view: list[np.ndarray]) -> np.ndarray:
    """Each cell's end after a case: the camera's views, then spatial_awareness's rules and zones.

    crop: crop_rules.crop_states' (its places and zones). per_view: scan_coverage.view_cells' answer per view.
    Returns: (N,) uint8 index into KINDS.
    """
    final = sc.combine(per_view)
    states = np.zeros(len(final), np.int8)
    states[final == sc.CLASSES.index("free")] = of.FREE
    states[final == sc.CLASSES.index("occupied")] = of.OCCUPIED
    grid = states.reshape(crop["shape"])
    of.drop_unknown_specks(grid)
    of.mark_probably_free(grid, 0.025)
    zone = crop["zones"]["above the base"] | crop["zones"]["base's shadow"]
    reason = {sc.CLASSES.index("never in view"): 3, sc.CLASSES.index("behind Billie's own body"): 4,
              sc.CLASSES.index("behind the gripper mask"): 6}  # fmt: skip
    kind = np.full(len(final), 5, np.uint8)  # hidden behind an object, unless another reason
    for cls, index in reason.items():
        kind[final == cls] = index
    kind[states == of.PROBABLY_FREE] = 1
    kind[(states == of.UNKNOWN) & zone] = 2
    kind[(states == of.FREE) | (states == of.OCCUPIED)] = 0
    return kind


def past_column_m(body: sc.RobotBody, to_billie: np.ndarray, views: list[tuple[float, ...]]) -> float:
    """How far the arm (with its gripper) reaches past the column above Billie's base in a case's views.

    body: The robot's body. to_billie: (4, 4) map -> Billie's frame. views: The views' joints. Returns: Meters.
    """
    arm_links = tuple(name for name in body.points if name.startswith("link") and name != "link_base")
    worst = 0.0
    for view in views:
        points = body.at(view, arm_links) @ to_billie[:3, :3].T + to_billie[:3, 3]
        out_x = np.maximum(np.abs(points[:, 0]) - of.BASE_HALF_LENGTH_M - of.CLEAR_MARGIN_M, 0)
        out_y = np.maximum(np.abs(points[:, 1]) - of.BASE_HALF_SIDE_M - of.CLEAR_MARGIN_M, 0)
        worst = max(worst, float(np.hypot(out_x, out_y).max()))
    return worst


def room_data(folder: str) -> dict:
    """One robot room's page data: its cells, the real map's occupied cells, and every case's end states.

    folder: The robot folder (crop_rules.crop_states'). Returns: The JSON-ready dict.
    """
    sc.use_camera(folder)
    crop = crop_rules.crop_states(folder)
    robot, pts, loc = crop["robot"], crop["pts"], crop["loc"]
    world = sc.World(crop["grid"], crop["lo"], 0.025, robot)
    camera = sc.camera_model(os.path.join(folder, "arm.urdf"), robot)
    body = sc.RobotBody(os.path.join(folder, "arm.urdf"), robot)
    keep, to_billie = sc.gripper_keep(), np.linalg.inv(billie_frame(robot))
    arm = np.array([robot["arm_to_base"][0] / 1000, robot["arm_to_base"][1] / 1000, robot["arm_base_z_m"] - sc.FLOOR_M])
    thin = np.zeros(crop["shape"], bool)
    thin[::THIN, ::THIN, ::THIN] = True
    on_base = (np.abs(loc[:, 0]) <= of.BASE_HALF_LENGTH_M) & (np.abs(loc[:, 1]) <= of.BASE_HALF_SIDE_M)
    shown = thin.reshape(-1) & (np.linalg.norm(loc - arm, axis=1) <= SHOWN_M) & ~on_base & (loc[:, 2] > 0.05)
    rendered: dict[tuple, np.ndarray] = {}  # a view's cells, by its joints (cases share views)

    def view_cells(view: tuple[float, ...]) -> np.ndarray:
        """One view's scan_coverage.view_cells answer, rendered once. view: Its joints."""
        if view not in rendered:
            pose = camera(view)
            depth, hit = sc.render_at(world, body, pose, view)
            rendered[view] = sc.view_cells(pts, pose, depth, sc.pixel_rules(depth, hit, pose, True), keep)
        return rendered[view]

    out = []
    for name, note, views in cases():
        per_view = [view_cells(view) for view in views]
        kind = end_states(crop, per_view)
        unknown = kind >= 3
        # which views saw each cell (free or occupied): bit i for view i
        seen_by = sum((cells[shown] <= 1).astype(np.uint8) << i for i, cells in enumerate(per_view))
        out.append({
            "name": name, "note": note, "codes": base64.b64encode(kind[shown].tobytes()).decode(),
            "seen_by": base64.b64encode(seen_by.astype(np.uint8).tobytes()).decode(),
            "joints": [[round(j, 2) for j in view] for view in views],
            "views": snap_shapes([camera(view) for view in views], to_billie),
            "front_unknown": round(float(unknown[crop["front"]].mean()), 4),
            "front_camera_only": round(float((kind != 0)[crop["front"]].mean()), 4),
            "reach_unknown": round(float(unknown[crop["reach"]].mean()), 4),
            "past_column_cm": round(past_column_m(body, to_billie, views) * 100, 1),
        })  # fmt: skip
        print(f"{folder.split('/')[2]} {name} ({note}): front unknown {out[-1]['front_unknown']:.1%}, "
              f"reach {out[-1]['reach_unknown']:.1%}, camera only {out[-1]['front_camera_only']:.1%}")  # fmt: skip
    occupied = thin.reshape(-1) & (np.linalg.norm(loc - arm, axis=1) <= SHOWN_M + 0.6) & (crop["states"].reshape(-1) == 1)
    cm = lambda cells: base64.b64encode(np.rint(loc[cells] * 100).astype("<i2").tobytes()).decode()  # noqa: E731
    # bit 0: ahead of Billie's centre (whatever the distance); bit 1: within the arm's reach
    ahead = loc[:, 0] > crop_rules.FRONT_FROM_X_M
    flags = (ahead[shown].astype(np.uint8) | (crop["reach"][shown].astype(np.uint8) << 1)).tobytes()
    return {"name": folder.split("/")[2], "cells": cm(shown), "flags": base64.b64encode(flags).decode(),
            "occupied": cm(occupied), "cases": out}  # fmt: skip


def main() -> None:
    """Builds every room's data and writes the page."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--robot", required=True, action="append", help="a robot folder (repeat for more rooms)")
    parser.add_argument("--out", required=True, help="the HTML page to write")
    args = parser.parse_args()
    data = {"cell_cm": 2.5 * THIN, "kinds": list(KINDS), "rooms": [room_data(folder) for folder in args.robot],
            "base_half_m": [of.BASE_HALF_LENGTH_M, of.BASE_HALF_SIDE_M, 0.55], "shown_m": SHOWN_M,
            "front_from_x_m": crop_rules.FRONT_FROM_X_M, "reach_m": crop_rules.REACH_M}  # fmt: skip
    page = open(_TEMPLATE).read().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write(page)
    print(f"{args.out}: {len(page) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
