"""Covers the kitchen's obstacles in the 3D map with capsules, for the capsule (arm_planner) side of the comparison.

Only the obstacle surface the arm can get to needs covering: a capsule may sit inside a counter, a
wall or the room behind it, where the arm never goes. So the fit works on three voxel sets:
- solid: the map's surfaces, with gaps narrower than the gripper (GAP_CLOSE_M) filled, since the
  planner's arm model cannot enter them anyway;
- open: the space connected to the kitchen that is not solid, observed or not (the capsule planner
  treats unseen space as free, so capsules must not fill it either);
- targets: solid voxels touching the open space, within reach of the kitchen base positions.
Each capsule keeps out of the open space except for at most `bulge_m`, and the greedy fit adds
capsules until every target voxel is inside one. Per task, the planner gets the capsules nearest
the arm base (select_capsules); coverage_in_reach says whether that set still covers everything.

    python -m Simulation.world_sim.map_capsules [--bulge-mm 25] [--out output/world_sim/maps/office-2-20261006]
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from Simulation.world_sim.office_map import (
    ARM_HEIGHT_M,
    DEFAULT_MAP,
    FLOOR_TOP_M,
    KITCHEN_SEED_XY,
    REACH_M,
    WORK_BAND_M,
    OfficeMap,
    base_positions,
    kitchen_floor,
    load_office_map,
)

# [m] Gaps narrower than twice this are filled: the gripper capsule (82.6mm radius) plus the 2cm
# planner margin cannot enter a gap under ~20cm. Rounded to whole 5cm voxels (2 voxels).
GAP_CLOSE_M = 0.10
# [m] Top of the space the arm can reach (arm base 0.44m + ~1.0m reach + gripper radius).
ARM_TOP_M = 1.6
# [m] Largest capsule radius: big enough to cover a wall from behind with few capsules, small
# enough that a capsule does not reach around corners into the kitchen.
MAX_RADIUS_M = 0.5
# [m] Longest capsule segment.
MAX_LENGTH_M = 3.0
# [m] ARM_TO_BASE_CALIBRATION [-180, 0, 0]: the arm base is 18cm behind the base center.
ARM_OFFSET_M = -0.18
# [m] Arm base height above the floor (DEFAULT_ARM_BASE_HEIGHT_M).
ARM_BASE_Z_M = 0.4405
# Seeds tried per greedy step; more finds slightly fewer capsules, slower.
_SEEDS_PER_STEP = 12
# Sphere options per seed (radius levels from the smallest to the biggest that holds it).
_CANDIDATES = 6
# Target voxels a kitchen-axis capsule must cover before angled directions are tried instead.
MIN_AXIS_GAIN = 15


@dataclass
class Capsules:
    """Capsules in the map frame (z = height above the floor), meters."""

    starts_m: np.ndarray  # (M, 3) one end of each capsule's segment
    ends_m: np.ndarray  # (M, 3) the other end
    radii_m: np.ndarray  # (M,) radius around the segment

    def to_objects(self) -> list[dict]:
        """The capsules as map-frame scenario objects (the set_obstacles / world_sim format, mm).

        Returns: One {"type": "capsule", "frame": "map", ...} dict per capsule.
        """
        return [
            {"name": f"room_{k:03d}", "type": "capsule", "role": "avoid", "frame": "map",
             "start_mm": (s * 1000).round(1).tolist(), "end_mm": (e * 1000).round(1).tolist(),
             "radius_mm": round(float(r) * 1000, 1)}
            for k, (s, e, r) in enumerate(zip(self.starts_m, self.ends_m, self.radii_m))
        ]  # fmt: skip


@dataclass
class KitchenVoxels:
    """The voxel sets of one crop of the map that the fit works on (see the module docstring)."""

    origin_m: np.ndarray  # (3,) map position of the crop's voxel (0, 0, 0)
    voxel_m: float  # voxel edge, meters
    open_space: np.ndarray  # (X, Y, Z) bool, space connected to the kitchen
    depth_m: np.ndarray  # (X, Y, Z) distance from each voxel to the nearest open voxel, meters
    targets_m: np.ndarray  # (N, 3) map positions of the target voxels


def segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Distance from points to one segment.

    points: (N, 3) points. start, end: (3,) segment ends.
    Returns: (N,) distances, same unit as the input.
    """
    axis = end - start
    t = np.clip((points - start) @ axis / max(float(axis @ axis), 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (start + t[:, None] * axis), axis=1)


def kitchen_voxels(office: OfficeMap, bases_xy: np.ndarray, unseen_open_top_m: float = WORK_BAND_M[1]) -> KitchenVoxels:
    """Builds the solid / open / target sets around the kitchen.

    office: The map.
    bases_xy: (B, 2) map x, y of the usable base positions; targets are kept within their reach.
    unseen_open_top_m: Unobserved space within reach counts as open up to this height; above it (and
        beyond reach) it is room for capsules. -inf: unobserved space is never open.
    Returns: The voxel sets, cropped to the kitchen plus reach and capsule room.
    """
    v = office.voxel_m
    floor = np.argwhere(kitchen_floor(office))
    pad = int(np.ceil((REACH_M + MAX_RADIUS_M + 0.3) / v))
    lo = np.maximum(floor.min(axis=0) - pad, 0)
    hi = np.minimum(floor.max(axis=0) + pad + 1, office.distances_m.shape[:2])
    crop = (slice(lo[0], hi[0]), slice(lo[1], hi[1]))
    origin = office.origin_m + np.array([lo[0], lo[1], 0]) * v
    z = office.heights()
    r = int(round(GAP_CLOSE_M / v))
    ball = np.linalg.norm(np.indices((2 * r + 1,) * 3) - r, axis=0) <= r
    solid = np.pad(office.solid[crop], r)  # padded so the closing does not eat the crop's edges
    solid = ndimage.binary_closing(solid, ball)[r:-r, r:-r, r:-r]
    columns = origin[:2] + np.indices(solid.shape[:2]).transpose(1, 2, 0) * v
    reach_xy = REACH_M + abs(ARM_OFFSET_M) + 0.1
    in_reach = cKDTree(bases_xy).query(columns.reshape(-1, 2))[0].reshape(solid.shape[:2]) <= reach_xy
    # Unobserved space is open only where the arm can get to it; behind walls it is room for capsules.
    unseen_open = in_reach[:, :, None] & (z <= unseen_open_top_m)[None, None, :]
    passable = ~solid & (z <= ARM_TOP_M)[None, None, :] & (office.known[crop] | unseen_open)
    labels, _ = ndimage.label(passable)
    seed = np.round((np.array([*KITCHEN_SEED_XY, ARM_HEIGHT_M]) - origin) / v).astype(int)
    open_space = labels == labels[tuple(seed)]
    touching = ndimage.binary_dilation(open_space, np.ones((3, 3, 3), bool))
    band = (z >= FLOOR_TOP_M) & (z <= ARM_TOP_M + 0.1)
    idx = np.argwhere(solid & touching & band[None, None, :])
    points = origin + idx * v
    near = cKDTree(bases_xy).query(points[:, :2])[0] <= reach_xy
    depth = ndimage.distance_transform_edt(~open_space) * v
    return KitchenVoxels(origin, v, open_space, depth.astype(np.float32), points[near])


def _depth_at(kv: KitchenVoxels, points: np.ndarray) -> np.ndarray:
    """Distance to the open space at points (nearest voxel); -1 outside the crop.

    kv: The voxel sets. points: (N, 3) map positions.
    Returns: (N,) meters.
    """
    idx = np.round((points - kv.origin_m) / kv.voxel_m).astype(int)
    inside = np.all((idx >= 0) & (idx < kv.depth_m.shape), axis=1)
    out = np.full(len(points), -1.0)
    out[inside] = kv.depth_m[tuple(idx[inside].T)]
    return out


def _candidates_for(kv: KitchenVoxels, seed: np.ndarray, bulge_m: float) -> list[tuple[np.ndarray, float]]:
    """Sphere centers and radii that hold a target voxel and stay out of the open space.

    Gives a spread of options, since the best capsule is not always the biggest sphere: a long tube
    inside a thin tabletop needs a small radius centered in the board.
    kv: The voxel sets. seed: (3,) map position of the target voxel. bulge_m: Allowed bulge into open space.
    Returns: Up to _CANDIDATES (center (3,), radius) pairs, meters.
    """
    v, n = kv.voxel_m, int(np.ceil((MAX_RADIUS_M + bulge_m) / kv.voxel_m))
    offsets = (np.indices((2 * n + 1,) * 3).reshape(3, -1).T - n) * v
    centers, dist = seed + offsets, np.linalg.norm(offsets, axis=1)
    depth = _depth_at(kv, centers)
    radii = np.minimum(depth + bulge_m, MAX_RADIUS_M)
    ok = np.flatnonzero((radii >= dist) & (depth > 0))
    out = []
    for r_target in np.linspace(radii[ok].min(), radii[ok].max(), _CANDIDATES):
        # Per radius level: the center that holds the seed with that radius, nearest the level.
        k = ok[np.argmin(np.abs(radii[ok] - r_target) + 1e-3 * dist[ok])]
        out.append((centers[k], float(radii[k])))
    return out


def kitchen_axes(office: OfficeMap) -> np.ndarray:
    """The kitchen's own directions: its long axis, the short axis across it, and vertical.

    office: The map.
    Returns: (3, 3) unit vectors (rows), from the kitchen floor's principal directions.
    """
    xy = office.xy(np.argwhere(kitchen_floor(office)))
    _, vectors = np.linalg.eigh(np.cov((xy - xy.mean(axis=0)).T))
    long = vectors[:, 1]
    return np.array([[long[0], long[1], 0.0], [-long[1], long[0], 0.0], [0.0, 0.0, 1.0]])


def _angled_directions() -> np.ndarray:
    """Fallback capsule directions for what the kitchen axes cover badly: every 15 deg and tilted.

    Returns: (D, 3) unit vectors (each axis once; a segment runs both ways).
    """
    out = [[np.cos(a), np.sin(a), 0.0] for a in np.radians(np.arange(0, 180, 15))]
    for a in np.radians(np.arange(0, 360, 30)):
        for e in np.radians([30.0, 60.0]):
            out.append([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])
    return np.asarray(out)


def _grow(kv: KitchenVoxels, center: np.ndarray, radius: float, axis: np.ndarray, bulge_m: float):
    """Stretches a sphere into a capsule along an axis while it stays out of the open space.

    kv: The voxel sets. center: (3,) sphere center. radius: Its radius. axis: (3,) unit direction.
    bulge_m: Allowed bulge into the open space.
    Returns: (start (3,), end (3,)) of the longest allowed segment through center.
    """
    steps = np.arange(kv.voxel_m / 2, MAX_LENGTH_M / 2, kv.voxel_m / 2)
    ends = []
    for sign in (-1.0, 1.0):
        ok = _depth_at(kv, center + sign * steps[:, None] * axis) + bulge_m >= radius
        reach = steps[np.argmin(ok)] - kv.voxel_m / 2 if not ok.all() else steps[-1]
        ends.append(center + sign * max(reach, 0.0) * axis)
    return ends[0], ends[1]


def _best_capsule(kv, tree, covered, seeds, directions, bulge_m):
    """The capsule, over all seeds, candidate spheres and directions, that covers the most uncovered targets.

    kv: The voxel sets. tree: cKDTree of the target voxels. covered: (N,) bool, already covered targets.
    seeds: Target indices to start from. directions: (D, 3) allowed axes. bulge_m: Allowed bulge.
    Returns: (number covered, (start, end, radius, covered target indices) or None).
    """
    best = (0, None)
    for s in seeds:
        for center, radius in _candidates_for(kv, kv.targets_m[s], bulge_m):
            for axis in directions:
                start, end = _grow(kv, center, radius, axis, bulge_m)
                half = np.linalg.norm(end - start) / 2
                near = np.asarray(tree.query_ball_point((start + end) / 2, radius + half), int)
                near = near[~covered[near]]
                hit = near[segment_distance(kv.targets_m[near], start, end) <= radius]
                if len(hit) > best[0]:
                    best = (len(hit), (start, end, radius, hit))
    return best


def fit_capsules(kv: KitchenVoxels, bulge_m: float, rng: np.random.Generator, axes: np.ndarray,
                 max_capsules: int = 400) -> tuple[Capsules, np.ndarray]:
    """Greedy cover of the target voxels, along the kitchen's axes where it can.

    Each step adds the capsule covering the most uncovered targets, using only the kitchen axes.
    When the best of those covers fewer than MIN_AXIS_GAIN voxels, angled directions are tried too.
    Capsules left fully redundant by later ones are removed at the end.
    kv: The voxel sets. bulge_m: How far a capsule may reach into the open space, meters.
    rng: Picks the seed voxels of each step. axes: (3, 3) kitchen directions (kitchen_axes).
    max_capsules: Stop after this many (the report then shows the uncovered share).
    Returns: (the capsules, (M,) bool: capsule is angled).
    """
    tree, covered = cKDTree(kv.targets_m), np.zeros(len(kv.targets_m), bool)
    angled_dirs, picked = _angled_directions(), []
    while not covered.all() and len(picked) < max_capsules:
        uncovered = np.flatnonzero(~covered)
        seeds = rng.choice(uncovered, min(_SEEDS_PER_STEP, len(uncovered)), replace=False)
        gain, best = _best_capsule(kv, tree, covered, seeds, axes, bulge_m)
        angled = False
        if gain < MIN_AXIS_GAIN:
            gain_a, best_a = _best_capsule(kv, tree, covered, seeds, angled_dirs, bulge_m)
            if gain_a >= 2 * max(gain, 1):  # an angled capsule only when it is clearly better
                best, angled = best_a, True
        start, end, radius, hit = best
        covered[hit] = True
        picked.append((start, end, radius, angled))
    s, e, r, a = (np.array(x) for x in zip(*picked))
    keep = _drop_redundant(Capsules(s, e, r), kv.targets_m)
    return Capsules(s[keep], e[keep], r[keep]), a[keep]


def _drop_redundant(capsules: Capsules, targets_m: np.ndarray) -> np.ndarray:
    """Which capsules to keep: drops, smallest first, any whose targets all lie in other capsules.

    capsules: The fitted capsules. targets_m: (N, 3) target voxels.
    Returns: (M,) bool keep mask.
    """
    inside = np.array([segment_distance(targets_m, s, e) <= r
                       for s, e, r in zip(capsules.starts_m, capsules.ends_m, capsules.radii_m)])  # fmt: skip
    counts, keep = inside.sum(axis=0), np.ones(len(inside), bool)
    for k in np.argsort(inside.sum(axis=1)):
        if np.all(counts[inside[k]] >= 2):
            keep[k] = False
            counts[inside[k]] -= 1
    return keep


def arm_base_position(base_xy: np.ndarray, heading_rad: float) -> np.ndarray:
    """Map position of the arm base for a Billie base pose.

    base_xy: (2,) base center, map meters. heading_rad: Base heading from the map +X.
    Returns: (3,) arm base position, meters (z above the floor).
    """
    forward = np.array([np.cos(heading_rad), np.sin(heading_rad)])
    return np.array([*(np.asarray(base_xy) + ARM_OFFSET_M * forward), ARM_BASE_Z_M])


def select_capsules(capsules: Capsules, arm_base_m: np.ndarray, limit: int) -> tuple[np.ndarray, int]:
    """The capsules the arm can reach from one base pose, nearest first, at most `limit` of them.

    capsules: The room capsules. arm_base_m: (3,) arm base, map meters. limit: Planner slots.
    Returns: (indices of the chosen capsules, how many were within reach in total).
    """
    gap = np.array([segment_distance(arm_base_m[None], s, e)[0] - r
                    for s, e, r in zip(capsules.starts_m, capsules.ends_m, capsules.radii_m)])  # fmt: skip
    in_reach = np.flatnonzero(gap <= REACH_M)
    return in_reach[np.argsort(gap[in_reach])][:limit], len(in_reach)


def coverage_in_reach(capsules: Capsules, chosen: np.ndarray, targets_m: np.ndarray, arm_base_m: np.ndarray) -> float:
    """Share of the target voxels within the arm's reach that the chosen capsules cover.

    capsules: The room capsules. chosen: Indices from select_capsules. targets_m: (N, 3) target voxels.
    arm_base_m: (3,) arm base, map meters.
    Returns: Covered share, 1.0 when nothing is in reach.
    """
    near = targets_m[np.linalg.norm(targets_m - arm_base_m, axis=1) <= REACH_M]
    if not len(near):
        return 1.0
    inside = np.zeros(len(near), bool)
    for k in chosen:
        inside |= segment_distance(near, capsules.starts_m[k], capsules.ends_m[k]) <= capsules.radii_m[k]
    return float(inside.mean())


def bulge_stats(capsules: Capsules, kv: KitchenVoxels) -> np.ndarray:
    """How far each capsule reaches into the open space (its radius minus the depth along its axis).

    capsules: The room capsules. kv: The voxel sets.
    Returns: (M,) deepest open-voxel center inside each capsule, measured from its surface, meters (0 = none).
    """
    tree = cKDTree(kv.origin_m + np.argwhere(kv.open_space) * kv.voxel_m)
    out = []
    for s, e, r in zip(capsules.starts_m, capsules.ends_m, capsules.radii_m):
        near = np.asarray(tree.query_ball_point((s + e) / 2, r + np.linalg.norm(e - s) / 2), int)
        depth = r - segment_distance(tree.data[near], s, e) if len(near) else np.zeros(1)
        out.append(max(float(depth.max()), 0.0))
    return np.asarray(out)


def location_table(capsules: Capsules, kv: KitchenVoxels, bases: np.ndarray, limits=(32, 64)) -> list[dict]:
    """Per base position and heading: capsules in reach and the coverage with each slot limit.

    capsules: The room capsules. kv: The voxel sets. bases: (B, 3) base positions (x, y, unseen share).
    limits: Planner slot counts to check.
    Returns: One row per (position, heading every 45 deg).
    """
    rows = []
    for x, y, unseen in bases:
        for heading in np.radians(np.arange(0, 360, 45)):
            arm = arm_base_position(np.array([x, y]), heading)
            row = {"x": x, "y": y, "heading_deg": round(float(np.degrees(heading))), "unseen": unseen}
            for limit in limits:
                chosen, row["in_reach"] = select_capsules(capsules, arm, limit)
                row[f"coverage_{limit}"] = coverage_in_reach(capsules, chosen, kv.targets_m, arm)
            rows.append(row)
    return rows


def main() -> None:
    """Fits the kitchen capsules, prints the coverage summary and writes room_capsules.json."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--bulge-mm", type=float, default=25.0, help="how far a capsule may reach into open space")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_MAP.parent)
    args = parser.parse_args()
    started = time.time()
    office = load_office_map(args.map)
    bases = base_positions(office)
    kv = kitchen_voxels(office, bases[:, :2])
    capsules, _ = fit_capsules(kv, args.bulge_mm / 1000.0, np.random.default_rng(args.seed), kitchen_axes(office))
    bulge = bulge_stats(capsules, kv)
    rows = location_table(capsules, kv, bases)
    in_reach = np.array([r["in_reach"] for r in rows])
    print(f"map {office.checksum[:12]}  bases {len(bases)}  target voxels {len(kv.targets_m)}  fit {time.time() - started:.0f}s")
    print(f"capsules {len(capsules.radii_m)}  radius mm min/median/max {np.percentile(capsules.radii_m * 1000, [0, 50, 100]).round()}")
    print(f"bulge into open space mm: median {np.median(bulge) * 1000:.0f}  max {bulge.max() * 1000:.0f}")
    print(f"capsules in reach per (position, heading): min {in_reach.min()} median {np.median(in_reach):.0f} max {in_reach.max()}")
    for limit in (32, 64):
        cov = np.array([r[f"coverage_{limit}"] for r in rows])
        print(f"  {limit} slots: fully covered at {(cov >= 1.0).mean() * 100:.0f}% of poses, worst coverage {cov.min() * 100:.1f}%")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "room_capsules.json").write_text(json.dumps({
        "map_checksum": office.checksum, "bulge_mm": args.bulge_mm, "gap_close_m": GAP_CLOSE_M,
        "max_radius_m": MAX_RADIUS_M, "capsules": capsules.to_objects(), "locations": rows}, indent=1))  # fmt: skip
    print(args.out / "room_capsules.json")


if __name__ == "__main__":
    main()
