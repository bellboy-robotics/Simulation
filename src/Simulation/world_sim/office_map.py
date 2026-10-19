"""The office-2 3D map (the mapper's ESDF) and the kitchen area the planner comparison runs in.

The map is the mapper's baked ESDF: a 5cm grid of distances to the nearest surface, NaN where the
camera never looked, in the AutoXing map frame with z = 0 on the floor. The kitchen is the free floor
connected to KITCHEN_SEED_XY. A base position is usable when Billie's base fits there and the space
the arm could move into, at working height, was observed by the map (see base_positions).
"""

import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

# The mapper's grid loader; the nvblox_mapper package is not installed here, so it is loaded by path.
_ESDF_GRID_PY = (
    Path(__file__).resolve().parents[3]
    / "billie-onboard/billie/nodes/nvblox_mapper/nvblox_mapper/esdf_grid.py"
)
# office-2 map loaded on billie-17 on 2026-10-07 (capture 2026-10-06 11:55), ESDF sha256 b5afaf38...
DEFAULT_MAP = Path("output/world_sim/maps/office-2-20261006/esdf_grid.esdfbin")
# The same map with every unknown voxel above FREE_ABOVE_M marked free (the user's second version).
FREE_ABOVE_MAP = Path("output/world_sim/maps/office-2-20261006-free12/esdf_grid.esdfbin")
# [m] Unknown space above this height counts as free in FREE_ABOVE_MAP (the top of WORK_BAND_M).
FREE_ABOVE_M = 1.2
# The ESDF branch's grid writer (billie_utils.esdf_grid there; not on arm_planner), loaded by path.
_ESDF_WRITER_PY = Path(__file__).resolve().parents[3] / "output/esdf_branch/billie/billie-utils/billie_utils/esdf_grid.py"

# [m] A point on the kitchen floor (map frame), between the island and the coffee corner.
KITCHEN_SEED_XY = (4.0, -1.8)
# [m] The kitchen is east of the corridor; this keeps the flood fill out of the rest of the office.
KITCHEN_MIN_X_M = 0.0
# [m] Voxels below this height are the floor itself, not obstacles.
FLOOR_TOP_M = 0.05
# [m] Obstacles up to this height count for the base footprint (Billie's column is ~1.6m tall).
BODY_TOP_M = 2.0
# [m] Billie's base needs this much free floor around its center (base radius ~0.35m plus slack).
BASE_CLEARANCE_M = 0.40
# [m] Horizontal radius around the base center the arm can reach (gripper at 1.01m + its 8cm radius).
REACH_M = 1.1
# [m] Working height band: tasks and the observed-space rule only look at 0.10-1.20m above the floor.
WORK_BAND_M = (0.10, 1.20)
# [m] Height the arm-reachable flood fill starts from (about the arm base and table height).
ARM_HEIGHT_M = 0.75
# Largest share of the arm-reachable space in the working band that may be unobserved (user: 1%).
MAX_UNSEEN_FRACTION = 0.01


@dataclass
class OfficeMap:
    """The 3D map as arrays, plus the helpers to go between voxel indices and map meters."""

    distances_m: np.ndarray  # (X, Y, Z) distance to the nearest surface, meters; NaN = never observed
    origin_m: np.ndarray  # (3,) map position of voxel (0, 0, 0), meters
    voxel_m: float  # voxel edge, meters
    checksum: str  # sha256 of the map file, recorded with every result

    @property
    def known(self) -> np.ndarray:
        """(X, Y, Z) bool: the voxel was observed."""
        return np.isfinite(self.distances_m)

    @property
    def solid(self) -> np.ndarray:
        """(X, Y, Z) bool: the voxel holds a surface (observed and within half a voxel of one)."""
        return self.known & (np.nan_to_num(self.distances_m, nan=np.inf) <= self.voxel_m / 2)

    def query(self, points_m: np.ndarray) -> np.ndarray:
        """Trilinear distance to the nearest surface at map points, like the mapper's own lookup.

        points_m: (N, 3) map points, meters (z above the floor).
        Returns: (N,) meters; NaN where a corner of the cell was never observed or outside the map.
        """
        coords = ((np.asarray(points_m) - self.origin_m) / self.voxel_m).T
        return ndimage.map_coordinates(self.distances_m, coords, order=1, mode="constant", cval=np.nan)

    def heights(self) -> np.ndarray:
        """(Z,) height above the floor of each voxel layer, meters."""
        return self.origin_m[2] + np.arange(self.distances_m.shape[2]) * self.voxel_m

    def index(self, xy_m: np.ndarray) -> np.ndarray:
        """Voxel (i, j) of map points.

        xy_m: (..., 2) map x, y in meters.
        Returns: (..., 2) int voxel indices.
        """
        return np.round((np.asarray(xy_m) - self.origin_m[:2]) / self.voxel_m).astype(int)

    def xy(self, ij: np.ndarray) -> np.ndarray:
        """Map x, y of voxel columns.

        ij: (..., 2) voxel indices.
        Returns: (..., 2) map x, y in meters.
        """
        return self.origin_m[:2] + np.asarray(ij) * self.voxel_m


def load_office_map(path: Path = DEFAULT_MAP) -> OfficeMap:
    """Reads the mapper's .esdfbin file.

    path: The esdf_grid.esdfbin file.
    Returns: The map, with NaN for unobserved voxels.
    """
    if _ESDF_GRID_PY.exists():
        spec = importlib.util.spec_from_file_location("nvblox_esdf_grid", _ESDF_GRID_PY)
        module = importlib.util.module_from_spec(spec)
        sys.modules["nvblox_esdf_grid"] = module  # dataclasses need the module registered
        spec.loader.exec_module(module)
        grid = module.load_esdf_grid(Path(path))
    else:  # a robot copy has no mapper sources; arm_plan_esdf's billie-utils has the same reader
        from billie_utils.esdf_grid import load_esdf_grid  # noqa: PLC0415

        grid = load_esdf_grid(Path(path))
    return OfficeMap(
        np.asarray(grid.distances_m, dtype=np.float32),
        np.asarray(grid.origin_m, dtype=np.float64),
        float(grid.voxel_size_m),
        str(grid.checksum),
    )


def free_above(office: OfficeMap, height_m: float = FREE_ABOVE_M) -> OfficeMap:
    """The map with every unknown voxel above a height made free.

    A filled voxel gets its distance to the nearest observed surface, as if the camera had seen empty
    space there; observed voxels and unknown ones below the height keep their values.
    office: The map. height_m: Height above the floor, meters.
    Returns: A new map (checksum "" until it is written, see write_office_map).
    """
    fill = ~office.known & (office.heights() > height_m)[None, None, :]
    to_surface = ndimage.distance_transform_edt(~office.solid) * office.voxel_m
    distances = np.where(fill, to_surface, office.distances_m).astype(np.float32)
    return OfficeMap(distances, office.origin_m.copy(), office.voxel_m, "")


def write_office_map(office: OfficeMap, path: Path, name: str) -> str:
    """Writes a map in the mapper's .esdfbin format (the format both planners read).

    office: The map. path: Output file (its folder is created). name: Grid name in the header.
    Returns: The file's sha256.
    """
    spec = importlib.util.spec_from_file_location("esdf_branch_esdf_grid", _ESDF_WRITER_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["esdf_branch_esdf_grid"] = module
    spec.loader.exec_module(module)
    raw = module.serialize_esdf_grid(name=name, frame_id="autoxing_map", origin_m=office.origin_m,
                                     voxel_size_m=office.voxel_m, distances_m=office.distances_m)  # fmt: skip
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def kitchen_floor(office: OfficeMap) -> np.ndarray:
    """The kitchen's free, observed floor: columns with no obstacle up to BODY_TOP_M, linked to the seed.

    office: The map.
    Returns: (X, Y) bool mask of kitchen floor columns.
    """
    z = office.heights()
    body = (z >= FLOOR_TOP_M) & (z <= BODY_TOP_M)
    obstacle = office.solid[:, :, body].any(axis=2)
    observed = office.known[:, :, body].mean(axis=2) > 0.5
    x = office.xy(np.indices(obstacle.shape).transpose(1, 2, 0))[..., 0]
    labels, _ = ndimage.label(observed & ~obstacle & (x > KITCHEN_MIN_X_M))
    seed = labels[tuple(office.index(KITCHEN_SEED_XY))]
    if seed == 0:
        raise ValueError(f"KITCHEN_SEED_XY {KITCHEN_SEED_XY} is not on free observed floor")
    return labels == seed


def unseen_fraction(office: OfficeMap, ij: tuple[int, int]) -> float:
    """Share of the space the arm could move into, around one base position, that the map never observed.

    Flood-fills the open voxels (not a surface) inside the reach cylinder from the arm's height, so
    the unobserved insides of counters and walls do not count; only open space the arm can get to.
    office: The map.
    ij: Voxel column of the base center.
    Returns: Unobserved share of the reachable open space in WORK_BAND_M (0 = all observed).
    """
    r = int(REACH_M / office.voxel_m)
    z = office.heights()
    z0, z1 = np.searchsorted(z, 0.0), np.searchsorted(z, 1.6)
    box = (slice(ij[0] - r, ij[0] + r + 1), slice(ij[1] - r, ij[1] + r + 1), slice(z0, z1))
    solid, known = office.solid[box], office.known[box]
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    cylinder = np.broadcast_to(((xx**2 + yy**2) <= r * r)[:, :, None], solid.shape)
    labels, _ = ndimage.label(~solid & cylinder)
    reach = labels == labels[r, r, np.searchsorted(z[z0:z1], ARM_HEIGHT_M)]
    band = (z[z0:z1] >= WORK_BAND_M[0]) & (z[z0:z1] < WORK_BAND_M[1])
    return float((reach & ~known)[:, :, band].sum() / max(reach[:, :, band].sum(), 1))


def base_positions(office: OfficeMap, stride: int = 4) -> np.ndarray:
    """Kitchen base positions where Billie fits and the arm's working space was observed.

    office: The map.
    stride: Check every stride-th voxel column (4 = every 20cm), to keep it quick.
    Returns: (N, 3) rows of map x, y (meters) and the unseen share (see unseen_fraction), only rows
        within MAX_UNSEEN_FRACTION.
    """
    z = office.heights()
    body = (z >= FLOOR_TOP_M) & (z <= BODY_TOP_M)
    clearance = ndimage.distance_transform_edt(~office.solid[:, :, body].any(axis=2)) * office.voxel_m
    fits = kitchen_floor(office) & (clearance >= BASE_CLEARANCE_M)
    rows = []
    for ij in np.argwhere(fits)[::stride]:
        share = unseen_fraction(office, tuple(ij))
        if share <= MAX_UNSEEN_FRACTION:
            rows.append([*office.xy(ij), share])
    return np.asarray(rows).reshape(-1, 3)
