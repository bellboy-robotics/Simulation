"""Rotatable 3D page: the office-2 mesh with the kitchen capsule sets on it.

Fits each capsule set (see map_capsules), saves it as room_capsules_<set>.json and writes one
self-contained HTML page (mesh embedded) to open in a browser. On the page: pick a capsule set, cut
the mesh at a height to look inside, and pick a base pose to see the capsules the planner would get
(nearest 32 or 64 in reach) and the target voxels they leave uncovered.

    python -m Simulation.world_sim.map_capsules_view [--out output/world_sim/maps/office-2-20261006]
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from Simulation.world_sim import map_capsules as mc
from Simulation.world_sim.office_map import DEFAULT_MAP, REACH_M, base_positions, load_office_map

# Capsule sets shown: (name, unseen space open up to this height (see kitchen_voxels), allowed bulge in m).
SETS = (
    ("A-25", 1.2, 0.025),
    ("A-50", 1.2, 0.05),
    ("B-25", -np.inf, 0.025),
    ("B-50", -np.inf, 0.05),
)
_TEMPLATE = Path(__file__).with_name("map_capsules_view.html")


def build_sets(office, bases: np.ndarray, out: Path) -> list[dict]:
    """Fits every capsule set, saves each as JSON and returns the page data.

    office: The map. bases: (B, 3) usable base positions (x, y, unseen share).
    out: Folder for room_capsules_<set>.json.
    Returns: One dict per set: name, capsules (start, end, radius in m), target voxels (mm), summary.
    """
    sets = []
    for name, unseen_top, bulge in SETS:
        kv = mc.kitchen_voxels(office, bases[:, :2], unseen_top)
        caps, angled = mc.fit_capsules(kv, bulge, np.random.default_rng(0), mc.kitchen_axes(office))
        rows = mc.location_table(caps, kv, bases)
        summary = {
            "capsules": len(caps.radii_m),
            "targets": len(kv.targets_m),
            "full32": float(np.mean([r["coverage_32"] >= 1 for r in rows])),
            "full64": float(np.mean([r["coverage_64"] >= 1 for r in rows])),
            "bulge_max_mm": float(mc.bulge_stats(caps, kv).max() * 1000),
            "angled": int(angled.sum()),
        }
        (out / f"room_capsules_{name}.json").write_text(json.dumps(
            {"set": name, "bulge_m": bulge, "unseen_open_top_m": unseen_top if np.isfinite(unseen_top) else None,
             "summary": summary, "capsules": caps.to_objects(), "locations": rows}, indent=1))  # fmt: skip
        sets.append({
            "name": name, "summary": summary,
            "caps": np.column_stack((caps.starts_m, caps.ends_m, caps.radii_m)).round(4).tolist(),
            "targets": (kv.targets_m * 1000).round().astype(int).ravel().tolist(),
        })  # fmt: skip
        print(f"{name}: {summary}")
    return sets


def main() -> None:
    """Fits the capsule sets and writes capsules_3d.html next to the map."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--out", type=Path, default=DEFAULT_MAP.parent)
    args = parser.parse_args()
    office = load_office_map(args.map)
    bases = base_positions(office)
    data = {
        "sets": build_sets(office, bases, args.out),
        "bases": bases[:, :2].round(3).tolist(),
        "reach_m": REACH_M,
        "arm_offset_m": mc.ARM_OFFSET_M,
        "arm_z_m": mc.ARM_BASE_Z_M,
        "checksum": office.checksum[:12],
    }
    glb = base64.b64encode((args.map.parent / "mesh.glb").read_bytes()).decode()
    html = _TEMPLATE.read_text().replace("__DATA__", json.dumps(data)).replace("__GLB__", glb)
    page = args.out / "capsules_3d.html"
    page.write_text(html)
    print(page)


if __name__ == "__main__":
    main()
