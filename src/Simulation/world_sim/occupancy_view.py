"""Shows a robot's live occupancy map (nvblox_mapper's occupancy.bin.gz) as an interactive 3D page in the browser.

    python -m Simulation.world_sim.occupancy_view --host bellboy@billie-07.bellboy
    python -m Simulation.world_sim.occupancy_view --file output/world_sim/occupancy/billie-07/occupancy.bin.gz

Fetches the snapshot (read only) and the mapper's camera trajectory from the robot, then draws the occupied cells
colored by height, with the camera path. The floor height is printed, since the map should put it near z = 0.
"""

import argparse
import gzip
import json
import os
import struct
import subprocess
import webbrowser

import numpy as np
import plotly.graph_objects as go

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# Where the mapper writes its live files on the robot host (mounted in the billie container as $BILLIE_DIR/mapper/live).
ROBOT_LIVE_DIR = "/home/bellboy/billie/mapper/live"
# Snapshot magic of the billie_occupancy_v1 format (nvblox_mapper occupancy_stream.MAGIC).
_MAGIC = b"BOCCUP1\0"
# One block record: int32 block index (x, y, z), then 8^3 int8 values, index x*64 + y*8 + z (>0 occupied, <0 free).
_RECORD = np.dtype([("block", "<i4", (3,)), ("v", "i1", (512,))])
# Cells drawn at most; above it the cells are merged into coarser boxes (a box is shown if any of its cells is
# occupied), so the browser stays responsive.
_MAX_POINTS = 600_000


def read_blocks(path: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """Reads an occupancy snapshot's blocks as the mapper sends them.

    path: The occupancy.bin.gz file.
    Returns: (its JSON header, (N, 3) int32 block indices, (N, 512) int8 cells: >0 occupied, <0 free, 0 unknown).
    """
    with open(path, "rb") as f:
        raw = gzip.decompress(f.read())
    if raw[:8] != _MAGIC:
        raise ValueError(f"{path} is not a billie_occupancy_v1 snapshot")
    size = struct.unpack("<I", raw[8:12])[0]
    records = np.frombuffer(raw[12 + size :], dtype=_RECORD)
    return json.loads(raw[12 : 12 + size]), records["block"].copy(), records["v"].copy()


def read_snapshot(path: str) -> tuple[dict, np.ndarray, float]:
    """Reads an occupancy snapshot's occupied cells.

    path: The occupancy.bin.gz file.
    Returns: (its JSON header, (N, 3) centers of the occupied cells in meters (map frame), the cell size in meters).
    """
    header, indices, values = read_blocks(path)
    voxel = float(header["voxel_size_m"])
    block, cell = np.nonzero(values > 0)
    local = np.stack(np.unravel_index(cell, (8, 8, 8)), axis=1)
    return header, (indices[block] * 8 + local + 0.5) * voxel, voxel


def fetch(host: str, folder: str) -> tuple[str, str | None]:
    """Copies the robot's live snapshot, camera trajectory and robot state (robot.json, when the mapper writes it).

    host: ssh destination, e.g. bellboy@billie-07.bellboy. folder: Local folder to copy them into.
    Returns: (the snapshot's local path, the trajectory's local path or None if the robot has none).
    """
    os.makedirs(folder, exist_ok=True)
    snapshot = os.path.join(folder, "occupancy.bin.gz")
    subprocess.run(["scp", "-q", f"{host}:{ROBOT_LIVE_DIR}/occupancy.bin.gz", snapshot], check=True)
    # trajectory.json and robot.json are root-only on the host, so they are read through the robot's running container.
    containers = subprocess.run(["ssh", host, "docker", "ps", "--format", "{{.Names}}"], capture_output=True,
                                text=True, check=True).stdout.split()  # fmt: skip
    container = next((c for c in ("billie", "billie-dev") if c in containers), None)
    copied = {}
    for name in ("trajectory.json", "robot.json"):
        local = os.path.join(folder, name)
        if os.path.exists(local):
            os.remove(local)  # never keep an older robot's file next to a new snapshot
        if container is None:
            continue
        result = subprocess.run(["ssh", host, "docker", "exec", container, "cat", f"{ROBOT_LIVE_DIR}/{name}"],
                                capture_output=True, text=True, check=False)  # fmt: skip
        if result.returncode == 0:
            with open(local, "w") as f:
                f.write(result.stdout)
            copied[name] = local
    return snapshot, copied.get("trajectory.json")


def floor_height(centers: np.ndarray, voxel: float) -> float:
    """The floor's height: the densest occupied layer between -50 and +20cm.

    centers: (N, 3) occupied cell centers. voxel: The cell size. Returns: Its z, meters (NaN if no cells there).
    """
    z = centers[:, 2]
    low = z[(z > -0.5) & (z < 0.2)]
    if len(low) == 0:
        return float("nan")
    layers, counts = np.unique(np.round(low / voxel), return_counts=True)
    return float(layers[np.argmax(counts)] * voxel)


def thin_out(centers: np.ndarray, voxel: float) -> tuple[np.ndarray, float]:
    """Merges cells into coarser boxes until at most _MAX_POINTS remain.

    centers: (N, 3) occupied cell centers. voxel: The cell size.
    Returns: (the box centers drawn, the box size in meters).
    """
    size = voxel
    while len(centers) > _MAX_POINTS:
        size *= 2
        centers = (np.unique(np.floor(centers / size), axis=0) + 0.5) * size
    return centers, size


def figure(centers: np.ndarray, size: float, floor_z: float, trajectory: np.ndarray | None, title: str) -> go.Figure:
    """The 3D page: occupied boxes colored by height above the floor, and the camera path.

    centers: (M, 3) box centers drawn. size: Their size, meters. floor_z: The floor's height, meters.
    trajectory: (K, 3) camera positions, or None. title: The page title.
    Returns: The plotly figure.
    """
    height = centers[:, 2] - floor_z
    fig = go.Figure(go.Scatter3d(
        x=centers[:, 0], y=centers[:, 1], z=centers[:, 2], mode="markers", name=f"occupied ({size * 100:g}cm)",
        marker={"size": 1.5, "color": height, "colorscale": "Viridis", "cmin": 0.0, "cmax": 2.0,
                "colorbar": {"title": "m above floor"}},
        hovertemplate="x %{x:.2f} y %{y:.2f} z %{z:.2f}<extra></extra>",
    ))  # fmt: skip
    if trajectory is not None and len(trajectory):
        fig.add_trace(go.Scatter3d(x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2], mode="lines+markers",
                                   name="camera", line={"color": "red", "width": 4}, marker={"size": 2, "color": "red"}))  # fmt: skip
    fig.update_layout(title=title, scene={"aspectmode": "data", "xaxis_title": "x (m)", "yaxis_title": "y (m)",
                                          "zaxis_title": "z (m)"}, margin={"l": 0, "r": 0, "t": 40, "b": 0})  # fmt: skip
    return fig


def main() -> None:
    """Command line: fetch (or read) a snapshot, print what it holds, write the 3D page and open it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--host", help="robot ssh destination, e.g. bellboy@billie-07.bellboy")
    source.add_argument("--file", help="a local occupancy.bin.gz")
    parser.add_argument("--zmax", type=float, default=2.5, help="hide cells higher than this above the floor (m)")
    parser.add_argument("--no-floor", action="store_true", help="hide the cells within 5cm of the floor")
    args = parser.parse_args()
    if args.host:
        folder = os.path.join(_REPO, "output", "world_sim", "occupancy", args.host.split("@")[-1].split(".")[0])
        path, trajectory_path = fetch(args.host, folder)
    else:
        path, trajectory_path = args.file, None
        folder = os.path.dirname(os.path.abspath(path))
    header, centers, voxel = read_snapshot(path)
    floor_z = floor_height(centers, voxel)
    above = centers[:, 2] - floor_z
    keep = (above <= args.zmax) & (above > (0.05 if args.no_floor else -1.0))
    shown, size = thin_out(centers[keep], voxel)
    trajectory = None
    if trajectory_path:
        with open(trajectory_path) as f:
            trajectory = np.array([[p["x"], p["y"], p["z"]] for p in json.load(f)])
    print(f"{path}: {header.get('frames_integrated')} frames, {len(centers)} occupied cells at {voxel * 100:g}cm "
          f"(drawn {len(shown)} at {size * 100:g}cm); floor at z = {floor_z * 100:.1f}cm; "
          f"x {centers[:, 0].min():.1f}..{centers[:, 0].max():.1f} y {centers[:, 1].min():.1f}..{centers[:, 1].max():.1f} m")  # fmt: skip
    page = os.path.join(folder, "occupancy.html")
    figure(shown, size, floor_z, trajectory, f"{os.path.basename(folder)} occupancy, {header.get('frames_integrated')} frames").write_html(page)  # fmt: skip
    print(f"Wrote {page}")
    webbrowser.open(f"file://{page}")


if __name__ == "__main__":
    main()
