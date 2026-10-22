"""Rebuilds a robot's arm field offline from a capture, with spatial_awareness's own code, and checks it against the
field the robot wrote.

    python -m Simulation.world_sim.capture_field output/world_sim/captures/billie-40/20261021-1547 [--write field.bin.gz]

A capture (see its README.md) holds the mapper's live files (live/occupancy.bin.gz, the robot's own
live/arm_esdf.bin.gz and robot.json), 30 s of AutoXing's websocket (autoxing_topics.jsonl.gz: the 2D map, the pose,
the lidar scans) and the robot's settings (settings.env). The rebuild follows the robot's export
(arm_esdf_export._build): the recorded scans are replayed in time order through the lidar-to-map alignment
(PoseAligner, for the correction that moves AutoXing's walls), the standing walls (StandingWalls: what the lidar sees
through) and the lidar grid (LidarGrid); then build_field() runs on the occupancy, placed where the robot's own field
says the base and the arm base were.
"""

import argparse
import base64
import gzip
import json
import os
import struct
import sys
from dataclasses import dataclass

import numpy as np

# The lidar intensity AutoXing gives a noise return (spatial_awareness/main.py NOISE_INTENSITY): such points are dropped.
NOISE_INTENSITY = 0.0
# [m] The arm field's cell edge and [cells] its crop edge (spatial_awareness/arm_esdf_export.py VOXEL_M, CROP_CELLS).
VOXEL_M = 0.025
CROP_CELLS = 118
# The robot's settings the field code reads when imported or run (the rest of settings.env is for the viewers).
_SETTINGS_USED = ("BILLIE_SYSID", "ARM_TO_BASE_CALIBRATION", "AUTOXING_BASE_TO_ARM_BASE_Z_M")


def _use_settings(folder: str) -> None:
    """Sets the capture's robot settings in this process before the field code is imported (it reads them then).

    folder: The capture folder (its settings.env). An empty value (unset on the robot) is left unset here too.
    """
    with open(os.path.join(folder, "settings.env")) as f:
        pairs = dict(line.rstrip("\n").split("=", 1) for line in f if "=" in line)
    os.environ.setdefault("BILLIE_DIR", os.path.join(folder, "_billie_dir"))  # pose_alignment's map-file path
    for name in _SETTINGS_USED:
        if pairs.get(name):
            os.environ[name] = pairs[name]


def _field_code():
    """spatial_awareness's field modules, from the local billie-onboard checkout. Returns: (occupancy_field,
    pose_alignment, map_alignment, area_free, SensorReading)."""
    from Simulation.world_sim import robot  # noqa: PLC0415 - after _use_settings

    sys.path.insert(0, os.path.join(robot._local_billie_root(), "billie", "nodes", "spatial_awareness"))
    from spatial_awareness import area_free, map_alignment, occupancy_field, pose_alignment  # noqa: PLC0415
    from spatial_awareness.models import SensorReading  # noqa: PLC0415

    return occupancy_field, pose_alignment, map_alignment, area_free, SensorReading


@dataclass
class Capture:
    """What a capture holds for one field build."""

    blocks: np.ndarray  # (N, 3) int32 occupancy block indices
    values: np.ndarray  # (N, 512) int8 cells per block (>0 occupied, <0 free, 0 unknown)
    occupancy_header: dict  # the occupancy snapshot's JSON header
    robot_field: dict  # the robot's own arm field: {"header", "values" (dims) int16, "unknown" bool, "scale_m"}
    messages: list[tuple[float, dict]]  # AutoXing's websocket messages, (receive time s, message), in time order


def _read_besdfq1(path: str) -> dict:
    """An arm field file (gzip BESDFQ1). path: The file. Returns: {"header", "values" int16 (dims), "unknown"
    bool (dims), "scale_m"}."""
    raw = gzip.decompress(open(path, "rb").read())
    size = struct.unpack("<I", raw[8:12])[0]
    header = json.loads(raw[12 : 12 + size])
    values = np.frombuffer(raw[12 + size :], "<i2").reshape(header["dimensions"])
    quantization = header.get("quantization") or {}
    unknown = values == quantization.get("unknown", -32768)
    return {"header": header, "values": values, "unknown": unknown, "scale_m": quantization.get("scale_m", 0.001)}


def load_capture(folder: str) -> Capture:
    """Reads a capture folder. folder: The capture (live/, autoxing_topics.jsonl.gz). Returns: The capture."""
    raw = gzip.decompress(open(os.path.join(folder, "live", "occupancy.bin.gz"), "rb").read())
    size = struct.unpack("<I", raw[8:12])[0]
    records = np.frombuffer(raw[12 + size :], dtype=np.dtype([("block", "<i4", (3,)), ("v", "i1", (512,))]))
    messages = []
    with gzip.open(os.path.join(folder, "autoxing_topics.jsonl.gz"), "rt") as f:
        for line in f:
            row = json.loads(line)
            messages.append((float(row["t"]), row["msg"]))
    messages.sort(key=lambda m: m[0])
    robot_field = _read_besdfq1(os.path.join(folder, "live", "arm_esdf.bin.gz"))
    return Capture(records["block"].copy(), records["v"].copy(), json.loads(raw[12 : 12 + size]), robot_field, messages)


def _lidar_frames(capture: Capture, area_free) -> list:
    """The recorded scans as spatial_awareness's tick frames, each placed with the latest pose before it.

    capture: The capture. area_free: spatial_awareness's area_free module (its Frame).
    Returns: area_free.Frame per /horizontal_laser_2d/matched scan, in time order (scans before the first pose dropped).
    """
    frames, pose = [], None
    for t, msg in capture.messages:
        if msg.get("topic") == "/tracked_pose":
            pose = msg
        elif msg.get("topic") == "/horizontal_laser_2d/matched" and pose is not None and msg.get("data"):
            points = np.frombuffer(base64.b64decode(msg["data"]), "<f4").reshape(-1, 4)
            points = points[points[:, 3] != NOISE_INTENSITY]
            stamp = float(msg.get("stamp") or t)
            frames.append(area_free.Frame(stamp, float(pose["pos"][0]), float(pose["pos"][1]), float(pose["ori"]),
                                          points[:, :2].astype(np.float32)))  # fmt: skip
    return frames


def replay_lidar(capture: Capture):
    """Replays the scans through the robot's alignment, standing walls and lidar grid, as its tick does.

    capture: The capture.
    Returns: (the aligned pose after the last scan (AlignedPose), the standing walls (WallMap or None), the lidar
        grid's snapshot (LidarLayer or None), the number of scans replayed).
    """
    occupancy_field, pose_alignment, _, area_free, SensorReading = _field_code()
    map_msg = next((m for _, m in capture.messages if m.get("topic") == "/map"), None)
    if map_msg is None:
        raise SystemExit("the capture has no /map message: AutoXing's walls cannot be stood")
    sources = pose_alignment.ReferenceSources()
    aligner, walls, grid = pose_alignment.PoseAligner(), occupancy_field.StandingWalls(), None
    reading = SensorReading(data=map_msg, timestamp=capture.messages[0][0])
    frames = _lidar_frames(capture, area_free)
    for frame in frames:
        references = sources.refresh(reading, frame.timestamp)
        aligner.update(frame, references)
        walls.set_reference(sources.autoxing_reference())
        robot_xy = np.array([frame.pose_x, frame.pose_y])
        pose = aligner.aligned_pose((frame.pose_x, frame.pose_y, frame.pose_theta), frame.timestamp)
        if walls.scan_due(frame.timestamp) and pose.aligned:
            raw = (pose.x_m - pose.correction_m[0], pose.y_m - pose.correction_m[1])
            correction = occupancy_field.Correction(pose.correction_rad, np.asarray(pose.correction_m, float),
                                                    np.asarray(raw))  # fmt: skip
            walls.add_scan(robot_xy, frame.points_xy, correction, frame.timestamp)
        grid = occupancy_field.follow_lidar(grid, robot_xy)
        if grid.scan_due(frame.timestamp):
            grid.add(robot_xy, frame.points_xy, frame.timestamp)
    last = frames[-1]
    pose = aligner.aligned_pose((last.pose_x, last.pose_y, last.pose_theta), last.timestamp)
    return pose, walls.standing(), grid.snapshot() if grid is not None else None, len(frames)


def rebuild_field(capture: Capture):
    """The arm field as the robot built it, from the capture (the robot's own field gives the placement).

    capture: The capture.
    Returns: (the Field (occupancy_field.Field), the aligned pose used, the number of scans replayed).
    """
    occupancy_field = _field_code()[0]
    header = capture.robot_field["header"]
    pose, walls, lidar, scans = replay_lidar(capture)
    base = tuple(float(v) for v in header["base"])  # AutoXing's raw pose: where the mapper placed its frames
    arm_xyz = np.asarray(header["arm_base_m"], dtype=float)
    area = occupancy_field.area_around(arm_xyz, (CROP_CELLS,) * 3, VOXEL_M)
    correction = occupancy_field.Correction(pose.correction_rad, np.asarray(pose.correction_m, float),
                                            np.asarray(base[:2]))  # fmt: skip
    cap = occupancy_field.CAP_M if header.get("capped") else None
    field = occupancy_field.build_field(area, capture.blocks, capture.values, walls, correction, base,
                                        float(arm_xyz[2]), cap, lidar)  # fmt: skip
    return field, pose, scans


def compare(field, robot_field: dict) -> dict:
    """How the rebuilt field agrees with the robot's own, cell by cell.

    field: The rebuilt Field. robot_field: The robot's (_read_besdfq1).
    Returns: {"same_origin", "occupied_robot", "occupied_rebuilt", "occupied_both", "occupied_iou", "unknown_agree",
        "distance_p50_cm", "distance_p99_cm"} (distances over cells known in both).
    """
    header = robot_field["header"]
    robot_occ = (robot_field["values"] <= 0) & ~robot_field["unknown"]
    ours_occ = (field.values <= 0) & ~field.unknown
    known = ~robot_field["unknown"] & ~field.unknown
    diff = np.abs(robot_field["values"].astype(float) * robot_field["scale_m"] - field.values)[known]
    both, either = int(np.count_nonzero(robot_occ & ours_occ)), int(np.count_nonzero(robot_occ | ours_occ))
    return {
        "same_origin": bool(np.allclose(header["origin_m"], field.origin_m)),
        "occupied_robot": int(robot_occ.sum()), "occupied_rebuilt": int(ours_occ.sum()), "occupied_both": both,
        "occupied_iou": both / either if either else 1.0,
        "unknown_agree": float(np.mean(robot_field["unknown"] == field.unknown)),
        "distance_p50_cm": float(np.percentile(diff, 50) * 100), "distance_p99_cm": float(np.percentile(diff, 99) * 100),
    }  # fmt: skip


def write_field(field, header: dict, path: str) -> None:
    """Writes the rebuilt field as the robot does (gzip BESDFQ1, never-observed cells as the unknown value).

    field: The rebuilt Field. header: The robot field's header (its extra keys are kept). path: The file to write.
    """
    from billie_utils.esdf_grid import serialize_esdf_grid  # noqa: PLC0415

    extra = {k: header[k] for k in ("map_epoch", "frames_integrated", "capped", "cap_m", "arm_base_m", "base") if k in header}
    extra["note"] = "rebuilt offline from a capture: " + field.note
    data = serialize_esdf_grid(name="arm_esdf", frame_id="autoxing_map", origin_m=field.origin_m, voxel_size_m=VOXEL_M,
                               distances_m=np.where(field.unknown, np.nan, field.values), extra_header=extra)  # fmt: skip
    with open(path, "wb") as f:
        f.write(gzip.compress(data, 6))


def main() -> None:
    """Rebuilds a capture's arm field, prints how it agrees with the robot's, and writes it with --write."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("capture", help="the capture folder (see its README.md)")
    parser.add_argument("--write", help="write the rebuilt field here (gzip BESDFQ1, like live/arm_esdf.bin.gz)")
    args = parser.parse_args()
    _use_settings(args.capture)
    capture = load_capture(args.capture)
    field, pose, scans = rebuild_field(capture)
    header = capture.robot_field["header"]
    print(f"robot's field: {header.get('note')}")
    print(f"rebuilt field: {field.note}")
    print(f"alignment after {scans} scans: aligned={pose.aligned} correction "
          f"({pose.correction_m[0]:+.3f}, {pose.correction_m[1]:+.3f}) m, {np.degrees(pose.correction_rad):+.2f} deg "
          f"{pose.reason}")  # fmt: skip
    for key, value in compare(field, capture.robot_field).items():
        print(f"  {key:18s} {value:.3f}" if isinstance(value, float) else f"  {key:18s} {value}")
    if args.write:
        write_field(field, header, args.write)
        print(f"wrote {args.write}")


if __name__ == "__main__":
    main()
