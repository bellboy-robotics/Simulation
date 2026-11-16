"""What a replay_policy recording does with the arm: its joints at chosen points, its base pose, its gripper, and how
far its path strays from a straight joints move to its last frame.

    python -m Simulation.flow_fix.recording_joints bellboy-robotics/B-standard_br_under_bed-20260128-200037-BILLIE-14

Downloads only the recording's meta and data files (no videos) with the Hugging Face login saved on this machine,
into --cache. Use it before replacing a replay with a `joints` action: the last frame is the joints to send, and a
large gap from the straight line means the recording went around something (add the mid point as a step first).
"""

import argparse
import os
import tempfile

import numpy as np
import pyarrow.parquet as pq

# [deg] A gap from the straight joints line this large on any joint is worth a via point before the last frame.
VIA_POINT_GAP_DEG = 15.0


def download(repo_id: str, cache: str) -> str:
    """The recording's first episode file, downloaded once. repo_id: The dataset. cache: Folder. Returns: Its path."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415 - only when downloading

    folder = os.path.join(cache, repo_id.replace("/", "__"))
    snapshot_download(repo_id, repo_type="dataset", allow_patterns=["meta/*", "data/*"], local_dir=folder)
    return os.path.join(folder, "data", "chunk-000", "episode_000000.parquet")


def column(table, name: str) -> np.ndarray:
    """A feature of every frame. table: The episode. name: e.g. "observation.xarm_joints". Returns: (frames, ...)."""
    return np.asarray(table.column(name).to_pylist(), dtype=float)


def straight_line_gap(joints: np.ndarray) -> tuple[np.ndarray, float]:
    """The largest gap between the recorded joints and a straight joints move from its first to its last frame.

    joints: (frames, 6) degrees. Returns: (the gap per joint at its worst frame, that frame's share of the move).
    """
    travel = np.r_[0.0, np.cumsum(np.abs(np.diff(joints, axis=0)).sum(1))]
    share = travel / max(travel[-1], 1e-9)
    line = joints[0] + share[:, None] * (joints[-1] - joints[0])
    worst = int(np.argmax(np.abs(joints - line).max(1)))
    return joints[worst] - line[worst], float(share[worst])


def report(path: str, points: list[float]) -> None:
    """Prints the recording's summary. path: The episode file. points: Shares of the move to print joints at."""
    table = pq.read_table(path)
    joints = column(table, "observation.xarm_joints")
    base = column(table, "observation.base_pose")
    grip = column(table, "observation.gripper").reshape(-1)
    print(f"{len(joints)} frames ({table.column('timestamp').to_pylist()[-1]:.1f}s)")
    print(f"base pose first {np.round(base[0], 4).tolist()} last {np.round(base[-1], 4).tolist()} (map x, y, heading)")
    print(f"gripper first {grip[0]:.0f} last {grip[-1]:.0f} (min {grip.min():.0f}, max {grip.max():.0f})")
    for p in points:
        i = min(int(round(p * (len(joints) - 1))), len(joints) - 1)
        print(f"joints at {p:.0%} (frame {i}): {np.round(joints[i], 2).tolist()}")
    tail = joints[-10:].max(0) - joints[-10:].min(0)
    print(f"still over the last 10 frames: {bool(np.all(tail < 0.1))} (span {np.round(tail, 3).tolist()})")
    gap, share = straight_line_gap(joints)
    need = bool(np.abs(gap).max() > VIA_POINT_GAP_DEG)
    print(f"largest gap from a straight joints move: {np.round(gap, 1).tolist()} deg at {share:.0%} of the move"
          f"{' -> add a via point (the joints at 50%) before the last frame' if need else ''}")  # fmt: skip


def main() -> None:
    """Parses the arguments, downloads the recording and prints its summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("repo_id", help="the replay_policy's repo_id")
    parser.add_argument("--points", type=float, nargs="+", default=[0.0, 0.5, 1.0], help="shares of the move")
    parser.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "flow_fix_recordings"))
    args = parser.parse_args()
    report(download(args.repo_id, args.cache), args.points)


if __name__ == "__main__":
    main()
