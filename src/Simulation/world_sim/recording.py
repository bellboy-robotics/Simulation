"""Replay frames from a recording, resampled to the brain's tick rate like replay_policy does."""

import json
import math
import os
from dataclasses import dataclass, replace

import numpy as np

# [Hz] The brain plays one replay frame per tick (NODE_TICKS_MILLIS["brain"] = 20ms).
BRAIN_TICK_RATE = 50
# Local cache of exported recordings (poses + joints + fps as JSON), also synced to robots.
RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "output", "world_sim", "recordings")


@dataclass
class ReplayState:
    """One replay frame (the brain's state.State, without gripper, thing and brain commands)."""

    position: np.ndarray  # (6,) recorded TCP pose [x, y, z mm, rx, ry, rz rad rotation vector]
    recorded_joints: np.ndarray | None  # (6,) recorded joints, degrees
    source_frame: int  # index of the recording frame it came from, -1 for planned (rerouted) frames

    def copy(self) -> "ReplayState":
        """A copy whose arrays can be changed independently."""
        joints = None if self.recorded_joints is None else self.recorded_joints.copy()
        return replace(self, position=self.position.copy(), recorded_joints=joints)


def recording_file(repo_id: str, folder: str) -> str:
    """Where a recording exported by export_recording lives.

    repo_id: Hugging Face dataset id. folder: Recordings folder.
    Returns: Path of its JSON file.
    """
    return os.path.join(folder, repo_id.replace("/", "__") + ".json")


def export_recording(repo_id: str, folder: str) -> str:
    """Saves a recording's poses, joints and fps as JSON, for machines without lerobot (e.g. a robot's planner venv).

    repo_id: Hugging Face dataset id. folder: Recordings folder (created if missing).
    Returns: Path of the written file (skipped if it exists).
    """
    path = recording_file(repo_id, folder)
    if not os.path.exists(path):
        from Simulation.ik_sim.dataset_reader import DatasetReader

        dataset = DatasetReader(repo_id, episode=0).dataset
        poses = [frame["observation.xarm_pose"].numpy().tolist() for frame in dataset]
        joints = [frame["observation.xarm_joints"].numpy()[:6].tolist() for frame in dataset]
        os.makedirs(folder, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"repo_id": repo_id, "fps": float(dataset.fps), "poses": poses, "joints_deg": joints}, f)
    return path


def load_recording(repo_id: str) -> tuple[list[ReplayState], float]:
    """Reads a recording's arm poses and joints: from $WORLD_SIM_RECORDINGS (export_recording) if there, else from Hugging Face.

    repo_id: Hugging Face dataset id, e.g. "bellboy-robotics/B-unknown-20260301-180408-BILLIE-12".
    Returns: (one state per recorded frame, the recording's fps).
    """
    path = recording_file(repo_id, os.environ.get("WORLD_SIM_RECORDINGS", RECORDINGS_DIR))
    if not os.path.exists(path):
        path = export_recording(repo_id, RECORDINGS_DIR)
    with open(path) as f:
        data = json.load(f)
    states = [
        ReplayState(position=np.asarray(p, dtype=np.float64), recorded_joints=np.asarray(j, dtype=np.float64), source_frame=i)
        for i, (p, j) in enumerate(zip(data["poses"], data["joints_deg"]))
    ]
    return states, float(data["fps"])


def select_poses(poses: list[ReplayState], fps: float, speed: float) -> list[ReplayState]:
    """Picks the frames to play so playback at the brain's tick rate runs at the requested speed.

    Port of replay_policy._select_poses without the brain-command special cases: frames are
    skipped when the recording is faster than the ticks, and repeated when it is slower.

    poses: The recorded frames.
    fps: The recording's frame rate.
    speed: Playback speed, 1.0 = as recorded.
    Returns: The frames to play, one per brain tick.
    """
    ratio = fps * speed / BRAIN_TICK_RATE
    frames = []
    ticks = 0
    while True:
        index = math.floor(ticks * ratio)
        if index >= len(poses):
            break
        frames.append(poses[index].copy())
        ticks += 1
    return frames
