"""PyBullet playback of a simulated run: the arm moving among the world objects.

Avoid objects are red, eef_touch objects green. A link turns orange inside the planner's 2cm
margin of an avoid object and red inside it.
"""

import os
import time

import numpy as np
import pybullet as p
from billie_utils.messages.pyroki_world_poc import WORLD_COL_MARGIN_M

from Simulation.ik_sim.pybullet_sim import set_pybullet
from Simulation.world_sim.analysis import Run, dense_path, link_distances
from Simulation.world_sim.report import ROLE_COLORS

# [s] Display time per path sample at speed 1: one brain tick, so replay frames play at recorded speed.
_SAMPLE_PERIOD_S = 0.02
_LINK_COLORS = {"clear": [0.7, 0.7, 0.7, 1], "near": [1.0, 0.6, 0.0, 1], "inside": [1.0, 0.0, 0.0, 1]}


def _urdf_path() -> str:
    """The planner's merged URDF (arm + base + gripper + TCP), generated locally if this machine has none.

    Returns: Path of /tmp/urdf/<XARM_SN>-with-tcp.urdf.
    """
    path = f"/tmp/urdf/{os.environ['XARM_SN']}-with-tcp.urdf"
    if not os.path.exists(path):
        from pyroki_planner.urdf import load_urdf

        load_urdf()
    return path


def _add_objects(client: int, run: Run) -> None:
    """Adds every world object's capsules as static visual bodies.

    client: PyBullet client id. run: The run (objects in meters, xArm base frame = PyBullet world).
    """
    for obj in run.objects:
        rgba = [int(ROLE_COLORS[obj["role"]][i : i + 2], 16) / 255 for i in (1, 3, 5)] + [0.5]
        for start, end, radius in zip(obj["starts_m"], obj["ends_m"], obj["radii_m"]):
            axis = end - start
            length = float(np.linalg.norm(axis))
            if length < 1e-6:
                shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=client)
                orientation = [0, 0, 0, 1]
            else:
                shape = p.createVisualShape(p.GEOM_CAPSULE, radius=radius, length=length, rgbaColor=rgba,
                                            physicsClientId=client)  # fmt: skip
                z = axis / length
                rot_axis = np.cross([0, 0, 1], z)
                angle = float(np.arccos(np.clip(z[2], -1, 1)))
                rot_axis = rot_axis / np.linalg.norm(rot_axis) if np.linalg.norm(rot_axis) > 1e-9 else np.array([1.0, 0, 0])
                orientation = p.getQuaternionFromAxisAngle(rot_axis.tolist(), angle)
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, basePosition=((start + end) / 2).tolist(),
                              baseOrientation=orientation, physicsClientId=client)  # fmt: skip


def play(run: Run, speed: float = 1.0, loop: bool = False) -> None:
    """Opens the PyBullet GUI and plays the run's arm path among its objects.

    run: The run to show.
    speed: Playback speed; 1 shows one path sample (a replay frame, or 2deg of a single move) per 20ms.
    loop: Replay forever (close the window to stop); otherwise stay on the last frame until closed.
    """
    arm = set_pybullet(_urdf_path(), gui=True, camera_distance=1.8, camera_yaw=60, camera_pitch=-25,
                       camera_target=(0.3, 0.0, 0.2))  # fmt: skip
    client, robot = arm["client"], arm["robot_id"]
    _add_objects(client, run)
    names = {p.getJointInfo(robot, j, physicsClientId=client)[12].decode(): j
             for j in range(p.getNumJoints(robot, physicsClientId=client))}  # fmt: skip
    link_ids = [names.get(name) for name in run.model.link_names]  # None for the URDF root link

    samples, owner = dense_path(run)
    distance = link_distances(run, samples, "avoid")
    state = np.where(distance < 0, 2, np.where(distance < WORLD_COL_MARGIN_M, 1, 0))
    colors = [_LINK_COLORS["clear"], _LINK_COLORS["near"], _LINK_COLORS["inside"]]
    text_id, shown = -1, np.full(len(link_ids), -1)
    while p.isConnected(client):
        for k, (q, i) in enumerate(zip(samples, owner)):
            if not p.isConnected(client):
                return
            for joint, angle in zip(arm["joint_indices"], np.deg2rad(q)):
                p.resetJointState(robot, joint, float(angle), physicsClientId=client)
            for li, (link, s) in enumerate(zip(link_ids, state[k])):
                if link is not None and shown[li] != s:
                    p.changeVisualShape(robot, link, rgbaColor=colors[s], physicsClientId=client)
                    shown[li] = s
            c = run.command[i]
            label = "start" if i < 0 else f"cmd {c} {run.commands[c]['spec']['cmd']}: {run.kind[i]} #{i}"
            replace = {"replaceItemUniqueId": text_id} if text_id >= 0 else {}
            text_id = p.addUserDebugText(label, [0, 0, 1.0], textSize=1.4, physicsClientId=client, **replace)
            time.sleep(_SAMPLE_PERIOD_S / speed)
        if not loop:
            break
    while p.isConnected(client):
        time.sleep(0.1)
