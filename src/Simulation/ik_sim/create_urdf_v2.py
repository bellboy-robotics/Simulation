import os
import numpy as np
import time
import pybullet as p
import pybullet_data

matches = [f for f in os.listdir("/tmp/urdf") if f.endswith("-with-tcp.urdf")]
if not matches:
    raise FileNotFoundError("No *-with-tcp.urdf found in /tmp/urdf — run the IK solver first")
else:
    print(matches)
urdf_path = f"/tmp/urdf/{matches[0]}"
print(f"loading: {urdf_path}")

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot = p.loadURDF(urdf_path, useFixedBase=True)
p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=45, cameraPitch=-20,
                              cameraTargetPosition=[0, 0, 0.3])

joint_sliders = []
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    if info[2] == p.JOINT_REVOLUTE:
        lower, upper = info[8], info[9]
        if lower >= upper:
            lower, upper = -3.14, 3.14
        slider = p.addUserDebugParameter(info[1].decode(), lower, upper, 0.0)
        if slider >= 0:
            joint_sliders.append((i, slider))

while p.isConnected():
    try:
        for joint_index, slider in joint_sliders:
            p.resetJointState(robot, joint_index, p.readUserDebugParameter(slider))
        p.stepSimulation()
    except p.error:
        if not p.isConnected():
            break
    time.sleep(1. / 60.)
