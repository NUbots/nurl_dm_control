from dm_control import mujoco
import numpy as np

with open("nubots.xml") as f:
    xml = f.read()

physics = mujoco.Physics.from_xml_string(xml)

# 初始化尝试
print("Before forward, ncon =", physics.data.ncon)
physics.forward()
print("After forward, ncon =", physics.data.ncon)

print("qpos:", physics.data.qpos)
print("qvel:", physics.data.qvel)

# 加载并强制设置初始高度
physics = mujoco.Physics.from_xml_string(xml)
physics.data.qpos[:3] = [0, 0, 80.]
physics.forward()
print("New ncon =", physics.data.ncon)