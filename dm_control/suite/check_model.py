
from dm_control import mujoco
import numpy as np

# 路径改为你本地文件路径
with open("nubots.xml") as f:
    xml = f.read()

try:
    physics = mujoco.Physics.from_xml_string(xml)
    model = physics.model
    data = physics.data

    print("✅ 模型载入成功")
    print("基本结构信息：")
    print(f"  关节数 njnt:     {model.njnt}")
    print(f"  自由度 nq:       {model.nq}")
    print(f"  速度向量维度 nv: {model.nv}")
    print(f"  刚体数 nbody:    {model.nbody}")
    print(f"  传感器数量:      {model.nsensor}")
    print(f"  geom 数量:       {model.ngeom}")

    print("\n✅ 状态检查：")
    print(f"  qpos shape: {data.qpos.shape}")
    print(f"  qvel shape: {data.qvel.shape}")
    print(f"  qpos 范围:  min={np.min(data.qpos):.5f}, max={np.max(data.qpos):.5f}")
    print(f"  qvel 范围:  min={np.min(data.qvel):.5f}, max={np.max(data.qvel):.5f}")
    print(f"  qpos 是否含 NaN？ {np.isnan(data.qpos).any()}")
    print(f"  qvel 是否含 NaN？ {np.isnan(data.qvel).any()}")

    # 尝试 forward，是否触发崩溃
    print("\n➡️ 尝试调用 physics.forward()")
    physics.forward()
    print("✅ forward() 成功")

except Exception as e:
    print("❌ 模型载入或 forward 失败：")
    print(str(e))
