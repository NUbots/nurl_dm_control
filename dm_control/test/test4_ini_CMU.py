from dm_control import suite
from dm_control import viewer
import numpy as np
import matplotlib.pyplot as plt  # 添加 matplotlib 导入

# 加载环境
env = suite.load(domain_name="nubots_CMU", task_name="stand")
action_spec = env.action_spec()

# 打印 action space 信息
print("Action Spec:", action_spec)

# 打印 observation 变量中的所有传感器信息
observation_spec = env.observation_spec()
print("Observation Spec (sensors):")
for key, value in observation_spec.items():
    print(f"  {key}: {value}")

# ✅ 设置初始化姿态（Base 位于 z = 0.6 处）
# 注意：reset 后设置才有效
time_step = env.reset()
env.physics.data.qpos[:7] = [0, 0, 0.6, 1, 0, 0, 0]  # xyz + 四元数(w, x, y, z)
env.physics.data.qpos[7:] = 0                      # 所有关节角度清零
env.physics.data.qvel[:] = 0                       # 所有关节速度清零
env.physics.forward()

# ✅ 定义一个随机策略
def random_policy(time_step):
    del time_step  # Unused
    return np.random.uniform(low=action_spec.minimum,
                             high=action_spec.maximum,
                             size=action_spec.shape)

# 用 for 循环控制动作并打印观测信息
num_steps = 100
for step in range(num_steps):

    action = random_policy(time_step)
    time_step = env.step(action)
    print(f"Step {step} observation:")
    for key, value in time_step.observation.items():
        print(f"  {key}: {value}")

    # 渲染图像并显示
    img = env.physics.render(height=480, width=640, camera_id=0)
    plt.imshow(img)
    plt.axis('off')
    plt.pause(0.001)
    plt.clf()

