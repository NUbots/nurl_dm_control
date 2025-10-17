from dm_control import suite
import numpy as np
from dm_control import suite
from dm_control.locomotion import soccer as dm_soccer
# from dm_control.soccer import load
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control import viewer
# 设置渲染后端为 GLFW
from dm_control import _render
_render.RENDER_BACKEND = 'glfw'


# 加载环境
env = dm_soccer.load(team_size=2,
                     time_limit=1.5,
                     disable_walker_contacts=False,
                     enable_field_box=True,
                     terminate_on_goal=False,
                     walker_type=dm_soccer.WalkerType.NUBOTS)
action_spec = env.action_spec()

# 打印 action space 信息
print("Action Spec:", action_spec)

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

# ✅ 启动交互式 viewer（ESC 退出）
viewer.launch(env, policy=random_policy)
