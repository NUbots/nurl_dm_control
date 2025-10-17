from dm_control import suite

env = suite.load(domain_name="nubots", task_name="stand")

# 设置高度或直接 forward 看初始接触
env.physics.data.qpos[:3] = [0, 0, 80]
env.physics.forward()

print(f"🟠 当前接触数: {env.physics.data.ncon}")

for i in range(env.physics.data.ncon):
    contact = env.physics.data.contact[i]

    # 获取 geom ID
    g1 = contact.geom1
    g2 = contact.geom2

    # 获取 geom 所属的 body ID
    b1 = env.physics.model.geom_bodyid[g1]
    b2 = env.physics.model.geom_bodyid[g2]

    # 获取 body 名称（可能为 None）
    body1_name = env.physics.model.id2name(b1, 'body')
    body2_name = env.physics.model.id2name(b2, 'body')

    print(f"❌ 接触对 {i+1}: body '{body1_name}' ⬄ body '{body2_name}'")

