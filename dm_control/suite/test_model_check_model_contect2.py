from dm_control import suite

env = suite.load(domain_name="nubots", task_name="stand")

env.physics.data.qpos[:3] = [0, 0, 80]
env.physics.forward()

print(f"🟠 当前接触数: {env.physics.data.ncon}")

pairs = set()

for i in range(env.physics.data.ncon):
    c = env.physics.data.contact[i]
    g1, g2 = c.geom1, c.geom2
    b1, b2 = env.physics.model.geom_bodyid[g1], env.physics.model.geom_bodyid[g2]

    body1 = env.physics.model.id2name(b1, 'body')
    body2 = env.physics.model.id2name(b2, 'body')
    geom1 = env.physics.model.id2name(g1, 'geom')
    geom2 = env.physics.model.id2name(g2, 'geom')

    pair = tuple(sorted((body1, body2)))
    if pair not in pairs:
        pairs.add(pair)
        print(f"❌ 接触: {body1} ({geom1}) ⬄ {body2} ({geom2})")
