from dm_control.suite import nubots
import numpy as np

env = nubots.stand()
ts = env.reset()

print("Initial observation keys:", ts.observation.keys())

for i in range(10):
    action = np.random.uniform(env.action_spec().minimum,
                               env.action_spec().maximum)
    ts = env.step(action)
    print(f"Step {i+1}: reward={ts.reward:.3f}")
