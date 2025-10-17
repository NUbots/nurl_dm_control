from dm_control import suite, viewer
from dm_control import _render
_render.RENDER_BACKEND = 'glfw'
import numpy as np

try:
    env = suite.load(domain_name="nubots", task_name="stand")
    action_spec = env.action_spec()

    def random_policy(time_step):
        return np.random.uniform(low=action_spec.minimum,
                                 high=action_spec.maximum,
                                 size=action_spec.shape)

    print("Launching viewer...")
    viewer.launch(env, policy=random_policy)

except Exception as e:
    print("❌ Failed to launch environment or viewer:")
    print(e)
