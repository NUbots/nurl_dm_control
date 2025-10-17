import numpy as np

class DMControlWrapper:
    def __init__(self, domain, task, seed=0):
        from dm_control import suite
        self.env = suite.load(domain, task, task_kwargs={'random': seed})
        self.action_spec = self.env.action_spec()
        self.observation_spec = self.env.observation_spec()
        self._reset_next = True

        # Flatten observation space
        self.obs_dim = sum(np.prod(v.shape) for v in self.observation_spec.values())
        self.act_dim = self.action_spec.shape[0]

    def _flatten_obs(self, obs_dict):
        return np.concatenate([v.ravel() for v in obs_dict.values()])

    def reset(self):
        time_step = self.env.reset()
        return self._flatten_obs(time_step.observation)

    def step(self, action):
        time_step = self.env.step(action)
        obs = self._flatten_obs(time_step.observation)
        reward = time_step.reward or 0.0
        done = time_step.last()
        return obs, reward, done, {}

    def render(self):
        return self.env.physics.render(height=240, width=320, camera_id=0)
