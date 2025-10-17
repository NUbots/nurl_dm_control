
import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np
_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = 0.020
_STAND_HEIGHT = 0.6
_WALK_SPEED = 1.0
_RUN_SPEED = 10

SUITE = containers.TaggedTasks()

def get_model_and_assets():
    """Returns the model XML and asset dict for NUBots."""
    return common.read_model('assets/nubots_CMU.xml'), common.ASSETS

@SUITE.add('benchmarking')
def stand(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = NUBotsTask(move_speed=0, pure_state=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)

@SUITE.add('benchmarking')
def walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = NUBotsTask(move_speed=_WALK_SPEED, pure_state=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)

@SUITE.add('benchmarking')
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = NUBotsTask(move_speed=_RUN_SPEED, pure_state=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)

@SUITE.add()
def run_pure_state(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = NUBotsTask(move_speed=_RUN_SPEED, pure_state=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)

class Physics(mujoco.Physics):
    def torso_upright(self):
        return self.named.data.xmat['torso', 'zz']

    def head_height(self):
        return self.named.data.xpos['head', 'z']

    def center_of_mass_position(self):
        return self.named.data.subtree_com['torso'].copy()

    def center_of_mass_velocity(self):
        return self.named.data.sensordata['torso_subtreelinvel'].copy()

    def torso_vertical_orientation(self):
        return self.named.data.xmat['torso', ['zx', 'zy', 'zz']]

    def joint_angles(self):
        return self.data.qpos[7:].copy()

    def extremities(self):
        torso_frame = self.named.data.xmat['torso'].reshape(3, 3)
        torso_pos = self.named.data.xpos['torso']
        positions = []
        for site in ['left_foot_touch', 'right_foot_touch']:
            rel_pos = self.named.data.site_xpos[site] - torso_pos
            positions.append(rel_pos.dot(torso_frame))
        return np.hstack(positions)

    # def extremities(self):
    #     """Returns end effector positions in egocentric frame."""
    #     torso_frame = self.named.data.xmat['torso'].reshape(3, 3)
    #     torso_pos = self.named.data.xpos['torso']
    #     positions = []
    #     for side in ('left_', 'right_'):
    #         for limb in ('foot'):
    #             torso_to_limb = self.named.data.xpos[side + limb] - torso_pos
    #             positions.append(torso_to_limb.dot(torso_frame))
    #     return np.hstack(positions)


class NUBotsTask(base.Task):
    def __init__(self, move_speed, pure_state, random=None):
        self._move_speed = move_speed
        self._pure_state = pure_state
        super().__init__(random=random)
        
    # def initialize_episode(self, physics):
    #     # 尝试自动抬高初始位置，直到无接触（避免穿透）
    #     max_attempts = 10
    #     for i in range(max_attempts):
    #         physics.data.qpos[:3] = [0, 0, 0.5 + i * 0.05]  # 不断上抬
    #         physics.forward()
    #         if physics.data.ncon == 0:
    #             break
    #     else:
    #         raise RuntimeError("Failed to find non-colliding initial height")

    #     super().initialize_episode(physics)
    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Args:
        physics: An instance of `Physics`.

        """
        # Find a collision-free random initial configuration.
        penetrating = True
        while penetrating:
            # randomizers.randomize_limited_and_rotational_joints(physics, self.random)
            # Check for collisions.
            physics.after_reset()
            penetrating = physics.data.ncon > 0
            print(f"Attempting to find non-colliding configuration, ncon = {physics.data.ncon}")
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['joint_angles'] = physics.joint_angles()
        obs['joint_velocities'] = physics.data.qvel.copy()
        obs['com_velocity'] = physics.center_of_mass_velocity()[[0, 1]]
        obs['upright'] = np.array([physics.torso_upright()])
        obs['head_height'] = np.array([physics.head_height()])
        obs['extremities'] = physics.extremities()
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        standing = rewards.tolerance(physics.head_height(),
                                    bounds=(_STAND_HEIGHT, float('inf')),
                                    margin=_STAND_HEIGHT/4)
        upright = rewards.tolerance(physics.torso_upright(),
                                    bounds=(0.9, float('inf')), sigmoid='linear',
                                    margin=1.9, value_at_margin=0)
        stand_reward = standing * upright
        small_control = rewards.tolerance(physics.control(), margin=1,
                                        value_at_margin=0,
                                        sigmoid='quadratic').mean()
        small_control = (4 + small_control) / 5
        if self._move_speed == 0:
            horizontal_velocity = physics.center_of_mass_velocity()[[0, 1]]
            dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()
            return small_control * stand_reward * dont_move
        else:
            com_velocity = np.linalg.norm(physics.center_of_mass_velocity()[[0, 1]])
            move = rewards.tolerance(com_velocity,
                                    bounds=(self._move_speed, float('inf')),
                                    margin=self._move_speed, value_at_margin=0,
                                    sigmoid='linear')
            move = (5*move + 1) / 6
            return small_control * stand_reward * move
