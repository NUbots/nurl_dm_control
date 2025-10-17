# Copyright 2025 Your Name
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""NUBots Domain."""

import collections
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base, common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers, rewards

_DEFAULT_TIME_LIMIT = 20
_CONTROL_TIMESTEP = 0.02
# Height of root body above which stand reward is 1.
_STAND_HEIGHT = 0.5
# Speed threshold for walk reward >=1
_WALK_SPEED = 1.0
_RUN_SPEED = 10

SUITE = containers.TaggedTasks()


def get_model_and_assets():
    """Returns the model XML and asset dict for NUBots."""
    return common.read_model('nubots.xml'), common.ASSETS


@SUITE.add('benchmarking')
def stand(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Stand task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = nubots(move_speed=0, pure_state=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add('benchmarking')
def walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Walk task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = nubots(move_speed=_WALK_SPEED, pure_state=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add('benchmarking')
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = nubots(move_speed=_RUN_SPEED, pure_state=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add()
def run_pure_state(time_limit=_DEFAULT_TIME_LIMIT, random=None,
                   environment_kwargs=None):
  """Returns the Run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = nubots(move_speed=_RUN_SPEED, pure_state=True, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)

class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Walker domain."""

  def torso_upright(self):
    """Returns projection from z-axes of torso to the z-axes of world."""
    return self.named.data.xmat['torso', 'zz']

  def head_height(self):
    """Returns the height of the torso."""
    return self.named.data.xpos['head', 'z']

  def center_of_mass_position(self):
    """Returns position of the center-of-mass."""
    return self.named.data.subtree_com['torso'].copy()

  def center_of_mass_velocity(self):
    """Returns the velocity of the center-of-mass."""
    return self.named.data.sensordata['torso_subtreelinvel'].copy()

  def torso_vertical_orientation(self):
    """Returns the z-projection of the torso orientation matrix."""
    return self.named.data.xmat['torso', ['zx', 'zy', 'zz']]

  def joint_angles(self):
    """Returns the state without global orientation or position."""
    return self.data.qpos[7:].copy()  # Skip the 7 DoFs of the free root joint.

  def extremities(self):
    """Returns end effector positions in egocentric frame."""
    torso_frame = self.named.data.xmat['torso'].reshape(3, 3)
    torso_pos = self.named.data.xpos['torso']
    positions = []
    for side in ('left_', 'right_'):
      for limb in ('hand', 'foot'):
        torso_to_limb = self.named.data.xpos[side + limb] - torso_pos
        positions.append(torso_to_limb.dot(torso_frame))
    return np.hstack(positions)



class nubots(base.Task):
    """A simple Task for NUBots model."""
    def __init__(self, move_speed, pure_state, random=None):
        self._move_speed = move_speed
        self._pure_state = pure_state
        super().__init__(random=random)

    def initialize_episode(self, physics):
        # Randomize until no interpenetration
        penetrating = True
        while penetrating:
            randomizers.randomize_limited_and_rotational_joints(physics, self.random)
            physics.after_reset()
            penetrating = physics.data.ncon > 0
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['joint_angles'] = physics.qpos.copy()
        obs['joint_velocities'] = physics.qvel.copy()
        # horizontal COM velocity
        com = physics.named.data.subtree_com['root']
        obs['com_velocity'] = com[[0, 1]]
        return obs

    def get_reward(self, physics):
        """Default reward placeholder; overridden by specific tasks."""
        return 0.0
