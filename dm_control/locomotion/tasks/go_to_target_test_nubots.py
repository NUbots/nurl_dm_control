# Copyright 2024 The dm_control Authors.
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

"""Tests for locomotion.tasks.go_to_target with NUBots humanoid."""

from absl.testing import absltest
from dm_control import composer
from dm_control.locomotion.arenas import floors
from dm_control.suite import nubots_CMU
import numpy as np

class GoToTargetNUBotsTest(absltest.TestCase):

    def test_observables(self):
        env = nubots_CMU.stand()
        timestep = env.reset()
        self.assertIn('joint_angles', timestep.observation)
        self.assertIn('head_height', timestep.observation)

    def test_reward_stand(self):
        env = nubots_CMU.stand()
        env.reset()
        zero_action = np.zeros_like(env.physics.data.ctrl)
        for _ in range(2):
            timestep = env.step(zero_action)
            # Standing reward should be positive if upright
            self.assertGreaterEqual(timestep.reward, 0)
        # Move the agent's head below the stand height to test reward drop
        env.physics.data.qpos[2] = 0.1
        env.physics.forward()
        timestep = env.step(zero_action)
        self.assertLess(timestep.reward, 0.5)

    def test_termination(self):
        env = nubots_CMU.stand()
        env.reset()
        zero_action = np.zeros_like(env.physics.data.ctrl)
        # Should not terminate in the first few steps
        for _ in range(5):
            timestep = env.step(zero_action)
            self.assertFalse(timestep.last())
        # Drop the agent far below the ground to trigger termination
        env.physics.data.qpos[2] = -1.0
        env.physics.forward()
        timestep = env.step(zero_action)
        self.assertTrue(timestep.last())

if __name__ == '__main__':
    absltest.main()
