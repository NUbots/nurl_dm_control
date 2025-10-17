# Copyright 2020 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""A CMU humanoid walker."""

import abc
import collections
import os

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base
from dm_control.locomotion.walkers import legacy_base
from dm_control.locomotion.walkers import rescale
from dm_control.locomotion.walkers import scaled_actuators
from dm_control.mujoco import wrapper as mj_wrapper
import numpy as np
from dm_control.suite import nubots_CMU

_XML_PATH = os.path.join(os.path.dirname(__file__),
                         'assets/nubots.xml')
_WALKER_GEOM_GROUP = 2
_WALKER_INVIS_GROUP = 1

_CMU_MOCAP_JOINTS = (
    'j_left_hip_yaw', 'j_left_hip_roll', 'j_left_hip_pitch', 'j_left_knee_pitch' ,'j_left_ankle_pitch',
    'j_left_ankle_roll', 'j_right_hip_yaw', 'j_right_hip_roll', 'j_right_hip_pitch' , 'j_right_knee_pitch',
    'j_right_ankle_pitch', 'j_right_ankle_roll', 'j_neck_yaw', 'j_head_pitch', 'j_left_shoulder_pitch', 
    'j_left_shoulder_roll', 'j_left_elbow_pitch', 'j_right_shoulder_pitch', 'j_right_shoulder_roll',
    'j_right_elbow_pitch' )





PositionActuatorParams = collections.namedtuple('PositionActuatorParams', ['name', 'forcerange', 'kp'])

_POSITION_ACTUATORS_NUBOTS = [
    PositionActuatorParams('j_left_hip_yaw',         [-80, 80], 80),     # lfemurrz
    PositionActuatorParams('j_left_hip_roll',        [-80, 80], 80),     # lfemurry
    PositionActuatorParams('j_left_hip_pitch',       [-120, 120], 120),  # lfemurrx
    PositionActuatorParams('j_left_knee_pitch',      [-80, 80], 80),     # ltibiarx
    PositionActuatorParams('j_left_ankle_pitch',     [-50, 50], 50),     # lfootrx
    PositionActuatorParams('j_left_ankle_roll',      [-50, 50], 50),     # lfootrz

    PositionActuatorParams('j_right_hip_yaw',        [-80, 80], 80),     # rfemurrz
    PositionActuatorParams('j_right_hip_roll',       [-80, 80], 80),     # rfemurry
    PositionActuatorParams('j_right_hip_pitch',      [-120, 120], 120),  # rfemurrx
    PositionActuatorParams('j_right_knee_pitch',     [-80, 80], 80),     # rtibiarx
    PositionActuatorParams('j_right_ankle_pitch',    [-50, 50], 50),     # rfootrx
    PositionActuatorParams('j_right_ankle_roll',     [-50, 50], 50),     # rfootrz

    PositionActuatorParams('j_neck_yaw',             [-20, 20], 20),     # headrz
    PositionActuatorParams('j_head_pitch',           [-20, 20], 20),     # headrx

    PositionActuatorParams('j_left_shoulder_pitch',  [-60, 60], 60),     # lhumerusrx
    PositionActuatorParams('j_left_shoulder_roll',   [-60, 60], 60),     # lhumerusry
    PositionActuatorParams('j_left_elbow_pitch',     [-60, 60], 60),     # lradiusrx

    PositionActuatorParams('j_right_shoulder_pitch', [-60, 60], 60),     # rhumerusrx
    PositionActuatorParams('j_right_shoulder_roll',  [-60, 60], 60),     # rhumerusry
    PositionActuatorParams('j_right_elbow_pitch',    [-60, 60], 60),     # rradiusrx
]



PositionActuatorParamsV2020 = collections.namedtuple('PositionActuatorParamsV2020', ['name', 'forcerange', 'kp', 'damping'])

_POSITION_ACTUATORS_NUBOTS_V2020 = [
    PositionActuatorParamsV2020('j_left_hip_yaw',         [-200, 200], 200, 10),   # lfemurrz
    PositionActuatorParamsV2020('j_left_hip_roll',        [-200, 200], 200, 10),   # lfemurry
    PositionActuatorParamsV2020('j_left_hip_pitch',       [-300, 300], 300, 15),   # lfemurrx
    PositionActuatorParamsV2020('j_left_knee_pitch',      [-160, 160], 160, 8),    # ltibiarx
    PositionActuatorParamsV2020('j_left_ankle_pitch',     [-120, 120], 120, 6),    # lfootrx
    PositionActuatorParamsV2020('j_left_ankle_roll',      [-50, 50],   50,  3),    # lfootrz

    PositionActuatorParamsV2020('j_right_hip_yaw',        [-200, 200], 200, 10),   # rfemurrz
    PositionActuatorParamsV2020('j_right_hip_roll',       [-200, 200], 200, 10),   # rfemurry
    PositionActuatorParamsV2020('j_right_hip_pitch',      [-300, 300], 300, 15),   # rfemurrx
    PositionActuatorParamsV2020('j_right_knee_pitch',     [-160, 160], 160, 8),    # rtibiarx
    PositionActuatorParamsV2020('j_right_ankle_pitch',    [-120, 120], 120, 6),    # rfootrx
    PositionActuatorParamsV2020('j_right_ankle_roll',     [-50, 50],   50,  3),    # rfootrz

    PositionActuatorParamsV2020('j_neck_yaw',             [-40, 40],   40,  2),    # headrz
    PositionActuatorParamsV2020('j_head_pitch',           [-40, 40],   40,  2),    # headrx

    PositionActuatorParamsV2020('j_left_shoulder_pitch',  [-120, 120], 120, 6),    # lhumerusrx
    PositionActuatorParamsV2020('j_left_shoulder_roll',   [-120, 120], 120, 6),    # lhumerusry
    PositionActuatorParamsV2020('j_left_elbow_pitch',     [-90, 90],   90,  5),    # lradiusrx

    PositionActuatorParamsV2020('j_right_shoulder_pitch', [-120, 120], 120, 6),    # rhumerusrx
    PositionActuatorParamsV2020('j_right_shoulder_roll',  [-120, 120], 120, 6),    # rhumerusry
    PositionActuatorParamsV2020('j_right_elbow_pitch',    [-90, 90],   90,  5),    # rradiusrx
]

# pylint: enable=bad-whitespace

_UPRIGHT_POS = (0.0, 0.0, 0.94)
_UPRIGHT_POS_V2020 = (0.0, 0.0, 1.143)
_UPRIGHT_QUAT = (0.859, 1.0, 1.0, 0.859)

# Height of head above which the humanoid is considered standing.
_STAND_HEIGHT = 1

_TORQUE_THRESHOLD = 60


class _CMUHumanoidBase(legacy_base.Walker_TORQUE_THRESHOLD, metaclass=abc.ABCMeta):
  """The abstract base class for walkers compatible with the CMU humanoid."""

  def _build(self,
             name='walker',
             marker_rgba=None,
             include_face=False,
             initializer=None):
    self._mjcf_root = mjcf.from_path(self._xml_path)
    if name:
      self._mjcf_root.model = name

    # Set corresponding marker color if specified.
    if marker_rgba is not None:
      for geom in self.marker_geoms:
        geom.set_attributes(rgba=marker_rgba)

    self._actuator_order = np.argsort(_CMU_MOCAP_JOINTS)
    self._inverse_order = np.argsort(self._actuator_order)

    super()._build(initializer=initializer)

    if include_face:
      head = self._mjcf_root.find('body', 'head')
      head.add(
          'geom',
          type='capsule',
          name='face',
          size=(0.065, 0.014),
          pos=(0.000341465, 0.048184, 0.01),
          quat=(0.717887, 0.696142, -0.00493334, 0),
          mass=0.,
          contype=0,
          conaffinity=0)

      face_forwardness = head.pos[1]-.02
      head_geom = self._mjcf_root.find('geom', 'head')
      nose_size = head_geom.size[0] / 4.75
      face = head.add(
          'body', name='face', pos=(0.0, 0.039, face_forwardness))
      face.add('geom',
               type='capsule',
               name='nose',
               size=(nose_size, 0.01),
               pos=(0.0, 0.0, 0.0),
               quat=(1, 0.7, 0, 0),
               mass=0.,
               contype=0,
               conaffinity=0,
               group=_WALKER_INVIS_GROUP)

  def _build_observables(self):
    return CMUHumanoidObservables(self)

  @property
  @abc.abstractmethod
  def _xml_path(self):
    raise NotImplementedError

  @composer.cached_property
  def mocap_joints(self):
    return tuple(
        self._mjcf_root.find('joint', name) for name in _CMU_MOCAP_JOINTS)

  @property
  def actuator_order(self):
    """Index of joints from the CMU mocap dataset sorted alphabetically by name.

    Actuators in this walkers are ordered alphabetically by name. This property
    provides a mapping between from actuator ordering to canonical CMU ordering.

    Returns:
      A list of integers corresponding to joint indices from the CMU dataset.
      Specifically, the n-th element in the list is the index of the CMU joint
      index that corresponds to the n-th actuator in this walker.
    """
    return self._actuator_order

  @property
  def actuator_to_joint_order(self):
    """Index of actuators corresponding to each CMU mocap joint.

    Actuators in this walkers are ordered alphabetically by name. This property
    provides a mapping between from canonical CMU ordering to actuator ordering.

    Returns:
      A list of integers corresponding to actuator indices within this walker.
      Specifically, the n-th element in the list is the index of the actuator
      in this walker that corresponds to the n-th joint from the CMU mocap
      dataset.
    """
    return self._inverse_order

  @property
  def upright_pose(self):
    return base.WalkerPose(xpos=_UPRIGHT_POS, xquat=_UPRIGHT_QUAT)

  @property
  def mjcf_model(self):
    return self._mjcf_root

  @composer.cached_property
  def actuators(self):
    return tuple(self._mjcf_root.find_all('actuator'))

  @composer.cached_property
  def root_body(self):
    return self._mjcf_root.find('body', 'root')

  @composer.cached_property
  def head(self):
    return self._mjcf_root.find('body', 'head')

  @composer.cached_property
  def left_arm_root(self):
      return self._mjcf_root.find('body', 'left_shoulder')

  @composer.cached_property
  def right_arm_root(self):
      return self._mjcf_root.find('body', 'right_shoulder')


  @composer.cached_property
  def ground_contact_geoms(self):
    return tuple(self._mjcf_root.find('body', 'left_foot').find_all('geom') +
                 self._mjcf_root.find('body', 'right_foot').find_all('geom'))

  @composer.cached_property
  def standing_height(self):
    return _STAND_HEIGHT

  @composer.cached_property
  def end_effectors(self):
    return (self._mjcf_root.find('body', 'right_lower_arm'),
            self._mjcf_root.find('body', 'left_lower_arm'),
            self._mjcf_root.find('body', 'right_foot'),
            self._mjcf_root.find('body', 'left_foot'))

  @composer.cached_property
  def observable_joints(self):
    return tuple(actuator.joint for actuator in self.actuators
                 if actuator.joint is not None)

  @composer.cached_property
  def bodies(self):
    return tuple(self._mjcf_root.find_all('body'))

  @composer.cached_property
  def mocap_tracking_bodies(self):
    """Collection of bodies for mocap tracking."""
    # remove root body
    root_body = self._mjcf_root.find('body', 'root')
    return tuple(
        b for b in self._mjcf_root.find_all('body') if b != root_body)

  @composer.cached_property
  def egocentric_camera(self):
    return self._mjcf_root.find('camera', 'Legocentric')

  @composer.cached_property
  def body_camera(self):
    return self._mjcf_root.find('camera', 'bodycam')

  @property
  def marker_geoms(self):
    return (self._mjcf_root.find('geom', 'rradius'),
            self._mjcf_root.find('geom', 'lradius'))


class CMUHumanoid(_CMUHumanoidBase):
  """A CMU humanoid walker."""

  @property
  def _xml_path(self):
    return _XML_PATH.format(model_version='2019')


class CMUHumanoidPositionControlled(CMUHumanoid):
  """A position-controlled CMU humanoid with control range scaled to [-1, 1]."""

  def _build(self, model_version='2019', **kwargs):
    self._version = model_version
    if 'scale_default' in kwargs:
      scale_default = kwargs['scale_default']
      del kwargs['scale_default']
    else:
      scale_default = False

    super()._build(**kwargs)

    if scale_default:
      # NOTE: This rescaling doesn't affect the attached hands
      rescale.rescale_humanoid(self, 1.2, 1.2, 1)

    # modify actuators
    if self._version == '2020':
      position_actuators = _POSITION_ACTUATORS_NUBOTS_V2020
    else:
      position_actuators = _POSITION_ACTUATORS_NUBOTS
    self._mjcf_root.default.general.forcelimited = 'true'
    self._mjcf_root.actuator.motor.clear()
    for actuator_params in position_actuators:
      associated_joint = self._mjcf_root.find('joint', actuator_params.name)
      if hasattr(actuator_params, 'damping'):
        associated_joint.damping = actuator_params.damping
      actuator = scaled_actuators.add_position_actuator(
          name=actuator_params.name,
          target=associated_joint,
          kp=actuator_params.kp,
          qposrange=associated_joint.range,
          ctrlrange=(-1, 1),
          forcerange=actuator_params.forcerange)
      if self._version == '2020':
        actuator.dyntype = 'filter'
        actuator.dynprm = [0.030]
    limits = zip(*(actuator.joint.range for actuator in self.actuators))  # pylint: disable=not-an-iterable
    lower, upper = (np.array(limit) for limit in limits)
    self._scale = upper - lower
    self._offset = upper + lower

  @property
  def _xml_path(self):
    return _XML_PATH.format(model_version=self._version)

  def cmu_pose_to_actuation(self, target_pose):
    """Creates the control signal corresponding a CMU mocap joints pose.

    Args:
      target_pose: An array containing the target position for each joint.
        These must be given in "canonical CMU order" rather than "qpos order",
        i.e. the order of `target_pose[self.actuator_order]` should correspond
        to the order of `physics.bind(self.actuators).ctrl`.

    Returns:
      An array of the same shape as `target_pose` containing inputs for position
      controllers. Writing these values into `physics.bind(self.actuators).ctrl`
      will cause the actuators to drive joints towards `target_pose`.
    """
    return (2 * target_pose[self.actuator_order] - self._offset) / self._scale


class CMUHumanoidPositionControlledV2020(CMUHumanoidPositionControlled):
  """A 2020 updated CMU humanoid walker; includes nose for head orientation."""

  def _build(self, **kwargs):
    super()._build(
        model_version='2020', scale_default=True, include_face=True, **kwargs)

  @property
  def upright_pose(self):
    return base.WalkerPose(xpos=_UPRIGHT_POS_V2020, xquat=_UPRIGHT_QUAT)


class CMUHumanoidObservables(legacy_base.WalkerObservables):
  """Observables for the Humanoid."""

  @composer.observable
  def body_camera(self):
    options = mj_wrapper.MjvOption()

    # Don't render this walker's geoms.
    options.geomgroup[_WALKER_GEOM_GROUP] = 0
    return observable.MJCFCamera(
        self._entity.body_camera, width=64, height=64, scene_option=options)

  @composer.observable
  def egocentric_camera(self):
    options = mj_wrapper.MjvOption()

    # Don't render this walker's geoms.
    options.geomgroup[_WALKER_INVIS_GROUP] = 0
    return observable.MJCFCamera(self._entity.egocentric_camera,
                                 width=64, height=64, scene_option=options)

  @composer.observable
  def head_height(self):
    return observable.MJCFFeature('xpos', self._entity.head)[2]

  @composer.observable
  def sensors_torque(self):
    return observable.MJCFFeature(
        'sensordata', self._entity.mjcf_model.sensor.torque,
        corruptor=lambda v, random_state: np.tanh(2 * v / _TORQUE_THRESHOLD))

  @composer.observable
  def actuator_activation(self):
    return observable.MJCFFeature('act',
                                  self._entity.mjcf_model.find_all('actuator'))

  @composer.observable
  def appendages_pos(self):
    """Equivalent to `end_effectors_pos` with the head's position appended."""
    def relative_pos_in_egocentric_frame(physics):
      end_effectors_with_head = (
          self._entity.end_effectors + (self._entity.head,))
      end_effector = physics.bind(end_effectors_with_head).xpos
      torso = physics.bind(self._entity.root_body).xpos
      xmat = np.reshape(physics.bind(self._entity.root_body).xmat, (3, 3))
      return np.reshape(np.dot(end_effector - torso, xmat), -1)
    return observable.Generic(relative_pos_in_egocentric_frame)

  @property
  def proprioception(self):
    return [
        self.joints_pos,
        self.joints_vel,
        self.actuator_activation,
        self.body_height,
        self.end_effectors_pos,
        self.appendages_pos,
        self.world_zaxis
    ] + self._collect_from_attachments('proprioception')

env = nubots_CMU.stand()

walker = None  # Not used, see below
arena = floors.Floor()
task = go_to_target.GoToTarget(walker=None, arena=arena, moving_target=False)
