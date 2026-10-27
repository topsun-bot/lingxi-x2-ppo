"""AgiBot Lingxi X2 Ultra (MJCF v1.3.0) training configuration.

The geometry, inertial parameters, joint limits, and effort limits come from
the official AgiBot X2 model.  Servo gains are simulation defaults and must be
identified on the target robot before a policy is allowed to drive hardware.
"""

from collections import defaultdict
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from src import SRC_PATH

X2_ULTRA_XML: Path = (
  SRC_PATH / "assets" / "robots" / "agibot_x2" / "xmls" / "x2_ultra.xml"
)
assert X2_ULTRA_XML.exists()

# Joint order in the official v1.3.0 MJCF (floating base excluded).  The ONNX
# file also carries the runtime-resolved joint/action names in its metadata.
X2_ULTRA_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_pitch_joint",
  "waist_roll_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_yaw_joint",
  "left_wrist_pitch_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_yaw_joint",
  "right_wrist_pitch_joint",
  "right_wrist_roll_joint",
  "head_yaw_joint",
  "head_pitch_joint",
)


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, X2_ULTRA_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  """Load the official model and normalize it for mjlab.

  The source MJCF contains torque motors and many redundant joint sensors.
  Mjlab creates position servos from the articulation configuration, so the
  source motors are removed.  Only the sensors used by the velocity task are
  retained.  Collision geoms are named here to keep the vendor XML diff small.
  """
  spec = mujoco.MjSpec.from_file(str(X2_ULTRA_XML))
  spec.assets = get_assets(spec.meshdir)

  for actuator in list(spec.actuators):
    spec.delete(actuator)

  required_sensors = {"imu_ang_vel", "imu_lin_vel", "imu_lin_acc", "root_angmom"}
  for sensor in list(spec.sensors):
    if sensor.name not in required_sensors:
      spec.delete(sensor)

  body_geom_count: dict[str, int] = defaultdict(int)
  foot_geom_count = {"left": 0, "right": 0}
  for geom in spec.geoms:
    if geom.group != 3:
      continue
    body_name = geom.parent.name
    if body_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
      side = body_name.split("_", maxsplit=1)[0]
      foot_geom_count[side] += 1
      geom.name = f"{side}_foot{foot_geom_count[side]}_collision"
    elif not geom.name:
      body_geom_count[body_name] += 1
      geom.name = f"{body_name}_{body_geom_count[body_name]}_collision"

  if tuple(foot_geom_count.values()) != (12, 12):
    raise ValueError(f"Unexpected X2 foot collision layout: {foot_geom_count}")
  return spec


def _position_actuator(
  names: tuple[str, ...], stiffness: float, damping: float, effort: float
) -> BuiltinPositionActuatorCfg:
  return BuiltinPositionActuatorCfg(
    target_names_expr=names,
    stiffness=stiffness,
    damping=damping,
    effort_limit=effort,
    armature=0.03,
  )


# Effort limits are from the official v1.3.0 MJCF.  Stiffness and damping are
# conservative simulation values pending system identification on real X2.
X2_ACTUATOR_HIP_KNEE = _position_actuator(
  (".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_joint"),
  stiffness=200.0,
  damping=5.0,
  effort=118.0,
)
X2_ACTUATOR_ANKLE_PITCH = _position_actuator(
  (".*_ankle_pitch_joint",), stiffness=60.0, damping=2.0, effort=36.0
)
X2_ACTUATOR_ANKLE_ROLL = _position_actuator(
  (".*_ankle_roll_joint",), stiffness=40.0, damping=2.0, effort=24.0
)
X2_ACTUATOR_WAIST_YAW = _position_actuator(
  ("waist_yaw_joint",), stiffness=200.0, damping=5.0, effort=118.0
)
X2_ACTUATOR_WAIST_PITCH_ROLL = _position_actuator(
  ("waist_pitch_joint", "waist_roll_joint"),
  stiffness=80.0,
  damping=4.0,
  effort=48.0,
)
X2_ACTUATOR_SHOULDER_PITCH_ROLL = _position_actuator(
  (".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"),
  stiffness=40.0,
  damping=2.0,
  effort=36.0,
)
X2_ACTUATOR_ARM = _position_actuator(
  (".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_yaw_joint"),
  stiffness=30.0,
  damping=1.5,
  effort=24.0,
)
X2_ACTUATOR_WRIST = _position_actuator(
  (".*_wrist_pitch_joint", ".*_wrist_roll_joint"),
  stiffness=8.0,
  damping=0.5,
  effort=2.2,
)
X2_ACTUATOR_HEAD_YAW = _position_actuator(
  ("head_yaw_joint",), stiffness=5.0, damping=0.3, effort=2.6
)
X2_ACTUATOR_HEAD_PITCH = _position_actuator(
  ("head_pitch_joint",), stiffness=2.0, damping=0.2, effort=0.6
)


HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.68),
  joint_pos={
    ".*_hip_pitch_joint": -0.20,
    ".*_knee_joint": 0.40,
    ".*_ankle_pitch_joint": -0.20,
    "left_shoulder_pitch_joint": 0.20,
    "right_shoulder_pitch_joint": 0.20,
    "left_shoulder_roll_joint": 0.15,
    "right_shoulder_roll_joint": -0.15,
    ".*_elbow_joint": -0.60,
  },
  joint_vel={".*": 0.0},
)


FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot(1[0-2]|[1-9])_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot(1[0-2]|[1-9])_collision$": 1},
  friction={r"^(left|right)_foot(1[0-2]|[1-9])_collision$": (0.6,)},
)


X2_ULTRA_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    X2_ACTUATOR_HIP_KNEE,
    X2_ACTUATOR_ANKLE_PITCH,
    X2_ACTUATOR_ANKLE_ROLL,
    X2_ACTUATOR_WAIST_YAW,
    X2_ACTUATOR_WAIST_PITCH_ROLL,
    X2_ACTUATOR_SHOULDER_PITCH_ROLL,
    X2_ACTUATOR_ARM,
    X2_ACTUATOR_WRIST,
    X2_ACTUATOR_HEAD_YAW,
    X2_ACTUATOR_HEAD_PITCH,
  ),
  soft_joint_pos_limit_factor=0.9,
)


# Hardware-validated gain center for the new from-scratch recipe.  Keep this
# articulation separate from the legacy training setup so existing policies
# retain their original deployment contract.
X2_ULTRA_KP090_KD080_KP_SCALE = 0.90
X2_ULTRA_KP090_KD080_KD_SCALE = 0.80


def _scaled_actuator(
  actuator: BuiltinPositionActuatorCfg,
) -> BuiltinPositionActuatorCfg:
  return BuiltinPositionActuatorCfg(
    target_names_expr=actuator.target_names_expr,
    transmission_type=actuator.transmission_type,
    stiffness=actuator.stiffness * X2_ULTRA_KP090_KD080_KP_SCALE,
    damping=actuator.damping * X2_ULTRA_KP090_KD080_KD_SCALE,
    effort_limit=actuator.effort_limit,
    armature=actuator.armature,
    frictionloss=actuator.frictionloss,
  )


X2_ULTRA_KP090_KD080_ARTICULATION = EntityArticulationInfoCfg(
  actuators=tuple(
    _scaled_actuator(actuator) for actuator in X2_ULTRA_ARTICULATION.actuators
  ),
  soft_joint_pos_limit_factor=X2_ULTRA_ARTICULATION.soft_joint_pos_limit_factor,
)


# Leg gains captured from the factory locomotion controller on the real X2
# Ultra.  The capture only exposed the 12 leg-joint gains, so the waist, arm,
# wrist, and head actuators deliberately retain the official simulation
# defaults above.  Keep each leg joint family separate because the factory
# controller does not use one shared hip/knee gain.
X2_ACTUATOR_FACTORY_HIP_PITCH = _position_actuator(
  (".*_hip_pitch_joint",), stiffness=120.0, damping=5.0, effort=118.0
)
X2_ACTUATOR_FACTORY_HIP_ROLL = _position_actuator(
  (".*_hip_roll_joint",), stiffness=100.0, damping=4.0, effort=118.0
)
X2_ACTUATOR_FACTORY_HIP_YAW = _position_actuator(
  (".*_hip_yaw_joint",), stiffness=100.0, damping=4.0, effort=118.0
)
X2_ACTUATOR_FACTORY_KNEE = _position_actuator(
  (".*_knee_joint",), stiffness=150.0, damping=5.0, effort=118.0
)
X2_ACTUATOR_FACTORY_ANKLE_PITCH = _position_actuator(
  (".*_ankle_pitch_joint",), stiffness=40.0, damping=2.0, effort=36.0
)
X2_ACTUATOR_FACTORY_ANKLE_ROLL = _position_actuator(
  (".*_ankle_roll_joint",), stiffness=40.0, damping=2.0, effort=24.0
)

X2_ULTRA_FACTORY_RECORDED_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    X2_ACTUATOR_FACTORY_HIP_PITCH,
    X2_ACTUATOR_FACTORY_HIP_ROLL,
    X2_ACTUATOR_FACTORY_HIP_YAW,
    X2_ACTUATOR_FACTORY_KNEE,
    X2_ACTUATOR_FACTORY_ANKLE_PITCH,
    X2_ACTUATOR_FACTORY_ANKLE_ROLL,
    X2_ACTUATOR_WAIST_YAW,
    X2_ACTUATOR_WAIST_PITCH_ROLL,
    X2_ACTUATOR_SHOULDER_PITCH_ROLL,
    X2_ACTUATOR_ARM,
    X2_ACTUATOR_WRIST,
    X2_ACTUATOR_HEAD_YAW,
    X2_ACTUATOR_HEAD_PITCH,
  ),
  soft_joint_pos_limit_factor=X2_ULTRA_ARTICULATION.soft_joint_pos_limit_factor,
)


def get_x2_ultra_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION_WITHOUT_SELF,),
    spec_fn=get_spec,
    articulation=X2_ULTRA_ARTICULATION,
  )


def get_x2_ultra_kp090_kd080_robot_cfg() -> EntityCfg:
  """Return the v1.3.0 X2 model with the hardware-validated gain center."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION_WITHOUT_SELF,),
    spec_fn=get_spec,
    articulation=X2_ULTRA_KP090_KD080_ARTICULATION,
  )


def get_x2_ultra_factory_recorded_robot_cfg() -> EntityCfg:
  """Return X2 with the leg PD gains recorded from factory locomotion."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION_WITHOUT_SELF,),
    spec_fn=get_spec,
    articulation=X2_ULTRA_FACTORY_RECORDED_ARTICULATION,
  )


def _make_action_scale(
  articulation: EntityArticulationInfoCfg,
) -> dict[str, float]:
  action_scale: dict[str, float] = {}
  for actuator_cfg in articulation.actuators:
    assert isinstance(actuator_cfg, BuiltinPositionActuatorCfg)
    assert actuator_cfg.effort_limit is not None
    for expression in actuator_cfg.target_names_expr:
      action_scale[expression] = (
        0.25 * actuator_cfg.effort_limit / actuator_cfg.stiffness
      )
  return action_scale


X2_ULTRA_ACTION_SCALE = _make_action_scale(X2_ULTRA_ARTICULATION)
X2_ULTRA_KP090_KD080_ACTION_SCALE = _make_action_scale(
  X2_ULTRA_KP090_KD080_ARTICULATION
)
X2_ULTRA_FACTORY_RECORDED_ACTION_SCALE = _make_action_scale(
  X2_ULTRA_FACTORY_RECORDED_ARTICULATION
)
