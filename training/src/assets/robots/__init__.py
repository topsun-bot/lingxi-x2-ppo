"""AgiBot Lingxi X2 Ultra model and actuator recipes."""

from .agibot_x2.x2_ultra_constants import (
  X2_ULTRA_ACTION_SCALE,
  X2_ULTRA_FACTORY_RECORDED_ACTION_SCALE,
  X2_ULTRA_JOINT_NAMES,
  X2_ULTRA_KP090_KD080_ACTION_SCALE,
  get_x2_ultra_factory_recorded_robot_cfg,
  get_x2_ultra_kp090_kd080_robot_cfg,
  get_x2_ultra_robot_cfg,
)

__all__ = [
  "X2_ULTRA_ACTION_SCALE",
  "X2_ULTRA_FACTORY_RECORDED_ACTION_SCALE",
  "X2_ULTRA_JOINT_NAMES",
  "X2_ULTRA_KP090_KD080_ACTION_SCALE",
  "get_x2_ultra_factory_recorded_robot_cfg",
  "get_x2_ultra_kp090_kd080_robot_cfg",
  "get_x2_ultra_robot_cfg",
]
