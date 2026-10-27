"""AgiBot Lingxi X2 Ultra velocity environment configurations."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.assets.robots import (
  X2_ULTRA_ACTION_SCALE,
  X2_ULTRA_FACTORY_RECORDED_ACTION_SCALE,
  X2_ULTRA_KP090_KD080_ACTION_SCALE,
  get_x2_ultra_factory_recorded_robot_cfg,
  get_x2_ultra_kp090_kd080_robot_cfg,
  get_x2_ultra_robot_cfg,
)
from src.tasks.velocity.mdp import (
  BehaviorVelocityCommandCfg,
  feet_swing_height,
  moving_command_relative_velocity_error_l1,
  moving_command_velocity_error_l2,
  yaw_tracking_error_l2,
  zero_command_settling_exp,
  zero_command_velocity_l2,
)
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def agibot_x2_ultra_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create an X2 Ultra rough-terrain velocity task."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 200
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.sim.nconmax = 64
  cfg.scene.entities = {"robot": get_x2_ultra_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision"
    for side in ("left", "right")
    for i in range(1, 13)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg,)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = X2_ULTRA_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.05

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis",)

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.50,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.50,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.08,
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.08,
    r".*shoulder_pitch.*": 0.18,
    r".*shoulder_roll.*": 0.12,
    r".*shoulder_yaw.*": 0.10,
    r".*elbow.*": 0.12,
    r".*wrist.*": 0.06,
    r".*head.*": 0.04,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.60,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.60,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.10,
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.10,
    r".*waist_pitch.*": 0.10,
    r".*shoulder_pitch.*": 0.30,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.08,
    r".*head.*": 0.05,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = (
    "torso_link",
  )
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_clearance"].params["target_height"] = 0.08
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if (
      cfg.scene.terrain is not None
      and cfg.scene.terrain.terrain_generator is not None
    ):
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def agibot_x2_ultra_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create an X2 Ultra flat-terrain velocity task."""
  cfg = agibot_x2_ultra_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def agibot_x2_ultra_kp090_kd080_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Progressive nominal locomotion curriculum for training X2 from scratch."""
  cfg = agibot_x2_ultra_flat_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_x2_ultra_kp090_kd080_robot_cfg()}

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = X2_ULTRA_KP090_KD080_ACTION_SCALE

  # Do not ask a randomly initialized policy to learn full-speed omnidirectional
  # locomotion under pushes and parameter variation at the same time.  Stage 1
  # learns a nominal gait first; later stages add behavior and robustness.
  for event_name in ("push_robot", "foot_friction", "encoder_bias", "base_com"):
    cfg.events.pop(event_name, None)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = False
  twist_cmd.rel_standing_envs = 0.10
  twist_cmd.ranges.lin_vel_x = (-0.20, 0.35)
  twist_cmd.ranges.lin_vel_y = (-0.05, 0.05)
  twist_cmd.ranges.ang_vel_z = (-0.20, 0.20)
  twist_cmd.ranges.heading = None

  if not play:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (-0.20, 0.35),
        "lin_vel_y": (-0.05, 0.05),
        "ang_vel_z": (-0.20, 0.20),
      },
      {
        "step": 1200 * 24,
        "lin_vel_x": (-0.40, 0.70),
        "lin_vel_y": (-0.20, 0.20),
        "ang_vel_z": (-0.40, 0.40),
      },
      {
        "step": 2800 * 24,
        "lin_vel_x": (-0.60, 1.20),
        "lin_vel_y": (-0.40, 0.40),
        "ang_vel_z": (-0.70, 0.70),
      },
      {
        "step": 4200 * 24,
        "lin_vel_x": (-1.00, 2.00),
        "lin_vel_y": (-1.00, 1.00),
        "ang_vel_z": (-1.00, 1.00),
      },
    ]

  cfg.rewards["track_linear_velocity"].weight = 2.0
  cfg.rewards["moving_command_velocity_error"] = RewardTermCfg(
    func=moving_command_velocity_error_l2,
    weight=-1.0,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "yaw_scale": 0.5,
    },
  )
  return cfg


def use_x2_ultra_factory_recorded_leg_gains(
  cfg: ManagerBasedRlEnvCfg,
) -> ManagerBasedRlEnvCfg:
  """Replace an X2 task's actuator recipe with factory-recorded leg gains.

  This small adapter lets every proven locomotion curriculum be reused without
  duplicating its command and reward configuration.  The action scale is
  recomputed from each actuator's Kp and effort limit.
  """
  cfg.scene.entities = {"robot": get_x2_ultra_factory_recorded_robot_cfg()}
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = X2_ULTRA_FACTORY_RECORDED_ACTION_SCALE
  return cfg


def agibot_x2_ultra_kp090_kd080_gait_bootstrap_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Break the stable-standing local optimum with sagittal gait training."""
  cfg = agibot_x2_ultra_kp090_kd080_flat_env_cfg(play=play)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_standing_envs = 0.10
  twist_cmd.ranges.lin_vel_x = (-0.25, 0.50)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  cfg.curriculum = {}

  cfg.rewards["track_linear_velocity"].weight = 4.0
  cfg.rewards["track_angular_velocity"].weight = 0.5
  cfg.rewards["pose"].weight = 0.25
  cfg.rewards["foot_gait"].weight = 1.0
  cfg.rewards["is_terminated"].weight = -100.0
  cfg.rewards["action_rate_l2"].weight = -0.02
  cfg.rewards["stand_still"].weight = -0.5
  cfg.rewards["moving_command_velocity_error"].weight = -4.0
  cfg.rewards["moving_command_relative_velocity_error"] = RewardTermCfg(
    func=moving_command_relative_velocity_error_l1,
    weight=-3.0,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "cross_track_scale": 1.0,
    },
  )
  return cfg


def agibot_x2_ultra_behavior_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the flat task used for stop/restart and direction-change training."""
  cfg = agibot_x2_ultra_flat_env_cfg(play=play)

  twist_cmd = BehaviorVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(2.0, 30.0),
    heading_command=False,
    debug_vis=True,
    ranges=BehaviorVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 2.0),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-1.0, 1.0),
      heading=None,
    ),
  )
  twist_cmd.viz.z_offset = 1.05
  cfg.commands["twist"] = twist_cmd

  # Long episodes let the structured sampler expose both prolonged standing and
  # complete stop/restart sequences in the same rollout.
  if not play:
    cfg.episode_length_s = 60.0
  cfg.curriculum.pop("command_vel", None)

  cfg.rewards["zero_command_velocity"] = RewardTermCfg(
    func=zero_command_velocity_l2,
    weight=-1.0,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "angular_scale": 0.25,
    },
  )
  cfg.rewards["moving_command_velocity_error"] = RewardTermCfg(
    func=moving_command_velocity_error_l2,
    weight=-0.5,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "yaw_scale": 0.5,
    },
  )
  # Foot slip matters during stationary phases as well as during locomotion.
  cfg.rewards["foot_slip"].params["command_threshold"] = -1.0

  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """From-scratch behavior task centered on the hardware-validated PD gains."""
  cfg = agibot_x2_ultra_behavior_flat_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_x2_ultra_kp090_kd080_robot_cfg()}

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = X2_ULTRA_KP090_KD080_ACTION_SCALE
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_sagittal_stabilize_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Stabilize a bootstrapped sagittal gait before omni-directional training."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_flat_env_cfg(play=play)

  # Keep this bridge stage nominal.  It teaches sustained walking, braking,
  # restarting, and long standing before physics and gain randomization.
  for event_name in ("push_robot", "foot_friction", "encoder_bias", "base_com"):
    cfg.events.pop(event_name, None)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-0.20, 0.35)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.standing_time_range = (2.0, 6.0)
  twist_cmd.moving_time_range = (3.0, 6.0)
  twist_cmd.initial_standing_probability = 0.35
  twist_cmd.stop_probability = 0.55
  twist_cmd.reverse_probability = 0.10
  twist_cmd.walk_speed_range = (0.15, 0.35)
  twist_cmd.backward_speed_range = (0.12, 0.20)
  twist_cmd.mode_weights = (0.60, 0.40, 0.0, 0.0, 0.0, 0.0)

  # The gait-bootstrap stage intentionally overweights velocity tracking.  This
  # stage restores a large fall penalty and zero-command accuracy without
  # removing the learned stepping response.
  cfg.rewards["track_linear_velocity"].weight = 3.0
  cfg.rewards["track_angular_velocity"].weight = 1.0
  cfg.rewards["pose"].weight = 0.5
  cfg.rewards["is_terminated"].weight = -250.0
  cfg.rewards["action_rate_l2"].weight = -0.03
  cfg.rewards["foot_gait"].weight = 0.75
  cfg.rewards["zero_command_velocity"].weight = -2.0
  cfg.rewards["moving_command_velocity_error"].weight = -1.5
  cfg.rewards["moving_command_relative_velocity_error"] = RewardTermCfg(
    func=moving_command_relative_velocity_error_l1,
    weight=-1.0,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "cross_track_scale": 1.0,
    },
  )
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_backward_stabilize_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Recover low-speed reverse walking without forgetting stops or forward gait."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_sagittal_stabilize_env_cfg(
    play=play
  )

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.stop_probability = 0.50
  twist_cmd.reverse_probability = 0.05
  twist_cmd.backward_speed_range = (0.15, 0.25)
  twist_cmd.mode_weights = (0.25, 0.75, 0.0, 0.0, 0.0, 0.0)

  cfg.rewards["track_linear_velocity"].weight = 4.0
  cfg.rewards["track_angular_velocity"].weight = 2.0
  cfg.rewards["body_orientation_l2"].weight = -2.0
  cfg.rewards["is_terminated"].weight = -300.0
  cfg.rewards["moving_command_velocity_error"].weight = -3.0
  cfg.rewards["moving_command_relative_velocity_error"].weight = -3.0
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_sagittal_balance_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Balance bidirectional speed while reducing tilt and unintended yaw."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_backward_stabilize_env_cfg(
    play=play
  )

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.backward_speed_range = (0.12, 0.20)
  twist_cmd.mode_weights = (0.50, 0.50, 0.0, 0.0, 0.0, 0.0)

  cfg.rewards["track_linear_velocity"].weight = 3.0
  cfg.rewards["track_angular_velocity"].weight = 4.0
  cfg.rewards["body_orientation_l2"].weight = -4.0
  cfg.rewards["body_ang_vel"].weight = -0.10
  cfg.rewards["angular_momentum"].weight = -0.04
  cfg.rewards["action_rate_l2"].weight = -0.05
  cfg.rewards["moving_command_velocity_error"].weight = -2.0
  cfg.rewards["moving_command_relative_velocity_error"].weight = -1.5
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_omni_low_speed_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Introduce low-speed lateral, turn, and mixed commands after sagittal pass."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_sagittal_balance_env_cfg(
    play=play
  )

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_y = (-0.20, 0.20)
  twist_cmd.ranges.ang_vel_z = (-0.30, 0.30)
  twist_cmd.lateral_speed_range = (0.08, 0.20)
  twist_cmd.turn_speed_range = (0.10, 0.30)
  twist_cmd.mixed_lateral_scale = 0.60
  twist_cmd.mixed_yaw_scale = 0.60
  twist_cmd.mode_weights = (0.28, 0.22, 0.20, 0.18, 0.0, 0.12)

  cfg.rewards["track_linear_velocity"].weight = 4.0
  cfg.rewards["is_terminated"].weight = -350.0
  cfg.rewards["moving_command_velocity_error"].weight = -3.0
  cfg.rewards["moving_command_relative_velocity_error"].weight = -2.5
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_lateral_bootstrap_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Bootstrap symmetric low-speed lateral stepping from sagittal balance."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_sagittal_balance_env_cfg(
    play=play
  )

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_y = (-0.25, 0.25)
  twist_cmd.lateral_speed_range = (0.12, 0.25)
  twist_cmd.stop_probability = 0.45
  twist_cmd.mode_weights = (0.10, 0.10, 0.80, 0.0, 0.0, 0.0)

  cfg.rewards["track_linear_velocity"].weight = 5.0
  cfg.rewards["pose"].weight = 0.35
  cfg.rewards["body_orientation_l2"].weight = -3.0
  cfg.rewards["is_terminated"].weight = -400.0
  cfg.rewards["moving_command_velocity_error"].weight = -4.0
  cfg.rewards["moving_command_relative_velocity_error"].weight = -4.0
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_lateral_balance_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Reduce lateral body tilt after symmetric side-stepping has emerged."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_lateral_bootstrap_env_cfg(
    play=play
  )

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.lateral_speed_range = (0.10, 0.20)
  twist_cmd.mode_weights = (0.15, 0.15, 0.70, 0.0, 0.0, 0.0)

  cfg.rewards["track_linear_velocity"].weight = 4.0
  cfg.rewards["track_angular_velocity"].weight = 5.0
  cfg.rewards["pose"].weight = 0.60
  cfg.rewards["body_orientation_l2"].weight = -8.0
  cfg.rewards["body_ang_vel"].weight = -0.15
  cfg.rewards["angular_momentum"].weight = -0.06
  cfg.rewards["action_rate_l2"].weight = -0.08
  cfg.rewards["is_terminated"].weight = -500.0
  cfg.rewards["moving_command_velocity_error"].weight = -3.0
  cfg.rewards["moving_command_relative_velocity_error"].weight = -2.5
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_turn_bootstrap_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Bootstrap symmetric low-speed turning after lateral balance passes."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_lateral_balance_env_cfg(
    play=play
  )

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.ranges.ang_vel_z = (-0.35, 0.35)
  twist_cmd.turn_speed_range = (0.15, 0.30)
  twist_cmd.mode_weights = (0.10, 0.10, 0.10, 0.70, 0.0, 0.0)

  cfg.rewards["track_angular_velocity"].weight = 6.0
  cfg.rewards["body_orientation_l2"].weight = -6.0
  cfg.rewards["is_terminated"].weight = -500.0
  cfg.rewards["moving_command_velocity_error"].weight = -4.0
  cfg.rewards["moving_command_relative_velocity_error"].weight = -4.0
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_turn_balance_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Fine-balance an established turn policy with extra right-turn replay."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_turn_bootstrap_env_cfg(
    play=play
  )

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.turn_positive_probability = 0.35
  twist_cmd.mode_weights = (0.05, 0.05, 0.05, 0.85, 0.0, 0.0)

  cfg.rewards["track_angular_velocity"].weight = 8.0
  cfg.rewards["is_terminated"].weight = -500.0
  cfg.rewards["moving_command_relative_velocity_error"].weight = -5.0
  return cfg


def agibot_x2_ultra_kp090_kd080_turn_seeded_bootstrap_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Seed pure-yaw motion in simulation to break the no-turn local optimum."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_turn_bootstrap_env_cfg(
    play=play
  )

  twist_cmd = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 6.0),
    heading_command=False,
    rel_standing_envs=0.10,
    init_velocity_prob=0.50 if not play else 0.0,
    debug_vis=True,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.0, 0.0),
      lin_vel_y=(0.0, 0.0),
      ang_vel_z=(-0.40, 0.40),
      heading=None,
    ),
  )
  twist_cmd.viz.z_offset = 1.05
  cfg.commands["twist"] = twist_cmd
  cfg.curriculum = {}
  return cfg


def agibot_x2_ultra_kp090_kd080_turn_gait_bootstrap_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Aggressively discover a turning gait before restoring stability costs."""
  cfg = agibot_x2_ultra_kp090_kd080_turn_seeded_bootstrap_env_cfg(
    play=play
  )

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.init_velocity_prob = 0.25 if not play else 0.0

  cfg.rewards["track_linear_velocity"].weight = 1.0
  cfg.rewards["track_angular_velocity"].weight = 8.0
  cfg.rewards["pose"].weight = 0.25
  cfg.rewards["body_orientation_l2"].weight = -1.0
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.025
  cfg.rewards["is_terminated"].weight = -100.0
  cfg.rewards["action_rate_l2"].weight = -0.02
  cfg.rewards["foot_gait"].weight = 1.0
  cfg.rewards["moving_command_velocity_error"].weight = -4.0
  cfg.rewards["moving_command_relative_velocity_error"].weight = -6.0
  return cfg


def agibot_x2_ultra_kp090_kd080_arc_turn_bootstrap_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Learn yaw control while walking forward before attempting pivot turns."""
  cfg = agibot_x2_ultra_kp090_kd080_turn_gait_bootstrap_env_cfg(
    play=play
  )

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_standing_envs = 0.05
  twist_cmd.init_velocity_prob = 0.25 if not play else 0.0
  twist_cmd.ranges.lin_vel_x = (0.15, 0.35)
  twist_cmd.ranges.ang_vel_z = (-0.40, 0.40)

  cfg.rewards["track_linear_velocity"].weight = 4.0
  cfg.rewards["track_angular_velocity"].weight = 8.0
  cfg.rewards["body_orientation_l2"].weight = -2.0
  cfg.rewards["is_terminated"].weight = -200.0
  cfg.rewards["moving_command_relative_velocity_error"].weight = -4.0
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_omni_rehearsal_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Rehearse all low-speed axes evenly to consolidate learned behaviors."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_omni_low_speed_env_cfg(
    play=play
  )

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.initial_standing_probability = 0.30
  twist_cmd.stop_probability = 0.50
  twist_cmd.reverse_probability = 0.05
  twist_cmd.mode_weights = (0.25, 0.25, 0.25, 0.25, 0.0, 0.0)

  cfg.rewards["track_linear_velocity"].weight = 4.0
  cfg.rewards["track_angular_velocity"].weight = 6.0
  cfg.rewards["pose"].weight = 0.50
  cfg.rewards["body_orientation_l2"].weight = -6.0
  cfg.rewards["body_ang_vel"].weight = -0.12
  cfg.rewards["angular_momentum"].weight = -0.05
  cfg.rewards["action_rate_l2"].weight = -0.06
  cfg.rewards["is_terminated"].weight = -500.0
  cfg.rewards["moving_command_velocity_error"].weight = -4.0
  cfg.rewards["moving_command_relative_velocity_error"].weight = -3.0
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_omni_consolidate_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Consolidate weak low-speed directions with biased replay and strong tracking."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_omni_rehearsal_env_cfg(
    play=play
  )

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.initial_standing_probability = 0.25
  twist_cmd.stop_probability = 0.45
  twist_cmd.walk_speed_range = (0.20, 0.35)
  twist_cmd.backward_speed_range = (0.15, 0.20)
  twist_cmd.lateral_speed_range = (0.12, 0.20)
  twist_cmd.turn_speed_range = (0.15, 0.30)
  twist_cmd.lateral_positive_probability = 0.30
  twist_cmd.turn_positive_probability = 0.60
  twist_cmd.mode_weights = (0.10, 0.30, 0.30, 0.30, 0.0, 0.0)

  cfg.rewards["track_linear_velocity"].weight = 5.0
  cfg.rewards["track_angular_velocity"].weight = 8.0
  cfg.rewards["pose"].weight = 0.35
  cfg.rewards["body_orientation_l2"].weight = -4.0
  cfg.rewards["action_rate_l2"].weight = -0.04
  cfg.rewards["foot_gait"].weight = 1.0
  cfg.rewards["is_terminated"].weight = -400.0
  cfg.rewards["moving_command_velocity_error"].weight = -5.0
  cfg.rewards["moving_command_relative_velocity_error"].weight = -5.0
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_omni_finalize_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Gently close the remaining right-turn gap without forgetting other axes."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_omni_rehearsal_env_cfg(
    play=play
  )

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.initial_standing_probability = 0.30
  twist_cmd.stop_probability = 0.55
  twist_cmd.reverse_probability = 0.05
  twist_cmd.walk_speed_range = (0.20, 0.35)
  twist_cmd.backward_speed_range = (0.15, 0.20)
  twist_cmd.lateral_speed_range = (0.12, 0.20)
  twist_cmd.turn_speed_range = (0.18, 0.30)
  twist_cmd.lateral_positive_probability = 0.50
  # The distilled seed is already within 0.006 rad/s of the right-turn gate.
  # A modest right bias closes that gap while all four learned modes are replayed.
  twist_cmd.turn_positive_probability = 0.30
  twist_cmd.mode_weights = (0.23, 0.23, 0.20, 0.34, 0.0, 0.0)

  cfg.rewards["track_linear_velocity"].weight = 5.0
  cfg.rewards["track_angular_velocity"].weight = 8.0
  cfg.rewards["body_orientation_l2"].weight = -7.0
  cfg.rewards["body_ang_vel"].weight = -0.15
  cfg.rewards["angular_momentum"].weight = -0.06
  cfg.rewards["action_rate_l2"].weight = -0.07
  cfg.rewards["is_terminated"].weight = -600.0
  cfg.rewards["moving_command_velocity_error"].weight = -4.0
  cfg.rewards["moving_command_relative_velocity_error"].weight = -4.0
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_full_speed_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Train one policy across the complete locomotion command envelope.

  The policy sees the complete command range during training.  Any conservative
  speed cap used on hardware belongs to the joystick bridge, not this task or
  the exported actor.
  """
  cfg = agibot_x2_ultra_behavior_kp090_kd080_flat_env_cfg(play=play)

  # First adapt the established full-speed gait to the selected PD center in a
  # nominal environment.  Parameter and delay randomization are added only
  # after nominal speed tracking and gait quality have passed.
  for event_name in ("push_robot", "foot_friction", "encoder_bias", "base_com"):
    cfg.events.pop(event_name, None)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-1.0, 2.0)
  twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)
  twist_cmd.standing_time_range = (2.0, 8.0)
  twist_cmd.moving_time_range = (3.0, 8.0)
  twist_cmd.initial_standing_probability = 0.15
  twist_cmd.stop_probability = 0.25
  twist_cmd.reverse_probability = 0.20
  twist_cmd.walk_speed_range = (0.35, 1.20)
  twist_cmd.backward_speed_range = (0.30, 0.80)
  twist_cmd.lateral_speed_range = (0.25, 0.80)
  twist_cmd.turn_speed_range = (0.30, 1.00)
  twist_cmd.run_speed_range = (1.30, 2.00)
  twist_cmd.mixed_lateral_scale = 0.60
  twist_cmd.mixed_yaw_scale = 0.60
  twist_cmd.lateral_positive_probability = 0.50
  twist_cmd.turn_positive_probability = 0.50
  twist_cmd.mixed_forward_probability = 0.75
  twist_cmd.mode_weights = (0.25, 0.16, 0.16, 0.14, 0.18, 0.11)

  cfg.rewards["track_linear_velocity"].weight = 4.0
  cfg.rewards["track_angular_velocity"].weight = 4.0
  cfg.rewards["pose"].weight = 0.35
  cfg.rewards["body_orientation_l2"].weight = -2.0
  cfg.rewards["body_ang_vel"].weight = -0.08
  cfg.rewards["angular_momentum"].weight = -0.03
  cfg.rewards["is_terminated"].weight = -350.0
  cfg.rewards["action_rate_l2"].weight = -0.035
  cfg.rewards["foot_gait"].weight = 1.25
  cfg.rewards["foot_clearance"].weight = -0.75
  cfg.rewards["foot_slip"].weight = -0.20
  cfg.rewards["soft_landing"].weight = -0.0005
  cfg.rewards["stand_still"].weight = -1.0
  cfg.rewards["zero_command_velocity"].weight = -1.5
  cfg.rewards["moving_command_velocity_error"].weight = -2.5
  cfg.rewards["moving_command_relative_velocity_error"] = RewardTermCfg(
    func=moving_command_relative_velocity_error_l1,
    weight=-2.0,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "cross_track_scale": 0.75,
    },
  )
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_full_speed_brake_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Replay full-speed motion with extra stop, restart and reversal coverage."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_full_speed_env_cfg(play=play)
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.standing_time_range = (1.5, 5.0)
  twist_cmd.moving_time_range = (3.0, 6.0)
  twist_cmd.initial_standing_probability = 0.20
  twist_cmd.stop_probability = 0.50
  twist_cmd.reverse_probability = 0.30
  twist_cmd.mode_weights = (0.23, 0.20, 0.15, 0.13, 0.18, 0.11)

  cfg.rewards["is_terminated"].weight = -500.0
  cfg.rewards["zero_command_velocity"].weight = -3.0
  cfg.rewards["stand_still"].weight = -1.5
  cfg.rewards["action_rate_l2"].weight = -0.05
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_full_speed_gait_refine_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Refine lateral foot clearance while replaying the full-speed envelope."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_full_speed_env_cfg(play=play)
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)

  # The full-duration gate found an asymmetric tap/low-clearance right foot
  # only for positive-y motion.  Bias that case without dropping sagittal,
  # turning, running, mixed-motion, stop, or reversal replay.
  twist_cmd.lateral_speed_range = (0.40, 0.80)
  twist_cmd.lateral_positive_probability = 0.70
  twist_cmd.mode_weights = (0.18, 0.14, 0.30, 0.12, 0.16, 0.10)

  cfg.rewards["foot_gait"].weight = 2.0
  cfg.rewards["foot_clearance"].weight = -1.20
  cfg.rewards["action_rate_l2"].weight = -0.045
  cfg.rewards["moving_command_velocity_error"].weight = -3.0
  cfg.rewards["moving_command_relative_velocity_error"].weight = -2.5
  return cfg


def agibot_x2_ultra_behavior_factory_recorded_full_speed_gait_refine_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Symmetrically refine high-speed gait for the factory-recorded leg gains.

  With the lower factory hip/ankle Kp and its correspondingly larger action
  scale, both lateral directions can converge to a one-leg tap pattern.  The
  older gain recipe only needed positive-y replay; this recipe must train both
  signs equally and weight the alternating-contact schedule more strongly.
  """
  cfg = agibot_x2_ultra_behavior_kp090_kd080_full_speed_gait_refine_env_cfg(
    play=play
  )
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.lateral_positive_probability = 0.50
  twist_cmd.stop_probability = 0.35
  twist_cmd.mode_weights = (0.14, 0.12, 0.40, 0.10, 0.14, 0.10)

  cfg.rewards["foot_gait"].weight = 5.0
  cfg.rewards["foot_clearance"].weight = -2.0
  cfg.rewards["is_terminated"].weight = -500.0
  cfg.rewards["action_rate_l2"].weight = -0.06
  return cfg


def agibot_x2_ultra_behavior_factory_recorded_right_gait_refine_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Raise the low-clearance left swing foot during negative-y locomotion."""
  cfg = agibot_x2_ultra_behavior_factory_recorded_full_speed_gait_refine_env_cfg(
    play=play
  )
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.lateral_positive_probability = 0.20
  twist_cmd.stop_probability = 0.40
  twist_cmd.mode_weights = (0.12, 0.10, 0.50, 0.08, 0.12, 0.08)

  cfg.rewards["foot_gait"].weight = 6.0
  cfg.rewards["foot_clearance"].weight = -3.0
  return cfg


def agibot_x2_ultra_behavior_factory_recorded_right_gait_finish_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Close the final negative-y swing-clearance gap without a long fine-tune."""
  cfg = agibot_x2_ultra_behavior_factory_recorded_right_gait_refine_env_cfg(
    play=play
  )
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.lateral_positive_probability = 0.10
  twist_cmd.stop_probability = 0.50
  twist_cmd.mode_weights = (0.08, 0.06, 0.66, 0.06, 0.08, 0.06)

  cfg.rewards["foot_gait"].weight = 4.0
  cfg.rewards["foot_clearance"].weight = -6.0
  cfg.rewards["pose"].weight = 0.20
  cfg.rewards["zero_command_velocity"].weight = -3.0
  cfg.rewards["stand_still"].weight = -1.5
  cfg.rewards["action_rate_l2"].weight = -0.04
  return cfg


def agibot_x2_ultra_behavior_robust_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Behavior fine-tuning with X2 sim-to-real disturbance randomization."""
  cfg = agibot_x2_ultra_behavior_flat_env_cfg(play=play)

  if play:
    # Evaluation is nominal unless a dedicated robustness evaluator injects
    # disturbances explicitly.
    cfg.events.pop("push_robot", None)
    return cfg

  # Replace the instantaneous velocity teleport with a short physical wrench
  # on the pelvis.  The first fine-tuning stage is intentionally moderate.
  cfg.events["push_robot"] = EventTermCfg(
    func=envs_mdp.apply_body_impulse,
    mode="step",
    params={
      "force_range": (-80.0, 80.0),
      "torque_range": (-10.0, 10.0),
      "duration_s": (0.05, 0.12),
      "cooldown_s": (3.0, 7.0),
      "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",)),
      "body_point_offset": (0.0, 0.0, 0.15),
    },
  )

  cfg.events["pd_gains"] = EventTermCfg(
    func=dr.pd_gains,
    mode="startup",
    params={
      "kp_range": (0.90, 1.10),
      "kd_range": (0.85, 1.15),
      "operation": "scale",
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.events["effort_limits"] = EventTermCfg(
    func=dr.effort_limits,
    mode="startup",
    params={
      "effort_limit_range": (0.90, 1.05),
      "operation": "scale",
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  # Apply the physically consistent mass/inertia perturbation before the
  # existing pelvis CoM offset so both randomizations remain effective.
  base_com_event = cfg.events.pop("base_com")
  cfg.events["pseudo_inertia"] = EventTermCfg(
    func=dr.pseudo_inertia,
    mode="startup",
    params={
      "alpha_range": (-0.04, 0.04),
      "d_range": (-0.02, 0.02),
      "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    },
  )
  cfg.events["base_com"] = base_com_event
  cfg.events["joint_friction"] = EventTermCfg(
    func=dr.joint_friction,
    mode="startup",
    params={
      "ranges": (0.0, 0.15),
      "operation": "abs",
      "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    },
  )

  # Sensor latency is measured in policy/control steps (20 ms each).
  for term_name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
    term = cfg.observations["actor"].terms[term_name]
    term.delay_min_lag = 0
    term.delay_max_lag = 2
    term.delay_hold_prob = 0.8
    term.delay_update_period = 5

  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_robust_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Robust stage around Kp x0.90 and Kd x0.80 after nominal convergence."""
  cfg = agibot_x2_ultra_behavior_robust_flat_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_x2_ultra_kp090_kd080_robot_cfg()}

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = X2_ULTRA_KP090_KD080_ACTION_SCALE

  if not play:
    # Absolute target ranges relative to the legacy setup are Kp 0.81-0.99
    # and Kd 0.70-0.90.  These scale factors are relative to the new center.
    cfg.events["pd_gains"].params["kp_range"] = (0.90, 1.10)
    cfg.events["pd_gains"].params["kd_range"] = (0.875, 1.125)
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_robust_omni_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Robust low-speed omni task centered on the recommended X2 gains."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_omni_rehearsal_env_cfg(
    play=play
  )
  if play:
    return cfg

  # Reuse the audited sim-to-real event definitions while retaining the exact
  # command distribution, reward and recommended-gain robot from omni rehearsal.
  robust_template = agibot_x2_ultra_behavior_robust_flat_env_cfg(play=False)
  randomized_events = (
    "push_robot",
    "pd_gains",
    "effort_limits",
    "pseudo_inertia",
    "base_com",
    "joint_friction",
    "foot_friction",
    "encoder_bias",
  )
  for event_name in randomized_events:
    cfg.events[event_name] = robust_template.events[event_name]

  # These multipliers are relative to the new Kp x0.90 / Kd x0.80 center.
  cfg.events["pd_gains"].params["kp_range"] = (0.90, 1.10)
  cfg.events["pd_gains"].params["kd_range"] = (0.875, 1.125)
  for term_name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
    source = robust_template.observations["actor"].terms[term_name]
    target = cfg.observations["actor"].terms[term_name]
    target.delay_min_lag = source.delay_min_lag
    target.delay_max_lag = source.delay_max_lag
    target.delay_hold_prob = source.delay_hold_prob
    target.delay_update_period = source.delay_update_period
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_robust_mild_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """First robust curriculum stage with mild parameter and sensor variation."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_robust_omni_env_cfg(play=play)
  if play:
    return cfg

  cfg.events.pop("push_robot", None)
  cfg.events["pd_gains"].params["kp_range"] = (0.97, 1.03)
  cfg.events["pd_gains"].params["kd_range"] = (0.95, 1.05)
  cfg.events["effort_limits"].params["effort_limit_range"] = (0.97, 1.03)
  cfg.events["pseudo_inertia"].params["alpha_range"] = (-0.015, 0.015)
  cfg.events["pseudo_inertia"].params["d_range"] = (-0.005, 0.005)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.01, 0.01),
    1: (-0.01, 0.01),
    2: (-0.01, 0.01),
  }
  cfg.events["joint_friction"].params["ranges"] = (0.0, 0.05)
  cfg.events["foot_friction"].params["ranges"] = (0.70, 1.20)
  cfg.events["encoder_bias"].params["bias_range"] = (-0.005, 0.005)
  for term_name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
    term = cfg.observations["actor"].terms[term_name]
    term.delay_max_lag = 1
    term.delay_hold_prob = 0.5
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_full_speed_robust_mild_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Full-speed command coverage with the audited mild sim-to-real variation."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_full_speed_env_cfg(play=play)
  if play:
    return cfg

  robust_template = agibot_x2_ultra_behavior_kp090_kd080_robust_mild_env_cfg(
    play=False
  )
  randomized_events = (
    "pd_gains",
    "effort_limits",
    "pseudo_inertia",
    "base_com",
    "joint_friction",
    "foot_friction",
    "encoder_bias",
  )
  for event_name in randomized_events:
    cfg.events[event_name] = robust_template.events[event_name]
  for term_name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
    source = robust_template.observations["actor"].terms[term_name]
    target = cfg.observations["actor"].terms[term_name]
    target.delay_min_lag = source.delay_min_lag
    target.delay_max_lag = source.delay_max_lag
    target.delay_hold_prob = source.delay_hold_prob
    target.delay_update_period = source.delay_update_period
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_history2_full_envelope_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Two-frame, full command envelope with mild sim-to-real randomization."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_full_speed_robust_mild_env_cfg(
    play=play
  )
  cfg.observations["actor"].history_length = 2

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.standing_time_range = (1.5, 5.0)
  twist_cmd.moving_time_range = (3.0, 7.0)
  twist_cmd.initial_standing_probability = 0.20
  twist_cmd.stop_probability = 0.35
  twist_cmd.reverse_probability = 0.20
  twist_cmd.walk_speed_range = (0.20, 1.20)
  twist_cmd.backward_speed_range = (0.15, 0.80)
  twist_cmd.lateral_speed_range = (0.12, 0.80)
  twist_cmd.turn_speed_range = (0.15, 1.00)
  twist_cmd.run_speed_range = (1.30, 2.00)
  twist_cmd.mode_weights = (0.24, 0.15, 0.15, 0.14, 0.20, 0.12)

  cfg.rewards["track_linear_velocity"].weight = 5.0
  cfg.rewards["track_angular_velocity"].weight = 6.0
  cfg.rewards["yaw_tracking_error"] = RewardTermCfg(
    func=yaw_tracking_error_l2,
    weight=-1.5,
    params={"command_name": "twist"},
  )
  cfg.rewards["body_orientation_l2"].weight = -4.0
  cfg.rewards["is_terminated"].weight = -600.0
  cfg.rewards["foot_gait"].weight = 4.0
  cfg.rewards["foot_clearance"].weight = -4.0
  cfg.rewards["foot_clearance"].params["target_height"] = 0.07
  cfg.rewards["action_rate_l2"].weight = -0.04
  cfg.rewards["zero_command_velocity"].weight = -3.0
  cfg.rewards["stand_still"].weight = -1.5
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_full_speed_leg_robust_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Full-speed replay with independent leg/load and sensor uncertainty.

  This stage targets real-hardware leg compliance without narrowing the command
  envelope or changing the trained action scale.  Randomization is independent
  across actuators and feet so either leg can become the weak/load-bearing side.
  """
  cfg = agibot_x2_ultra_behavior_kp090_kd080_full_speed_robust_mild_env_cfg(
    play=play
  )
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.standing_time_range = (1.5, 4.0)
  twist_cmd.moving_time_range = (3.0, 7.0)
  twist_cmd.initial_standing_probability = 0.25
  twist_cmd.stop_probability = 0.35

  cfg.rewards["body_orientation_l2"].weight = -4.0
  cfg.rewards["is_terminated"].weight = -500.0
  cfg.rewards["stand_still"].weight = -1.5
  cfg.rewards["zero_command_velocity"].weight = -3.0
  cfg.rewards["action_rate_l2"].weight = -0.04

  if not play:
    cfg.events["pd_gains"].params["kp_range"] = (0.85, 1.10)
    cfg.events["pd_gains"].params["kd_range"] = (0.85, 1.15)
    cfg.events["effort_limits"].params["effort_limit_range"] = (0.90, 1.05)
    cfg.events["pseudo_inertia"].params["alpha_range"] = (-0.025, 0.025)
    cfg.events["pseudo_inertia"].params["d_range"] = (-0.01, 0.01)
    cfg.events["base_com"].params["ranges"] = {
      0: (-0.015, 0.015),
      1: (-0.04, 0.04),
      2: (-0.015, 0.015),
    }
    cfg.events["joint_friction"].params["ranges"] = (0.0, 0.10)
    cfg.events["encoder_bias"].params["bias_range"] = (-0.015, 0.015)
    for term_name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
      term = cfg.observations["actor"].terms[term_name]
      term.delay_max_lag = 2
      term.delay_hold_prob = 0.70
      term.delay_update_period = 5

    cfg.events.pop("foot_friction", None)
    for side in ("left", "right"):
      cfg.events[f"{side}_foot_friction"] = EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
          "asset_cfg": SceneEntityCfg(
            "robot",
            geom_names=tuple(
              f"{side}_foot{i}_collision" for i in range(1, 13)
            ),
          ),
          "operation": "abs",
          "ranges": (0.50, 1.40),
          "shared_random": True,
        },
      )

  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_full_speed_robust_finalize_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Close robust stand, stop, and right-turn gait gaps with full-speed replay."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_full_speed_robust_mild_env_cfg(
    play=play
  )
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.standing_time_range = (1.5, 5.0)
  twist_cmd.moving_time_range = (3.0, 6.0)
  twist_cmd.initial_standing_probability = 0.25
  twist_cmd.stop_probability = 0.55
  twist_cmd.reverse_probability = 0.25
  twist_cmd.turn_positive_probability = 0.35
  twist_cmd.mode_weights = (0.20, 0.17, 0.15, 0.20, 0.16, 0.12)

  cfg.rewards["track_angular_velocity"].weight = 5.0
  cfg.rewards["foot_gait"].weight = 1.75
  cfg.rewards["foot_clearance"].weight = -1.0
  cfg.rewards["zero_command_velocity"].weight = -4.0
  cfg.rewards["stand_still"].weight = -2.0
  cfg.rewards["action_rate_l2"].weight = -0.05
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_full_speed_robust_stop_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Final low-rate refinement for robust zero-command settling."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_full_speed_robust_finalize_env_cfg(
    play=play
  )
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.standing_time_range = (1.0, 3.5)
  twist_cmd.moving_time_range = (2.5, 5.0)
  twist_cmd.initial_standing_probability = 0.30
  twist_cmd.stop_probability = 0.80
  twist_cmd.reverse_probability = 0.15
  twist_cmd.mode_weights = (0.21, 0.17, 0.15, 0.18, 0.17, 0.12)

  cfg.rewards["track_linear_velocity"].weight = 5.0
  cfg.rewards["track_angular_velocity"].weight = 5.0
  cfg.rewards["zero_command_velocity"].weight = -8.0
  cfg.rewards["stand_still"].weight = -3.0
  cfg.rewards["action_rate_l2"].weight = -0.035
  cfg.rewards["is_terminated"].weight = -500.0
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_history2_full_stop_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Two-frame full-speed stop/restart refinement with gait preservation."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_full_speed_robust_stop_env_cfg(
    play=play
  )
  cfg.observations["actor"].history_length = 2
  cfg.rewards["yaw_tracking_error"] = RewardTermCfg(
    func=yaw_tracking_error_l2,
    weight=-1.5,
    params={"command_name": "twist"},
  )
  cfg.rewards["foot_gait"].weight = 4.0
  cfg.rewards["foot_clearance"].weight = -4.0
  cfg.rewards["foot_clearance"].params["target_height"] = 0.07
  cfg.rewards["is_terminated"].weight = -600.0
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_history2_full_settle_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Full-envelope history policy with threshold-aware stop refinement."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_history2_full_envelope_env_cfg(
    play=play
  )
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.standing_time_range = (1.0, 3.0)
  twist_cmd.moving_time_range = (2.5, 5.5)
  twist_cmd.initial_standing_probability = 0.30
  twist_cmd.stop_probability = 0.75
  twist_cmd.reverse_probability = 0.20
  twist_cmd.mode_weights = (0.22, 0.16, 0.15, 0.17, 0.18, 0.12)

  cfg.rewards["zero_command_velocity"].weight = -5.0
  cfg.rewards["zero_command_velocity"].params["angular_scale"] = 1.0
  cfg.rewards["zero_command_settling"] = RewardTermCfg(
    func=zero_command_settling_exp,
    weight=4.0,
    params={
      "command_name": "twist",
      "planar_std": 0.15,
      "yaw_std": 0.20,
    },
  )
  cfg.rewards["stand_still"].weight = -2.0
  cfg.rewards["action_rate_l2"].weight = -0.04
  cfg.rewards["is_terminated"].weight = -600.0
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_history2_full_gait_settle_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Joint replay of symmetric low-speed gait and full-speed robust settling."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_history2_full_settle_env_cfg(
    play=play
  )
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.standing_time_range = (1.0, 3.5)
  twist_cmd.moving_time_range = (3.0, 6.0)
  twist_cmd.initial_standing_probability = 0.25
  twist_cmd.stop_probability = 0.60
  twist_cmd.mode_weights = (0.28, 0.18, 0.13, 0.16, 0.15, 0.10)

  cfg.rewards["foot_gait"].weight = 9.0
  cfg.rewards["foot_clearance"].weight = -8.0
  cfg.rewards["foot_clearance"].params["target_height"] = 0.065
  cfg.rewards["foot_swing_height"] = RewardTermCfg(
    func=feet_swing_height,
    weight=-1.0,
    params={
      "sensor_name": "feet_ground_contact",
      "target_height": 0.06,
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot")
      ),
    },
  )
  cfg.rewards["soft_landing"].weight = -0.001
  cfg.rewards["action_rate_l2"].weight = -0.035
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_history2_full_gait_stop_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Short joint objective for symmetric gait and post-run robust braking."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_history2_full_gait_settle_env_cfg(
    play=play
  )
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.standing_time_range = (1.0, 3.0)
  twist_cmd.moving_time_range = (2.5, 5.5)
  twist_cmd.initial_standing_probability = 0.30
  twist_cmd.stop_probability = 0.80
  twist_cmd.mode_weights = (0.22, 0.16, 0.14, 0.16, 0.20, 0.12)

  cfg.rewards["zero_command_velocity"].weight = -8.0
  cfg.rewards["zero_command_settling"].weight = 8.0
  cfg.rewards["stand_still"].weight = -2.5
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_robust_medium_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Second robust curriculum stage with moderate pushes and variation."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_robust_omni_env_cfg(play=play)
  if play:
    return cfg

  cfg.events["push_robot"].params["force_range"] = (-40.0, 40.0)
  cfg.events["push_robot"].params["torque_range"] = (-5.0, 5.0)
  cfg.events["push_robot"].params["cooldown_s"] = (4.0, 8.0)
  cfg.events["pd_gains"].params["kp_range"] = (0.95, 1.05)
  cfg.events["pd_gains"].params["kd_range"] = (0.9375, 1.0625)
  cfg.events["effort_limits"].params["effort_limit_range"] = (0.95, 1.05)
  cfg.events["pseudo_inertia"].params["alpha_range"] = (-0.025, 0.025)
  cfg.events["pseudo_inertia"].params["d_range"] = (-0.01, 0.01)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.025, 0.025),
    1: (-0.025, 0.025),
    2: (-0.025, 0.025),
  }
  cfg.events["joint_friction"].params["ranges"] = (0.0, 0.10)
  cfg.events["foot_friction"].params["ranges"] = (0.50, 1.40)
  cfg.events["encoder_bias"].params["bias_range"] = (-0.01, 0.01)
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_robust_actuator_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Ablation task containing only mild actuator uncertainty."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_robust_mild_env_cfg(play=play)
  if not play:
    for name in (
      "pseudo_inertia",
      "base_com",
      "joint_friction",
      "foot_friction",
      "encoder_bias",
    ):
      cfg.events.pop(name, None)
    for term_name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
      term = cfg.observations["actor"].terms[term_name]
      term.delay_max_lag = 0
      term.delay_hold_prob = 0.0
      term.delay_update_period = 0
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_robust_dynamics_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Ablation task containing only mild mass, CoM and friction uncertainty."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_robust_mild_env_cfg(play=play)
  if not play:
    for name in ("pd_gains", "effort_limits", "encoder_bias"):
      cfg.events.pop(name, None)
    for term_name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
      term = cfg.observations["actor"].terms[term_name]
      term.delay_max_lag = 0
      term.delay_hold_prob = 0.0
      term.delay_update_period = 0
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_robust_sensor_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Ablation task containing only mild encoder error and observation delay."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_robust_mild_env_cfg(play=play)
  if not play:
    for name in (
      "pd_gains",
      "effort_limits",
      "pseudo_inertia",
      "base_com",
      "joint_friction",
      "foot_friction",
    ):
      cfg.events.pop(name, None)
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_sensor_bias_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Ablation task containing encoder bias without observation delay."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_robust_sensor_env_cfg(play=play)
  if not play:
    for term_name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
      term = cfg.observations["actor"].terms[term_name]
      term.delay_min_lag = 0
      term.delay_max_lag = 0
      term.delay_hold_prob = 0.0
      term.delay_update_period = 0
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_sensor_delay_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Ablation task containing one-step observation delay without encoder bias."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_robust_sensor_env_cfg(play=play)
  if not play:
    cfg.events.pop("encoder_bias", None)
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_sensor_stability_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Preserve nominal low-speed gait while adapting to sensor bias and delay."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_robust_sensor_env_cfg(play=play)
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.standing_time_range = (1.5, 4.0)
  twist_cmd.moving_time_range = (3.0, 6.0)
  twist_cmd.initial_standing_probability = 0.35
  twist_cmd.stop_probability = 0.55
  twist_cmd.reverse_probability = 0.15
  twist_cmd.walk_speed_range = (0.25, 0.35)
  twist_cmd.backward_speed_range = (0.15, 0.20)
  twist_cmd.lateral_speed_range = (0.12, 0.20)
  twist_cmd.turn_speed_range = (0.15, 0.30)
  twist_cmd.mode_weights = (0.30, 0.24, 0.18, 0.20, 0.0, 0.08)

  cfg.rewards["track_linear_velocity"].weight = 5.0
  cfg.rewards["track_angular_velocity"].weight = 8.0
  cfg.rewards["yaw_tracking_error"] = RewardTermCfg(
    func=yaw_tracking_error_l2,
    weight=-3.0,
    params={"command_name": "twist"},
  )
  cfg.rewards["body_orientation_l2"].weight = -7.0
  cfg.rewards["body_ang_vel"].weight = -0.15
  cfg.rewards["zero_command_velocity"].weight = -5.0
  cfg.rewards["zero_command_velocity"].params["angular_scale"] = 2.0
  cfg.rewards["stand_still"].weight = -2.0
  cfg.rewards["moving_command_velocity_error"].weight = -4.0
  cfg.rewards["moving_command_velocity_error"].params["yaw_scale"] = 2.0
  cfg.rewards["moving_command_relative_velocity_error"].weight = -3.0
  cfg.rewards["action_rate_l2"].weight = -0.06
  cfg.rewards["is_terminated"].weight = -600.0
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_sensor_history2_stability_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Sensor-stability task with one previous actor-observation frame.

  The observation manager stores history per term in chronological order.  A
  length of two therefore gives the actor both the delayed measurement and its
  immediately preceding value without changing the action interface.
  """
  cfg = agibot_x2_ultra_behavior_kp090_kd080_sensor_stability_env_cfg(play=play)
  cfg.observations["actor"].history_length = 2
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_sensor_history2_gait_recover_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Recover foot clearance and alternating contacts after history adaptation."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_sensor_history2_stability_env_cfg(
    play=play
  )
  cfg.rewards["foot_gait"].weight = 3.0
  cfg.rewards["foot_clearance"].weight = -4.0
  cfg.rewards["foot_clearance"].params["target_height"] = 0.06
  cfg.rewards["action_rate_l2"].weight = -0.04
  cfg.rewards["track_linear_velocity"].weight = 5.5
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_sensor_history2_backward_clearance_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Target the remaining low backward swing while retaining forward replay."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_sensor_history2_gait_recover_env_cfg(
    play=play
  )
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.walk_speed_range = (0.25, 0.40)
  twist_cmd.backward_speed_range = (0.15, 0.25)
  twist_cmd.stop_probability = 0.35
  twist_cmd.reverse_probability = 0.30
  twist_cmd.mode_weights = (0.16, 0.64, 0.03, 0.03, 0.0, 0.14)

  cfg.rewards["foot_gait"].weight = 6.0
  cfg.rewards["foot_clearance"].weight = -12.0
  cfg.rewards["foot_clearance"].params["target_height"] = 0.055
  cfg.rewards["track_linear_velocity"].weight = 6.0
  cfg.rewards["action_rate_l2"].weight = -0.03
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_sensor_history2_backward_finish_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Close the last millimetres of backward swing clearance at very low LR."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_sensor_history2_backward_clearance_env_cfg(
    play=play
  )
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.mode_weights = (0.05, 0.85, 0.01, 0.01, 0.0, 0.08)
  cfg.rewards["foot_gait"].weight = 8.0
  cfg.rewards["foot_clearance"].weight = -20.0
  cfg.rewards["foot_clearance"].params["target_height"] = 0.06
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_sensor_history2_nominal_finish_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Calibrate clean-input backward clearance after robust history training."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_sensor_history2_backward_finish_env_cfg(
    play=play
  )
  cfg.events.pop("encoder_bias", None)
  for term_name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
    term = cfg.observations["actor"].terms[term_name]
    term.delay_min_lag = 0
    term.delay_max_lag = 0
    term.delay_hold_prob = 0.0
    term.delay_update_period = 0
  cfg.observations["actor"].enable_corruption = False

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.mode_weights = (0.04, 0.90, 0.01, 0.01, 0.0, 0.04)
  cfg.rewards["foot_clearance"].weight = -24.0
  cfg.rewards["foot_clearance"].params["target_height"] = 0.065
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_sensor_history2_turn_finish_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Add clean-input turn gait margin without dropping sagittal replay."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_sensor_history2_nominal_finish_env_cfg(
    play=play
  )
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.turn_speed_range = (0.15, 0.35)
  twist_cmd.turn_positive_probability = 0.50
  twist_cmd.mode_weights = (0.07, 0.07, 0.05, 0.76, 0.0, 0.05)
  cfg.rewards["foot_gait"].weight = 10.0
  cfg.rewards["foot_clearance"].weight = -24.0
  cfg.rewards["foot_clearance"].params["target_height"] = 0.06
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_robust_mild_sagittal_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Mild robust adaptation focused on forward, backward and stops."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_robust_mild_env_cfg(play=play)
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.mode_weights = (0.50, 0.50, 0.0, 0.0, 0.0, 0.0)
  cfg.rewards["track_linear_velocity"].weight = 5.0
  cfg.rewards["track_angular_velocity"].weight = 4.0
  cfg.rewards["body_orientation_l2"].weight = -5.0
  cfg.rewards["is_terminated"].weight = -600.0
  cfg.rewards["moving_command_velocity_error"].weight = -4.0
  cfg.rewards["moving_command_relative_velocity_error"].weight = -4.0
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_robust_mild_lateral_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Mild robust adaptation focused on symmetric lateral stepping."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_robust_mild_env_cfg(play=play)
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.mode_weights = (0.10, 0.10, 0.80, 0.0, 0.0, 0.0)
  cfg.rewards["track_linear_velocity"].weight = 6.0
  cfg.rewards["track_angular_velocity"].weight = 5.0
  cfg.rewards["body_orientation_l2"].weight = -8.0
  cfg.rewards["is_terminated"].weight = -600.0
  cfg.rewards["moving_command_velocity_error"].weight = -5.0
  cfg.rewards["moving_command_relative_velocity_error"].weight = -5.0
  return cfg


def agibot_x2_ultra_behavior_kp090_kd080_robust_mild_turn_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Mild robust adaptation focused on symmetric pivot turning."""
  cfg = agibot_x2_ultra_behavior_kp090_kd080_robust_mild_env_cfg(play=play)
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_y = (-0.05, 0.05)
  twist_cmd.mode_weights = (0.10, 0.10, 0.10, 0.70, 0.0, 0.0)
  cfg.rewards["track_linear_velocity"].weight = 4.0
  cfg.rewards["track_angular_velocity"].weight = 8.0
  cfg.rewards["body_orientation_l2"].weight = -7.0
  cfg.rewards["is_terminated"].weight = -600.0
  cfg.rewards["moving_command_velocity_error"].weight = -5.0
  cfg.rewards["moving_command_relative_velocity_error"].weight = -5.0
  return cfg


def agibot_x2_ultra_behavior_robust_low_speed_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Robust fine-tuning focused on the deployed low-speed joystick envelope."""
  cfg = agibot_x2_ultra_behavior_robust_flat_env_cfg(play=play)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-0.20, 0.35)
  twist_cmd.ranges.lin_vel_y = (-0.20, 0.20)
  twist_cmd.ranges.ang_vel_z = (-0.30, 0.30)
  twist_cmd.standing_time_range = (1.0, 2.5)
  twist_cmd.moving_time_range = (2.0, 4.0)
  twist_cmd.initial_standing_probability = 0.40
  twist_cmd.stop_probability = 0.65
  twist_cmd.reverse_probability = 0.10
  twist_cmd.walk_speed_range = (0.30, 0.35)
  twist_cmd.backward_speed_range = (0.18, 0.20)
  twist_cmd.lateral_speed_range = (0.12, 0.20)
  twist_cmd.turn_speed_range = (0.15, 0.30)
  # Forward/backward and mixed low-speed transitions dominate this stage.
  twist_cmd.mode_weights = (0.46, 0.46, 0.02, 0.02, 0.0, 0.04)

  # At low speeds the generic exponential reward is easy to satisfy while the
  # robot remains almost still.  Strengthen the direct tracking gradient.
  cfg.rewards["track_linear_velocity"].weight = 2.0
  cfg.rewards["moving_command_velocity_error"].weight = -2.0
  cfg.rewards["zero_command_velocity"].weight = -1.5
  cfg.rewards["moving_command_relative_velocity_error"] = RewardTermCfg(
    func=moving_command_relative_velocity_error_l1,
    weight=-1.0,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "cross_track_scale": 0.5,
    },
  )

  return cfg


def agibot_x2_ultra_behavior_robust_backward_refine_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Short refinement stage for reliable 0 to -0.20 m/s transitions."""
  cfg = agibot_x2_ultra_behavior_robust_low_speed_env_cfg(play=play)
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)
  twist_cmd.standing_time_range = (0.8, 2.0)
  twist_cmd.moving_time_range = (2.5, 4.0)
  twist_cmd.walk_speed_range = (0.32, 0.35)
  twist_cmd.backward_speed_range = (0.19, 0.20)
  twist_cmd.mode_weights = (0.22, 0.74, 0.01, 0.01, 0.0, 0.02)
  cfg.rewards["track_linear_velocity"].weight = 2.5
  cfg.rewards["moving_command_relative_velocity_error"].weight = -3.0
  return cfg


def agibot_x2_ultra_behavior_robust_lateral_refine_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Fine-tune pure lateral and forward-lateral stability at joystick speeds."""
  cfg = agibot_x2_ultra_behavior_robust_low_speed_env_cfg(play=play)
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)

  # Cover the real-hardware failure envelope (vx=0.35, |vy|=0.32-0.42)
  # while retaining low-speed starts, stops, and both lateral directions.
  twist_cmd.ranges.lin_vel_x = (-0.20, 0.35)
  twist_cmd.ranges.lin_vel_y = (-0.45, 0.45)
  twist_cmd.ranges.ang_vel_z = (-0.15, 0.15)
  twist_cmd.standing_time_range = (1.0, 2.5)
  twist_cmd.moving_time_range = (3.0, 6.0)
  twist_cmd.initial_standing_probability = 0.25
  twist_cmd.stop_probability = 0.40
  twist_cmd.reverse_probability = 0.15
  twist_cmd.walk_speed_range = (0.25, 0.35)
  twist_cmd.backward_speed_range = (0.18, 0.20)
  twist_cmd.lateral_speed_range = (0.15, 0.45)
  twist_cmd.turn_speed_range = (0.10, 0.15)
  twist_cmd.mixed_lateral_scale = 1.0
  twist_cmd.mixed_yaw_scale = 0.0
  # Forward, backward, pure lateral, turn, run, forward-lateral diagonal.
  twist_cmd.mode_weights = (0.10, 0.05, 0.40, 0.02, 0.0, 0.43)

  # Preserve velocity tracking while strongly discouraging lateral tipping,
  # cross-track drift, foot slip, and irrecoverable roll/pitch excursions.
  cfg.rewards["track_linear_velocity"].weight = 3.0
  cfg.rewards["moving_command_velocity_error"].weight = -3.0
  cfg.rewards["moving_command_relative_velocity_error"].weight = -2.0
  cfg.rewards["moving_command_relative_velocity_error"].params[
    "cross_track_scale"
  ] = 1.0
  cfg.rewards["body_orientation_l2"].weight = -2.5
  cfg.rewards["body_ang_vel"].weight = -0.15
  cfg.rewards["foot_slip"].weight = -0.50
  cfg.rewards["is_terminated"].weight = -300.0
  cfg.terminations["fell_over"].params["limit_angle"] = math.radians(45.0)

  if not play:
    # Randomize each foot independently.  The previous shared coefficient
    # could not reproduce the asymmetric grip that excites lateral falls.
    cfg.events.pop("foot_friction", None)
    for side in ("left", "right"):
      cfg.events[f"{side}_foot_friction"] = EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
          "asset_cfg": SceneEntityCfg(
            "robot",
            geom_names=tuple(
              f"{side}_foot{i}_collision" for i in range(1, 13)
            ),
          ),
          "operation": "abs",
          "ranges": (0.35, 1.60),
          "shared_random": True,
        },
      )

  return cfg


def agibot_x2_ultra_behavior_robust_lateral_balance_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Short refinement for high positive-y diagonal motion and clean braking."""
  cfg = agibot_x2_ultra_behavior_robust_lateral_refine_env_cfg(play=play)
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)

  twist_cmd.standing_time_range = (0.8, 2.0)
  twist_cmd.moving_time_range = (2.5, 4.0)
  twist_cmd.initial_standing_probability = 0.20
  twist_cmd.stop_probability = 0.65
  twist_cmd.reverse_probability = 0.05
  twist_cmd.walk_speed_range = (0.30, 0.35)
  twist_cmd.lateral_speed_range = (0.25, 0.45)
  twist_cmd.lateral_positive_probability = 0.70
  twist_cmd.mixed_forward_probability = 1.0
  # Keep both pure lateral directions, but focus on forward + lateral commands.
  twist_cmd.mode_weights = (0.03, 0.02, 0.25, 0.0, 0.0, 0.70)

  cfg.rewards["body_orientation_l2"].weight = -4.0
  cfg.rewards["body_ang_vel"].weight = -0.25
  cfg.rewards["zero_command_velocity"].weight = -2.5
  cfg.rewards["foot_slip"].weight = -0.60
  cfg.rewards["action_rate_l2"].weight = -0.07
  cfg.rewards["is_terminated"].weight = -400.0
  return cfg


def agibot_x2_ultra_behavior_robust_lateral_brake_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Short refinement for diagonal-to-zero transitions under disturbances."""
  cfg = agibot_x2_ultra_behavior_robust_lateral_balance_env_cfg(play=play)
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)

  twist_cmd.standing_time_range = (0.8, 1.8)
  twist_cmd.moving_time_range = (2.5, 4.0)
  twist_cmd.initial_standing_probability = 0.15
  twist_cmd.stop_probability = 0.85
  twist_cmd.reverse_probability = 0.0
  twist_cmd.walk_speed_range = (0.30, 0.35)
  twist_cmd.lateral_speed_range = (0.18, 0.25)
  twist_cmd.lateral_positive_probability = 0.75
  twist_cmd.mixed_forward_probability = 1.0
  twist_cmd.mode_weights = (0.08, 0.08, 0.12, 0.02, 0.0, 0.70)

  cfg.rewards["zero_command_velocity"].weight = -8.0
  cfg.rewards["stand_still"].weight = -2.0
  cfg.rewards["action_rate_l2"].weight = -0.08
  return cfg


def agibot_x2_ultra_behavior_robust_lateral_backward_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Joint replay of backward, lateral, diagonal, and stop behaviors."""
  cfg = agibot_x2_ultra_behavior_robust_lateral_brake_env_cfg(play=play)
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, BehaviorVelocityCommandCfg)

  twist_cmd.standing_time_range = (0.8, 2.0)
  twist_cmd.moving_time_range = (2.5, 4.5)
  twist_cmd.stop_probability = 0.65
  twist_cmd.reverse_probability = 0.10
  twist_cmd.walk_speed_range = (0.25, 0.35)
  twist_cmd.backward_speed_range = (0.18, 0.20)
  twist_cmd.lateral_speed_range = (0.15, 0.45)
  twist_cmd.lateral_positive_probability = 0.65
  twist_cmd.mixed_forward_probability = 1.0
  twist_cmd.mode_weights = (0.05, 0.40, 0.18, 0.02, 0.0, 0.35)

  cfg.rewards["moving_command_relative_velocity_error"].weight = -3.0
  cfg.rewards["zero_command_velocity"].weight = -4.0
  cfg.rewards["action_rate_l2"].weight = -0.07
  return cfg
