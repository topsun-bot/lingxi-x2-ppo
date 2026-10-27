"""Headless, deterministic behavior evaluation for the AgiBot X2 Ultra policy."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

TASK_ID = "Agibot-X2-Ultra-Behavior-Flat"
ROBUST_TASK_ID = "Agibot-X2-Ultra-Behavior-Robust-Low-Speed"
LATERAL_ROBUST_TASK_ID = "Agibot-X2-Ultra-Behavior-Robust-Lateral-Refine"


@dataclass(frozen=True)
class EvalConfig:
  checkpoint_file: str
  task_id: str | None = None
  fallback_checkpoint_file: str | None = None
  zero_command_checkpoint_file: str | None = None
  lateral_gate_threshold: float = 0.10
  num_envs: int = 32
  seed: int = 42
  device: str | None = None
  output_file: str | None = None
  duration_scale: float = 1.0
  profile: Literal[
    "full",
    "braking",
    "standing",
    "low-speed",
    "sagittal-gait-core",
    "omni-low-speed",
    "lateral-low-speed",
    "turn-low-speed",
    "arc-turn-low-speed",
    "lateral",
    "lateral-full-speed",
    "run-full-speed",
    "lateral-core",
    "lateral-brake-sequence",
  ] = "full"
  environment: Literal["nominal", "robust"] = "nominal"
  standing_duration_s: float = 300.0
  actuator_gain_profile: Literal["training", "native-walk"] = "training"
  actuator_gain_scale: float = 1.0


@dataclass(frozen=True)
class Stage:
  name: str
  command: tuple[float, float, float]
  duration_s: float
  kind: str


STAGES = (
  Stage("long_stand", (0.0, 0.0, 0.0), 60.0, "stand"),
  Stage("forward_walk", (0.8, 0.0, 0.0), 8.0, "linear"),
  Stage("brake_after_forward", (0.0, 0.0, 0.0), 5.0, "brake"),
  Stage("restart_forward", (0.8, 0.0, 0.0), 8.0, "restart"),
  Stage("backward_walk", (-0.6, 0.0, 0.0), 8.0, "linear"),
  Stage("lateral_left", (0.0, 0.6, 0.0), 8.0, "linear"),
  Stage("lateral_right", (0.0, -0.6, 0.0), 8.0, "linear"),
  Stage("turn_left", (0.0, 0.0, 0.7), 8.0, "turn"),
  Stage("turn_right", (0.0, 0.0, -0.7), 8.0, "turn"),
  Stage("run", (1.8, 0.0, 0.0), 10.0, "run"),
  Stage("brake_after_run", (0.0, 0.0, 0.0), 6.0, "brake"),
  Stage("restart_after_stop", (0.8, 0.0, 0.0), 8.0, "restart"),
  Stage("switch_forward", (0.7, 0.0, 0.0), 2.0, "switch_linear"),
  Stage("switch_backward", (-0.6, 0.0, 0.0), 2.0, "switch_linear"),
  Stage("switch_left", (0.0, 0.6, 0.0), 2.0, "switch_linear"),
  Stage("switch_right", (0.0, -0.6, 0.0), 2.0, "switch_linear"),
  Stage("switch_turn_left", (0.0, 0.0, 0.7), 2.0, "switch_turn"),
  Stage("switch_turn_right", (0.0, 0.0, -0.7), 2.0, "switch_turn"),
  Stage("final_stand", (0.0, 0.0, 0.0), 20.0, "stand"),
)

LOW_SPEED_STAGES = (
  Stage("low_long_stand", (0.0, 0.0, 0.0), 20.0, "stand"),
  Stage("creep_forward", (0.15, 0.0, 0.0), 6.0, "low_linear"),
  Stage("brake_after_creep", (0.0, 0.0, 0.0), 4.0, "brake"),
  Stage("slow_forward", (0.35, 0.0, 0.0), 8.0, "low_linear"),
  Stage("brake_after_slow_forward", (0.0, 0.0, 0.0), 5.0, "brake"),
  Stage("restart_slow_forward", (0.35, 0.0, 0.0), 8.0, "low_restart"),
  Stage("slow_backward", (-0.20, 0.0, 0.0), 8.0, "low_linear"),
  Stage("brake_after_slow_backward", (0.0, 0.0, 0.0), 5.0, "brake"),
  Stage("restart_slow_backward", (-0.20, 0.0, 0.0), 8.0, "low_restart"),
  Stage("low_final_stand", (0.0, 0.0, 0.0), 20.0, "stand"),
)

SAGITTAL_GAIT_CORE_STAGES = (
  Stage("gait_core_initial_stand", (0.0, 0.0, 0.0), 4.0, "stand"),
  Stage("gait_core_forward", (0.35, 0.0, 0.0), 6.0, "low_linear"),
  Stage("gait_core_brake", (0.0, 0.0, 0.0), 3.0, "brake"),
  Stage("gait_core_restart", (0.35, 0.0, 0.0), 6.0, "low_restart"),
  Stage("gait_core_backward", (-0.20, 0.0, 0.0), 6.0, "low_linear"),
  Stage("gait_core_final_stand", (0.0, 0.0, 0.0), 4.0, "stand"),
)

OMNI_LOW_SPEED_STAGES = (
  Stage("omni_low_initial_stand", (0.0, 0.0, 0.0), 10.0, "stand"),
  Stage("omni_low_forward", (0.35, 0.0, 0.0), 6.0, "low_linear"),
  Stage("omni_low_brake_forward", (0.0, 0.0, 0.0), 4.0, "brake"),
  Stage("omni_low_backward", (-0.20, 0.0, 0.0), 6.0, "low_linear"),
  Stage("omni_low_lateral_left", (0.0, 0.15, 0.0), 6.0, "lateral"),
  Stage("omni_low_brake_left", (0.0, 0.0, 0.0), 4.0, "brake"),
  Stage("omni_low_lateral_right", (0.0, -0.15, 0.0), 6.0, "lateral"),
  Stage("omni_low_turn_left", (0.0, 0.0, 0.25), 6.0, "turn"),
  Stage("omni_low_brake_turn", (0.0, 0.0, 0.0), 4.0, "brake"),
  Stage("omni_low_turn_right", (0.0, 0.0, -0.25), 6.0, "turn"),
  Stage("omni_low_final_stand", (0.0, 0.0, 0.0), 10.0, "stand"),
)

ARC_TURN_LOW_SPEED_STAGES = (
  Stage("arc_initial_stand", (0.0, 0.0, 0.0), 10.0, "stand"),
  Stage("arc_turn_left", (0.25, 0.0, 0.25), 8.0, "arc"),
  Stage("arc_brake_left", (0.0, 0.0, 0.0), 4.0, "brake"),
  Stage("arc_turn_right", (0.25, 0.0, -0.25), 8.0, "arc"),
  Stage("arc_final_stand", (0.0, 0.0, 0.0), 10.0, "stand"),
)

LATERAL_STAGES = (
  Stage("lateral_initial_stand", (0.0, 0.0, 0.0), 10.0, "stand"),
  Stage("lateral_left_020", (0.0, 0.20, 0.0), 6.0, "lateral"),
  Stage("lateral_brake_left", (0.0, 0.0, 0.0), 4.0, "brake"),
  Stage("lateral_right_020", (0.0, -0.20, 0.0), 6.0, "lateral"),
  Stage("lateral_left_035", (0.0, 0.35, 0.0), 8.0, "lateral"),
  Stage("lateral_right_035", (0.0, -0.35, 0.0), 8.0, "lateral"),
  Stage("diagonal_left_020", (0.35, 0.20, 0.0), 8.0, "diagonal"),
  Stage("diagonal_brake_left", (0.0, 0.0, 0.0), 4.0, "brake"),
  Stage("diagonal_right_020", (0.35, -0.20, 0.0), 8.0, "diagonal"),
  Stage("diagonal_left_035", (0.35, 0.35, 0.0), 8.0, "diagonal"),
  Stage("diagonal_right_035", (0.35, -0.35, 0.0), 8.0, "diagonal"),
  # Exact envelope surrounding the real-hardware failure at |vy|=0.42 m/s.
  Stage("diagonal_left_042", (0.35, 0.42, 0.0), 8.0, "diagonal"),
  Stage("diagonal_right_042", (0.35, -0.42, 0.0), 8.0, "diagonal"),
  Stage("lateral_final_stand", (0.0, 0.0, 0.0), 10.0, "stand"),
)

FULL_SPEED_LATERAL_STAGES = (
  Stage("lateral_fast_initial_stand", (0.0, 0.0, 0.0), 5.0, "stand"),
  Stage("lateral_fast_left", (0.0, 0.6, 0.0), 8.0, "lateral"),
  Stage("lateral_fast_brake_left", (0.0, 0.0, 0.0), 3.0, "brake"),
  Stage("lateral_fast_right", (0.0, -0.6, 0.0), 8.0, "lateral"),
  Stage("lateral_fast_final_stand", (0.0, 0.0, 0.0), 5.0, "stand"),
)

FULL_SPEED_RUN_STAGES = (
  Stage("run_fast_initial_stand", (0.0, 0.0, 0.0), 5.0, "stand"),
  Stage("run_fast", (1.8, 0.0, 0.0), 10.0, "run"),
  Stage("run_fast_brake", (0.0, 0.0, 0.0), 6.0, "brake"),
  Stage("run_fast_final_stand", (0.0, 0.0, 0.0), 5.0, "stand"),
)

LATERAL_CORE_STAGE_NAMES = {
  "lateral_initial_stand",
  "diagonal_left_035",
  "diagonal_brake_left",
  "diagonal_left_042",
  "lateral_final_stand",
}

BRAKING_STAGE_NAMES = {
  "forward_walk",
  "brake_after_forward",
  "restart_forward",
  "run",
  "brake_after_run",
  "restart_after_stop",
}


def _apply_actuator_gain_profile(
  env_cfg,
  profile: Literal["training", "native-walk"],
) -> None:
  """Override only leg PD gains while preserving trained action scaling.

  The native-walk values were observed on the X2 HAL joint command topics
  during native locomotion.  Keeping the policy action scale unchanged makes
  this an actuator-gain A/B test that matches a deployment-side gain override.
  """
  if profile == "training":
    return

  robot_cfg = env_cfg.scene.entities["robot"]
  actuators = robot_cfg.articulation.actuators
  overridden: list[BuiltinPositionActuatorCfg] = []
  replaced_expressions: set[str] = set()
  native_walk_gains = {
    ".*_hip_pitch_joint": (120.0, 5.0),
    ".*_hip_roll_joint": (100.0, 4.0),
    ".*_hip_yaw_joint": (100.0, 4.0),
    ".*_knee_joint": (150.0, 5.0),
    ".*_ankle_pitch_joint": (40.0, 2.0),
    ".*_ankle_roll_joint": (40.0, 2.0),
  }
  for actuator in actuators:
    if not isinstance(actuator, BuiltinPositionActuatorCfg):
      overridden.append(actuator)
      continue
    matching = tuple(
      expression
      for expression in actuator.target_names_expr
      if expression in native_walk_gains
    )
    if not matching:
      overridden.append(actuator)
      continue
    if len(matching) != len(actuator.target_names_expr):
      raise ValueError(
        "Native gain override matched only part of actuator group "
        f"{actuator.target_names_expr}"
      )
    for expression in matching:
      stiffness, damping = native_walk_gains[expression]
      overridden.append(
        replace(
          actuator,
          target_names_expr=(expression,),
          stiffness=stiffness,
          damping=damping,
        )
      )
      replaced_expressions.add(expression)

  missing = set(native_walk_gains) - replaced_expressions
  if missing:
    raise ValueError(f"Native gain override did not find actuators: {sorted(missing)}")
  robot_cfg.articulation.actuators = tuple(overridden)


def _apply_actuator_gain_scale(env_cfg, scale: float) -> None:
  """Scale PD gains without changing policy actions or effort limits."""
  if not 0.0 < scale <= 2.0:
    raise ValueError(f"Actuator gain scale must be in (0, 2], got {scale}")
  if scale == 1.0:
    return
  robot_cfg = env_cfg.scene.entities["robot"]
  robot_cfg.articulation.actuators = tuple(
    replace(
      actuator,
      stiffness=actuator.stiffness * scale,
      damping=actuator.damping * scale,
    )
    if isinstance(actuator, BuiltinPositionActuatorCfg)
    else actuator
    for actuator in robot_cfg.articulation.actuators
  )


def _quantile(values: torch.Tensor, q: float) -> float:
  return float(torch.quantile(values.float(), q).item())


def _stage_passes(
  stage: Stage,
  result: dict[str, object],
  environment: Literal["nominal", "robust"],
) -> tuple[bool, list[str]]:
  failures: list[str] = []

  def require(condition: bool, message: str) -> None:
    if not condition:
      failures.append(message)

  survival = float(result["survival_rate"])
  tilt = float(result["tilt_p95_deg"])
  linear_rmse = float(result["linear_tracking_rmse"])
  yaw_rmse = float(result["yaw_tracking_rmse"])
  planar_rms = float(result["planar_speed_rms"])
  yaw_rms = float(result["yaw_rate_rms"])

  require(survival >= 0.95, f"survival {survival:.3f} < 0.95")
  require(tilt <= 20.0, f"tilt p95 {tilt:.2f} deg > 20 deg")

  if stage.kind == "stand":
    require(planar_rms <= 0.12, f"planar RMS {planar_rms:.3f} > 0.12 m/s")
    require(yaw_rms <= 0.15, f"yaw RMS {yaw_rms:.3f} > 0.15 rad/s")
  elif stage.kind == "brake":
    settle = float(result["settling_time_p90_s"])
    settle_limit = 3.5 if environment == "robust" else 2.0
    require(
      settle <= settle_limit,
      f"settling p90 {settle:.2f} s > {settle_limit:.1f} s",
    )
    require(planar_rms <= 0.15, f"settled planar RMS {planar_rms:.3f} > 0.15 m/s")
    require(yaw_rms <= 0.20, f"settled yaw RMS {yaw_rms:.3f} > 0.20 rad/s")
  elif stage.kind in {"linear", "restart"}:
    require(linear_rmse <= 0.40, f"linear RMSE {linear_rmse:.3f} > 0.40 m/s")
    require(yaw_rmse <= 0.35, f"yaw RMSE {yaw_rmse:.3f} > 0.35 rad/s")
    if stage.kind == "restart":
      response = float(result["response_time_p90_s"])
      response_limit = 3.5 if environment == "robust" else 2.5
      require(
        response <= response_limit,
        f"response p90 {response:.2f} s > {response_limit:.1f} s",
      )
  elif stage.kind in {"low_linear", "low_restart"}:
    mean_velocity = result["mean_actual_velocity"]
    assert isinstance(mean_velocity, list)
    command = torch.tensor(stage.command[:2])
    target_speed = float(torch.linalg.vector_norm(command).item())
    actual = torch.tensor(mean_velocity[:2])
    signed_progress = float(torch.dot(actual, command).item() / target_speed)
    lateral_error = float(
      torch.linalg.vector_norm(actual - signed_progress * command / target_speed).item()
    )
    require(
      signed_progress >= 0.70 * target_speed,
      f"signed progress {signed_progress:.3f} < 70% of {target_speed:.3f} m/s",
    )
    require(linear_rmse <= 0.22, f"low-speed linear RMSE {linear_rmse:.3f} > 0.22 m/s")
    require(lateral_error <= 0.12, f"lateral drift {lateral_error:.3f} > 0.12 m/s")
    require(yaw_rmse <= 0.25, f"yaw RMSE {yaw_rmse:.3f} > 0.25 rad/s")
    result["signed_progress_mps"] = signed_progress
    result["lateral_drift_mps"] = lateral_error
    if stage.kind == "low_restart":
      response = float(result["response_time_p90_s"])
      response_limit = 3.5 if environment == "robust" else 2.5
      require(
        response <= response_limit,
        f"response p90 {response:.2f} s > {response_limit:.1f} s",
      )
  elif stage.kind in {"lateral", "diagonal"}:
    mean_velocity = result["mean_actual_velocity"]
    assert isinstance(mean_velocity, list)
    command = torch.tensor(stage.command[:2])
    target_speed = float(torch.linalg.vector_norm(command).item())
    actual = torch.tensor(mean_velocity[:2])
    direction = command / target_speed
    signed_progress = float(torch.dot(actual, direction).item())
    cross_track = float(
      torch.linalg.vector_norm(actual - signed_progress * direction).item()
    )
    tilt_limit = 15.0 if environment == "robust" else 12.0
    rmse_limit = 0.25 if environment == "robust" else 0.20
    require(
      signed_progress >= 0.65 * target_speed,
      f"signed progress {signed_progress:.3f} < 65% of {target_speed:.3f} m/s",
    )
    require(linear_rmse <= rmse_limit, f"linear RMSE {linear_rmse:.3f} > {rmse_limit:.2f} m/s")
    require(cross_track <= 0.15, f"cross-track drift {cross_track:.3f} > 0.15 m/s")
    require(yaw_rmse <= 0.30, f"yaw RMSE {yaw_rmse:.3f} > 0.30 rad/s")
    require(tilt <= tilt_limit, f"tilt p95 {tilt:.2f} deg > {tilt_limit:.1f} deg")
    result["signed_progress_mps"] = signed_progress
    result["cross_track_drift_mps"] = cross_track
  elif stage.kind == "run":
    mean_velocity = result["mean_actual_velocity"]
    assert isinstance(mean_velocity, list)
    require(float(mean_velocity[0]) >= 1.25, "mean run speed < 1.25 m/s")
    require(linear_rmse <= 0.65, f"run RMSE {linear_rmse:.3f} > 0.65 m/s")
  elif stage.kind == "turn":
    mean_angular_velocity = result["mean_actual_angular_velocity"]
    assert isinstance(mean_angular_velocity, list)
    target_yaw = abs(stage.command[2])
    signed_yaw_progress = mean_angular_velocity[2] * (
      1.0 if stage.command[2] > 0.0 else -1.0
    )
    require(
      signed_yaw_progress >= 0.65 * target_yaw,
      f"signed yaw progress {signed_yaw_progress:.3f} < 65% of {target_yaw:.3f} rad/s",
    )
    require(yaw_rmse <= 0.45, f"yaw RMSE {yaw_rmse:.3f} > 0.45 rad/s")
    require(planar_rms <= 0.35, f"turn planar RMS {planar_rms:.3f} > 0.35 m/s")
    result["signed_yaw_progress_radps"] = signed_yaw_progress
  elif stage.kind == "arc":
    mean_velocity = result["mean_actual_velocity"]
    assert isinstance(mean_velocity, list)
    mean_angular_velocity = result["mean_actual_angular_velocity"]
    assert isinstance(mean_angular_velocity, list)
    signed_linear_progress = mean_velocity[0] * (
      1.0 if stage.command[0] > 0.0 else -1.0
    )
    signed_yaw_progress = mean_angular_velocity[2] * (
      1.0 if stage.command[2] > 0.0 else -1.0
    )
    require(
      signed_linear_progress >= 0.65 * abs(stage.command[0]),
      f"signed linear progress {signed_linear_progress:.3f} < 65% target",
    )
    require(
      signed_yaw_progress >= 0.65 * abs(stage.command[2]),
      f"signed yaw progress {signed_yaw_progress:.3f} < 65% target",
    )
    require(linear_rmse <= 0.25, f"arc linear RMSE {linear_rmse:.3f} > 0.25 m/s")
    require(yaw_rmse <= 0.35, f"arc yaw RMSE {yaw_rmse:.3f} > 0.35 rad/s")
    require(tilt <= 20.0, f"tilt p95 {tilt:.2f} deg > 20 deg")
    result["signed_linear_progress_mps"] = signed_linear_progress
    result["signed_yaw_progress_radps"] = signed_yaw_progress
  elif stage.kind == "switch_linear":
    require(linear_rmse <= 0.65, f"switch linear RMSE {linear_rmse:.3f} > 0.65 m/s")
    require(yaw_rmse <= 0.45, f"switch yaw RMSE {yaw_rmse:.3f} > 0.45 rad/s")
  elif stage.kind == "switch_turn":
    require(yaw_rmse <= 0.60, f"switch yaw RMSE {yaw_rmse:.3f} > 0.60 rad/s")
    require(planar_rms <= 0.45, f"switch planar RMS {planar_rms:.3f} > 0.45 m/s")

  # A stable but shuffling or one-sided policy is not an acceptable walking
  # policy.  Apply gait gates only to full-duration movement stages; shortened
  # checkpoint scans remain velocity/stability screens.
  gait_kinds = {
    "linear",
    "restart",
    "low_linear",
    "low_restart",
    "lateral",
    "diagonal",
    "turn",
    "arc",
    "run",
  }
  if stage.kind in gait_kinds and stage.duration_s >= 4.0:
    touchdown_rates = result["step_frequency_per_foot_hz"]
    landing_heights = result["mean_landing_swing_height_m"]
    assert isinstance(touchdown_rates, list)
    assert isinstance(landing_heights, list)
    require(
      min(touchdown_rates) >= 0.25,
      f"foot touchdown rate {min(touchdown_rates):.3f} < 0.25 Hz",
    )
    require(
      max(touchdown_rates) <= 4.0,
      f"foot touchdown rate {max(touchdown_rates):.3f} > 4.0 Hz",
    )
    touchdown_rate_asymmetry = abs(touchdown_rates[0] - touchdown_rates[1])
    require(
      touchdown_rate_asymmetry <= 0.80,
      f"left/right touchdown-rate asymmetry {touchdown_rate_asymmetry:.3f} "
      "> 0.80 Hz",
    )
    asymmetry_limit = 0.25 if stage.kind == "turn" else 0.20
    contact_asymmetry = float(result["contact_duty_asymmetry"])
    require(
      contact_asymmetry <= asymmetry_limit,
      f"left/right contact-duty asymmetry {contact_asymmetry:.3f} "
      f"> {asymmetry_limit:.2f}",
    )
    single_support = float(result["single_support_fraction"])
    require(
      single_support >= 0.15,
      f"single-support fraction {single_support:.3f} < 0.15",
    )
    flight_limit = 0.75 if stage.kind == "run" else 0.50
    flight = float(result["flight_fraction"])
    require(flight <= flight_limit, f"flight fraction {flight:.3f} > {flight_limit:.2f}")
    valid_landing_heights = [height for height in landing_heights if height > 0.0]
    require(
      len(valid_landing_heights) == 2,
      "missing a landing-height measurement for one foot",
    )
    if len(valid_landing_heights) == 2:
      require(
        min(valid_landing_heights) >= 0.025,
        f"swing height {min(valid_landing_heights):.3f} < 0.025 m",
      )
      require(
        max(valid_landing_heights) <= 0.20,
        f"swing height {max(valid_landing_heights):.3f} > 0.20 m",
      )
    slip = float(result["mean_contact_slip_mps"])
    require(slip <= 0.35, f"contact slip {slip:.3f} > 0.35 m/s")

  return not failures, failures


def _evaluate_stage(
  env: RslRlVecEnvWrapper,
  policy,
  stage: Stage,
  environment: Literal["nominal", "robust"],
) -> dict[str, object]:
  unwrapped = env.unwrapped
  command_term = unwrapped.command_manager.get_term("twist")
  assert command_term is not None
  robot = unwrapped.scene["robot"]
  contact_sensor = unwrapped.scene["feet_ground_contact"]
  foot_site_ids, _ = robot.find_sites(
    ("left_foot", "right_foot"), preserve_order=True
  )
  command = torch.tensor(stage.command, device=unwrapped.device)
  steps = round(stage.duration_s / unwrapped.step_dt)
  steady_start = int(steps * (0.5 if stage.duration_s > 2.0 else 0.4))

  sample_count = torch.zeros(unwrapped.num_envs, device=unwrapped.device)
  actual_sum = torch.zeros(unwrapped.num_envs, 3, device=unwrapped.device)
  angular_sum = torch.zeros(unwrapped.num_envs, 3, device=unwrapped.device)
  linear_error_sum = torch.zeros(unwrapped.num_envs, device=unwrapped.device)
  yaw_error_sum = torch.zeros(unwrapped.num_envs, device=unwrapped.device)
  planar_speed_sum = torch.zeros(unwrapped.num_envs, device=unwrapped.device)
  yaw_rate_sum = torch.zeros(unwrapped.num_envs, device=unwrapped.device)
  max_tilt = torch.zeros(unwrapped.num_envs, device=unwrapped.device)
  fell = torch.zeros(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device)
  alive = torch.ones(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device)

  contact_data = contact_sensor.data.found
  assert contact_data is not None
  previous_contact = contact_data > 0
  swing_peak = torch.zeros(
    unwrapped.num_envs, len(foot_site_ids), device=unwrapped.device
  )
  contact_samples = torch.zeros(len(foot_site_ids), device=unwrapped.device)
  touchdown_count = torch.zeros(len(foot_site_ids), device=unwrapped.device)
  landing_peak_sum = torch.zeros(len(foot_site_ids), device=unwrapped.device)
  landing_peak_count = torch.zeros(len(foot_site_ids), device=unwrapped.device)
  alive_samples = torch.tensor(0.0, device=unwrapped.device)
  single_support_samples = torch.tensor(0.0, device=unwrapped.device)
  double_support_samples = torch.tensor(0.0, device=unwrapped.device)
  flight_samples = torch.tensor(0.0, device=unwrapped.device)
  slip_sum = torch.tensor(0.0, device=unwrapped.device)
  slip_samples = torch.tensor(0.0, device=unwrapped.device)
  landing_force_sum = torch.tensor(0.0, device=unwrapped.device)
  landing_force_samples = torch.tensor(0.0, device=unwrapped.device)

  consecutive = torch.zeros(unwrapped.num_envs, dtype=torch.long, device=unwrapped.device)
  transition_time = torch.full(
    (unwrapped.num_envs,), stage.duration_s + unwrapped.step_dt, device=unwrapped.device
  )
  transition_window = max(1, round(0.5 / unwrapped.step_dt))
  progress_history = torch.zeros(
    transition_window, unwrapped.num_envs, device=unwrapped.device
  )
  progress_valid_history = torch.zeros(
    transition_window,
    unwrapped.num_envs,
    dtype=torch.bool,
    device=unwrapped.device,
  )

  for step in range(steps):
    command_term.vel_command_b[:] = command
    command_term.is_standing_env[:] = torch.linalg.vector_norm(command) < 0.1
    command_term.is_heading_env[:] = False
    command_term.time_left[:] = 1.0e6
    observations = env.get_observations()
    with torch.inference_mode():
      actions = policy(observations)
      _, _, dones, _ = env.step(actions)
    dones = dones.bool()

    actual = robot.data.root_link_lin_vel_b
    angular = robot.data.root_link_ang_vel_b
    planar_speed = torch.norm(actual[:, :2], dim=1)
    yaw_rate = torch.abs(angular[:, 2])
    gravity_z = torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0)
    tilt = torch.acos(gravity_z)
    fell |= dones
    alive &= ~dones
    max_tilt = torch.maximum(max_tilt, tilt)

    contact_data = contact_sensor.data.found
    assert contact_data is not None
    in_contact = contact_data > 0
    foot_heights = robot.data.site_pos_w[:, foot_site_ids, 2]
    swing_peak = torch.where(
      ~in_contact, torch.maximum(swing_peak, foot_heights), swing_peak
    )

    if stage.kind == "brake":
      reached = (planar_speed < 0.15) & (yaw_rate < 0.20) & ~dones
      consecutive = torch.where(reached, consecutive + 1, torch.zeros_like(consecutive))
      newly_reached = (consecutive == transition_window) & (
        transition_time > stage.duration_s
      )
    else:
      linear_target = torch.norm(command[:2])
      if linear_target > 0.0:
        progress = torch.sum(actual[:, :2] * command[:2], dim=1) / linear_target
        target_progress = 0.7 * linear_target
      elif torch.abs(command[2]) > 0.0:
        progress = angular[:, 2] * torch.sign(command[2])
        target_progress = 0.7 * torch.abs(command[2])
      else:
        progress = -torch.maximum(planar_speed / 0.15, yaw_rate / 0.20)
        target_progress = torch.tensor(-1.0, device=unwrapped.device)
      slot = step % transition_window
      progress_history[slot] = progress
      progress_valid_history[slot] = ~dones
      window_ready = step + 1 >= transition_window
      reached = (
        window_ready
        & (progress_history.mean(dim=0) >= target_progress)
        & progress_valid_history.all(dim=0)
      )
      newly_reached = reached & (transition_time > stage.duration_s)
    transition_time[newly_reached] = (
      step + 1 - transition_window
    ) * unwrapped.step_dt

    if step < steady_start:
      previous_contact = in_contact
      continue
    valid_gait = alive.unsqueeze(1)
    touchdowns = (~previous_contact) & in_contact & valid_gait
    touchdown_count += touchdowns.float().sum(dim=0)
    landing_peak_sum += (swing_peak * touchdowns.float()).sum(dim=0)
    landing_peak_count += touchdowns.float().sum(dim=0)
    swing_peak = torch.where(in_contact, torch.zeros_like(swing_peak), swing_peak)
    previous_contact = in_contact

    live_count = alive.float().sum()
    alive_samples += live_count
    contact_samples += (in_contact & valid_gait).float().sum(dim=0)
    support_count = in_contact.sum(dim=1)
    single_support_samples += ((support_count == 1) & alive).float().sum()
    double_support_samples += ((support_count == 2) & alive).float().sum()
    flight_samples += ((support_count == 0) & alive).float().sum()
    foot_slip = torch.norm(
      robot.data.site_lin_vel_w[:, foot_site_ids, :2], dim=-1
    )
    slip_mask = in_contact & valid_gait
    slip_sum += (foot_slip * slip_mask.float()).sum()
    slip_samples += slip_mask.float().sum()
    contact_force = contact_sensor.data.force
    if contact_force is not None:
      force_magnitude = torch.norm(contact_force, dim=-1)
      landing_force_sum += (force_magnitude * touchdowns.float()).sum()
      landing_force_samples += touchdowns.float().sum()

    valid = (~dones).float()
    sample_count += valid
    actual_sum += actual[:, (0, 1, 2)] * valid.unsqueeze(1)
    angular_sum += angular[:, (0, 1, 2)] * valid.unsqueeze(1)
    linear_error_sum += (
      torch.sum(torch.square(actual[:, :2] - command[:2]), dim=1) * valid
    )
    yaw_error_sum += torch.square(angular[:, 2] - command[2]) * valid
    planar_speed_sum += torch.sum(torch.square(actual[:, :2]), dim=1) * valid
    yaw_rate_sum += torch.square(angular[:, 2]) * valid

  total_samples = torch.clamp(sample_count.sum(), min=1.0)
  gait_samples = torch.clamp(alive_samples, min=1.0)
  measured_gait_duration_s = max(
    unwrapped.step_dt, (steps - steady_start) * unwrapped.step_dt
  )
  landing_peak_denominator = torch.clamp(landing_peak_count, min=1.0)
  result: dict[str, object] = {
    "command": list(stage.command),
    "duration_s": stage.duration_s,
    "kind": stage.kind,
    "survival_rate": float((~fell).float().mean().item()),
    "tilt_p95_deg": _quantile(torch.rad2deg(max_tilt), 0.95),
    "mean_actual_velocity": [
      float(value)
      for value in (actual_sum.sum(dim=0) / total_samples).cpu().tolist()
    ],
    "mean_actual_angular_velocity": [
      float(value)
      for value in (angular_sum.sum(dim=0) / total_samples).cpu().tolist()
    ],
    "linear_tracking_rmse": float(
      torch.sqrt(linear_error_sum.sum() / total_samples).item()
    ),
    "yaw_tracking_rmse": float(
      torch.sqrt(yaw_error_sum.sum() / total_samples).item()
    ),
    "planar_speed_rms": float(
      torch.sqrt(planar_speed_sum.sum() / total_samples).item()
    ),
    "yaw_rate_rms": float(torch.sqrt(yaw_rate_sum.sum() / total_samples).item()),
    "foot_contact_duty_factor": [
      float(value) for value in (contact_samples / gait_samples).cpu().tolist()
    ],
    "contact_duty_asymmetry": float(
      torch.abs(contact_samples[0] - contact_samples[1]).div(gait_samples).item()
    ),
    "single_support_fraction": float(
      (single_support_samples / gait_samples).item()
    ),
    "double_support_fraction": float(
      (double_support_samples / gait_samples).item()
    ),
    "flight_fraction": float((flight_samples / gait_samples).item()),
    "touchdowns_per_foot_per_env": [
      float(value)
      for value in (touchdown_count / unwrapped.num_envs).cpu().tolist()
    ],
    "step_frequency_per_foot_hz": [
      float(value)
      for value in (
        touchdown_count / unwrapped.num_envs / measured_gait_duration_s
      )
      .cpu()
      .tolist()
    ],
    "mean_landing_swing_height_m": [
      float(value)
      for value in (landing_peak_sum / landing_peak_denominator).cpu().tolist()
    ],
    "mean_contact_slip_mps": float(
      (slip_sum / torch.clamp(slip_samples, min=1.0)).item()
    ),
    "mean_landing_force_n": float(
      (landing_force_sum / torch.clamp(landing_force_samples, min=1.0)).item()
    ),
  }
  if stage.kind == "brake":
    result["settling_time_p90_s"] = _quantile(transition_time, 0.90)
  if stage.kind in {"restart", "low_restart"}:
    result["response_time_p90_s"] = _quantile(transition_time, 0.90)
    result["response_confirmation_window_s"] = transition_window * unwrapped.step_dt
  passed, failures = _stage_passes(stage, result, environment)
  result["passed"] = passed
  result["failures"] = failures
  return result


def run_evaluation(cfg: EvalConfig) -> dict[str, object]:
  configure_torch_backends()
  checkpoint = Path(cfg.checkpoint_file).expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  if cfg.task_id is not None:
    task_id = cfg.task_id
  elif cfg.environment == "robust":
    task_id = (
      LATERAL_ROBUST_TASK_ID
      if cfg.profile in {"lateral", "lateral-core", "lateral-brake-sequence"}
      else ROBUST_TASK_ID
    )
  else:
    task_id = TASK_ID
  env_cfg = load_env_cfg(task_id, play=cfg.environment == "nominal")
  _apply_actuator_gain_profile(env_cfg, cfg.actuator_gain_profile)
  _apply_actuator_gain_scale(env_cfg, cfg.actuator_gain_scale)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  env_cfg.episode_length_s = int(1e9)
  env_cfg.events.pop("randomize_terrain", None)
  agent_cfg = load_rl_cfg(task_id)
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  env.reset()

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  primary_policy = runner.get_inference_policy(device=device)
  policy = primary_policy
  fallback_checkpoint: Path | None = None
  if cfg.fallback_checkpoint_file:
    fallback_checkpoint = Path(cfg.fallback_checkpoint_file).expanduser().resolve()
    if not fallback_checkpoint.is_file():
      raise FileNotFoundError(fallback_checkpoint)
    fallback_runner = runner_cls(env, asdict(agent_cfg), device=device)
    fallback_runner.load(
      str(fallback_checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    fallback_policy = fallback_runner.get_inference_policy(device=device)

    def routed_policy(observations):
      actor_observations = torch.cat(
        [observations[group] for group in primary_policy.obs_groups], dim=-1
      )
      use_lateral = (
        torch.abs(actor_observations[:, 7]) > cfg.lateral_gate_threshold
      ).unsqueeze(1)
      return torch.where(
        use_lateral,
        primary_policy(observations),
        fallback_policy(observations),
      )

    policy = routed_policy

  zero_command_checkpoint: Path | None = None
  if cfg.zero_command_checkpoint_file:
    zero_command_checkpoint = (
      Path(cfg.zero_command_checkpoint_file).expanduser().resolve()
    )
    if not zero_command_checkpoint.is_file():
      raise FileNotFoundError(zero_command_checkpoint)
    zero_runner = runner_cls(env, asdict(agent_cfg), device=device)
    zero_runner.load(
      str(zero_command_checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    zero_policy = zero_runner.get_inference_policy(device=device)
    moving_policy = policy

    def zero_routed_policy(observations):
      actor_observations = torch.cat(
        [observations[group] for group in primary_policy.obs_groups], dim=-1
      )
      if actor_observations.shape[1] == 104:
        command_start = 6
      elif actor_observations.shape[1] == 208:
        # Each observation term contains oldest-to-newest history.  Two
        # 3-vector terms precede the two-frame command term.
        command_start = 15
      else:
        raise RuntimeError(
          f"Unsupported actor observation width {actor_observations.shape[1]}"
        )
      command = actor_observations[:, command_start : command_start + 3]
      use_zero = (torch.linalg.vector_norm(command, dim=1) < 0.1).unsqueeze(1)
      return torch.where(
        use_zero,
        zero_policy(observations),
        moving_policy(observations),
      )

    policy = zero_routed_policy

  stage_results: dict[str, dict[str, object]] = {}
  if cfg.profile == "full":
    selected_stages = STAGES
  elif cfg.profile == "braking":
    selected_stages = tuple(
      stage for stage in STAGES if stage.name in BRAKING_STAGE_NAMES
    )
  elif cfg.profile == "standing":
    selected_stages = tuple(stage for stage in STAGES if stage.name == "long_stand")
  elif cfg.profile == "sagittal-gait-core":
    selected_stages = SAGITTAL_GAIT_CORE_STAGES
  elif cfg.profile == "omni-low-speed":
    selected_stages = OMNI_LOW_SPEED_STAGES
  elif cfg.profile == "lateral-low-speed":
    selected_stages = tuple(
      stage
      for stage in OMNI_LOW_SPEED_STAGES
      if stage.name
      in {
        "omni_low_initial_stand",
        "omni_low_lateral_left",
        "omni_low_brake_left",
        "omni_low_lateral_right",
        "omni_low_final_stand",
      }
    )
  elif cfg.profile == "turn-low-speed":
    selected_stages = tuple(
      stage
      for stage in OMNI_LOW_SPEED_STAGES
      if stage.name
      in {
        "omni_low_initial_stand",
        "omni_low_turn_left",
        "omni_low_brake_turn",
        "omni_low_turn_right",
        "omni_low_final_stand",
      }
    )
  elif cfg.profile == "arc-turn-low-speed":
    selected_stages = ARC_TURN_LOW_SPEED_STAGES
  elif cfg.profile == "lateral":
    selected_stages = LATERAL_STAGES
  elif cfg.profile == "lateral-full-speed":
    selected_stages = FULL_SPEED_LATERAL_STAGES
  elif cfg.profile == "run-full-speed":
    selected_stages = FULL_SPEED_RUN_STAGES
  elif cfg.profile == "lateral-core":
    selected_stages = tuple(
      stage for stage in LATERAL_STAGES if stage.name in LATERAL_CORE_STAGE_NAMES
    )
  elif cfg.profile == "lateral-brake-sequence":
    # Preserve the full lateral profile's disturbance timing through the
    # diagonal brake stage while omitting unrelated stages after it.
    selected_stages = LATERAL_STAGES[:8]
  else:
    selected_stages = LOW_SPEED_STAGES
  try:
    for configured_stage in selected_stages:
      duration_s = (
        cfg.standing_duration_s
        if cfg.profile == "standing" and configured_stage.name == "long_stand"
        else configured_stage.duration_s
      )
      stage = replace(
        configured_stage,
        duration_s=max(0.2, duration_s * cfg.duration_scale),
      )
      result = _evaluate_stage(env, policy, stage, cfg.environment)
      stage_results[stage.name] = result
      marker = "PASS" if result["passed"] else "FAIL"
      print(
        f"[{marker}] {stage.name:24s} survival={result['survival_rate']:.3f} "
        f"lin_rmse={result['linear_tracking_rmse']:.3f} "
        f"yaw_rmse={result['yaw_tracking_rmse']:.3f}"
      )
  finally:
    env.close()

  requirement_stages = {
    "long_zero_command_stand": ("long_stand", "final_stand"),
    "forward_backward_lateral_turn_run": (
      "forward_walk",
      "backward_walk",
      "lateral_left",
      "lateral_right",
      "turn_left",
      "turn_right",
      "run",
    ),
    "brake_after_command_cancel": ("brake_after_forward", "brake_after_run"),
    "restart_after_stop": ("restart_forward", "restart_after_stop"),
    "continuous_direction_switching": (
      "switch_forward",
      "switch_backward",
      "switch_left",
      "switch_right",
      "switch_turn_left",
      "switch_turn_right",
    ),
  }
  if cfg.profile == "standing":
    requirement_stages = {"long_zero_command_stand": ("long_stand",)}
  elif cfg.profile == "braking":
    requirement_stages = {
      "braking_tracking": ("forward_walk", "run"),
      "braking_settling": ("brake_after_forward", "brake_after_run"),
      "braking_restart": ("restart_forward", "restart_after_stop"),
    }
  elif cfg.profile == "sagittal-gait-core":
    requirement_stages = {
      "gait_core_standing": (
        "gait_core_initial_stand",
        "gait_core_final_stand",
      ),
      "gait_core_tracking": (
        "gait_core_forward",
        "gait_core_restart",
        "gait_core_backward",
      ),
      "gait_core_braking": ("gait_core_brake",),
    }
  elif cfg.profile == "low-speed":
    requirement_stages = {
      "low_speed_standing": ("low_long_stand", "low_final_stand"),
      # 0.15 m/s is retained as a diagnostic. Deployment acceptance follows
      # the actual joystick limits: +0.35 m/s and -0.20 m/s.
      "low_speed_tracking": ("slow_forward", "slow_backward"),
      "low_speed_braking": (
        "brake_after_slow_forward",
        "brake_after_slow_backward",
      ),
      "low_speed_restart": ("restart_slow_forward", "restart_slow_backward"),
    }
  elif cfg.profile == "omni-low-speed":
    requirement_stages = {
      "omni_low_standing": ("omni_low_initial_stand", "omni_low_final_stand"),
      "omni_low_sagittal": ("omni_low_forward", "omni_low_backward"),
      "omni_low_lateral": (
        "omni_low_lateral_left",
        "omni_low_lateral_right",
      ),
      "omni_low_turn": ("omni_low_turn_left", "omni_low_turn_right"),
      "omni_low_braking": (
        "omni_low_brake_forward",
        "omni_low_brake_left",
        "omni_low_brake_turn",
      ),
    }
  elif cfg.profile == "lateral-low-speed":
    requirement_stages = {
      "lateral_low_standing": (
        "omni_low_initial_stand",
        "omni_low_final_stand",
      ),
      "lateral_low_tracking": (
        "omni_low_lateral_left",
        "omni_low_lateral_right",
      ),
      "lateral_low_braking": ("omni_low_brake_left",),
    }
  elif cfg.profile == "turn-low-speed":
    requirement_stages = {
      "turn_low_standing": (
        "omni_low_initial_stand",
        "omni_low_final_stand",
      ),
      "turn_low_tracking": ("omni_low_turn_left", "omni_low_turn_right"),
      "turn_low_braking": ("omni_low_brake_turn",),
    }
  elif cfg.profile == "arc-turn-low-speed":
    requirement_stages = {
      "arc_turn_standing": ("arc_initial_stand", "arc_final_stand"),
      "arc_turn_tracking": ("arc_turn_left", "arc_turn_right"),
      "arc_turn_braking": ("arc_brake_left",),
    }
  elif cfg.profile == "lateral":
    requirement_stages = {
      "lateral_standing": ("lateral_initial_stand", "lateral_final_stand"),
      "pure_lateral_tracking": (
        "lateral_left_020",
        "lateral_right_020",
        "lateral_left_035",
        "lateral_right_035",
      ),
      "forward_lateral_diagonal": (
        "diagonal_left_020",
        "diagonal_right_020",
        "diagonal_left_035",
        "diagonal_right_035",
        "diagonal_left_042",
        "diagonal_right_042",
      ),
      "lateral_braking": ("lateral_brake_left", "diagonal_brake_left"),
    }
  elif cfg.profile == "lateral-full-speed":
    requirement_stages = {
      "lateral_fast_standing": (
        "lateral_fast_initial_stand",
        "lateral_fast_final_stand",
      ),
      "lateral_fast_tracking_and_gait": (
        "lateral_fast_left",
        "lateral_fast_right",
      ),
      "lateral_fast_braking": ("lateral_fast_brake_left",),
    }
  elif cfg.profile == "run-full-speed":
    requirement_stages = {
      "run_fast_standing": (
        "run_fast_initial_stand",
        "run_fast_final_stand",
      ),
      "run_fast_tracking_and_gait": ("run_fast",),
      "run_fast_braking": ("run_fast_brake",),
    }
  elif cfg.profile == "lateral-core":
    requirement_stages = {
      "lateral_core_standing": (
        "lateral_initial_stand",
        "lateral_final_stand",
      ),
      "positive_diagonal_tracking": (
        "diagonal_left_035",
        "diagonal_left_042",
      ),
      "positive_diagonal_braking": ("diagonal_brake_left",),
    }
  elif cfg.profile == "lateral-brake-sequence":
    requirement_stages = {
      "lateral_sequence_standing": ("lateral_initial_stand",),
      "lateral_sequence_tracking": (
        "lateral_left_020",
        "lateral_right_020",
        "lateral_left_035",
        "lateral_right_035",
        "diagonal_left_020",
      ),
      "lateral_sequence_braking": (
        "lateral_brake_left",
        "diagonal_brake_left",
      ),
    }
  requirements = {}
  for requirement, names in requirement_stages.items():
    if all(name in stage_results for name in names):
      requirements[requirement] = all(
        bool(stage_results[name]["passed"]) for name in names
      )
  report: dict[str, object] = {
    "task": task_id,
    "checkpoint": str(checkpoint),
    "fallback_checkpoint": (
      str(fallback_checkpoint) if fallback_checkpoint is not None else None
    ),
    "zero_command_checkpoint": (
      str(zero_command_checkpoint)
      if zero_command_checkpoint is not None
      else None
    ),
    "lateral_gate_threshold": cfg.lateral_gate_threshold,
    "num_envs": cfg.num_envs,
    "seed": cfg.seed,
    "profile": cfg.profile,
    "environment": cfg.environment,
    "actuator_gain_profile": cfg.actuator_gain_profile,
    "actuator_gain_scale": cfg.actuator_gain_scale,
    "requirements": requirements,
    "passed": all(requirements.values()),
    "stages": stage_results,
  }

  output = (
    Path(cfg.output_file).expanduser().resolve()
    if cfg.output_file
    else checkpoint.parent / f"{checkpoint.stem}_behavior_eval.json"
  )
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
  print(f"Report: {output}")
  print(f"Overall: {'PASS' if report['passed'] else 'FAIL'}")
  return report


def main() -> None:
  import mjlab.tasks  # noqa: F401

  import src.tasks  # noqa: F401

  cfg = tyro.cli(EvalConfig)
  report = run_evaluation(cfg)
  raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
  main()
