from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  wrap_to_pi,
)

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformVelocityCommand(CommandTerm):
  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.entity_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    self.vel_command_b[env_ids, :] *= (torch.norm(self.vel_command_b[env_ids, :], dim=1) > 0.1).unsqueeze(1)
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
    init_vel_env_ids = env_ids[init_vel_mask]
    if len(init_vel_env_ids) > 0:
      root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
      root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
      lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
      root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

  def _update_command(self) -> None:
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0

  # GUI.

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    """Create velocity joystick sliders in the Viser viewer."""
    from viser import Icon

    ranges = self.cfg.ranges

    axes = [
      ("lin_vel_x", ranges.lin_vel_x[1]),
      ("lin_vel_y", ranges.lin_vel_y[1]),
      ("ang_vel_z", ranges.ang_vel_z[1]),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, max_val in axes:
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=max_val,
          step=0.1,
          min=0.1,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-max_val,
          max=max_val,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    # Store GUI state for compute() override.
    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for i, s in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = s.value

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      # Skip if robot appears uninitialized (at origin).
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      # Helper to transform local to world coordinates.
      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

      # Command linear velocity arrow (blue).
      cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      cmd_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
      )
      visualizer.add_arrow(
        cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      # Command angular velocity arrow (green).
      cmd_ang_from = cmd_lin_from
      cmd_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
      )
      visualizer.add_arrow(
        cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
      )

      # Actual linear velocity arrow (cyan).
      act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      act_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
      )
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      # Actual angular velocity arrow (light green).
      act_ang_from = act_lin_from
      act_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
      )
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  entity_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 1.0
  init_velocity_prob: float = 0.0

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    return UniformVelocityCommand(self, env)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )


class BehaviorVelocityCommand(UniformVelocityCommand):
  """Sample deliberate stand, stop, restart, and direction-change sequences.

  Uniform independent commands make braking and restarting relatively rare.  This
  sampler keeps a small state per environment so that every standing interval is
  followed by motion, and a configurable fraction of moving intervals is followed
  by a zero command.  Remaining moving-to-moving transitions exercise direction
  changes, including explicit reversals.
  """

  cfg: BehaviorVelocityCommandCfg

  def __init__(self, cfg: BehaviorVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._mode_weights = torch.tensor(
      cfg.mode_weights, device=self.device, dtype=torch.float32
    )

  def _sample_magnitude(
    self, count: int, value_range: tuple[float, float]
  ) -> torch.Tensor:
    return torch.empty(count, device=self.device).uniform_(*value_range)

  def _sample_sign(
    self, count: int, positive_probability: float = 0.5
  ) -> torch.Tensor:
    return torch.where(
      torch.rand(count, device=self.device) < positive_probability,
      torch.ones(count, device=self.device),
      -torch.ones(count, device=self.device),
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    previous = self.vel_command_b[env_ids].clone()
    first_command = self.command_counter[env_ids] == 0
    was_standing = self.is_standing_env[env_ids] | (
      torch.norm(previous, dim=1) <= self.cfg.command_deadzone
    )

    random_values = torch.rand(len(env_ids), device=self.device)
    make_standing = first_command & (
      random_values < self.cfg.initial_standing_probability
    )
    moving_transition = ~first_command & ~was_standing
    make_standing |= moving_transition & (
      random_values < self.cfg.stop_probability
    )

    self.vel_command_b[env_ids] = 0.0
    self.is_heading_env[env_ids] = False
    self.is_standing_env[env_ids] = make_standing

    standing_ids = env_ids[make_standing]
    moving_local_ids = (~make_standing).nonzero(as_tuple=False).flatten()
    moving_ids = env_ids[moving_local_ids]
    if len(standing_ids) > 0:
      self.time_left[standing_ids] = self._sample_magnitude(
        len(standing_ids), self.cfg.standing_time_range
      )
    if len(moving_ids) > 0:
      self.time_left[moving_ids] = self._sample_magnitude(
        len(moving_ids), self.cfg.moving_time_range
      )

    reverse_local_mask = (
      ~first_command[moving_local_ids]
      & ~was_standing[moving_local_ids]
      & (
        torch.rand(len(moving_local_ids), device=self.device)
        < self.cfg.reverse_probability
      )
    )
    reverse_ids = moving_ids[reverse_local_mask]
    reverse_source = moving_local_ids[reverse_local_mask]
    if len(reverse_ids) > 0:
      reversed_command = -previous[reverse_source]
      reversed_command[:, 0].clamp_(*self.cfg.ranges.lin_vel_x)
      reversed_command[:, 1].clamp_(*self.cfg.ranges.lin_vel_y)
      reversed_command[:, 2].clamp_(*self.cfg.ranges.ang_vel_z)
      self.vel_command_b[reverse_ids] = reversed_command

    sample_ids = moving_ids[~reverse_local_mask]
    if len(sample_ids) == 0:
      return

    modes = torch.multinomial(
      self._mode_weights, len(sample_ids), replacement=True
    )
    for mode in range(len(self.cfg.mode_weights)):
      mode_ids = sample_ids[modes == mode]
      count = len(mode_ids)
      if count == 0:
        continue
      if mode == 0:  # Forward walk.
        self.vel_command_b[mode_ids, 0] = self._sample_magnitude(
          count, self.cfg.walk_speed_range
        )
      elif mode == 1:  # Backward walk.
        self.vel_command_b[mode_ids, 0] = -self._sample_magnitude(
          count, self.cfg.backward_speed_range
        )
      elif mode == 2:  # Lateral motion in both directions.
        self.vel_command_b[mode_ids, 1] = self._sample_sign(
          count, self.cfg.lateral_positive_probability
        ) * self._sample_magnitude(count, self.cfg.lateral_speed_range)
      elif mode == 3:  # Turning in both directions.
        self.vel_command_b[mode_ids, 2] = self._sample_sign(
          count, self.cfg.turn_positive_probability
        ) * self._sample_magnitude(count, self.cfg.turn_speed_range)
      elif mode == 4:  # Forward running.
        self.vel_command_b[mode_ids, 0] = self._sample_magnitude(
          count, self.cfg.run_speed_range
        )
      else:  # Mixed translation and turning.
        self.vel_command_b[mode_ids, 0] = self._sample_sign(
          count, self.cfg.mixed_forward_probability
        ) * self._sample_magnitude(count, self.cfg.walk_speed_range)
        self.vel_command_b[mode_ids, 1] = (
          self.cfg.mixed_lateral_scale
          * self._sample_sign(count, self.cfg.lateral_positive_probability)
          * self._sample_magnitude(count, self.cfg.lateral_speed_range)
        )
        self.vel_command_b[mode_ids, 2] = self.cfg.mixed_yaw_scale * self._sample_sign(
          count
        ) * self._sample_magnitude(count, self.cfg.turn_speed_range)


@dataclass(kw_only=True)
class BehaviorVelocityCommandCfg(UniformVelocityCommandCfg):
  """Configuration for behavior-focused velocity command sequences."""

  standing_time_range: tuple[float, float] = (8.0, 20.0)
  moving_time_range: tuple[float, float] = (3.0, 8.0)
  initial_standing_probability: float = 0.20
  stop_probability: float = 0.30
  reverse_probability: float = 0.30
  command_deadzone: float = 0.1
  walk_speed_range: tuple[float, float] = (0.35, 1.0)
  backward_speed_range: tuple[float, float] = (0.30, 0.80)
  lateral_speed_range: tuple[float, float] = (0.30, 0.80)
  turn_speed_range: tuple[float, float] = (0.35, 1.00)
  run_speed_range: tuple[float, float] = (1.30, 2.00)
  mixed_lateral_scale: float = 0.6
  mixed_yaw_scale: float = 0.6
  lateral_positive_probability: float = 0.5
  turn_positive_probability: float = 0.5
  mixed_forward_probability: float = 0.5
  # Forward, backward, lateral, turn, run, and mixed motion.
  mode_weights: tuple[float, float, float, float, float, float] = (
    0.23,
    0.15,
    0.18,
    0.14,
    0.20,
    0.10,
  )

  def build(self, env: ManagerBasedRlEnv) -> BehaviorVelocityCommand:
    return BehaviorVelocityCommand(self, env)

  def __post_init__(self):
    super().__post_init__()
    if self.heading_command:
      raise ValueError("BehaviorVelocityCommand requires heading_command=False.")
    for name in (
      "initial_standing_probability",
      "stop_probability",
      "reverse_probability",
      "lateral_positive_probability",
      "turn_positive_probability",
      "mixed_forward_probability",
    ):
      value = getattr(self, name)
      if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")
    for name in (
      "standing_time_range",
      "moving_time_range",
      "walk_speed_range",
      "backward_speed_range",
      "lateral_speed_range",
      "turn_speed_range",
      "run_speed_range",
    ):
      low, high = getattr(self, name)
      if low < 0.0 or high < low:
        raise ValueError(f"Invalid {name}: {(low, high)}.")
    if len(self.mode_weights) != 6 or sum(self.mode_weights) <= 0.0:
      raise ValueError("mode_weights must contain six values with a positive sum.")
    for name in ("mixed_lateral_scale", "mixed_yaw_scale"):
      value = getattr(self, name)
      if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")
