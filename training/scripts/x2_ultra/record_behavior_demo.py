"""Record a labeled headless video of the final X2 behavior policy."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import mediapy as media
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from PIL import Image, ImageDraw, ImageFont

TASK_ID = "Agibot-X2-Ultra-Behavior-Flat"


@dataclass(frozen=True)
class DemoStage:
  label: str
  command: tuple[float, float, float]
  duration_s: float


STAGES = (
  DemoStage("ZERO COMMAND STAND", (0.0, 0.0, 0.0), 3.0),
  DemoStage("FORWARD", (0.8, 0.0, 0.0), 4.0),
  DemoStage("BRAKE TO STAND", (0.0, 0.0, 0.0), 3.0),
  DemoStage("RESTART", (0.8, 0.0, 0.0), 3.0),
  DemoStage("BACKWARD", (-0.6, 0.0, 0.0), 3.0),
  DemoStage("LATERAL LEFT", (0.0, 0.6, 0.0), 3.0),
  DemoStage("LATERAL RIGHT", (0.0, -0.6, 0.0), 3.0),
  DemoStage("TURN LEFT", (0.0, 0.0, 0.7), 3.0),
  DemoStage("TURN RIGHT", (0.0, 0.0, -0.7), 3.0),
  DemoStage("RUN", (1.8, 0.0, 0.0), 4.0),
  DemoStage("FINAL BRAKE", (0.0, 0.0, 0.0), 3.0),
)

LOW_SPEED_STAGES = (
  DemoStage("ZERO COMMAND STAND", (0.0, 0.0, 0.0), 3.0),
  DemoStage("FORWARD 0.35 M/S", (0.35, 0.0, 0.0), 4.0),
  DemoStage("BRAKE TO STAND", (0.0, 0.0, 0.0), 3.0),
  DemoStage("RESTART 0.35 M/S", (0.35, 0.0, 0.0), 3.0),
  DemoStage("BACKWARD -0.20 M/S", (-0.20, 0.0, 0.0), 3.0),
  DemoStage("LATERAL LEFT 0.15 M/S", (0.0, 0.15, 0.0), 3.0),
  DemoStage("LATERAL RIGHT -0.15 M/S", (0.0, -0.15, 0.0), 3.0),
  DemoStage("TURN LEFT 0.25 RAD/S", (0.0, 0.0, 0.25), 3.0),
  DemoStage("TURN RIGHT -0.25 RAD/S", (0.0, 0.0, -0.25), 3.0),
  DemoStage("FINAL BRAKE", (0.0, 0.0, 0.0), 3.0),
)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
  return ImageFont.truetype(path, size) if path.is_file() else ImageFont.load_default()


def _annotate(
  frame: np.ndarray,
  stage: DemoStage,
  elapsed_s: float,
  linear_velocity: np.ndarray,
  yaw_rate: float,
) -> np.ndarray:
  if frame.dtype != np.uint8:
    frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
  image = Image.fromarray(frame)
  draw = ImageDraw.Draw(image, "RGBA")
  draw.rounded_rectangle((14, 14, 626, 91), radius=10, fill=(0, 0, 0, 172))
  draw.text((28, 23), f"X2 ULTRA  |  {stage.label}", font=_font(22), fill="white")
  command = stage.command
  detail = (
    f"t={elapsed_s:04.1f}s   cmd [{command[0]:+.1f}, {command[1]:+.1f}, "
    f"{command[2]:+.1f}]   actual [{linear_velocity[0]:+.2f}, "
    f"{linear_velocity[1]:+.2f}, {yaw_rate:+.2f}]"
  )
  draw.text((28, 59), detail, font=_font(14), fill=(220, 240, 255))
  return np.asarray(image)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint_file", type=Path)
  parser.add_argument("output_file", type=Path)
  parser.add_argument("--zero-command-checkpoint-file", type=Path, default=None)
  parser.add_argument("--task-id", default=TASK_ID)
  parser.add_argument("--profile", choices=("full", "low-speed"), default="full")
  parser.add_argument("--device", default=None)
  parser.add_argument("--width", type=int, default=640)
  parser.add_argument("--height", type=int, default=360)
  parser.add_argument("--fps", type=int, default=25)
  args = parser.parse_args()

  import mjlab.tasks  # noqa: F401

  import src.tasks  # noqa: F401

  checkpoint = args.checkpoint_file.expanduser().resolve()
  output = args.output_file.expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  output.parent.mkdir(parents=True, exist_ok=True)

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(args.task_id, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = 42
  env_cfg.events.pop("randomize_terrain", None)
  env_cfg.viewer.width = args.width
  env_cfg.viewer.height = args.height
  env_cfg.viewer.distance = 3.2
  env_cfg.viewer.azimuth = 135.0
  env_cfg.viewer.elevation = -10.0

  agent_cfg = load_rl_cfg(args.task_id)
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  env.reset()
  runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)
  zero_command_policy = None
  if args.zero_command_checkpoint_file is not None:
    zero_checkpoint = args.zero_command_checkpoint_file.expanduser().resolve()
    if not zero_checkpoint.is_file():
      raise FileNotFoundError(zero_checkpoint)
    zero_runner = runner_cls(env, asdict(agent_cfg), device=device)
    zero_runner.load(
      str(zero_checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    zero_command_policy = zero_runner.get_inference_policy(device=device)
  command_term = env.unwrapped.command_manager.get_term("twist")
  assert command_term is not None
  robot = env.unwrapped.scene["robot"]

  control_fps = round(1.0 / env.unwrapped.step_dt)
  frame_stride = max(1, round(control_fps / args.fps))
  output_fps = control_fps / frame_stride
  frames: list[np.ndarray] = []
  elapsed_s = 0.0
  try:
    stages = LOW_SPEED_STAGES if args.profile == "low-speed" else STAGES
    for stage in stages:
      command = torch.tensor(stage.command, device=env.unwrapped.device)
      steps = round(stage.duration_s / env.unwrapped.step_dt)
      for step in range(steps):
        command_term.vel_command_b[:] = command
        command_term.is_standing_env[:] = torch.linalg.vector_norm(command) < 0.1
        command_term.is_heading_env[:] = False
        command_term.time_left[:] = 1.0e6
        observations = env.get_observations()
        with torch.inference_mode():
          active_policy = (
            zero_command_policy
            if zero_command_policy is not None
            and torch.linalg.vector_norm(command) < 0.1
            else policy
          )
          actions = active_policy(observations)
        env.step(actions)
        if step % frame_stride == 0:
          rendered = base_env.render()
          if rendered is None:
            raise RuntimeError("Offscreen renderer returned no frame.")
          frame = rendered[0] if rendered.ndim == 4 else rendered
          linear_velocity = robot.data.root_link_lin_vel_b[0, :2].cpu().numpy()
          yaw_rate = float(robot.data.root_link_ang_vel_b[0, 2].item())
          frames.append(
            _annotate(frame, stage, elapsed_s, linear_velocity, yaw_rate)
          )
        elapsed_s += env.unwrapped.step_dt
  finally:
    env.close()

  media.write_video(str(output), frames, fps=output_fps)
  print("X2_DEMO_VIDEO_OK")
  print(f"file={output} duration={len(frames) / output_fps:.1f}s fps={output_fps:g}")


if __name__ == "__main__":
  main()
