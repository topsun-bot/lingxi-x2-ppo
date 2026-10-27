"""Compare X2 checkpoint and ONNX actions on observations from the simulator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from onnx.reference import ReferenceEvaluator

TASK_ID = "Agibot-X2-Ultra-Behavior-Kp090-Kd080-Omni-Rehearsal"
COMMANDS = (
  (0.0, 0.0, 0.0),
  (0.35, 0.0, 0.0),
  (-0.20, 0.0, 0.0),
  (0.8, 0.0, 0.0),
  (-0.6, 0.0, 0.0),
  (0.0, 0.6, 0.0),
  (0.0, 0.20, 0.0),
  (0.35, 0.20, 0.0),
  (0.35, 0.42, 0.0),
  (0.0, 0.0, 0.7),
  (1.8, 0.0, 0.0),
)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint_file", type=Path)
  parser.add_argument("onnx_file", type=Path)
  parser.add_argument("--task-id", default=TASK_ID)
  parser.add_argument("--fallback-checkpoint-file", type=Path, default=None)
  parser.add_argument("--zero-command-checkpoint-file", type=Path, default=None)
  parser.add_argument("--lateral-gate-threshold", type=float, default=0.10)
  parser.add_argument("--device", default=None)
  parser.add_argument("--tolerance", type=float, default=1.0e-4)
  args = parser.parse_args()

  import mjlab.tasks  # noqa: F401

  import src.tasks  # noqa: F401

  checkpoint = args.checkpoint_file.expanduser().resolve()
  onnx_file = args.onnx_file.expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  if not onnx_file.is_file():
    raise FileNotFoundError(onnx_file)

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(args.task_id, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = 42
  env_cfg.events.pop("randomize_terrain", None)
  agent_cfg = load_rl_cfg(args.task_id)
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  env.reset()

  runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  torch_policy = runner.get_inference_policy(device=device)
  fallback_policy = None
  if args.fallback_checkpoint_file is not None:
    fallback_checkpoint = args.fallback_checkpoint_file.expanduser().resolve()
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
  onnx_policy = ReferenceEvaluator(onnx.load(onnx_file, load_external_data=False))
  command_term = env.unwrapped.command_manager.get_term("twist")
  assert command_term is not None

  max_error = 0.0
  comparisons = 0
  try:
    for command_values in COMMANDS:
      command = torch.tensor(command_values, device=env.unwrapped.device)
      for _ in range(5):
        command_term.vel_command_b[:] = command
        command_term.is_standing_env[:] = torch.linalg.vector_norm(command) < 0.1
        command_term.is_heading_env[:] = False
        command_term.time_left[:] = 1.0e6
        observations = env.get_observations()
        with torch.inference_mode():
          expected_actions = torch_policy(observations)
          if fallback_policy is not None:
            actor_width = observations["actor"].shape[1]
            command_start = 6 if actor_width == 104 else 15
            use_lateral = (
              torch.abs(observations["actor"][:, command_start + 1])
              > args.lateral_gate_threshold
            ).unsqueeze(1)
            expected_actions = torch.where(
              use_lateral,
              expected_actions,
              fallback_policy(observations),
            )
          if zero_command_policy is not None:
            actor_width = observations["actor"].shape[1]
            command_start = 6 if actor_width == 104 else 15
            current_command = observations["actor"][
              :, command_start : command_start + 3
            ]
            use_zero = (
              torch.linalg.vector_norm(current_command, dim=1) < 0.1
            ).unsqueeze(1)
            expected_actions = torch.where(
              use_zero,
              zero_command_policy(observations),
              expected_actions,
            )
          expected = expected_actions.cpu().numpy()
        actor_observation = observations["actor"].cpu().numpy().astype(np.float32)
        actual = onnx_policy.run(None, {"obs": actor_observation})[0]
        error = float(np.max(np.abs(expected - actual)))
        max_error = max(max_error, error)
        comparisons += expected.size
        env.step(torch.from_numpy(actual).to(env.unwrapped.device))
  finally:
    env.close()

  if max_error > args.tolerance:
    raise AssertionError(
      f"ONNX parity error {max_error:.8g} exceeds tolerance {args.tolerance:.8g}"
    )
  print("X2_ONNX_PARITY_OK")
  print(f"comparisons={comparisons} max_abs_error={max_error:.8g}")


if __name__ == "__main__":
  main()
