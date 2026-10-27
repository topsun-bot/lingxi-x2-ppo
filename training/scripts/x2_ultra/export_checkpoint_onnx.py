"""Export an arbitrary X2 RSL-RL checkpoint to ONNX with task metadata."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from src.tasks.velocity.rl.runner import _preserve_numeric_metadata_precision


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint_file", type=Path)
  parser.add_argument("output_file", type=Path)
  parser.add_argument(
    "--task-id",
    default="Agibot-X2-Ultra-Behavior-Kp090-Kd080-Omni-Rehearsal",
  )
  parser.add_argument("--device", default=None)
  args = parser.parse_args()

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  checkpoint = args.checkpoint_file.expanduser().resolve()
  output = args.output_file.expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
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
  try:
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=base_env.device,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    runner.export_policy_to_onnx(str(output.parent), output.name)
    metadata = get_base_metadata(base_env, "local_export")
    _preserve_numeric_metadata_precision(metadata)
    history_length = env_cfg.observations["actor"].history_length
    history_length = int(history_length) if history_length else 1
    metadata.update(
      {
        "metadata_contract": (
          "agibot_x2_velocity_v2" if history_length > 1
          else "agibot_x2_velocity_v1"
        ),
        "policy_dt": "0.02",
        "phase_period": "0.6",
        "observation_dims": "3,3,3,2,31,31,31",
        "observation_history_length": str(history_length),
      }
    )
    attach_metadata_to_onnx(str(output), metadata)
  finally:
    env.close()
  print(output)


if __name__ == "__main__":
  main()
