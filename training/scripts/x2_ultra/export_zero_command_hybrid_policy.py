"""Export moving and zero-command X2 actors as one routed ONNX policy."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from torch import nn

from src.tasks.velocity.rl.runner import _preserve_numeric_metadata_precision


class ZeroCommandHybridPolicy(nn.Module):
  """Use a settling specialist at zero command and locomotion otherwise."""

  def __init__(
    self,
    moving: nn.Module,
    settling: nn.Module,
    command_start: int,
    threshold: float,
  ) -> None:
    super().__init__()
    self.moving = moving
    self.settling = settling
    self.command_start = command_start
    self.threshold = threshold

  def forward(self, observations: torch.Tensor) -> torch.Tensor:
    command = observations[
      :, self.command_start : self.command_start + 3
    ]
    use_settling = torch.linalg.vector_norm(command, dim=1, keepdim=True) < self.threshold
    return torch.where(
      use_settling,
      self.settling(observations),
      self.moving(observations),
    )


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("moving_checkpoint", type=Path)
  parser.add_argument("settling_checkpoint", type=Path)
  parser.add_argument("output_file", type=Path)
  parser.add_argument(
    "--task-id",
    default=(
      "Agibot-X2-Ultra-Behavior-Factory-Recorded-History2-Full-Envelope"
    ),
  )
  parser.add_argument("--threshold", type=float, default=0.10)
  parser.add_argument("--device", default=None)
  args = parser.parse_args()

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  moving_checkpoint = args.moving_checkpoint.expanduser().resolve()
  settling_checkpoint = args.settling_checkpoint.expanduser().resolve()
  output = args.output_file.expanduser().resolve()
  for checkpoint in (moving_checkpoint, settling_checkpoint):
    if not checkpoint.is_file():
      raise FileNotFoundError(checkpoint)

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(args.task_id, play=True)
  history_length = env_cfg.observations["actor"].history_length
  history_length = int(history_length) if history_length else 1
  if history_length not in (1, 2):
    raise ValueError(f"Unsupported observation history length {history_length}")
  input_width = 104 * history_length
  command_start = 6 if history_length == 1 else 15
  env_cfg.scene.num_envs = 1
  env_cfg.seed = 42
  env_cfg.events.pop("randomize_terrain", None)
  agent_cfg = load_rl_cfg(args.task_id)
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  env.reset()
  runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
  try:
    moving_runner = runner_cls(env, asdict(agent_cfg), device=device)
    moving_runner.load(
      str(moving_checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    settling_runner = runner_cls(env, asdict(agent_cfg), device=device)
    settling_runner.load(
      str(settling_checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    moving = moving_runner.get_inference_policy(device=device).as_onnx(False)
    settling = settling_runner.get_inference_policy(device=device).as_onnx(False)
    model = ZeroCommandHybridPolicy(
      moving, settling, command_start, args.threshold
    ).cpu().eval()
    metadata = get_base_metadata(base_env, "zero_command_hybrid")
    _preserve_numeric_metadata_precision(metadata)
  finally:
    env.close()

  output.parent.mkdir(parents=True, exist_ok=True)
  torch.onnx.export(
    model,
    torch.zeros(1, input_width),
    output,
    export_params=True,
    opset_version=18,
    input_names=["obs"],
    output_names=["actions"],
    dynamic_axes={},
    dynamo=False,
  )
  metadata.update(
    {
      "hybrid_gate": f"norm(current_command)<{args.threshold:.9g}",
      "hybrid_moving_checkpoint_sha256": sha256(moving_checkpoint),
      "hybrid_settling_checkpoint_sha256": sha256(settling_checkpoint),
      "metadata_contract": (
        "agibot_x2_velocity_v2" if history_length > 1
        else "agibot_x2_velocity_v1"
      ),
      "observation_dims": "3,3,3,2,31,31,31",
      "observation_history_length": str(history_length),
      "phase_period": "0.6",
      "policy_dt": "0.02",
    }
  )
  attach_metadata_to_onnx(str(output), metadata)
  print(output)


if __name__ == "__main__":
  main()
