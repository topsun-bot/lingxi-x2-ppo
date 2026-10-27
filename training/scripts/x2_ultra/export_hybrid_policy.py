"""Export lateral and fallback actors as one routed ONNX policy."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path

import onnx
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from torch import nn

TASK_ID = "Agibot-X2-Ultra-Behavior-Flat"


class HybridOnnxPolicy(nn.Module):
  """Route lateral commands to a specialist and all other commands to fallback."""

  def __init__(self, lateral: nn.Module, fallback: nn.Module, threshold: float):
    super().__init__()
    self.lateral = lateral
    self.fallback = fallback
    self.threshold = threshold

  def forward(self, observations: torch.Tensor) -> torch.Tensor:
    use_lateral = torch.abs(observations[:, 7:8]) > self.threshold
    return torch.where(
      use_lateral,
      self.lateral(observations),
      self.fallback(observations),
    )


def add_metadata(path: Path, values: dict[str, str]) -> None:
  model = onnx.load(path, load_external_data=False)
  retained = {item.key: item.value for item in model.metadata_props}
  retained.update(values)
  del model.metadata_props[:]
  for key in sorted(retained):
    entry = model.metadata_props.add()
    entry.key = key
    entry.value = retained[key]
  onnx.checker.check_model(model)
  onnx.save(model, path)


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("lateral_checkpoint", type=Path)
  parser.add_argument("fallback_checkpoint", type=Path)
  parser.add_argument("output_file", type=Path)
  parser.add_argument("--threshold", type=float, default=0.10)
  parser.add_argument("--device", default=None)
  args = parser.parse_args()

  import mjlab.tasks  # noqa: F401

  import src.tasks  # noqa: F401

  lateral_checkpoint = args.lateral_checkpoint.expanduser().resolve()
  fallback_checkpoint = args.fallback_checkpoint.expanduser().resolve()
  output = args.output_file.expanduser().resolve()
  for checkpoint in (lateral_checkpoint, fallback_checkpoint):
    if not checkpoint.is_file():
      raise FileNotFoundError(checkpoint)

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = 42
  env_cfg.events.pop("randomize_terrain", None)
  agent_cfg = load_rl_cfg(TASK_ID)
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  env.reset()
  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  try:
    lateral_runner = runner_cls(env, asdict(agent_cfg), device=device)
    lateral_runner.load(
      str(lateral_checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    fallback_runner = runner_cls(env, asdict(agent_cfg), device=device)
    fallback_runner.load(
      str(fallback_checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    lateral = lateral_runner.get_inference_policy(device=device).as_onnx(False)
    fallback = fallback_runner.get_inference_policy(device=device).as_onnx(False)
    model = HybridOnnxPolicy(lateral, fallback, args.threshold).cpu().eval()
  finally:
    env.close()

  output.parent.mkdir(parents=True, exist_ok=True)
  torch.onnx.export(
    model,
    torch.zeros(1, 104),
    output,
    export_params=True,
    opset_version=18,
    input_names=["obs"],
    output_names=["actions"],
    dynamic_axes={},
    dynamo=False,
  )
  add_metadata(
    output,
    {
      "hybrid_gate": f"abs(command_vy)>{args.threshold:.9g}",
      "hybrid_lateral_checkpoint_sha256": sha256(lateral_checkpoint),
      "hybrid_fallback_checkpoint_sha256": sha256(fallback_checkpoint),
    },
  )
  print(output)


if __name__ == "__main__":
  main()
