"""Bake small raw-command gain corrections into an X2 actor checkpoint."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class CalibrationConfig:
  checkpoint_file: str
  output_file: str
  lin_x_scale: float = 1.0
  lin_y_scale: float = 1.0
  yaw_scale: float = 1.0
  normalizer_epsilon: float = 1.0e-2


def run_calibration(cfg: CalibrationConfig) -> dict[str, object]:
  scales = (cfg.lin_x_scale, cfg.lin_y_scale, cfg.yaw_scale)
  if any(scale <= 0.0 for scale in scales):
    raise ValueError("All command scales must be positive")

  source = Path(cfg.checkpoint_file).expanduser().resolve()
  if not source.is_file():
    raise FileNotFoundError(source)
  checkpoint = torch.load(source, map_location="cpu", weights_only=False)
  output_checkpoint = copy.deepcopy(checkpoint)
  actor_state = output_checkpoint["actor_state_dict"]

  weight = actor_state["mlp.0.weight"].clone()
  bias = actor_state["mlp.0.bias"].clone()
  mean = actor_state["obs_normalizer._mean"].reshape(-1)
  std = actor_state["obs_normalizer._std"].reshape(-1)
  if weight.shape[1] != 104 or mean.numel() != 104 or std.numel() != 104:
    raise ValueError("Expected the X2 104-element actor observation layout")

  # Actor observations are [ang_vel(3), gravity(3), command(3), ...].
  # Scaling a first-layer column changes the raw command slope.  The bias
  # correction preserves the exact pre-activation at a zero raw command.
  command_indices = (6, 7, 8)
  for index, scale in zip(command_indices, scales, strict=True):
    original_column = weight[:, index].clone()
    weight[:, index] = original_column * scale
    bias += (
      (scale - 1.0)
      * original_column
      * mean[index]
      / (std[index] + cfg.normalizer_epsilon)
    )

  actor_state["mlp.0.weight"] = weight
  actor_state["mlp.0.bias"] = bias
  infos = dict(output_checkpoint.get("infos") or {})
  infos["command_calibration"] = {
    "source_checkpoint": str(source),
    "lin_x_scale": cfg.lin_x_scale,
    "lin_y_scale": cfg.lin_y_scale,
    "yaw_scale": cfg.yaw_scale,
    "zero_command_bias_compensated": True,
  }
  output_checkpoint["infos"] = infos

  output = Path(cfg.output_file).expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(output_checkpoint, output)
  report = {
    "source_checkpoint": str(source),
    "output_checkpoint": str(output),
    "scales": {
      "lin_x": cfg.lin_x_scale,
      "lin_y": cfg.lin_y_scale,
      "yaw": cfg.yaw_scale,
    },
    "zero_command_bias_compensated": True,
  }
  report_path = output.with_suffix(".json")
  report_path.write_text(json.dumps(report, indent=2) + "\n")
  print(f"[INFO] Saved calibrated checkpoint: {output}")
  print(f"[INFO] Report: {report_path}")
  return report


def main() -> None:
  import tyro

  run_calibration(tyro.cli(CalibrationConfig))


if __name__ == "__main__":
  main()
