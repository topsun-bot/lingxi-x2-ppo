"""Linearly blend two compatible X2 actor checkpoints."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class BlendConfig:
  base_checkpoint: str
  donor_checkpoint: str
  output_file: str
  donor_weight: float


def run_blend(cfg: BlendConfig) -> dict[str, object]:
  if not 0.0 <= cfg.donor_weight <= 1.0:
    raise ValueError("donor_weight must be in [0, 1]")
  base_path = Path(cfg.base_checkpoint).expanduser().resolve()
  donor_path = Path(cfg.donor_checkpoint).expanduser().resolve()
  for path in (base_path, donor_path):
    if not path.is_file():
      raise FileNotFoundError(path)

  base = torch.load(base_path, map_location="cpu", weights_only=False)
  donor = torch.load(donor_path, map_location="cpu", weights_only=False)
  output_checkpoint = copy.deepcopy(base)
  base_actor = base["actor_state_dict"]
  donor_actor = donor["actor_state_dict"]
  if base_actor.keys() != donor_actor.keys():
    raise ValueError("Actor checkpoint keys do not match")

  alpha = cfg.donor_weight
  blended_actor: dict[str, torch.Tensor] = {}
  for key, base_value in base_actor.items():
    donor_value = donor_actor[key]
    if base_value.shape != donor_value.shape:
      raise ValueError(f"Actor tensor shape mismatch for {key}")
    if torch.is_floating_point(base_value):
      blended_actor[key] = torch.lerp(base_value, donor_value, alpha)
    else:
      blended_actor[key] = copy.deepcopy(base_value)
  output_checkpoint["actor_state_dict"] = blended_actor

  infos = dict(output_checkpoint.get("infos") or {})
  infos["checkpoint_blend"] = {
    "base_checkpoint": str(base_path),
    "donor_checkpoint": str(donor_path),
    "donor_weight": alpha,
    "actor_only": True,
  }
  output_checkpoint["infos"] = infos
  output_path = Path(cfg.output_file).expanduser().resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(output_checkpoint, output_path)
  report = {
    "output_checkpoint": str(output_path),
    **infos["checkpoint_blend"],
  }
  report_path = output_path.with_suffix(".json")
  report_path.write_text(json.dumps(report, indent=2) + "\n")
  print(f"[INFO] Saved blended checkpoint: {output_path}")
  print(f"[INFO] Report: {report_path}")
  return report


def main() -> None:
  import tyro

  run_blend(tyro.cli(BlendConfig))


if __name__ == "__main__":
  main()
