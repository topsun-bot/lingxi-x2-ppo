"""Interpolate policy MLP weights while preserving deployment normalization."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--base", type=Path, required=True)
  parser.add_argument("--refined", type=Path, required=True)
  parser.add_argument("--alpha", type=float, required=True)
  parser.add_argument(
    "--command-alpha",
    type=float,
    default=None,
    help="Optional interpolation ratio for command columns 6:9 of mlp.0.weight.",
  )
  parser.add_argument(
    "--command-indices",
    type=int,
    nargs="+",
    default=[6, 7, 8],
    help="First-layer actor input columns controlled by --command-alpha.",
  )
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if not 0.0 <= args.alpha <= 1.0:
    raise ValueError("--alpha must be in [0, 1]")
  if args.command_alpha is not None and not 0.0 <= args.command_alpha <= 1.0:
    raise ValueError("--command-alpha must be in [0, 1]")
  if any(index < 0 for index in args.command_indices):
    raise ValueError("--command-indices must be non-negative")

  base = torch.load(args.base, map_location="cpu", weights_only=False)
  refined = torch.load(args.refined, map_location="cpu", weights_only=False)
  base_actor = base["actor_state_dict"]
  refined_actor = refined["actor_state_dict"]
  if base_actor.keys() != refined_actor.keys():
    raise ValueError("Actor state dictionaries have different keys")
  command_key = "mlp.0.weight"
  command_indices = args.command_indices
  if any(index >= base_actor[command_key].shape[1] for index in command_indices):
    raise ValueError("--command-indices contains an out-of-range actor input")
  base_command_columns = base_actor[command_key][:, command_indices].clone()

  blended_keys: list[str] = []
  for key, base_value in base_actor.items():
    refined_value = refined_actor[key]
    if base_value.shape != refined_value.shape:
      raise ValueError(f"Shape mismatch for {key}")
    if key.startswith("mlp.") and torch.is_floating_point(base_value):
      base_actor[key] = torch.lerp(base_value, refined_value, args.alpha)
      blended_keys.append(key)

  if args.command_alpha is not None:
    base_actor[command_key][:, command_indices] = torch.lerp(
      base_command_columns,
      refined_actor[command_key][:, command_indices],
      args.command_alpha,
    )

  base["iter"] = int(refined["iter"])
  infos = dict(base.get("infos") or {})
  infos["actor_interpolation"] = {
    "base": str(args.base.resolve()),
    "refined": str(args.refined.resolve()),
    "alpha": args.alpha,
    "command_alpha": args.command_alpha,
    "command_indices": command_indices,
    "blended_keys": blended_keys,
    "normalization_source": "base",
  }
  base["infos"] = infos
  args.output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(base, args.output)
  print(f"Saved {args.output} with alpha={args.alpha:.3f}")


if __name__ == "__main__":
  main()
