"""Expand a feed-forward actor checkpoint to term-wise observation history.

The upgraded actor is initially identical to the source actor: its first layer
uses only the newest value in every observation term, while the older history
slots start with zero weight.  PPO can then learn temporal corrections without
destroying a working gait at initialization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _expand_termwise(value: torch.Tensor, term_dims: list[int], history: int) -> torch.Tensor:
  """Repeat a 1-D observation statistic in oldest-to-newest term order."""
  chunks = torch.split(value, term_dims, dim=-1)
  return torch.cat(
    [torch.cat([chunk] * history, dim=-1) for chunk in chunks], dim=-1
  )


def _expand_first_layer(
  weight: torch.Tensor, term_dims: list[int], history: int
) -> torch.Tensor:
  expanded = weight.new_zeros((weight.shape[0], weight.shape[1] * history))
  old_start = 0
  new_start = 0
  for dim in term_dims:
    # CircularBuffer exposes chronological history, so the newest frame is the
    # final block for each term.
    newest_start = new_start + (history - 1) * dim
    expanded[:, newest_start : newest_start + dim] = weight[
      :, old_start : old_start + dim
    ]
    old_start += dim
    new_start += history * dim
  return expanded


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--history", type=int, default=2)
  parser.add_argument(
    "--term-dims",
    type=int,
    nargs="+",
    default=[3, 3, 3, 2, 31, 31, 31],
    help="Actor observation term dimensions in manager concatenation order.",
  )
  args = parser.parse_args()
  if args.history < 2:
    raise ValueError("--history must be at least 2")

  checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
  actor = checkpoint["actor_state_dict"]
  old_dim = actor["mlp.0.weight"].shape[1]
  if sum(args.term_dims) != old_dim:
    raise ValueError(
      f"term dimensions sum to {sum(args.term_dims)}, expected actor input {old_dim}"
    )

  actor["mlp.0.weight"] = _expand_first_layer(
    actor["mlp.0.weight"], args.term_dims, args.history
  )
  for key in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
    actor[key] = _expand_termwise(actor[key], args.term_dims, args.history)

  # Adam's first/second moments for the expanded first layer have the old
  # shape.  Keep the optimizer parameter groups, but clear all moments so the
  # resumed run starts with a shape-safe optimizer state.
  optimizer = checkpoint.get("optimizer_state_dict")
  if isinstance(optimizer, dict):
    optimizer["state"] = {}

  infos = dict(checkpoint.get("infos") or {})
  infos["actor_history_upgrade"] = {
    "source": str(args.input.resolve()),
    "history": args.history,
    "term_dims": args.term_dims,
    "ordering": "term-wise oldest-to-newest",
    "initial_policy": "newest frame only",
    "optimizer_state": "reset",
  }
  checkpoint["infos"] = infos
  args.output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(checkpoint, args.output)
  print(f"Saved {args.output} with actor input {old_dim} -> {old_dim * args.history}")


if __name__ == "__main__":
  main()
