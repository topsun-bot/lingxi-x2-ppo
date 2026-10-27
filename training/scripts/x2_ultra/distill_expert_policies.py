"""Distill sagittal, lateral, and turning X2 policies into one actor checkpoint."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from torch import nn


@dataclass(frozen=True)
class DistillConfig:
  sagittal_checkpoint: str
  lateral_checkpoint: str
  turn_checkpoint: str
  output_file: str
  initial_checkpoint: str | None = None
  sagittal_task_id: str = (
    "Agibot-X2-Ultra-Behavior-Kp090-Kd080-Sagittal-Balance"
  )
  lateral_task_id: str = (
    "Agibot-X2-Ultra-Behavior-Kp090-Kd080-Lateral-Balance"
  )
  turn_task_id: str = "Agibot-X2-Ultra-Behavior-Kp090-Kd080-Turn-Bootstrap"
  num_envs: int = 256
  steps_per_expert: int = 600
  sample_stride: int = 2
  epochs: int = 80
  batch_size: int = 4096
  learning_rate: float = 3.0e-4
  validation_fraction: float = 0.10
  sagittal_loss_weight: float = 1.0
  lateral_loss_weight: float = 1.0
  turn_loss_weight: float = 1.0
  seed: int = 42
  device: str | None = None


@dataclass(frozen=True)
class ExpertSpec:
  name: str
  task_id: str
  checkpoint: Path


def _collect_expert(
  spec: ExpertSpec,
  cfg: DistillConfig,
  device: str,
  seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  env_cfg = load_env_cfg(spec.task_id)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = seed
  agent_cfg = load_rl_cfg(spec.task_id)
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  env.reset()

  runner_cls = load_runner_cls(spec.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(spec.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)
  observations_out: list[torch.Tensor] = []
  actions_out: list[torch.Tensor] = []
  try:
    for step in range(cfg.steps_per_expert):
      observations = env.get_observations()
      with torch.inference_mode():
        actions = policy(observations)
      if step % cfg.sample_stride == 0:
        observations_out.append(observations["actor"].detach().cpu().clone())
        actions_out.append(actions.detach().cpu().clone())
      env.step(actions)
  finally:
    env.close()
  del runner, policy, env, base_env
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
  observations = torch.cat(observations_out)
  actions = torch.cat(actions_out)
  print(
    f"[INFO] Collected {len(observations)} samples from {spec.name}: "
    f"{spec.checkpoint}"
  )
  return observations, actions


def _make_actor(actor_state: dict[str, torch.Tensor], device: str) -> nn.Sequential:
  dims = [
    actor_state["mlp.0.weight"].shape[1],
    actor_state["mlp.0.weight"].shape[0],
    actor_state["mlp.2.weight"].shape[0],
    actor_state["mlp.4.weight"].shape[0],
    actor_state["mlp.6.weight"].shape[0],
  ]
  actor = nn.Sequential(
    nn.Linear(dims[0], dims[1]),
    nn.ELU(),
    nn.Linear(dims[1], dims[2]),
    nn.ELU(),
    nn.Linear(dims[2], dims[3]),
    nn.ELU(),
    nn.Linear(dims[3], dims[4]),
  ).to(device)
  actor.load_state_dict(
    {
      key.removeprefix("mlp."): value
      for key, value in actor_state.items()
      if key.startswith("mlp.")
    }
  )
  return actor


def run_distillation(cfg: DistillConfig) -> dict[str, object]:
  configure_torch_backends()
  torch.manual_seed(cfg.seed)
  expert_loss_weights = torch.tensor(
    [
      cfg.sagittal_loss_weight,
      cfg.lateral_loss_weight,
      cfg.turn_loss_weight,
    ],
    dtype=torch.float32,
  )
  if torch.any(expert_loss_weights <= 0):
    raise ValueError("All expert loss weights must be positive")
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  experts = (
    ExpertSpec(
      "sagittal",
      cfg.sagittal_task_id,
      Path(cfg.sagittal_checkpoint).expanduser().resolve(),
    ),
    ExpertSpec(
      "lateral",
      cfg.lateral_task_id,
      Path(cfg.lateral_checkpoint).expanduser().resolve(),
    ),
    ExpertSpec(
      "turn",
      cfg.turn_task_id,
      Path(cfg.turn_checkpoint).expanduser().resolve(),
    ),
  )
  for expert in experts:
    if not expert.checkpoint.is_file():
      raise FileNotFoundError(expert.checkpoint)

  observations_by_expert: list[torch.Tensor] = []
  actions_by_expert: list[torch.Tensor] = []
  labels: list[torch.Tensor] = []
  for index, expert in enumerate(experts):
    observations, actions = _collect_expert(
      expert, cfg, device, cfg.seed + index
    )
    observations_by_expert.append(observations)
    actions_by_expert.append(actions)
    labels.append(torch.full((len(observations),), index, dtype=torch.long))

  observations = torch.cat(observations_by_expert)
  actions = torch.cat(actions_by_expert)
  expert_labels = torch.cat(labels)
  generator = torch.Generator().manual_seed(cfg.seed)
  permutation = torch.randperm(len(observations), generator=generator)
  validation_count = max(1, round(len(observations) * cfg.validation_fraction))
  validation_indices = permutation[:validation_count]
  train_indices = permutation[validation_count:]

  train_observations = observations[train_indices]
  normalizer_mean = train_observations.mean(dim=0, keepdim=True)
  normalizer_var = train_observations.var(dim=0, unbiased=False, keepdim=True)
  normalizer_std = torch.sqrt(normalizer_var)
  del train_observations

  initial_checkpoint = (
    Path(cfg.initial_checkpoint).expanduser().resolve()
    if cfg.initial_checkpoint is not None
    else experts[2].checkpoint
  )
  if not initial_checkpoint.is_file():
    raise FileNotFoundError(initial_checkpoint)
  base_checkpoint = torch.load(
    initial_checkpoint, map_location="cpu", weights_only=False
  )
  actor_state = base_checkpoint["actor_state_dict"]
  actor = _make_actor(actor_state, device)
  optimizer = torch.optim.AdamW(
    actor.parameters(), lr=cfg.learning_rate, weight_decay=1.0e-6
  )
  loss_fn = nn.MSELoss()
  eps = 1.0e-2
  mean_device = normalizer_mean.to(device)
  std_device = normalizer_std.to(device)
  expert_loss_weights_device = expert_loss_weights.to(device)

  best_validation_loss = float("inf")
  best_actor_state: dict[str, torch.Tensor] | None = None
  for epoch in range(cfg.epochs):
    actor.train()
    order = train_indices[torch.randperm(len(train_indices), generator=generator)]
    train_loss_sum = 0.0
    train_samples = 0
    for start in range(0, len(order), cfg.batch_size):
      indices = order[start : start + cfg.batch_size]
      x = observations[indices].to(device)
      y = actions[indices].to(device)
      prediction = actor((x - mean_device) / (std_device + eps))
      sample_loss = torch.mean((prediction - y) ** 2, dim=1)
      sample_weights = expert_loss_weights_device[
        expert_labels[indices].to(device)
      ]
      loss = torch.sum(sample_loss * sample_weights) / torch.sum(sample_weights)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
      optimizer.step()
      train_loss_sum += float(loss.item()) * len(indices)
      train_samples += len(indices)

    actor.eval()
    validation_loss_sum = 0.0
    validation_samples = 0
    with torch.inference_mode():
      for start in range(0, len(validation_indices), cfg.batch_size):
        indices = validation_indices[start : start + cfg.batch_size]
        x = observations[indices].to(device)
        y = actions[indices].to(device)
        prediction = actor((x - mean_device) / (std_device + eps))
        loss = loss_fn(prediction, y)
        validation_loss_sum += float(loss.item()) * len(indices)
        validation_samples += len(indices)
    train_loss = train_loss_sum / max(1, train_samples)
    validation_loss = validation_loss_sum / max(1, validation_samples)
    if validation_loss < best_validation_loss:
      best_validation_loss = validation_loss
      best_actor_state = copy.deepcopy(actor.state_dict())
    if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == cfg.epochs:
      print(
        f"[INFO] Epoch {epoch + 1:03d}/{cfg.epochs}: "
        f"train_mse={train_loss:.6f} val_mse={validation_loss:.6f}"
      )

  assert best_actor_state is not None
  actor.load_state_dict(best_actor_state)
  actor.eval()
  validation_by_expert: dict[str, float] = {}
  with torch.inference_mode():
    for index, expert in enumerate(experts):
      indices = validation_indices[expert_labels[validation_indices] == index]
      x = observations[indices].to(device)
      y = actions[indices].to(device)
      prediction = actor((x - mean_device) / (std_device + eps))
      validation_by_expert[expert.name] = float(loss_fn(prediction, y).item())

  output_checkpoint = copy.deepcopy(base_checkpoint)
  output_actor_state = output_checkpoint["actor_state_dict"]
  for key, value in best_actor_state.items():
    output_actor_state[f"mlp.{key}"] = value.detach().cpu()
  output_actor_state["obs_normalizer._mean"] = normalizer_mean.cpu()
  output_actor_state["obs_normalizer._var"] = normalizer_var.cpu()
  output_actor_state["obs_normalizer._std"] = normalizer_std.cpu()
  output_actor_state["obs_normalizer.count"] = torch.tensor(
    len(train_indices), dtype=torch.long
  )
  output_checkpoint["iter"] = max(
    int(torch.load(e.checkpoint, map_location="cpu", weights_only=False)["iter"])
    for e in experts
  )
  infos = dict(output_checkpoint.get("infos") or {})
  infos["distillation"] = {
    "initial_checkpoint": str(initial_checkpoint),
    "experts": {
      expert.name: str(expert.checkpoint) for expert in experts
    },
    "samples": len(observations),
    "epochs": cfg.epochs,
    "expert_loss_weights": {
      expert.name: float(expert_loss_weights[index])
      for index, expert in enumerate(experts)
    },
    "validation_mse": best_validation_loss,
    "validation_mse_by_expert": validation_by_expert,
  }
  output_checkpoint["infos"] = infos

  output_path = Path(cfg.output_file).expanduser().resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(output_checkpoint, output_path)
  report = {
    "output_checkpoint": str(output_path),
    "initial_checkpoint": str(initial_checkpoint),
    "samples": len(observations),
    "expert_loss_weights": {
      expert.name: float(expert_loss_weights[index])
      for index, expert in enumerate(experts)
    },
    "validation_mse": best_validation_loss,
    "validation_mse_by_expert": validation_by_expert,
    "experts": {expert.name: str(expert.checkpoint) for expert in experts},
  }
  report_path = output_path.with_suffix(".json")
  report_path.write_text(json.dumps(report, indent=2) + "\n")
  print(f"[INFO] Saved distilled checkpoint: {output_path}")
  print(f"[INFO] Report: {report_path}")
  return report


def main() -> None:
  import tyro

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  run_distillation(tyro.cli(DistillConfig))


if __name__ == "__main__":
  main()
