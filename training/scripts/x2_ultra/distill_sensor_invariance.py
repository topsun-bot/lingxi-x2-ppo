"""Distill a velocity actor that preserves clean behavior under sensor error."""

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


SENSOR_SLICES = ((0, 3), (3, 6), (11, 42), (42, 73))
JOINT_POSITION_SLICE = (11, 42)
STABILITY_ACTION_INDICES = (1, 2, 5, 7, 8, 11, 12)
JOINT_MIRROR_PERMUTATION = (
  6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5,
  12, 13, 14,
  22, 23, 24, 25, 26, 27, 28,
  15, 16, 17, 18, 19, 20, 21,
  29, 30,
)
JOINT_MIRROR_SIGNS = (
  1, -1, -1, 1, 1, -1,
  1, -1, -1, 1, 1, -1,
  -1, 1, -1,
  1, -1, -1, 1, -1, 1, -1,
  1, -1, -1, 1, -1, 1, -1,
  -1, 1,
)


@dataclass(frozen=True)
class SensorInvariantConfig:
  teacher_checkpoint: str
  output_file: str
  task_id: str = "Agibot-X2-Ultra-Behavior-Factory-Recorded-Sensor-Stability"
  num_envs: int = 256
  collection_steps: int = 1500
  sample_stride: int = 2
  epochs: int = 30
  batch_size: int = 4096
  learning_rate: float = 5.0e-5
  clean_anchor_weight: float = 6.0
  delayed_loss_weight: float = 1.0
  stability_action_weight: float = 3.0
  symmetry_loss_weight: float = 0.0
  delay_probability: float = 0.25
  encoder_bias: float = 0.003
  validation_fraction: float = 0.10
  seed: int = 42
  device: str | None = None


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


def _augment_observations(
  observations: torch.Tensor,
  previous_observations: torch.Tensor,
  delay_probability: float,
  encoder_bias: float,
  generator: torch.Generator,
) -> torch.Tensor:
  augmented = observations.clone()
  for start, end in SENSOR_SLICES:
    delayed = torch.rand(
      (len(augmented), 1), generator=generator, dtype=torch.float32
    ) < delay_probability
    augmented[:, start:end] = torch.where(
      delayed, previous_observations[:, start:end], augmented[:, start:end]
    )
  if encoder_bias > 0.0:
    start, end = JOINT_POSITION_SLICE
    bias = (
      torch.rand(
        (len(augmented), end - start),
        generator=generator,
        dtype=torch.float32,
      )
      * 2.0
      - 1.0
    ) * encoder_bias
    augmented[:, start:end] += bias
  return augmented


def _mirror_joint_block(values: torch.Tensor) -> torch.Tensor:
  permutation = torch.tensor(JOINT_MIRROR_PERMUTATION, device=values.device)
  signs = torch.tensor(
    JOINT_MIRROR_SIGNS, device=values.device, dtype=values.dtype
  )
  return values[:, permutation] * signs


def _mirror_observations(observations: torch.Tensor) -> torch.Tensor:
  mirrored = observations.clone()
  mirrored[:, 0:3] = observations[:, 0:3] * observations.new_tensor((-1, 1, -1))
  mirrored[:, 3:6] = observations[:, 3:6] * observations.new_tensor((1, -1, 1))
  mirrored[:, 6:9] = observations[:, 6:9] * observations.new_tensor((1, -1, -1))
  mirrored[:, 9:11] = -observations[:, 9:11]
  mirrored[:, 11:42] = _mirror_joint_block(observations[:, 11:42])
  mirrored[:, 42:73] = _mirror_joint_block(observations[:, 42:73])
  mirrored[:, 73:104] = _mirror_joint_block(observations[:, 73:104])
  return mirrored


def _mirror_actions(actions: torch.Tensor) -> torch.Tensor:
  return _mirror_joint_block(actions)


def _collect_teacher_data(
  cfg: SensorInvariantConfig,
  checkpoint: Path,
  device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  env_cfg = load_env_cfg(cfg.task_id, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  env_cfg.episode_length_s = int(1e9)
  env_cfg.events.pop("randomize_terrain", None)
  agent_cfg = load_rl_cfg(cfg.task_id)
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  env.reset()
  runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  observations_out: list[torch.Tensor] = []
  previous_out: list[torch.Tensor] = []
  actions_out: list[torch.Tensor] = []
  previous = env.get_observations()["actor"].detach().clone()
  try:
    for step in range(cfg.collection_steps):
      observations = env.get_observations()
      actor_observations = observations["actor"]
      with torch.inference_mode():
        actions = policy(observations)
      if step % cfg.sample_stride == 0:
        observations_out.append(actor_observations.detach().cpu().clone())
        previous_out.append(previous.detach().cpu().clone())
        actions_out.append(actions.detach().cpu().clone())
      env.step(actions)
      previous = actor_observations.detach().clone()
  finally:
    env.close()
  del runner, policy, env, base_env
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
  observations = torch.cat(observations_out)
  previous_observations = torch.cat(previous_out)
  actions = torch.cat(actions_out)
  print(f"[INFO] Collected {len(observations)} clean teacher samples")
  return observations, previous_observations, actions


def run_distillation(cfg: SensorInvariantConfig) -> dict[str, object]:
  if not 0.0 <= cfg.delay_probability <= 1.0:
    raise ValueError("delay_probability must be in [0, 1]")
  if cfg.clean_anchor_weight <= 0.0 or cfg.delayed_loss_weight <= 0.0:
    raise ValueError("loss weights must be positive")
  if cfg.stability_action_weight < 1.0:
    raise ValueError("stability_action_weight must be at least 1")
  if cfg.symmetry_loss_weight < 0.0:
    raise ValueError("symmetry_loss_weight must be non-negative")
  configure_torch_backends()
  torch.manual_seed(cfg.seed)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  checkpoint = Path(cfg.teacher_checkpoint).expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)

  observations, previous_observations, actions = _collect_teacher_data(
    cfg, checkpoint, device
  )
  base_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
  actor_state = base_checkpoint["actor_state_dict"]
  actor = _make_actor(actor_state, device)
  mean = actor_state["obs_normalizer._mean"].reshape(1, -1).to(device)
  std = actor_state["obs_normalizer._std"].reshape(1, -1).to(device)
  eps = 1.0e-2
  action_weights = torch.ones(actions.shape[1], device=device)
  action_weights[list(STABILITY_ACTION_INDICES)] = cfg.stability_action_weight
  symmetry_action_weights = torch.zeros(actions.shape[1], device=device)
  symmetry_action_weights[:13] = action_weights[:13]

  def weighted_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.square(prediction - target) * action_weights)

  def symmetry_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.square(prediction - target) * symmetry_action_weights)

  generator = torch.Generator().manual_seed(cfg.seed)
  permutation = torch.randperm(len(observations), generator=generator)
  validation_count = max(1, round(len(observations) * cfg.validation_fraction))
  validation_indices = permutation[:validation_count]
  train_indices = permutation[validation_count:]
  validation_generator = torch.Generator().manual_seed(cfg.seed + 10_000)
  validation_augmented = _augment_observations(
    observations[validation_indices],
    previous_observations[validation_indices],
    cfg.delay_probability,
    cfg.encoder_bias,
    validation_generator,
  )

  initial_parameters = {
    name: parameter.detach().clone() for name, parameter in actor.named_parameters()
  }
  optimizer = torch.optim.AdamW(actor.parameters(), lr=cfg.learning_rate)
  best_score = float("inf")
  best_actor_state: dict[str, torch.Tensor] | None = None
  best_clean_mse = float("inf")
  best_delayed_mse = float("inf")

  for epoch in range(cfg.epochs):
    actor.train()
    order = train_indices[torch.randperm(len(train_indices), generator=generator)]
    total_loss = 0.0
    total_samples = 0
    augmentation_generator = torch.Generator().manual_seed(cfg.seed + epoch + 1)
    for start in range(0, len(order), cfg.batch_size):
      indices = order[start : start + cfg.batch_size]
      clean_cpu = observations[indices]
      augmented_cpu = _augment_observations(
        clean_cpu,
        previous_observations[indices],
        cfg.delay_probability,
        cfg.encoder_bias,
        augmentation_generator,
      )
      clean = clean_cpu.to(device)
      augmented = augmented_cpu.to(device)
      target = actions[indices].to(device)
      clean_prediction = actor((clean - mean) / (std + eps))
      delayed_prediction = actor((augmented - mean) / (std + eps))
      clean_loss = weighted_mse(clean_prediction, target)
      delayed_loss = weighted_mse(delayed_prediction, target)
      if cfg.symmetry_loss_weight > 0.0:
        mirrored_target = _mirror_actions(target)
        mirrored_clean = _mirror_observations(clean)
        mirrored_augmented = _mirror_observations(augmented)
        symmetry_loss = symmetry_mse(
          actor((mirrored_clean - mean) / (std + eps)), mirrored_target
        ) + symmetry_mse(
          actor((mirrored_augmented - mean) / (std + eps)), mirrored_target
        )
      else:
        symmetry_loss = clean_loss.new_zeros(())
      parameter_anchor = sum(
        torch.mean((parameter - initial_parameters[name]) ** 2)
        for name, parameter in actor.named_parameters()
      )
      loss = (
        cfg.clean_anchor_weight * clean_loss
        + cfg.delayed_loss_weight * delayed_loss
        + cfg.symmetry_loss_weight * symmetry_loss
        + 1.0e-4 * parameter_anchor
      )
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
      optimizer.step()
      total_loss += float(loss.item()) * len(indices)
      total_samples += len(indices)

    actor.eval()
    clean_sum = 0.0
    delayed_sum = 0.0
    samples = 0
    with torch.inference_mode():
      for start in range(0, validation_count, cfg.batch_size):
        indices = validation_indices[start : start + cfg.batch_size]
        clean = observations[indices].to(device)
        augmented = validation_augmented[start : start + len(indices)].to(device)
        target = actions[indices].to(device)
        clean_prediction = actor((clean - mean) / (std + eps))
        delayed_prediction = actor((augmented - mean) / (std + eps))
        clean_sum += float(
          torch.sum(torch.square(clean_prediction - target) * action_weights).item()
        )
        delayed_sum += float(
          torch.sum(torch.square(delayed_prediction - target) * action_weights).item()
        )
        samples += target.numel()
    clean_mse = clean_sum / samples
    delayed_mse = delayed_sum / samples
    score = cfg.clean_anchor_weight * clean_mse + delayed_mse
    if score < best_score:
      best_score = score
      best_clean_mse = clean_mse
      best_delayed_mse = delayed_mse
      best_actor_state = copy.deepcopy(actor.state_dict())
    if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == cfg.epochs:
      print(
        f"[INFO] Epoch {epoch + 1:03d}/{cfg.epochs}: "
        f"train={total_loss / total_samples:.8f} "
        f"clean_val={clean_mse:.8f} delayed_val={delayed_mse:.8f}"
      )

  assert best_actor_state is not None
  output_checkpoint = copy.deepcopy(base_checkpoint)
  output_actor_state = output_checkpoint["actor_state_dict"]
  for key, value in best_actor_state.items():
    output_actor_state[f"mlp.{key}"] = value.detach().cpu()
  infos = dict(output_checkpoint.get("infos") or {})
  infos["sensor_invariance_distillation"] = {
    "teacher_checkpoint": str(checkpoint),
    "task_id": cfg.task_id,
    "samples": len(observations),
    "epochs": cfg.epochs,
    "clean_anchor_weight": cfg.clean_anchor_weight,
    "delayed_loss_weight": cfg.delayed_loss_weight,
    "stability_action_weight": cfg.stability_action_weight,
    "stability_action_indices": STABILITY_ACTION_INDICES,
    "symmetry_loss_weight": cfg.symmetry_loss_weight,
    "delay_probability": cfg.delay_probability,
    "encoder_bias": cfg.encoder_bias,
    "clean_validation_mse": best_clean_mse,
    "delayed_validation_mse": best_delayed_mse,
  }
  output_checkpoint["infos"] = infos

  output = Path(cfg.output_file).expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(output_checkpoint, output)
  report = infos["sensor_invariance_distillation"] | {
    "output_checkpoint": str(output)
  }
  output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
  print(f"[INFO] Saved sensor-invariant checkpoint: {output}")
  print(json.dumps(report, indent=2))
  return report


def main() -> None:
  import tyro

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  run_distillation(tyro.cli(SensorInvariantConfig))


if __name__ == "__main__":
  main()
