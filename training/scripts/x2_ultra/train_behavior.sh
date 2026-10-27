#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
python_bin="${PYTHON_BIN:-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
# Prefer this checkout's source tree even when the Python environment is an
# editable install owned by another checkout (for example on a training host).
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

num_envs="${NUM_ENVS:-1024}"
max_iterations="${MAX_ITERATIONS:-5001}"
save_interval="${SAVE_INTERVAL:-100}"
run_name="${RUN_NAME:-x2_ultra_behavior_v1}"
task_id="${TASK_ID:-Agibot-X2-Ultra-Behavior-Flat}"

args=(
  "$task_id"
  "--env.scene.num-envs=$num_envs"
  "--agent.max-iterations=$max_iterations"
  "--agent.save-interval=$save_interval"
  "--agent.run-name=$run_name"
)

echo "[INFO] Task: $task_id"

if [[ -n "${LEARNING_RATE:-}" ]]; then
  args+=("--agent.algorithm.learning-rate=$LEARNING_RATE")
fi
if [[ -n "${PPO_SCHEDULE:-}" ]]; then
  args+=("--agent.algorithm.schedule=$PPO_SCHEDULE")
fi

if [[ -n "${PRETRAINED_CHECKPOINT:-}" ]]; then
  checkpoint="$(readlink -f "$PRETRAINED_CHECKPOINT")"
  if [[ ! -s "$checkpoint" ]]; then
    echo "Pretrained checkpoint is missing or empty: $checkpoint" >&2
    exit 1
  fi
  pretrained_iteration="$("$python_bin" - "$checkpoint" <<'PY'
import sys

import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint["iter"]))
PY
)"
  experiment_name="$("$python_bin" - "$task_id" <<'PY'
import sys

import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import load_rl_cfg

import src.tasks  # noqa: F401

print(load_rl_cfg(sys.argv[1]).experiment_name)
PY
)"
  staging_run="logs/rsl_rl/$experiment_name/pretrained_base"
  staged_checkpoint="model_${pretrained_iteration}.pt"
  mkdir -p "$staging_run"
  cp --reflink=auto "$checkpoint" "$staging_run/$staged_checkpoint"
  args+=(
    --agent.resume=True
    --agent.load-run=pretrained_base
    "--agent.load-checkpoint=$staged_checkpoint"
  )
fi

exec "$python_bin" scripts/train.py "${args[@]}"
