#!/usr/bin/env bash
set -euo pipefail

export TASK_ID=Agibot-X2-Ultra-Behavior-Robust-Lateral-Balance
export PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:?Set PRETRAINED_CHECKPOINT to the selected lateral-refine checkpoint}"
export NUM_ENVS="${NUM_ENVS:-128}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-60}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
export RUN_NAME="${RUN_NAME:-x2_ultra_behavior_robust_lateral_balance}"
export FREEZE_OBS_NORMALIZATION="${FREEZE_OBS_NORMALIZATION:-1}"
export RESET_OPTIMIZER_ON_LOAD="${RESET_OPTIMIZER_ON_LOAD:-1}"
export LEARNING_RATE="${LEARNING_RATE:-0.00005}"
export PPO_SCHEDULE="${PPO_SCHEDULE:-fixed}"

exec "$(dirname "$0")/train_behavior.sh"
