#!/usr/bin/env bash
set -euo pipefail

export TASK_ID=Agibot-X2-Ultra-Behavior-Robust-Lateral-Refine
export PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-artifacts/x2_ultra_behavior_robust_v3/final_candidate/model_candidate.pt}"
export NUM_ENVS="${NUM_ENVS:-128}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-240}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
export RUN_NAME="${RUN_NAME:-x2_ultra_behavior_robust_lateral_refine_from_candidate}"
export FREEZE_OBS_NORMALIZATION="${FREEZE_OBS_NORMALIZATION:-1}"
export RESET_OPTIMIZER_ON_LOAD="${RESET_OPTIMIZER_ON_LOAD:-1}"
export LEARNING_RATE="${LEARNING_RATE:-0.0002}"
export PPO_SCHEDULE="${PPO_SCHEDULE:-fixed}"

exec "$(dirname "$0")/train_behavior.sh"
