#!/usr/bin/env bash
set -euo pipefail

export TASK_ID=Agibot-X2-Ultra-Behavior-Robust-Flat
export PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-artifacts/x2_ultra_behavior_v2/model_7100.pt}"
export NUM_ENVS="${NUM_ENVS:-128}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-300}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
export RUN_NAME="${RUN_NAME:-x2_ultra_behavior_robust_stage1_from7100}"

exec "$(dirname "$0")/train_behavior.sh"
