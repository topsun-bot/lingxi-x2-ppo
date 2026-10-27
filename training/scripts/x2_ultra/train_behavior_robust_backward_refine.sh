#!/usr/bin/env bash
set -euo pipefail

export TASK_ID=Agibot-X2-Ultra-Behavior-Robust-Backward-Refine
export PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-logs/rsl_rl/x2_ultra_behavior/2026-09-03_15-58-22_x2_ultra_behavior_robust_low_speed_stage2b_exact_from7200/model_7215.pt}"
export NUM_ENVS="${NUM_ENVS:-128}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-60}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
export RUN_NAME="${RUN_NAME:-x2_ultra_behavior_robust_backward_refine_from7215}"
export FREEZE_OBS_NORMALIZATION="${FREEZE_OBS_NORMALIZATION:-1}"

exec "$(dirname "$0")/train_behavior.sh"
