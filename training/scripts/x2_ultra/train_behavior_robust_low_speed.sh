#!/usr/bin/env bash
set -euo pipefail

export TASK_ID=Agibot-X2-Ultra-Behavior-Robust-Low-Speed
export PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-logs/rsl_rl/x2_ultra_behavior/2026-09-03_15-36-18_x2_ultra_behavior_robust_stage1_from7100/model_7200.pt}"
export NUM_ENVS="${NUM_ENVS:-128}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-150}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
export RUN_NAME="${RUN_NAME:-x2_ultra_behavior_robust_low_speed_stage2_from7200}"
export FREEZE_OBS_NORMALIZATION="${FREEZE_OBS_NORMALIZATION:-1}"

exec "$(dirname "$0")/train_behavior.sh"
