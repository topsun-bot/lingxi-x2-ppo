#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ -z "${PRETRAINED_CHECKPOINT:-}" ]]; then
  echo "Set PRETRAINED_CHECKPOINT to the best robust-finalize checkpoint." >&2
  exit 1
fi

export TASK_ID=Agibot-X2-Ultra-Behavior-Kp090-Kd080-Full-Speed-Robust-Stop
export RUN_NAME="${RUN_NAME:-x2_ultra_kp090_kd080_stage5c_full_speed_robust_stop_v1}"
export NUM_ENVS="${NUM_ENVS:-1024}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
export LEARNING_RATE="${LEARNING_RATE:-0.000005}"
export PPO_SCHEDULE="${PPO_SCHEDULE:-fixed}"
export RESET_OPTIMIZER_ON_LOAD="${RESET_OPTIMIZER_ON_LOAD:-1}"

exec scripts/x2_ultra/train_behavior.sh
