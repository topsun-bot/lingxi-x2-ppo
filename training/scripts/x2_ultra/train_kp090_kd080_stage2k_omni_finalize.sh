#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ -z "${PRETRAINED_CHECKPOINT:-}" ]]; then
  echo "Set PRETRAINED_CHECKPOINT to the accepted distilled omni model." >&2
  exit 1
fi

export TASK_ID=Agibot-X2-Ultra-Behavior-Kp090-Kd080-Omni-Finalize
export RUN_NAME="${RUN_NAME:-x2_ultra_kp090_kd080_stage2k_omni_finalize_v1}"
export NUM_ENVS="${NUM_ENVS:-1024}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
export LEARNING_RATE="${LEARNING_RATE:-2e-5}"
export PPO_SCHEDULE="${PPO_SCHEDULE:-fixed}"
export RESET_OPTIMIZER_ON_LOAD="${RESET_OPTIMIZER_ON_LOAD:-1}"

exec scripts/x2_ultra/train_behavior.sh
