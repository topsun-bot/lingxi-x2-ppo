#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ -z "${PRETRAINED_CHECKPOINT:-}" ]]; then
  echo "Set PRETRAINED_CHECKPOINT to the accepted sagittal-balance model." >&2
  exit 1
fi

export TASK_ID=Agibot-X2-Ultra-Behavior-Kp090-Kd080-Lateral-Bootstrap
export RUN_NAME="${RUN_NAME:-x2_ultra_kp090_kd080_stage2b_lateral_bootstrap_v1}"
export NUM_ENVS="${NUM_ENVS:-1024}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-400}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
export RESET_OPTIMIZER_ON_LOAD="${RESET_OPTIMIZER_ON_LOAD:-1}"

exec scripts/x2_ultra/train_behavior.sh
