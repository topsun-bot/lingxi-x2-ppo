#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export TASK_ID=Agibot-X2-Ultra-Factory-Recorded-Flat
export RUN_NAME="${RUN_NAME:-x2_ultra_factory_recorded_stage1_from_scratch_v1}"
export NUM_ENVS="${NUM_ENVS:-1024}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-5001}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"

# This entry point is intentionally unable to resume an older policy.
unset PRETRAINED_CHECKPOINT

exec scripts/x2_ultra/train_behavior.sh
