#!/usr/bin/env bash
set -euo pipefail

export NUM_ENVS="${NUM_ENVS:-16}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-2}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
export RUN_NAME="${RUN_NAME:-x2_ultra_smoke}"

exec "$(dirname "$0")/train_flat.sh"
