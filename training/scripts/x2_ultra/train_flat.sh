#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
python_bin="${PYTHON_BIN:-python}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1

num_envs="${NUM_ENVS:-1024}"
max_iterations="${MAX_ITERATIONS:-10001}"
save_interval="${SAVE_INTERVAL:-100}"
run_name="${RUN_NAME:-x2_ultra_flat_v1}"

exec "$python_bin" scripts/train.py Agibot-X2-Ultra-Flat \
  --env.scene.num-envs="$num_envs" \
  --agent.max-iterations="$max_iterations" \
  --agent.save-interval="$save_interval" \
  --agent.run-name="$run_name"
