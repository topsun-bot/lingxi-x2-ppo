#!/usr/bin/env bash
set -euo pipefail
deployment_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo_root=$(cd "${deployment_root}/.." && pwd)
workspace=${X2_ROS_WORKSPACE:-${deployment_root}/workspace}
model_path=${MODEL_PATH:-${repo_root}/models/history2_hybrid_v1/policy.onnx}
config_path=${X2_CONFIG:-${deployment_root}/config/dry_run.yaml}
: "${ONNXRUNTIME_ROOT:?Set ONNXRUNTIME_ROOT}"
: "${AIMDK_MSGS_PREFIX:?Set AIMDK_MSGS_PREFIX}"
test -f "${workspace}/install/setup.bash" || { echo "Build ROS 2 adapter first" >&2; exit 1; }
test -f "${model_path}" || { echo "Missing MODEL_PATH: ${model_path}" >&2; exit 1; }
set +u
source "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
source "${workspace}/install/setup.bash"
set -u
export AMENT_PREFIX_PATH="${AIMDK_MSGS_PREFIX}:${AMENT_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${ONNXRUNTIME_ROOT}/lib:${AIMDK_MSGS_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
exec "${workspace}/install/lingxi_x2_deployment/lib/lingxi_x2_deployment/x2_policy_controller" \
  --config "${config_path}" --model "${model_path}" "$@"
