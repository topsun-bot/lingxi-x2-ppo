#!/usr/bin/env bash
set -euo pipefail
deployment_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace=${X2_ROS_WORKSPACE:-${deployment_root}/workspace}
: "${ONNXRUNTIME_ROOT:?Set ONNXRUNTIME_ROOT to a matching CPU architecture C/C++ release}"
: "${AIMDK_MSGS_PREFIX:?Set AIMDK_MSGS_PREFIX to your externally installed official AIMDK interfaces}"
ros_setup=${ROS_SETUP:-/opt/ros/humble/setup.bash}
test -f "${ros_setup}" || { echo "Missing ROS_SETUP: ${ros_setup}" >&2; exit 1; }
test -f "${AIMDK_MSGS_PREFIX}/share/aimdk_msgs/cmake/aimdk_msgsConfig.cmake" || {
  echo "Official aimdk_msgs CMake package not found" >&2; exit 1;
}
set +u
source "${ros_setup}"
set -u
export AMENT_PREFIX_PATH="${AIMDK_MSGS_PREFIX}:${AMENT_PREFIX_PATH:-}"
export CMAKE_PREFIX_PATH="${AIMDK_MSGS_PREFIX}:${CMAKE_PREFIX_PATH:-}"
mkdir -p "${workspace}"
cd "${workspace}"
colcon build --base-paths "${deployment_root}" \
  --packages-select lingxi_x2_deployment \
  --cmake-args -DX2_BUILD_ROS2=ON -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Release "-DONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}" \
  "-Daimdk_msgs_DIR=${AIMDK_MSGS_PREFIX}/share/aimdk_msgs/cmake"
