#!/usr/bin/env bash
set -euo pipefail
deployment_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=${X2_CORE_BUILD_DIR:-${deployment_root}/build-core}
cmake -S "${deployment_root}" -B "${build_dir}" -DX2_BUILD_ROS2=OFF -DBUILD_TESTING=ON
cmake --build "${build_dir}" --parallel
ctest --test-dir "${build_dir}" --output-on-failure
