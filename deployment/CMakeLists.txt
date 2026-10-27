cmake_minimum_required(VERSION 3.16)
project(lingxi_x2_deployment VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(X2_BUILD_ROS2 "Build the optional ROS 2 + external AIMDK controller" OFF)

add_library(x2_policy_core src/x2_velocity_policy_core.cc)
target_include_directories(x2_policy_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_options(x2_policy_core PRIVATE -Wall -Wextra -Wpedantic)

include(CTest)
if(BUILD_TESTING)
  add_executable(x2_policy_core_test tests/x2_velocity_policy_core_test.cc)
  target_link_libraries(x2_policy_core_test PRIVATE x2_policy_core)
  target_compile_options(x2_policy_core_test PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME policy_contract COMMAND x2_policy_core_test)
endif()

if(X2_BUILD_ROS2)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(std_srvs REQUIRED)
  # Obtain the vendor interfaces separately. They are not redistributed here.
  find_package(aimdk_msgs REQUIRED)
  find_package(yaml-cpp REQUIRED)
  set(ONNXRUNTIME_ROOT "" CACHE PATH "Extracted ONNX Runtime C/C++ release")
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}/include" REQUIRED)
  find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime libonnxruntime.so.1 libonnxruntime.so.1.22.0
    HINTS "${ONNXRUNTIME_ROOT}/lib" REQUIRED)
  add_executable(x2_policy_controller src/ros2_controller.cpp)
  target_include_directories(x2_policy_controller PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
  target_link_libraries(x2_policy_controller x2_policy_core yaml-cpp "${ONNXRUNTIME_LIBRARY}")
  target_compile_options(x2_policy_controller PRIVATE -Wall -Wextra -Wpedantic)
  ament_target_dependencies(x2_policy_controller rclcpp sensor_msgs std_srvs aimdk_msgs)
  install(TARGETS x2_policy_controller DESTINATION lib/${PROJECT_NAME})
  install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
  ament_package()
endif()
