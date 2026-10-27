// Project-owned ROS 2 integration. Vendor interfaces are external dependencies.
#include "x2_rl_deploy_controller/x2_velocity_policy_core.h"

#include <aimdk_msgs/msg/joint_command_array.hpp>
#include <aimdk_msgs/msg/joint_state_array.hpp>
#include <onnxruntime_cxx_api.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Time = Clock::time_point;
using namespace x2::deploy;
enum class Mode { Disarmed, Damping, Prepare, Policy };

double elapsed(Time now, Time then) {
  return std::chrono::duration<double>(now - then).count();
}

template <class T>
T setting(const YAML::Node& node, const char* key, T fallback) {
  return node[key] ? node[key].as<T>() : fallback;
}

struct Configuration {
  bool output{false}, simulation{false}, allow_hardware{false}, deadman{true};
  Mode initial{Mode::Disarmed};
  double publish_dt{0.002}, state_timeout{0.1}, joy_timeout{0.3};
  double tilt_degrees{35}, pose_error{0.45}, transition_seconds{3};
  double gain{1}, damping{5};
  std::string imu_topic{"/aima/hal/imu/torso/state"}, imu_frame{"unverified"};
  SafetyLimits limits;
  int deadman_button{4}, disarm_button{0}, damping_button{1}, prepare_button{2}, policy_button{3};
  int forward_axis{1}, lateral_axis{0}, yaw_axis{3};
  double forward_sign{-1}, lateral_sign{-1}, yaw_sign{-1};
};

Configuration read_configuration(const std::string& path) {
  const auto yaml = YAML::LoadFile(path);
  Configuration config;
  config.output = setting(yaml, "output_enabled", false);
  config.simulation = setting(yaml, "simulation_mode", false);
  config.allow_hardware = setting(yaml, "allow_hardware_output", false);
  config.deadman = setting(yaml, "require_deadman", true);
  const auto mode = setting(yaml, "start_mode", std::string("disarmed"));
  if (mode == "rl") config.initial = Mode::Policy;
  else if (mode == "damping") config.initial = Mode::Damping;
  else if (mode != "disarmed") throw std::runtime_error("Invalid start_mode");
  config.publish_dt = setting(yaml, "publish_dt", config.publish_dt);
  config.state_timeout = setting(yaml, "state_timeout", config.state_timeout);
  config.joy_timeout = setting(yaml, "joy_timeout", config.joy_timeout);
  config.tilt_degrees = setting(yaml, "max_tilt_degrees", config.tilt_degrees);
  config.pose_error = setting(yaml, "max_default_pose_error", config.pose_error);
  config.transition_seconds = setting(yaml, "transition_duration", config.transition_seconds);
  config.gain = setting(yaml, "gain_scale", config.gain);
  config.damping = setting(yaml, "damping_gain", config.damping);
  config.imu_topic = setting(yaml, "imu_topic", config.imu_topic);
  config.imu_frame = setting(yaml, "imu_frame", config.imu_frame);
  if (config.imu_topic.empty() || (config.imu_frame != "pelvis" && config.imu_frame != "unverified")) {
    throw std::runtime_error("imu_topic is required; imu_frame must be pelvis or unverified");
  }
  if (const auto safety = yaml["safety"]) {
    config.limits.max_command_x = setting(safety, "max_command_x", config.limits.max_command_x);
    config.limits.max_command_y = setting(safety, "max_command_y", config.limits.max_command_y);
    config.limits.max_command_yaw = setting(safety, "max_command_yaw", config.limits.max_command_yaw);
    config.limits.action_clip = setting(safety, "action_clip", config.limits.action_clip);
    config.limits.action_scale_factor = setting(safety, "action_scale_factor", config.limits.action_scale_factor);
    config.limits.joint_limit_margin = setting(safety, "joint_limit_margin", config.limits.joint_limit_margin);
    config.limits.max_position_deviation = setting(safety, "max_position_deviation", config.limits.max_position_deviation);
    config.limits.max_target_velocity = setting(safety, "max_target_velocity", config.limits.max_target_velocity);
  }
  if (const auto joy = yaml["joystick"]) {
    config.deadman_button = setting(joy, "deadman_button", config.deadman_button);
    config.disarm_button = setting(joy, "disarm_button", config.disarm_button);
    config.damping_button = setting(joy, "damping_button", config.damping_button);
    config.prepare_button = setting(joy, "default_button", config.prepare_button);
    config.policy_button = setting(joy, "rl_button", config.policy_button);
    config.forward_axis = setting(joy, "forward_axis", config.forward_axis);
    config.lateral_axis = setting(joy, "lateral_axis", config.lateral_axis);
    config.yaw_axis = setting(joy, "yaw_axis", config.yaw_axis);
    config.forward_sign = setting(joy, "forward_axis_sign", config.forward_sign);
    config.lateral_sign = setting(joy, "lateral_axis_sign", config.lateral_sign);
    config.yaw_sign = setting(joy, "yaw_axis_sign", config.yaw_sign);
  }
  for (double value : {config.publish_dt, config.state_timeout, config.joy_timeout,
                      config.tilt_degrees, config.pose_error, config.transition_seconds,
                      config.gain, config.damping, config.limits.max_command_x,
                      config.limits.max_command_y, config.limits.max_command_yaw,
                      config.limits.action_clip, config.limits.action_scale_factor,
                      config.limits.joint_limit_margin, config.limits.max_position_deviation,
                      config.limits.max_target_velocity, config.forward_sign,
                      config.lateral_sign, config.yaw_sign}) {
    if (!std::isfinite(value)) throw std::runtime_error("Configuration contains non-finite values");
  }
  if (config.publish_dt < 0.001 || config.publish_dt > 0.02 ||
      config.state_timeout <= 0 || config.joy_timeout <= 0 ||
      config.tilt_degrees <= 0 || config.tilt_degrees >= 90 ||
      config.pose_error <= 0 || config.transition_seconds <= 0 ||
      config.gain <= 0 || config.gain > 2 || config.damping < 0 ||
      config.limits.max_command_x < 0 || config.limits.max_command_y < 0 ||
      config.limits.max_command_yaw < 0 || config.limits.action_clip <= 0 ||
      config.limits.action_scale_factor < 0 || config.limits.action_scale_factor > 1 ||
      config.limits.joint_limit_margin < 0 || config.limits.max_position_deviation <= 0 ||
      config.limits.max_target_velocity <= 0) {
    throw std::runtime_error("Configuration values are outside supported ranges");
  }
  for (int value : {config.deadman_button, config.disarm_button, config.damping_button,
                    config.prepare_button, config.policy_button, config.forward_axis,
                    config.lateral_axis, config.yaw_axis}) {
    if (value < 0 || value > 63) throw std::runtime_error("Invalid joystick index");
  }
  return config;
}

class OnnxPolicy {
 public:
  explicit OnnxPolicy(const std::string& path)
      : environment_(ORT_LOGGING_LEVEL_ERROR, "lingxi-x2"), session_(nullptr) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_ = Ort::Session(environment_, path.c_str(), options);
    if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1) {
      throw std::runtime_error("Expected one ONNX input and one output");
    }
    Ort::AllocatorWithDefaultOptions allocator;
    if (std::string(session_.GetInputNameAllocated(0, allocator).get()) != "obs" ||
        std::string(session_.GetOutputNameAllocated(0, allocator).get()) != "actions") {
      throw std::runtime_error("Expected ONNX names obs -> actions");
    }
    const auto info = session_.GetModelMetadata();
    std::unordered_map<std::string, std::string> metadata;
    for (const auto& key : info.GetCustomMetadataMapKeysAllocated(allocator)) {
      auto value = info.LookupCustomMetadataMapAllocated(key.get(), allocator);
      if (value) metadata.emplace(key.get(), value.get());
    }
    contract_ = ParsePolicyContract(metadata);
    const auto input_info = session_.GetInputTypeInfo(0);
    const auto output_info = session_.GetOutputTypeInfo(0);
    const auto input = input_info.GetTensorTypeAndShapeInfo();
    const auto output = output_info.GetTensorTypeAndShapeInfo();
    shape_ = {1, static_cast<int64_t>(kSingleObservationCount * contract_.observation_history_length)};
    if (input.GetShape() != std::vector<int64_t>(shape_.begin(), shape_.end()) ||
        output.GetShape() != std::vector<int64_t>({1, static_cast<int64_t>(kJointCount)}) ||
        input.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        output.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error("ONNX tensor types/shapes disagree with metadata");
    }
    if (std::set<std::string>(contract_.joint_names.begin(), contract_.joint_names.end()).size() != kJointCount) {
      throw std::runtime_error("Duplicate model joint names");
    }
  }

  const PolicyContract& contract() const { return contract_; }

  JointVector evaluate(ObservationVector observation) {
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(memory, observation.data(), observation.size(), shape_.data(), shape_.size());
    const char* inputs[] = {"obs"};
    const char* outputs[] = {"actions"};
    const auto result = session_.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
    JointVector action{};
    const auto* data = result.front().GetTensorData<float>();
    for (std::size_t index = 0; index < action.size(); ++index) {
      if (!std::isfinite(data[index])) throw std::runtime_error("Non-finite ONNX action");
      action[index] = data[index];
    }
    return action;
  }

 private:
  Ort::Env environment_;
  Ort::Session session_;
  std::array<int64_t, 2> shape_{};
  PolicyContract contract_;
};

class Controller final : public rclcpp::Node {
 public:
  Controller(Configuration config, const std::string& model, bool hardware)
      : Node("lingxi_x2_policy"), config_(std::move(config)), policy_(model), core_(policy_.contract()), mode_(config_.initial) {
    if (config_.output && !config_.simulation &&
        (!hardware || !config_.allow_hardware || !config_.deadman || mode_ != Mode::Disarmed)) {
      throw std::runtime_error("Hardware requires --hardware, allow_hardware_output, require_deadman, and start_mode=disarmed");
    }
    if (hardware && config_.simulation) throw std::runtime_error("--hardware conflicts with simulation_mode");
    if (config_.output && config_.imu_frame != "pelvis") {
      throw std::runtime_error("Output requires confirmation that IMU orientation/angular velocity match the pelvis/root training frame");
    }
    if (config_.publish_dt > core_.contract().policy_dt) throw std::runtime_error("publish_dt exceeds policy_dt");
    for (std::size_t index = 0; index < kJointCount; ++index) {
      index_by_name_.emplace(core_.contract().joint_names[index], index);
      const auto& name = core_.contract().joint_names[index];
      if (name.find("hip_") != std::string::npos || name.find("knee_") != std::string::npos ||
          name.find("ankle_") != std::string::npos) area_[index] = 0;
      else if (name.find("waist_") == 0) area_[index] = 1;
      else if (name.find("head_") == 0) area_[index] = 3;
      else if (name.find("shoulder_") != std::string::npos || name.find("elbow_") != std::string::npos ||
               name.find("wrist_") != std::string::npos) area_[index] = 2;
      else throw std::runtime_error("Unknown HAL joint area: " + name);
    }
    previous_target_ = core_.contract().default_position;
    joint_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    imu_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    joy_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    const auto qos = rclcpp::SensorDataQoS();
    rclcpp::SubscriptionOptions options;
    options.callback_group = joint_group_;
    const std::array<std::string, 4> areas{"leg", "waist", "arm", "head"};
    for (std::size_t area = 0; area < areas.size(); ++area) {
      subscriptions_[area] = create_subscription<aimdk_msgs::msg::JointStateArray>(
          "/aima/hal/joint/" + areas[area] + "/state", qos,
          [this](const aimdk_msgs::msg::JointStateArray::SharedPtr message) { receive_joints(*message); }, options);
      if (config_.output) {
        publishers_[area] = create_publisher<aimdk_msgs::msg::JointCommandArray>(
            "/aima/hal/joint/" + areas[area] + "/command", qos);
      }
    }
    options.callback_group = imu_group_;
    imu_subscription_ = create_subscription<sensor_msgs::msg::Imu>(
        config_.imu_topic, qos, [this](const sensor_msgs::msg::Imu::SharedPtr message) {
          std::lock_guard<std::mutex> lock(mutex_);
          state_.orientation_wxyz = {message->orientation.w, message->orientation.x, message->orientation.y, message->orientation.z};
          state_.angular_velocity = {message->angular_velocity.x, message->angular_velocity.y, message->angular_velocity.z};
          imu_time_ = Clock::now();
        }, options);
    options.callback_group = joy_group_;
    joy_subscription_ = create_subscription<sensor_msgs::msg::Joy>(
        "/joy", qos, [this](const sensor_msgs::msg::Joy::SharedPtr message) { receive_joy(*message); }, options);
    estop_service_ = create_service<std_srvs::srv::Trigger>(
        "/x2/policy/estop", [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                                   std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
          std::lock_guard<std::mutex> lock(mutex_);
          estop_ = true;
          mode_ = Mode::Damping;
          response->success = true;
          response->message = "Software E-stop latched; restart required to re-arm";
        });
    timer_ = create_wall_timer(std::chrono::duration<double>(config_.publish_dt), [this]() { tick(); });
    RCLCPP_INFO(get_logger(), "Policy: obs=%zu actions=31 dt=%.3f, command period=%.3f, gain=%.2f",
                core_.observation_count(), core_.contract().policy_dt, config_.publish_dt, config_.gain);
    RCLCPP_WARN(get_logger(), "%s", config_.output ?
                (config_.simulation ? "SIMULATION OUTPUT enabled" : "HARDWARE OUTPUT armed only by explicit joystick transitions") :
                "DRY RUN: no motor command publishers exist");
    if (config_.imu_frame == "unverified") {
      RCLCPP_WARN(get_logger(), "IMU source unverified: diagnostic inference only; confirm the sensor rigid body and frame convention before output");
    }
  }

 private:
  void receive_joints(const aimdk_msgs::msg::JointStateArray& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto now = Clock::now();
    for (const auto& joint : message.joints) {
      const auto found = index_by_name_.find(joint.name);
      if (found == index_by_name_.end()) continue;
      const auto index = found->second;
      state_.position[index] = joint.position;
      state_.velocity[index] = joint.velocity;
      joint_error_[index] = joint.error_code != 0;
      joint_time_[index] = now;
    }
  }

  void receive_joy(const sensor_msgs::msg::Joy& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto pressed = [&message](int index) {
      return static_cast<std::size_t>(index) < message.buttons.size() && message.buttons[index] != 0;
    };
    const auto rising = [this, &pressed](int index) {
      return pressed(index) && (static_cast<std::size_t>(index) >= previous_buttons_.size() || !previous_buttons_[index]);
    };
    const auto axis = [&message](int index) {
      if (static_cast<std::size_t>(index) >= message.axes.size()) return 0.0;
      const double value = message.axes[index];
      return std::isfinite(value) ? std::clamp(value, -1.0, 1.0) : 0.0;
    };
    command_ = {config_.forward_sign * axis(config_.forward_axis) * config_.limits.max_command_x,
                config_.lateral_sign * axis(config_.lateral_axis) * config_.limits.max_command_y,
                config_.yaw_sign * axis(config_.yaw_axis) * config_.limits.max_command_yaw};
    joy_time_ = Clock::now();
    deadman_ = pressed(config_.deadman_button);
    if (!previous_buttons_.empty() && !estop_) {
      if (rising(config_.disarm_button)) { mode_ = Mode::Disarmed; fault_ = false; }
      else if (rising(config_.damping_button)) { estop_ = true; mode_ = Mode::Damping; }
      else if (!fault_ && (!config_.deadman || deadman_)) {
        if (rising(config_.prepare_button)) mode_ = Mode::Prepare;
        else if (rising(config_.policy_button)) mode_ = Mode::Policy;
      }
    }
    previous_buttons_ = message.buttons;
  }

  void publish(const JointVector& target, bool damping) {
    if (!config_.output) return;
    std::array<aimdk_msgs::msg::JointCommandArray, 4> messages;
    for (std::size_t index = 0; index < kJointCount; ++index) {
      aimdk_msgs::msg::JointCommand joint;
      joint.name = core_.contract().joint_names[index];
      joint.position = damping ? 0.0 : target[index];
      joint.velocity = 0.0;
      joint.effort = 0.0;
      joint.stiffness = damping ? 0.0 : config_.gain * core_.contract().stiffness[index];
      joint.damping = damping ? config_.damping : config_.gain * core_.contract().damping[index];
      messages[area_[index]].joints.push_back(joint);
    }
    for (std::size_t area = 0; area < messages.size(); ++area) publishers_[area]->publish(messages[area]);
  }

  void tick() {
    const auto now = Clock::now();
    RobotState state;
    VelocityCommand command;
    Mode requested;
    std::string invalid;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      requested = mode_;
      state = state_;
      command = elapsed(now, joy_time_) <= config_.joy_timeout ? command_ : VelocityCommand{};
      if (requested == Mode::Disarmed) { active_ = Mode::Disarmed; return; }
      if (requested == Mode::Damping) { publish({}, true); active_ = Mode::Damping; return; }
      if (imu_time_ == Time{} || elapsed(now, imu_time_) > config_.state_timeout) invalid = "configured IMU missing/stale";
      for (std::size_t index = 0; index < kJointCount && invalid.empty(); ++index) {
        if (joint_time_[index] == Time{} || elapsed(now, joint_time_[index]) > config_.state_timeout) {
          invalid = "joint missing/stale: " + core_.contract().joint_names[index];
        } else if (joint_error_[index]) invalid = "joint fault: " + core_.contract().joint_names[index];
        else if (!std::isfinite(state.position[index]) || !std::isfinite(state.velocity[index])) invalid = "non-finite joint state";
      }
      if (config_.deadman && (!deadman_ || elapsed(now, joy_time_) > config_.joy_timeout)) invalid = "deadman released or joystick stale";
    }
    // Wait for the first complete sensor set without publishing position targets.
    if (!ever_ready_ && !invalid.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Waiting for state: %s", invalid.c_str());
      return;
    }
    try {
      if (!invalid.empty()) throw std::runtime_error(invalid);
      const double tilt = TiltAngleRadians(state) * 180.0 / std::acos(-1.0);
      if (tilt > config_.tilt_degrees) throw std::runtime_error("tilt exceeds configured threshold");
      ever_ready_ = true;
      if (requested != active_) {
        if (requested == Mode::Policy && config_.output && !config_.simulation) {
          for (std::size_t index = 0; index < kJointCount; ++index) {
            if (std::abs(state.position[index] - core_.contract().default_position[index]) > config_.pose_error) {
              throw std::runtime_error("RL entry pose mismatch: " + core_.contract().joint_names[index]);
            }
          }
        }
        active_ = requested;
        mode_start_ = now;
        prepare_start_ = state.position;
        previous_target_ = state.position;
        previous_action_.fill(0);
        core_.ResetObservationHistory();
        next_inference_ = Time{};
        last_inference_ = Time{};
        RCLCPP_INFO(get_logger(), "Mode: %s", requested == Mode::Prepare ? "prepare" : "policy");
      }
      if (requested == Mode::Prepare) {
        const double t = std::clamp(elapsed(now, mode_start_) / config_.transition_seconds, 0.0, 1.0);
        const double blend = t * t * (3.0 - 2.0 * t);
        for (std::size_t index = 0; index < kJointCount; ++index) {
          previous_target_[index] = prepare_start_[index] + blend * (core_.contract().default_position[index] - prepare_start_[index]);
        }
      } else {
        command = ClampVelocityCommand(command, config_.limits);
        if (next_inference_ == Time{} || now >= next_inference_) {
          const auto started = Clock::now();
          previous_action_ = policy_.evaluate(core_.BuildObservation(state, command, elapsed(now, mode_start_), previous_action_));
          for (auto& action : previous_action_) action = std::clamp(action, -config_.limits.action_clip, config_.limits.action_clip);
          const double milliseconds = elapsed(Clock::now(), started) * 1000;
          inference_max_ = std::max(inference_max_, milliseconds);
          if (last_inference_ != Time{}) inference_interval_max_ = std::max(inference_interval_max_, elapsed(now, last_inference_) * 1000);
          last_inference_ = now;
          const auto period = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(core_.contract().policy_dt));
          if (next_inference_ == Time{}) next_inference_ = now;
          next_inference_ += period;
          // Keep a fixed deadline without accumulating 500 Hz timer jitter.
          // Skip missed periods; never fabricate multiple history frames.
          if (next_inference_ <= now) {
            const auto skipped = (now - next_inference_) / period + 1;
            missed_inferences_ += static_cast<std::size_t>(skipped);
            next_inference_ += period * skipped;
          }
          ++inference_count_;
        }
        previous_target_ = core_.ComputeTargets(previous_action_, previous_target_, config_.publish_dt, config_.limits);
      }
      // Recheck stop state after inference before publishing a position command.
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (estop_ || fault_ || mode_ == Mode::Damping) { publish({}, true); return; }
        if (mode_ != requested) return;
        publish(previous_target_, false);
      }
      if (elapsed(now, log_time_) >= 2) {
        RCLCPP_INFO(get_logger(), "Policy health: tilt=%.2f cmd=[%.2f %.2f %.2f] inference_max_ms=%.3f interval_max_ms=%.3f samples=%zu skipped=%zu",
                    tilt, command.x, command.y, command.yaw, inference_max_, inference_interval_max_, inference_count_, missed_inferences_);
        log_time_ = now;
        inference_max_ = 0;
        inference_interval_max_ = 0;
        inference_count_ = 0;
        missed_inferences_ = 0;
      }
    } catch (const std::exception& error) {
      std::lock_guard<std::mutex> lock(mutex_);
      fault_ = true;
      mode_ = Mode::Damping;
      active_ = Mode::Damping;
      publish({}, true);
      RCLCPP_ERROR(get_logger(), "Fault latched, damping: %s", error.what());
    }
  }

  Configuration config_;
  OnnxPolicy policy_;
  X2VelocityPolicyCore core_;
  std::mutex mutex_;
  RobotState state_;
  VelocityCommand command_;
  std::unordered_map<std::string, std::size_t> index_by_name_;
  std::array<std::size_t, kJointCount> area_{};
  std::array<Time, kJointCount> joint_time_{};
  std::array<bool, kJointCount> joint_error_{};
  Time imu_time_{}, joy_time_{}, mode_start_{}, next_inference_{}, last_inference_{}, log_time_{};
  Mode mode_, active_{Mode::Disarmed};
  bool deadman_{false}, fault_{false}, estop_{false}, ever_ready_{false};
  std::vector<int32_t> previous_buttons_;
  JointVector prepare_start_{}, previous_target_{}, previous_action_{};
  double inference_max_{0}, inference_interval_max_{0};
  std::size_t inference_count_{0}, missed_inferences_{0};
  std::array<rclcpp::Subscription<aimdk_msgs::msg::JointStateArray>::SharedPtr, 4> subscriptions_;
  std::array<rclcpp::Publisher<aimdk_msgs::msg::JointCommandArray>::SharedPtr, 4> publishers_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscription_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr estop_service_;
  rclcpp::CallbackGroup::SharedPtr joint_group_, imu_group_, joy_group_;
  rclcpp::TimerBase::SharedPtr timer_;
};
}  // namespace

int main(int argc, char** argv) {
  std::string config_path, model_path;
  bool hardware = false;
  try {
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--hardware") hardware = true;
      else if ((argument == "--config" || argument == "--model") && index + 1 < argc) {
        (argument == "--config" ? config_path : model_path) = argv[++index];
      } else if (argument == "--ros-args") break;
      else throw std::runtime_error("Usage: x2_policy_controller --config FILE --model FILE [--hardware] [--ros-args ...]");
    }
    if (config_path.empty() || model_path.empty()) throw std::runtime_error("--config and --model are required");
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Controller>(read_configuration(config_path), model_path, hardware);
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions{}, 3);
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "x2_policy_controller: " << error.what() << '\n';
    if (rclcpp::ok()) rclcpp::shutdown();
    return 1;
  }
}
