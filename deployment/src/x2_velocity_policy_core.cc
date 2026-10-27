#include "x2_rl_deploy_controller/x2_velocity_policy_core.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace x2::deploy {
namespace {

constexpr double kPi = 3.14159265358979323846;

void RequireFinite(double value, const char* label) {
  if (!std::isfinite(value)) {
    throw std::runtime_error(std::string("Non-finite ") + label);
  }
}

double SymmetricClamp(double value, double magnitude) {
  if (!(magnitude >= 0.0) || !std::isfinite(magnitude)) {
    throw std::runtime_error("Invalid symmetric clamp magnitude");
  }
  return std::clamp(value, -magnitude, magnitude);
}

const std::string& RequireMetadata(
    const std::unordered_map<std::string, std::string>& metadata,
    const std::string& key) {
  const auto it = metadata.find(key);
  if (it == metadata.end() || it->second.empty()) {
    throw std::runtime_error("Missing ONNX metadata: " + key);
  }
  return it->second;
}

std::vector<std::string> SplitCsv(const std::string& text) {
  std::vector<std::string> values;
  std::istringstream stream(text);
  std::string value;
  while (std::getline(stream, value, ',')) values.push_back(value);
  return values;
}

double ParseNumber(const std::string& text, const std::string& key) {
  try {
    std::size_t end = 0;
    const double value = std::stod(text, &end);
    if (end != text.size() || !std::isfinite(value)) {
      throw std::runtime_error("invalid");
    }
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid numeric ONNX metadata " + key + ": " + text);
  }
}

JointVector ParseJointVector(
    const std::unordered_map<std::string, std::string>& metadata,
    const std::string& key) {
  const auto values = SplitCsv(RequireMetadata(metadata, key));
  if (values.size() != kJointCount) {
    throw std::runtime_error(
        "ONNX metadata " + key + " must have " + std::to_string(kJointCount) + " values");
  }
  JointVector result{};
  for (std::size_t index = 0; index < kJointCount; ++index) {
    result[index] = ParseNumber(values[index], key);
  }
  return result;
}

std::array<double, 3> ProjectGravity(const std::array<double, 4>& q_wxyz) {
  const double norm = std::sqrt(
      q_wxyz[0] * q_wxyz[0] + q_wxyz[1] * q_wxyz[1] +
      q_wxyz[2] * q_wxyz[2] + q_wxyz[3] * q_wxyz[3]);
  if (!std::isfinite(norm) || norm < 0.9 || norm > 1.1) {
    throw std::runtime_error("IMU quaternion norm outside [0.9, 1.1]");
  }

  const double w = q_wxyz[0] / norm;
  const double x = q_wxyz[1] / norm;
  const double y = q_wxyz[2] / norm;
  const double z = q_wxyz[3] / norm;
  return {
      2.0 * (w * y - z * x),
      -2.0 * (w * x + z * y),
      1.0 - 2.0 * (w * w + z * z),
  };
}

}  // namespace

PolicyContract ParsePolicyContract(
    const std::unordered_map<std::string, std::string>& metadata) {
  const auto& metadata_contract = RequireMetadata(metadata, "metadata_contract");
  if (metadata_contract != "agibot_x2_velocity_v1" &&
      metadata_contract != "agibot_x2_velocity_v2") {
    throw std::runtime_error("Unsupported ONNX metadata_contract");
  }
  if (RequireMetadata(metadata, "observation_names") !=
      "base_ang_vel,projected_gravity,command,phase,joint_pos,joint_vel,actions") {
    throw std::runtime_error("Unexpected ONNX observation_names layout");
  }
  if (RequireMetadata(metadata, "observation_dims") != "3,3,3,2,31,31,31") {
    throw std::runtime_error("Unexpected ONNX observation_dims layout");
  }

  PolicyContract contract;
  const auto names = SplitCsv(RequireMetadata(metadata, "joint_names"));
  if (names.size() != kJointCount) {
    throw std::runtime_error("ONNX joint_names must have 31 entries");
  }
  for (std::size_t index = 0; index < kJointCount; ++index) {
    if (names[index].empty()) throw std::runtime_error("Empty ONNX joint name");
    contract.joint_names[index] = names[index];
  }
  contract.default_position = ParseJointVector(metadata, "default_joint_pos");
  contract.action_scale = ParseJointVector(metadata, "action_scale");
  contract.stiffness = ParseJointVector(metadata, "joint_stiffness");
  contract.damping = ParseJointVector(metadata, "joint_damping");
  contract.lower_limit = ParseJointVector(metadata, "joint_limit_lower");
  contract.upper_limit = ParseJointVector(metadata, "joint_limit_upper");
  contract.policy_dt = ParseNumber(RequireMetadata(metadata, "policy_dt"), "policy_dt");
  contract.phase_period = ParseNumber(RequireMetadata(metadata, "phase_period"), "phase_period");
  if (metadata_contract == "agibot_x2_velocity_v2") {
    const double history_length = ParseNumber(
        RequireMetadata(metadata, "observation_history_length"),
        "observation_history_length");
    if (history_length != 2.0) {
      throw std::runtime_error("X2 velocity v2 requires two observation frames");
    }
    contract.observation_history_length = 2;
  } else {
    contract.observation_history_length = 1;
  }
  return contract;
}

VelocityCommand ClampVelocityCommand(
    const VelocityCommand& requested_command,
    const SafetyLimits& limits) {
  RequireFinite(requested_command.x, "command x");
  RequireFinite(requested_command.y, "command y");
  RequireFinite(requested_command.yaw, "command yaw");
  return {
      SymmetricClamp(requested_command.x, limits.max_command_x),
      SymmetricClamp(requested_command.y, limits.max_command_y),
      SymmetricClamp(requested_command.yaw, limits.max_command_yaw),
  };
}

double TiltAngleRadians(const RobotState& state) {
  const auto gravity = ProjectGravity(state.orientation_wxyz);
  return std::acos(std::clamp(-gravity[2], -1.0, 1.0));
}

X2VelocityPolicyCore::X2VelocityPolicyCore(PolicyContract contract)
    : contract_(std::move(contract)) {
  if (!std::isfinite(contract_.policy_dt) || !std::isfinite(contract_.phase_period) ||
      contract_.policy_dt <= 0.0 || contract_.phase_period <= 0.0) {
    throw std::runtime_error("Policy dt and phase period must be positive");
  }
  if (contract_.observation_history_length != 1 && contract_.observation_history_length != 2) {
    throw std::runtime_error("Only one or two observation frames are supported");
  }
  std::set<std::string> joint_names;
  for (std::size_t index = 0; index < kJointCount; ++index) {
    if (contract_.joint_names[index].empty() || !joint_names.insert(contract_.joint_names[index]).second) {
      throw std::runtime_error("Empty or duplicate joint name in policy contract");
    }
    RequireFinite(contract_.default_position[index], "default position");
    RequireFinite(contract_.action_scale[index], "action scale");
    RequireFinite(contract_.stiffness[index], "stiffness");
    RequireFinite(contract_.damping[index], "damping");
    if (contract_.stiffness[index] < 0 || contract_.damping[index] < 0) {
      throw std::runtime_error("Negative PD gain in policy contract");
    }
    if (!(contract_.lower_limit[index] < contract_.upper_limit[index])) {
      throw std::runtime_error("Invalid joint limits for " + contract_.joint_names[index]);
    }
    if (contract_.default_position[index] < contract_.lower_limit[index] ||
        contract_.default_position[index] > contract_.upper_limit[index]) {
      throw std::runtime_error("Default position outside limits for " + contract_.joint_names[index]);
    }
  }
}

ObservationVector X2VelocityPolicyCore::BuildObservation(
    const RobotState& state,
    const VelocityCommand& requested_command,
    double phase_time,
    const JointVector& previous_action) {
  RequireFinite(phase_time, "phase time");
  for (double value : state.angular_velocity) RequireFinite(value, "angular velocity");
  for (double value : state.position) RequireFinite(value, "joint position");
  for (double value : state.velocity) RequireFinite(value, "joint velocity");
  for (double value : previous_action) RequireFinite(value, "previous action");

  const auto gravity = ProjectGravity(state.orientation_wxyz);
  const std::array<double, 3> command{
      requested_command.x,
      requested_command.y,
      requested_command.yaw,
  };
  for (double value : command) RequireFinite(value, "velocity command");

  std::array<float, kSingleObservationCount> current{};
  std::size_t offset = 0;
  for (double value : state.angular_velocity) current[offset++] = static_cast<float>(value);
  for (double value : gravity) current[offset++] = static_cast<float>(value);
  for (double value : command) current[offset++] = static_cast<float>(value);

  const double command_norm = std::sqrt(
      command[0] * command[0] + command[1] * command[1] + command[2] * command[2]);
  if (command_norm < kCommandDeadzone) {
    current[offset++] = 0.0F;
    current[offset++] = 0.0F;
  } else {
    const double phase = std::fmod(std::max(0.0, phase_time), contract_.phase_period) /
                         contract_.phase_period;
    current[offset++] = static_cast<float>(std::sin(2.0 * kPi * phase));
    current[offset++] = static_cast<float>(std::cos(2.0 * kPi * phase));
  }

  for (std::size_t index = 0; index < kJointCount; ++index) {
    current[offset++] = static_cast<float>(
        state.position[index] - contract_.default_position[index]);
  }
  for (double value : state.velocity) current[offset++] = static_cast<float>(value);
  for (double value : previous_action) current[offset++] = static_cast<float>(value);

  if (offset != kSingleObservationCount) {
    throw std::runtime_error("Internal observation layout error");
  }

  ObservationVector observation;
  observation.reserve(observation_count());
  if (contract_.observation_history_length == 1) {
    observation.assign(current.begin(), current.end());
  } else {
    const auto& previous = observation_history_initialized_
                               ? previous_observation_
                               : current;
    constexpr std::array<std::size_t, 7> term_dims{3, 3, 3, 2, 31, 31, 31};
    std::size_t term_offset = 0;
    for (const std::size_t term_dim : term_dims) {
      observation.insert(
          observation.end(), previous.begin() + term_offset,
          previous.begin() + term_offset + term_dim);
      observation.insert(
          observation.end(), current.begin() + term_offset,
          current.begin() + term_offset + term_dim);
      term_offset += term_dim;
    }
  }
  previous_observation_ = current;
  observation_history_initialized_ = true;
  return observation;
}

void X2VelocityPolicyCore::ResetObservationHistory() {
  previous_observation_.fill(0.0F);
  observation_history_initialized_ = false;
}

JointVector X2VelocityPolicyCore::ComputeTargets(
    const JointVector& action,
    const JointVector& previous_target,
    double publish_dt,
    const SafetyLimits& limits) const {
  if (!(publish_dt > 0.0) || !std::isfinite(publish_dt)) {
    throw std::runtime_error("Publish dt must be finite and positive");
  }
  for (double value : {limits.action_clip, limits.action_scale_factor,
                       limits.joint_limit_margin, limits.max_position_deviation,
                       limits.max_target_velocity}) {
    RequireFinite(value, "target limit");
  }
  if (limits.action_scale_factor < 0.0 || limits.action_scale_factor > 1.0 ||
      limits.joint_limit_margin < 0.0 || limits.max_position_deviation <= 0.0 ||
      limits.max_target_velocity <= 0.0) {
    throw std::runtime_error("Invalid safety limits");
  }

  JointVector target{};
  const double max_step = limits.max_target_velocity * publish_dt;
  for (std::size_t index = 0; index < kJointCount; ++index) {
    RequireFinite(action[index], "policy action");
    RequireFinite(previous_target[index], "previous target");
    const double clipped_action = SymmetricClamp(action[index], limits.action_clip);
    double desired = contract_.default_position[index] +
                     limits.action_scale_factor * contract_.action_scale[index] * clipped_action;
    desired = std::clamp(
        desired,
        contract_.default_position[index] - limits.max_position_deviation,
        contract_.default_position[index] + limits.max_position_deviation);
    const double lower = contract_.lower_limit[index] + limits.joint_limit_margin;
    const double upper = contract_.upper_limit[index] - limits.joint_limit_margin;
    if (!(lower < upper)) {
      throw std::runtime_error("Joint margin consumes limits for " + contract_.joint_names[index]);
    }
    desired = std::clamp(desired, lower, upper);
    target[index] = previous_target[index] +
                    SymmetricClamp(desired - previous_target[index], max_step);
  }
  return target;
}

}  // namespace x2::deploy
