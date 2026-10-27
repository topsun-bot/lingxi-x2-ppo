#include "x2_rl_deploy_controller/x2_velocity_policy_core.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace {

x2::deploy::PolicyContract MakeContract() {
  x2::deploy::PolicyContract contract;
  contract.policy_dt = 0.02;
  contract.phase_period = 0.6;
  for (std::size_t index = 0; index < x2::deploy::kJointCount; ++index) {
    contract.joint_names[index] = "joint_" + std::to_string(index);
    contract.default_position[index] = static_cast<double>(index) * 0.01;
    contract.action_scale[index] = 0.2;
    contract.stiffness[index] = 20.0;
    contract.damping[index] = 1.0;
    contract.lower_limit[index] = -1.0;
    contract.upper_limit[index] = 1.0;
  }
  return contract;
}

bool Near(double lhs, double rhs, double tolerance = 1.0e-6) {
  return std::abs(lhs - rhs) <= tolerance;
}

}  // namespace

int main() {
  const auto contract = MakeContract();
  x2::deploy::X2VelocityPolicyCore core(contract);
  x2::deploy::RobotState state;
  x2::deploy::JointVector previous_action{};
  for (std::size_t index = 0; index < x2::deploy::kJointCount; ++index) {
    state.position[index] = contract.default_position[index] + 0.01;
    state.velocity[index] = 0.02;
    previous_action[index] = 0.03;
  }
  state.angular_velocity = {0.1, -0.2, 0.3};

  const auto standing = core.BuildObservation(state, {}, 0.15, previous_action);
  assert(Near(standing[0], 0.1));
  assert(Near(standing[1], -0.2));
  assert(Near(standing[2], 0.3));
  assert(Near(standing[3], 0.0));
  assert(Near(standing[4], 0.0));
  assert(Near(standing[5], -1.0));
  assert(Near(standing[9], 0.0));
  assert(Near(standing[10], 0.0));
  assert(Near(standing[11], 0.01));
  assert(Near(standing[42], 0.02));
  assert(Near(standing[73], 0.03));

  const x2::deploy::VelocityCommand moving{0.2, 0.0, 0.0};
  const auto walking = core.BuildObservation(state, moving, 0.15, previous_action);
  assert(Near(walking[6], 0.2));
  assert(Near(walking[9], 1.0));
  assert(Near(walking[10], 0.0));

  auto history_contract = contract;
  history_contract.observation_history_length = 2;
  x2::deploy::X2VelocityPolicyCore history_core(history_contract);
  const auto history_first = history_core.BuildObservation(
      state, {}, 0.15, previous_action);
  assert(history_first.size() == 208);
  assert(Near(history_first[0], 0.1));
  assert(Near(history_first[3], 0.1));
  assert(Near(history_first[12], 0.0));
  assert(Near(history_first[15], 0.0));

  state.angular_velocity[0] = 0.4;
  const auto history_second = history_core.BuildObservation(
      state, moving, 0.15, previous_action);
  assert(Near(history_second[0], 0.1));
  assert(Near(history_second[3], 0.4));
  assert(Near(history_second[12], 0.0));
  assert(Near(history_second[15], 0.2));
  history_core.ResetObservationHistory();
  const auto history_reset = history_core.BuildObservation(
      state, moving, 0.15, previous_action);
  assert(Near(history_reset[0], 0.4));
  assert(Near(history_reset[3], 0.4));
  assert(Near(history_reset[12], 0.2));
  assert(Near(history_reset[15], 0.2));

  x2::deploy::JointVector action{};
  x2::deploy::JointVector previous_target = contract.default_position;
  action.fill(1.0);
  x2::deploy::SafetyLimits limits;
  limits.action_scale_factor = 0.2;
  limits.max_target_velocity = 1.0;
  const auto target = core.ComputeTargets(action, previous_target, 0.002, limits);
  assert(Near(target[0], 0.002));

  bool rejected_nan = false;
  try {
    action[0] = std::numeric_limits<double>::quiet_NaN();
    static_cast<void>(core.ComputeTargets(action, previous_target, 0.002, limits));
  } catch (const std::runtime_error&) {
    rejected_nan = true;
  }
  assert(rejected_nan);

  x2::deploy::SafetyLimits command_limits;
  const auto clamped = x2::deploy::ClampVelocityCommand({1.0, -1.0, 2.0}, command_limits);
  assert(Near(clamped.x, 0.2));
  assert(Near(clamped.y, -0.1));
  assert(Near(clamped.yaw, 0.2));
  assert(Near(x2::deploy::TiltAngleRadians(state), 0.0));

  std::unordered_map<std::string, std::string> metadata{
      {"metadata_contract", "agibot_x2_velocity_v1"},
      {"observation_names", "base_ang_vel,projected_gravity,command,phase,joint_pos,joint_vel,actions"},
      {"observation_dims", "3,3,3,2,31,31,31"},
      {"policy_dt", "0.02"},
      {"phase_period", "0.6"},
  };
  std::string names;
  std::string defaults;
  std::string scales;
  std::string gains;
  std::string lower;
  std::string upper;
  for (std::size_t index = 0; index < x2::deploy::kJointCount; ++index) {
    const std::string separator = index == 0 ? "" : ",";
    names += separator + "joint_" + std::to_string(index);
    defaults += separator + std::to_string(static_cast<double>(index) * 0.01);
    scales += separator + "0.2";
    gains += separator + "2.0";
    lower += separator + "-1.0";
    upper += separator + "1.0";
  }
  metadata["joint_names"] = names;
  metadata["default_joint_pos"] = defaults;
  metadata["action_scale"] = scales;
  metadata["joint_stiffness"] = gains;
  metadata["joint_damping"] = gains;
  metadata["joint_limit_lower"] = lower;
  metadata["joint_limit_upper"] = upper;
  const auto parsed = x2::deploy::ParsePolicyContract(metadata);
  assert(parsed.joint_names[30] == "joint_30");
  assert(Near(parsed.default_position[30], 0.3));

  metadata["metadata_contract"] = "agibot_x2_velocity_v2";
  metadata["observation_history_length"] = "2";
  const auto parsed_history = x2::deploy::ParsePolicyContract(metadata);
  assert(parsed_history.observation_history_length == 2);

  bool rejected_contract = false;
  try {
    metadata["observation_dims"] = "104";
    static_cast<void>(x2::deploy::ParsePolicyContract(metadata));
  } catch (const std::runtime_error&) {
    rejected_contract = true;
  }
  assert(rejected_contract);

  for (const auto history : {std::size_t{0}, std::size_t{3}}) {
    auto invalid = MakeContract();
    invalid.observation_history_length = history;
    bool rejected = false;
    try { x2::deploy::X2VelocityPolicyCore unsupported(invalid); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
  }
  {
    auto invalid = MakeContract();
    invalid.joint_names[1] = invalid.joint_names[0];
    bool rejected = false;
    try { x2::deploy::X2VelocityPolicyCore duplicate(invalid); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
  }
  {
    auto invalid_limits = limits;
    invalid_limits.action_scale_factor = std::numeric_limits<double>::quiet_NaN();
    action.fill(1.0);
    bool rejected = false;
    try { static_cast<void>(core.ComputeTargets(action, previous_target, 0.002, invalid_limits)); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
  }
  return 0;
}
