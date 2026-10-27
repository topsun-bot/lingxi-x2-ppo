#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace x2::deploy {

inline constexpr std::size_t kJointCount = 31;
inline constexpr std::size_t kSingleObservationCount = 104;
inline constexpr double kCommandDeadzone = 0.1;

using JointVector = std::array<double, kJointCount>;
using ObservationVector = std::vector<float>;

struct PolicyContract {
  std::array<std::string, kJointCount> joint_names{};
  JointVector default_position{};
  JointVector action_scale{};
  JointVector stiffness{};
  JointVector damping{};
  JointVector lower_limit{};
  JointVector upper_limit{};
  double policy_dt{0.02};
  double phase_period{0.6};
  std::size_t observation_history_length{1};
};

struct RobotState {
  JointVector position{};
  JointVector velocity{};
  std::array<double, 3> angular_velocity{};
  // ROS IMU orientation converted to scalar-first order.
  std::array<double, 4> orientation_wxyz{1.0, 0.0, 0.0, 0.0};
};

struct VelocityCommand {
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
};

struct SafetyLimits {
  double max_command_x{0.2};
  double max_command_y{0.1};
  double max_command_yaw{0.2};
  double action_clip{1.0};
  double action_scale_factor{0.2};
  double joint_limit_margin{0.03};
  double max_position_deviation{0.35};
  double max_target_velocity{1.0};
};

[[nodiscard]] PolicyContract ParsePolicyContract(
    const std::unordered_map<std::string, std::string>& metadata);

[[nodiscard]] VelocityCommand ClampVelocityCommand(
    const VelocityCommand& requested_command,
    const SafetyLimits& limits);

[[nodiscard]] double TiltAngleRadians(const RobotState& state);

class X2VelocityPolicyCore {
 public:
  explicit X2VelocityPolicyCore(PolicyContract contract);

  [[nodiscard]] ObservationVector BuildObservation(
      const RobotState& state,
      const VelocityCommand& requested_command,
      double phase_time,
      const JointVector& previous_action);

  void ResetObservationHistory();

  [[nodiscard]] std::size_t observation_count() const {
    return kSingleObservationCount * contract_.observation_history_length;
  }

  [[nodiscard]] JointVector ComputeTargets(
      const JointVector& action,
      const JointVector& previous_target,
      double publish_dt,
      const SafetyLimits& limits) const;

  [[nodiscard]] const PolicyContract& contract() const { return contract_; }

 private:
  PolicyContract contract_;
  std::array<float, kSingleObservationCount> previous_observation_{};
  bool observation_history_initialized_{false};
};

}  // namespace x2::deploy
