# Policy and deployment contract

The release boundary between Python training and C++ deployment is `models/history2_hybrid_v1/policy.onnx`. Treat its metadata as part of the model, not optional documentation.

## Tensors and timing

| Field | Released value |
| --- | --- |
| Robot model | AgiBot X2 Ultra v1.3.0, 31 actuated joints |
| Input | `obs`, float32, `[1,208]` |
| Output | `actions`, float32, `[1,31]` |
| Contract | `agibot_x2_velocity_v2` |
| Policy period | 0.02 s / 50 Hz |
| Simulation | 0.005 s physics step, decimation 4 |
| Gait phase period | 0.6 s |
| Normalization | Included inside the ONNX graph |

The historical robot adapter published joint commands at 500 Hz while updating policy targets at 50 Hz. This does not change the policy's time step. A different output scheduler must preserve sensor freshness and policy timing.

## Observation order

History is stored **per term**, oldest first. Flattening two full 104-element vectors produces the wrong input.

| Observation term | Previous sample | Current sample | Values per sample |
| --- | --- | --- | ---: |
| Body angular velocity | `[0:3]` | `[3:6]` | 3 |
| Projected gravity | `[6:9]` | `[9:12]` | 3 |
| Body velocity command | `[12:15]` | `[15:18]` | 3 |
| Gait phase | `[18:20]` | `[20:22]` | 2 |
| Joint position minus default | `[22:53]` | `[53:84]` | 31 |
| Joint velocity | `[84:115]` | `[115:146]` | 31 |
| Previous policy action | `[146:177]` | `[177:208]` | 31 |

Slices use Python's start-inclusive/end-exclusive notation. On reset, duplicate the first valid sample into both history positions. Clear action history and gait state when initializing the policy. Do not append samples on every 500 Hz command publication; history advances with the 50 Hz policy step.

The training body frame is the **pelvis/root frame**: the `imu_ang_vel` sensor is attached to the pelvis in the MJCF, and projected gravity comes from the root orientation. Projected gravity is the world-down unit vector expressed in that frame. An IMU physically attached above the three waist joints is not equivalent during waist motion: both orientation and relative angular velocity need compensation. The ROS adapter makes the IMU source/frame explicit. Verify its actual SDK sensor mapping and calibration; the word `torso` in a vendor topic name does not by itself establish which rigid body is measured.

Commands are `[vx, vy, yaw_rate]` in m/s, m/s and rad/s. Joint arrays must be mapped **by name**, not by assumed bus order. Phase uses the training implementation's sine/cosine clock; see the independent core tests for reset and command-dependent behavior.

## Actions and PD gains

```text
q_target[j] = q_default[j] + action_scale[j] × action[j]
action_scale[j] = 0.25 × effort_limit[j] / Kp[j]
```

The action is an unconstrained residual, not a normalized servo angle. Training used no `[-1,1]` action clip. Raw outputs beyond that interval occur during normal locomotion; imposing that clip changes the controller. Joint mechanical limits, finite-value checks and any explicit experimental envelope are separate deployment concerns and must be documented when comparing results.

The low-level controller applies PD position control using metadata gains, conceptually `tau = Kp × (q_target − q) + Kd × (dq_target − dq)`. The released target velocity and feedforward torque are zero; actuator limits and firmware behavior remain robot-side responsibilities. Action scaling is not an extra observation normalization pass.

The recorded leg gain center is:

| Joint family, both legs | Kp | Kd | Model effort limit (Nm) | Action scale (rad/unit) |
| --- | ---: | ---: | ---: | ---: |
| Hip pitch | 120 | 5 | 118 | 0.245833333 |
| Hip roll | 100 | 4 | 118 | 0.295 |
| Hip yaw | 100 | 4 | 118 | 0.295 |
| Knee | 150 | 5 | 118 | 0.196666667 |
| Ankle pitch | 40 | 2 | 36 | 0.225 |
| Ankle roll | 40 | 2 | 24 | 0.15 |

These leg gains came from one factory-controller recording. Waist/arm/wrist/head gains are simulation defaults. Neither the captured gains nor model effort values constitute motor-system identification or universal firmware ratings.

Metadata includes `joint_names`, `default_joint_pos`, `action_scale`, `joint_stiffness`, `joint_damping`, `joint_effort_limit`, `joint_limit_lower`, `joint_limit_upper`, `observation_names`, `observation_dims`, `observation_history_length`, `policy_dt` and `phase_period`. Reject incompatible dimensions, missing arrays, duplicates, non-finite values and unsupported contracts.

## Expert routing

The ONNX graph evaluates the current command at indices `[15:18]`. A norm below `0.1` selects the settling actor; otherwise it selects the moving actor. Each branch contains its own trained observation normalization. The norm mixes linear and angular command components exactly as implemented; it is a routing heuristic, not a physical speed magnitude.

There is no branch blending or hysteresis. A sudden command cancellation can therefore cause a discontinuous action transition. This is a known research limitation and one candidate contributor to real turn-to-stop instability; the archived simulation passes do not establish causation or eliminate that risk.
