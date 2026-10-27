# Hardware validation status

This repository is an experimental research release. It does not claim general, untethered, all-direction stable locomotion on X2 hardware.

## Historical observations

The pre-publication integration was tested with one X2 Ultra. Standing and some hand-controller forward walking were achieved. Sensor callbacks and inference scheduling were investigated; a 30-second sensor-only dry-run showed no stale-state/timeout or non-finite-action event, with typical inference around 0.4 ms. Those timings characterize the historical integration and machine, not a benchmark for every build of the public adapter.

A real turn followed by command cancellation produced delayed tilt divergence, despite finite ONNX outputs and no observed IMU timeout in that run. The evidence did not establish a single cause. Policy state-distribution mismatch, branch transitions, dynamics/calibration and command sequencing need controlled investigation. It would be incorrect to infer that increasing Kp/Kd alone solves it. A historical +5% gain experiment also did not establish improved stability.

The training angular-velocity sensor and root gravity use the pelvis frame. Confirm the actual SDK IMU mapping on the target system before comparing observations. The historical topic contains the word `torso`, but official simulator mappings distinguish body/root and chest sensors, so the topic name alone is not evidence of a mismatch. An IMU physically above the waist would require orientation and relative-angular-velocity compensation. The public adapter exposes the source/frame explicitly; it does not implement unvalidated waist compensation.

The public release separates independently authored policy code from vendor code with unclear redistribution terms. The new ROS 2 integration has dry-run defaults and tests, but has **not** completed the historical adapter's hardware regression. The public code is therefore a reproducible integration starting point, not a drop-in certified replacement for the factory controller.

## Priorities

1. Add sustained turn → zero-command recovery sequences, both directions and multiple durations, to training and evaluation.
2. Compare hard expert routing with hysteresis, smooth blending or distillation into one actor.
3. Record synchronized commands, base orientation/angular velocity, joint state, targets, gains, actions and timing during repeatable robot trials.
4. Validate the public adapter's coordinate convention, named-joint mapping, SDK message contract, fault/stop path and actual low-level command semantics on the target firmware.
5. Expand independent regression across hardware, seeds, command profiles and randomization before changing release status.

Use the [deployment guide](../deployment/README.md) for the actual software interface and default behavior. Follow the robot manufacturer's operation requirements for physical testing; a simulation pass alone does not authorize a motion sequence.
