# X2 task catalog

This package registers 92 X2 velocity tasks. Only the X2 task family is
shown by `scripts/list_envs.py`, `scripts/train.py` and `scripts/play.py`.
Definitions and exact reward/randomization settings live in
[`env_cfgs.py`](src/tasks/velocity/config/x2_ultra/env_cfgs.py);
registration aliases live in
[`__init__.py`](src/tasks/velocity/config/x2_ultra/__init__.py).

## Choose a gain recipe first

| Task family | Gain center | Intended use |
| --- | --- | --- |
| `Factory-Recorded` | Captured 12 leg-joint Kp/Kd; simulation defaults elsewhere | Released History2 hybrid and new factory-gain experiments |
| `Kp090-Kd080` | Earlier simulation recipe ×0.90 Kp / ×0.80 Kd | Reproduce earlier staged experiments |
| Neither suffix | Original simulation gain center | Baseline and historical robustness experiments |

A matching observation width alone does not make two checkpoints compatible.
Keep the joint order, gain recipe, action scale, history and task objective
consistent. Task IDs containing `History2` use 208-input actors; others use
104 inputs. `Factory-Recorded` aliases replace the underlying robot and action
scale while retaining the selected curriculum's commands/rewards.

## Released temporal objectives

| Task suffix after `Agibot-X2-Ultra-Behavior-Factory-Recorded-` | Objective |
| --- | --- |
| `History2-Full-Envelope` | Full-speed temporal policy with mild randomization; shared export/evaluation contract |
| `History2-Full-Settle` | Cancellation, braking and zero-command settling; released settling expert objective |
| `History2-Full-Gait-Settle` | Symmetric alternating feet with full-envelope replay; released moving expert objective |
| `History2-Full-Stop` | Historical full-speed braking refinement |
| `History2-Full-Gait-Stop` | Historical joint gait/braking objective |

`train_history2.sh envelope|moving|settling` selects the first three. The actor
history is term-wise chronological; the critic history is independently
specified by each environment.

## Factory-recorded tasks

| Task ID | Actor inputs |
| --- | ---: |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Backward-Stabilize` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Flat` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Full-Speed` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Full-Speed-Brake` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Full-Speed-Gait-Refine` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Full-Speed-Leg-Robust` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Full-Speed-Robust-Finalize` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Full-Speed-Robust-Mild` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Full-Speed-Robust-Stop` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-History2-Full-Envelope` | 208 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-History2-Full-Gait-Settle` | 208 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-History2-Full-Gait-Stop` | 208 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-History2-Full-Settle` | 208 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-History2-Full-Stop` | 208 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Lateral-Balance` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Lateral-Bootstrap` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Omni-Consolidate` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Omni-Finalize` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Omni-Low-Speed` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Omni-Rehearsal` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Right-Gait-Finish` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Right-Gait-Refine` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Robust-Actuator` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Robust-Dynamics` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Robust-Medium` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Robust-Mild` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Robust-Mild-Lateral` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Robust-Mild-Sagittal` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Robust-Mild-Turn` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Robust-Sensor` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Sagittal-Balance` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Sagittal-Stabilize` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Sensor-Bias` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Sensor-Delay` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Sensor-History2-Backward-Clearance` | 208 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Sensor-History2-Backward-Finish` | 208 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Sensor-History2-Gait-Recover` | 208 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Sensor-History2-Nominal-Finish` | 208 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Sensor-History2-Stability` | 208 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Sensor-History2-Turn-Finish` | 208 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Sensor-Stability` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Turn-Balance` | 104 |
| `Agibot-X2-Ultra-Behavior-Factory-Recorded-Turn-Bootstrap` | 104 |
| `Agibot-X2-Ultra-Factory-Recorded-Arc-Turn-Bootstrap` | 104 |
| `Agibot-X2-Ultra-Factory-Recorded-Flat` | 104 |
| `Agibot-X2-Ultra-Factory-Recorded-Gait-Bootstrap` | 104 |
| `Agibot-X2-Ultra-Factory-Recorded-Turn-Gait-Bootstrap` | 104 |
| `Agibot-X2-Ultra-Factory-Recorded-Turn-Seeded-Bootstrap` | 104 |

## Historical gain-center tasks

These remain available to reproduce previous experiments and compare recipes.
They are not a list of stages that must all be run, and they are not the gain
contract of the released factory-recorded model.

| Task ID | Actor inputs |
| --- | ---: |
| `Agibot-X2-Ultra-Behavior-Flat` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Backward-Stabilize` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Flat` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Full-Speed` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Full-Speed-Brake` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Full-Speed-Gait-Refine` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Full-Speed-Leg-Robust` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Full-Speed-Robust-Finalize` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Full-Speed-Robust-Mild` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Full-Speed-Robust-Stop` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Lateral-Balance` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Lateral-Bootstrap` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Omni-Consolidate` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Omni-Finalize` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Omni-Low-Speed` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Omni-Rehearsal` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Robust-Actuator` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Robust-Dynamics` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Robust-Flat` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Robust-Medium` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Robust-Mild` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Robust-Mild-Lateral` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Robust-Mild-Sagittal` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Robust-Mild-Turn` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Robust-Omni` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Robust-Sensor` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Sagittal-Balance` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Sagittal-Stabilize` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Turn-Balance` | 104 |
| `Agibot-X2-Ultra-Behavior-Kp090-Kd080-Turn-Bootstrap` | 104 |
| `Agibot-X2-Ultra-Behavior-Robust-Backward-Refine` | 104 |
| `Agibot-X2-Ultra-Behavior-Robust-Flat` | 104 |
| `Agibot-X2-Ultra-Behavior-Robust-Lateral-Backward` | 104 |
| `Agibot-X2-Ultra-Behavior-Robust-Lateral-Balance` | 104 |
| `Agibot-X2-Ultra-Behavior-Robust-Lateral-Brake` | 104 |
| `Agibot-X2-Ultra-Behavior-Robust-Lateral-Refine` | 104 |
| `Agibot-X2-Ultra-Behavior-Robust-Low-Speed` | 104 |
| `Agibot-X2-Ultra-Flat` | 104 |
| `Agibot-X2-Ultra-Kp090-Kd080-Arc-Turn-Bootstrap` | 104 |
| `Agibot-X2-Ultra-Kp090-Kd080-Flat` | 104 |
| `Agibot-X2-Ultra-Kp090-Kd080-Gait-Bootstrap` | 104 |
| `Agibot-X2-Ultra-Kp090-Kd080-Turn-Gait-Bootstrap` | 104 |
| `Agibot-X2-Ultra-Kp090-Kd080-Turn-Seeded-Bootstrap` | 104 |
| `Agibot-X2-Ultra-Rough` | 104 |

## Run a selected task

```bash
TASK_ID=Agibot-X2-Ultra-Behavior-Factory-Recorded-Full-Speed \
PRETRAINED_CHECKPOINT=/path/to/selected.pt \
NUM_ENVS=1024 MAX_ITERATIONS=501 \
  scripts/x2_ultra/train_behavior.sh
```

For a fresh network, omit `PRETRAINED_CHECKPOINT`. The dedicated
`train_factory_recorded_stage1_from_scratch.sh` launcher explicitly clears it.
The generic launcher defaults to `Agibot-X2-Ultra-Behavior-Flat` if `TASK_ID`
is unset, so set `TASK_ID` deliberately for factory-recorded experiments.
