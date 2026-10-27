# Training · Lingxi X2 Ultra

GPU-parallel PPO locomotion training with **mjlab + MuJoCo Warp + RSL-RL**.
This package owns the robot model, reward functions, command curricula,
domain randomization, evaluation and ONNX export. Hardware deployment lives
in [`../deployment`](../deployment/README.md).

The released policy combines a moving expert and a settling expert in one
ONNX graph. Both use two-frame, term-wise observation history. The
[simulation video](../docs/media/simulation.mp4) demonstrates forward and
backward walking, lateral walking, turning, running, stopping and restarting.

## Install

The recorded training stack used Ubuntu 22.04, Python 3.12, an NVIDIA GPU,
PyTorch 2.10.0 with CUDA 12.8, mjlab 1.2.0 and MuJoCo/Warp 3.5.0. Python 3.12
is the recommended version; this package allows Python 3.11–3.12. A 16 GB GPU
was used for the original 1,024-environment runs. Reduce `NUM_ENVS` if your GPU
has less memory.

Run these commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ./training
cd training
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
```

The versions directly used by training, export and recording are pinned in
[`pyproject.toml`](pyproject.toml). This is a tested dependency set, not a full
transitive lockfile. The optional `runtime` extra installs ONNX Runtime; the
included ONNX checks use ONNX's reference evaluator and do not require it.
Video recording also requires an `ffmpeg` executable on `PATH`.

All commands below run from `training/` with the environment activated.
Shell entry points use `python` from `PATH`, or `PYTHON_BIN` when set. They
never rely on a particular computer, account, IP address or virtualenv path.

```bash
python scripts/list_envs.py --keyword Factory-Recorded
python scripts/x2_ultra/check_model.py
NUM_ENVS=16 MAX_ITERATIONS=2 scripts/x2_ultra/smoke_train.sh
```

`check_model.py` checks the 31-joint contract, 24 foot collision geometries,
IMU sensors and model dimensions (`nq=38`, `nv=37`, `nu=31`). The smoke command
performs two training iterations; it is a pipeline check, not a usable policy.

## Train from scratch

Start with the factory-recorded leg gain recipe:

```bash
NUM_ENVS=1024 MAX_ITERATIONS=5001 \
  scripts/x2_ultra/train_factory_recorded_stage1_from_scratch.sh
```

This entry point explicitly clears `PRETRAINED_CHECKPOINT`. Its task is
`Agibot-X2-Ultra-Factory-Recorded-Flat`. Checkpoints and exported ONNX files
are written under `logs/rsl_rl/x2_ultra_velocity/<timestamp>_<run_name>/`.

The leg gain center is symmetric left/right:

| Joint | Kp | Kd |
| --- | ---: | ---: |
| Hip pitch | 120 | 5 |
| Hip roll | 100 | 4 |
| Hip yaw | 100 | 4 |
| Knee | 150 | 5 |
| Ankle pitch | 40 | 2 |
| Ankle roll | 40 | 2 |

Only the 12 leg-joint gain pairs were captured from the factory controller.
Waist, arm, wrist and head gains remain the project's simulation values.
They are not claimed as factory-identified values.

Base training alone is not the full released recipe. The available curricula
progress through sagittal balance, lateral/turn bootstrapping, omnidirectional
rehearsal, full-speed behavior and randomization. Select each checkpoint using
the behavior evaluations before proceeding to the next objective. All
historical task IDs and their gain recipes are preserved in
[`TASKS.md`](TASKS.md).

## Fine-tune behaviors

Choose a compatible task and supply the previous checkpoint:

```bash
TASK_ID=Agibot-X2-Ultra-Behavior-Factory-Recorded-Omni-Rehearsal \
PRETRAINED_CHECKPOINT=/path/to/selected_model.pt \
NUM_ENVS=1024 MAX_ITERATIONS=1001 \
RUN_NAME=x2_omni_rehearsal \
  scripts/x2_ultra/train_behavior.sh
```

`MAX_ITERATIONS` is the number of **additional PPO iterations** requested
for this run. Checkpoint names retain the loaded iteration counter. The
launcher stages the source checkpoint in the local experiment directory;
it does not modify the supplied checkpoint.

Task IDs containing `Kp090-Kd080` preserve an earlier gain center. Their
specialized shell wrappers remain available for historical experiments.
For the released factory-recorded recipe, set `TASK_ID` explicitly with the
generic launcher rather than using a `train_kp090_kd080_*.sh` wrapper.

Useful controls include `NUM_ENVS`, `SAVE_INTERVAL`, `RUN_NAME`,
`LEARNING_RATE` and `PPO_SCHEDULE`. `RESET_OPTIMIZER_ON_LOAD=1` restarts Adam
moments when changing objectives. `FREEZE_OBS_NORMALIZATION=1` freezes running
observation statistics for controlled experiments; it is not enabled by default.

## Two-frame history and the released hybrid

The original actor has 104 inputs. The temporal actor has 208 inputs in
**term-wise oldest-to-newest order**: previous/current angular velocity,
previous/current gravity, previous/current command, and so on. It is not
the concatenation of two complete 104-element observations. The critic
retains its task-defined privileged observations.

To upgrade a selected 104-input checkpoint without changing its initial
actor output:

```bash
python scripts/x2_ultra/upgrade_checkpoint_actor_history.py \
  --input /path/to/single_frame_model.pt \
  --output checkpoints/history2_seed.pt

PRETRAINED_CHECKPOINT=checkpoints/history2_seed.pt \
  scripts/x2_ultra/train_history2.sh envelope
```

The upgrade zero-initializes old-frame first-layer weights, copies the
newest-frame weights and normalization statistics, and clears optimizer
moments. This provides an initialization for temporal fine-tuning.

Once the full-envelope policy passes its evaluations, refine two branches
from a selected temporal checkpoint:

```bash
PRETRAINED_CHECKPOINT=/path/to/selected_history2.pt \
  scripts/x2_ultra/train_history2.sh moving

PRETRAINED_CHECKPOINT=/path/to/selected_history2.pt \
  scripts/x2_ultra/train_history2.sh settling
```

The moving objective uses `History2-Full-Gait-Settle` with a default learning
rate of `1e-6`; the settling objective uses `History2-Full-Settle` at `3e-6`.
Both shortcuts default to 501 additional iterations. `FROM_SCRATCH=1` can
initialize a new temporal policy directly, but that is an alternate training
route, not the provenance of the released weights.

The released moving and settling checkpoints are `model_moving_4740.pt`
and `model_settling_4720.pt` in [`../models/history2_hybrid_v1`](../models/history2_hybrid_v1/).
The graph uses the settling expert when `norm(current_command) < 0.1`,
and the moving expert otherwise. The norm operates on `[vx, vy, yaw_rate]`
as implemented; it is an engineering gate, not a dimensionally normalized
physical quantity. This is hard routing with no interpolation or hysteresis.

The original experiment involved manual checkpoint selection and multiple
curriculum stages. The release includes the selected checkpoints and source
objectives; the commands above do not promise bit-identical weights from one
fresh training run or a different GPU.

## Evaluate

Run the full behavior sequence using both released experts:

```bash
python scripts/x2_ultra/evaluate_behaviors.py \
  --checkpoint-file ../models/history2_hybrid_v1/model_moving_4740.pt \
  --zero-command-checkpoint-file ../models/history2_hybrid_v1/model_settling_4720.pt \
  --task-id Agibot-X2-Ultra-Behavior-Factory-Recorded-History2-Full-Envelope \
  --profile full --environment robust --seed 42 --num-envs 32 \
  --output-file reports/full_robust_seed42.json
```

Repeat with `--environment nominal`, `--profile omni-low-speed`, and seeds
`42` and `7` to reproduce the eight published evaluation configurations.
Reports include velocity tracking, survival, tilt, contact timing, gait
symmetry, swing height, braking and restart behavior. A failing behavior gate
returns a nonzero exit status. `--duration-scale` is useful for a short pipeline
check; acceptance numbers must use the default complete durations.

The released reports are simulation evidence. Hardware walking, turning and
command-cancel transitions still require robot-specific validation.

## Export ONNX

Export the routed experts and attach the full deployment contract:

```bash
python scripts/x2_ultra/export_zero_command_hybrid_policy.py \
  ../models/history2_hybrid_v1/model_moving_4740.pt \
  ../models/history2_hybrid_v1/model_settling_4720.pt \
  exports/hybrid_raw.onnx

python scripts/x2_ultra/prepare_deploy_onnx.py \
  exports/hybrid_raw.onnx exports/policy.onnx --gain-recipe factory-recorded

python scripts/x2_ultra/check_onnx.py exports/policy.onnx

python scripts/x2_ultra/check_onnx_parity.py \
  ../models/history2_hybrid_v1/model_moving_4740.pt exports/policy.onnx \
  --zero-command-checkpoint-file ../models/history2_hybrid_v1/model_settling_4720.pt \
  --task-id Agibot-X2-Ultra-Behavior-Factory-Recorded-History2-Full-Envelope
```

For a single expert, use `export_checkpoint_onnx.py CHECKPOINT OUTPUT
--task-id TASK_ID`, then `prepare_deploy_onnx.py` with its matching gain recipe.

The actor runs at 50 Hz (`dt=0.005`, `decimation=4`). Its output is a
31-element joint-position residual:

```text
joint_target = default_joint_position + action_scale * action
action_scale[j] = 0.25 * effort_limit[j] / Kp[j]
```

Normalization lives inside the exported graph. Deployment must read the joint
order, default pose, gains, action scales, joint limits and observation-history
metadata from ONNX. Do not apply observation normalization twice. The released
model contract is `obs[1,208] → actions[1,31]` with a 0.6-second gait phase period.

## Record a simulation video

```bash
python scripts/x2_ultra/record_behavior_demo.py \
  ../models/history2_hybrid_v1/model_moving_4740.pt \
  videos/simulation.mp4 \
  --zero-command-checkpoint-file ../models/history2_hybrid_v1/model_settling_4720.pt \
  --task-id Agibot-X2-Ultra-Behavior-Factory-Recorded-History2-Full-Envelope \
  --profile full --width 960 --height 540 --fps 25
```

The labeled demo lasts 35 seconds and shows commanded and actual velocities.
It is a demonstration sequence, separate from the longer acceptance suite.

## Implementation map

| Location | Responsibility |
| --- | --- |
| `src/assets/robots/agibot_x2/` | Official v1.3.0 MJCF/meshes, gain recipes and joint order |
| `src/tasks/velocity/config/x2_ultra/` | X2 tasks, curricula, randomization and PPO configuration |
| `src/tasks/velocity/mdp/` | Behavior command generator, rewards, observations and termination terms |
| `src/tasks/velocity/rl/runner.py` | Checkpoint handling and precise ONNX metadata export |
| `scripts/train.py`, `scripts/play.py` | Training and interactive playback |
| `scripts/x2_ultra/` | Staged launchers, evaluation, history upgrade, export and recording |

The actor/critic MLPs use `512 → 256 → 128` hidden layers and ELU, with running
observation normalization. PPO uses 24 steps per environment, five epochs,
four minibatches, clip 0.2, gamma 0.99 and GAE lambda 0.95. Behavior curricula
explicitly sample standing, cancellation, restart, reversal, lateral motion,
turning and running. Sim-to-real randomization varies PD gains, effort limits,
inertia, center of mass, joint and contact friction, encoder bias and sensor
delay. Exact strengths are task-specific and defined in `env_cfgs.py`.

Only X2 velocity tasks are included. IPMAN motion imitation, other robots,
local experiment logs, virtualenvs, caches and vendor deployment binaries are
outside this package. Upstream training code derives from
[`unitreerobotics/unitree_rl_mjlab`](https://github.com/unitreerobotics/unitree_rl_mjlab)
at commit `1425b15f73bd4095f0df53709d7c389c3eb9e790` under Apache-2.0.
Official AgiBot model assets retain their
[`OFFICIAL_MODEL_LICENSE`](src/assets/robots/agibot_x2/OFFICIAL_MODEL_LICENSE)
(Mulan PSL v2).
