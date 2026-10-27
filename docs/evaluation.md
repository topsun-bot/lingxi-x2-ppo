# Simulation evaluation

The files in [`evaluations/`](evaluations/) are archived results for the released History2 Hybrid policy, generated on **2026-09-09**. Publication preparation normalized checkpoint locations to public repository paths; measurements and pass/fail fields were retained. See [provenance](provenance.md) for original report hashes.

## Coverage

| Profile | Environment | Seed 42 | Seed 7 |
| --- | --- | --- | --- |
| Full | Nominal | [PASS](evaluations/eval_full_nominal_seed42_n32.json) | [PASS](evaluations/eval_full_nominal_seed7_n32.json) |
| Full | Robust | [PASS](evaluations/eval_full_robust_seed42_n32.json) | [PASS](evaluations/eval_full_robust_seed7_n32.json) |
| Omni low speed | Nominal | [PASS](evaluations/eval_omni_low_speed_nominal_seed42_n32.json) | [PASS](evaluations/eval_omni_low_speed_nominal_seed7_n32.json) |
| Omni low speed | Robust | [PASS](evaluations/eval_omni_low_speed_robust_seed42_n32.json) | [PASS](evaluations/eval_omni_low_speed_robust_seed7_n32.json) |

Each configuration runs 32 parallel environments. There are 120 **stage results** across the eight reports, not 120 independent physical trials. Every reported stage passed its configured gate, with a minimum survival rate of 1.0 and a maximum stage tilt p95 of 12.2148°.

Full-profile stages cover a 60-second zero-command stand; forward, backward and lateral motion; turning; running; braking; restart; repeated direction changes; and final standing. Low-speed evaluation checks behavior near practical teleoperation command levels. Each JSON contains its own commands, durations, metrics and failures. Thresholds are implemented in `training/scripts/x2_ultra/evaluate_behaviors.py`; changing them changes what PASS means.

## Selected measurements

| Metric | Seed 42 | Seed 7 |
| --- | ---: | ---: |
| Robust full: walking → stop time p90 | 2.278 s | 2.632 s |
| Robust full: running → stop time p90 | 2.350 s | 2.568 s |

The robust configuration's braking gate is 3.5 s; nominal uses 2.0 s. Archived forward walking at 0.35 m/s showed nominal left/right per-foot contact frequencies of about 1.667 / 1.667 Hz. The archived export parity check compared 1,705 scalar action outputs and reported maximum absolute error `5.9604645e-7`. The release verifier performs an independent shape/metadata/finite-output smoke check; it is not a substitute for that rollout parity test.

## Reproduce

Install the [training environment](../training/README.md), then run from `training/`:

```bash
export MUJOCO_GL=egl
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/x2_ultra/evaluate_behaviors.py \
  --checkpoint-file ../models/history2_hybrid_v1/model_moving_4740.pt \
  --zero-command-checkpoint-file ../models/history2_hybrid_v1/model_settling_4720.pt \
  --task-id Agibot-X2-Ultra-Behavior-Factory-Recorded-History2-Full-Envelope \
  --profile full --environment robust --seed 42 --num-envs 32 \
  --output-file reports/full_robust_seed42.json
```

Repeat the combinations of `full` / `omni-low-speed`, `nominal` / `robust`, and seeds `42` / `7`. Use full durations for acceptance. Shortened durations are useful only for checking the pipeline. Regenerated reports can vary with GPU, backend versions and nondeterminism; archived PASS is not a promise of bit-identical retraining.

The exact training envelopes and domain randomization are defined in `training/src/tasks/velocity/config/x2_ultra/env_cfgs.py`. Broad training stages and the final mild-randomization stages have different ranges. Do not interpret robust evaluation as coverage of every earlier perturbation range.

## Limits

This evidence is conditional on the included MJCF, physics backend, command sequences and randomization distributions. It does not validate unknown payloads, robot calibration, terrain, impacts, hardware delays or firmware. The [hardware-status record](hardware-status.md) includes observed failures that the simulator did not predict. Release CI checks source syntax, deployment core, artifacts and CPU ONNX inference; it does not run the full GPU suite.
