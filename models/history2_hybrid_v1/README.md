# History2 Hybrid v1 · model card

An experimental, 31-joint velocity policy for **AgiBot Lingxi X2 Ultra v1.3.0**. This directory contains the actual released weights, not download placeholders.

| Artifact | Purpose |
| --- | --- |
| `policy.onnx` | Deployment graph, including actor normalization and both expert branches |
| `model_moving_4740.pt` | Moving expert; checkpoint 4740 |
| `model_settling_4720.pt` | Settling expert; checkpoint 4720 |
| `manifest.json` | Shapes, provenance, checksums and validation status |

The policy takes float32 `obs[1,208]` and returns `actions[1,31]` every 20 ms. Each observation term contains its previous sample followed by its current sample. `norm(current_command) < 0.1` selects the settling expert; all other commands select the moving expert. This is a hard switch without blending or hysteresis. See the [policy contract](../../docs/policy-contract.md) before integrating.

## Training and intended use

Both actors descend from a PPO locomotion policy trained from scratch in mjlab / MuJoCo Warp. Training proceeds through gait acquisition, multidirectional behavior, domain randomization, and two-frame observation fine-tuning. The leg PD gains are values recorded from one robot's factory-controller command stream; upper-body gains retain simulation defaults. They are **not** manufacturer specifications for every X2.

The release is intended for simulation research, reproducible evaluation and engineering integration. Archived evaluation consists of eight nominal/robust reports across two seeds, with 32 parallel environments per report. All 120 reported stage results passed their configured gates. Full reports and their limits are in [evaluation](../../docs/evaluation.md).

## Hardware status and limits

Historical tests established standing and some forward motion on one robot. Turning followed by command cancellation could still cause delayed loss of balance. All-direction untethered walking has not passed final acceptance. The public ROS 2 integration was separated from the private/vendor-derived adapter during release preparation and requires its own hardware regression. Simulation video must not be interpreted as a real-robot performance claim.

The exported graph includes no deployment-side `[-1,1]` action clipping. Its raw outputs can exceed that interval. Preserve the action scaling, named joint order, gains and observation semantics in ONNX metadata. Different firmware, joint order, IMU convention or gain recipe requires validation.

## Verify

From the repository root:

```bash
python tools/check_release.py
python tools/check_release.py --onnx
```

The second command needs `numpy` and `onnx`. Checkpoint files are PyTorch artifacts; load only trusted checkpoints. Source-code licensing is described in [LICENSE](../../LICENSE), with exceptions in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
