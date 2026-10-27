# Provenance and release record

The released policy and evaluation evidence are historical artifacts from
**2026-09-09**. The repository was curated for public release on
**2026-10-27**. Release preparation did not retrain the policy or rerun the
archived GPU behavior suite. A historical pass is distinguished from a fresh
test of the public code throughout this repository.

## Source lineage

| Component | Origin | Recorded revision / version |
| --- | --- | --- |
| Training framework adaptation | [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) | `1425b15f73bd4095f0df53709d7c389c3eb9e790` |
| Robot MJCF and meshes | [AgibotTech/agibot_x2_urdf](https://github.com/AgibotTech/agibot_x2_urdf) | `77f43eb0904dae4c48ccd9154fee824f8ffd4d38`, X2 Ultra v1.3.0 |
| PPO and vectorized simulation | mjlab, MuJoCo Warp and RSL-RL | Recorded package versions in [`training/pyproject.toml`](../training/pyproject.toml) |
| Public deployment integration | Project policy core and separately authored ROS 2 integration | Public release source in [`deployment/`](../deployment/) |

Upstream training source is Apache-2.0. The robot-description assets retain
MulanPSL-2.0. Exact scope and MJCF modifications are documented in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). No unrelated Unitree robot
meshes, vendor simulator binaries or external SDK source packages are part
of this release.

## Policy lineage

The training record starts with the project's factory-recorded-gain PPO
recipe initialized from random weights. Its original manifest records
`random_initialization: true`, `resume: false` and seed 42. Subsequent stages
reuse this project's checkpoints for gait acquisition, multidirectional
commands, full-speed behavior, randomization and two-frame history
fine-tuning. The released History2 policy is therefore a later fine-tuned
policy, not a claim that the two final experts were independently initialized
from scratch in their last training stage.

Both released experts have the same observation/action contract:

- `model_moving_4740.pt`: moving expert, checkpoint iteration 4740;
- `model_settling_4720.pt`: settling expert, checkpoint iteration 4720;
- `policy.onnx`: the two actors and their observation normalization in one
  ONNX graph. `norm(current_command) < 0.1` selects settling, otherwise moving.

The 208 input values contain two frames of the 104-value observation, laid
out previous/current within each observation term. The graph emits 31 raw
actions at a 20 ms policy period. The export records both checkpoint hashes
in ONNX metadata, allowing the graph to be matched to the bundled experts.
The [model card](../models/history2_hybrid_v1/README.md) and
[manifest](../models/history2_hybrid_v1/manifest.json) describe the complete
contract and its limitations.

The leg gains were measured from one robot's factory-controller command
stream. The observation did not reveal a manufacturer training policy or
training dataset; no manufacturer policy weights are included. Upper-body
gains remain simulation values. These gain values are not asserted to be
manufacturer specifications for every X2.

The project releases these self-trained weights under Apache-2.0. This does
not relicense the robot assets or any independently obtained manufacturer
software. Weights are research artifacts; final all-direction, untethered
hardware stability has not been established.

## Immutable artifact checksums

The weights were copied without changing their bytes. Paths below are
relative to the repository root; hashes use SHA-256.

```text
33f7b37300d83a9a8676665e0413cbd430bb19dbc98a52aa1b14029abfb79c40  models/history2_hybrid_v1/policy.onnx
0097415d4d5c75f6c53e82c0d607a6fdce9d95c1f731624778eede577bc23652  models/history2_hybrid_v1/model_moving_4740.pt
3b8a60efbf9a3fb31ef84c50eb8fb1150a61f1172dc331ed02e95e7c1fc41fc3  models/history2_hybrid_v1/model_settling_4720.pt
ddad1a4052491980d75fe33a4d4df709496eefff3279071cc7a36e1746bdf8ef  docs/media/simulation.mp4
```

The MP4 is the archived 35-second MuJoCo demonstration, 960 × 540 at 25 fps,
with command/actual-velocity annotations and no audio. The README GIF is a
derived preview. Both depict simulation, not a real-robot recording.

## Archived evaluation and redaction

Eight original JSON reports cover nominal/robust environments and full/
low-speed omnidirectional profiles with seeds 42 and 7. Each report uses
32 parallel environments. Across the archived reports, all 120 stage
results passed the configured gates, the minimum stage survival rate was
100%, and the largest stage tilt p95 was 12.2148 degrees. These statements
refer to the archived simulation runs, not a new hardware acceptance test.

The public copies in [`docs/evaluations/`](evaluations/) replace **only** the
two top-level checkpoint path values:

| JSON key | Public replacement |
| --- | --- |
| `checkpoint` | `models/history2_hybrid_v1/model_moving_4740.pt` |
| `zero_command_checkpoint` | `models/history2_hybrid_v1/model_settling_4720.pt` |

Private absolute paths were removed. All other JSON values, including stage
measurements, commands, seeds, thresholds and pass/fail results, were retained.
Release preparation compared the parsed original and public reports and
confirmed that only these two keys changed in each of the eight files.
The byte hashes necessarily differ after path replacement.

The following hashes identify the **original, unredacted report bytes**.
Filenames identify their public counterparts; the original private path
values are deliberately not reproduced.

```text
6eebc6514de2e8fc4579da00a86c9b6c861919989457a07c274a34b4888b0416  eval_full_nominal_seed42_n32.json
a0d6b194fa075da79fe1af1450eb850fb49c053b7eccea031e118bd9f5a5bf97  eval_full_nominal_seed7_n32.json
caef1185b381ca75a931b128507183d1a5e74f463374c6b38ee6de3532e4e2de  eval_full_robust_seed42_n32.json
4479dfe4ce1bdcf5f85497af795e47defa32df3f143688fccbfcca4e192cae3b  eval_full_robust_seed7_n32.json
efe2634107fb644eeb16ecced4b73736a6c15b73b0e18e5c162856441214bbcd  eval_omni_low_speed_nominal_seed42_n32.json
33571aefbb9819f1b5eef748ea393d9d0f7da3c8d4199ac6acdad8d73ac392b1  eval_omni_low_speed_nominal_seed7_n32.json
5e0688e800786d011e161236be5f92c29319419ebec901a3b80e1f6e35ccace7  eval_omni_low_speed_robust_seed42_n32.json
bce3c49539b8a66c92aeda66fa1ecf51297afb60d85ccba19e9dbfeb242dc3bb  eval_omni_low_speed_robust_seed7_n32.json
```

For comparison, SHA-256 of the **public, redacted copies** is:

```text
78ba4be115fce65d9a6a9da08e588edd945b8311e3f20f0cf8c8bca7ccfec4ca  eval_full_nominal_seed42_n32.json
e762c4432886d3d3c234ea2b3552a706c267837133877ebb49a50d92410a00c8  eval_full_nominal_seed7_n32.json
26484ff44d6935e969c6fd593aa510d50840b66f7ee494dfb451ed26373cf27f  eval_full_robust_seed42_n32.json
8f57c5f0130deeb10432e61ec1f0735b76bf108f53df2b0b1e94063a638ceb3c  eval_full_robust_seed7_n32.json
d5a4ade46403e6bbe1e584e5435700fbd86acaab652f89455305c481427144f7  eval_omni_low_speed_nominal_seed42_n32.json
32ed0e590bd46897f5795f4a8e10eb7c5c7d0b470515c559e3e6d9d1cb5e08a9  eval_omni_low_speed_nominal_seed7_n32.json
0cf56f7b8e55756e2f81447120af35bafc703cd3649ce6c54f2ca6bb0d4307ea  eval_omni_low_speed_robust_seed42_n32.json
f69da8276b672ec5773b81e9036df7e38e00f5aa2bbe75bf4a3a62a186a9582c  eval_omni_low_speed_robust_seed7_n32.json
```

## Deployment release boundary

The private development adapter incorporated the manufacturer's X2 RL Deploy
example. That historical adapter is not republished as a licensed vendor
distribution here. The public release retains the project's policy core and
uses a separately authored ROS 2 controller that links to an external SDK.
The new integration requires its own hardware regression; historical robot
results do not certify the newly packaged controller.

On 2026-10-27, release preparation inspected the official download manifest
at [sdk-downloads.json](https://x2-aimdk.agibot.com/sdk-downloads.json) and the
linked [X2 RL Deploy v1.1 archive](https://x2-aimdk.agibot.com/downloads/aimdk-extra-x2_rl_deploy-v1.1.zip).
The inspected archive had SHA-256
`cbf772de7875a70f85b4bd4f3097e763065d64fbebc0c811e65833a3c2339f17`.
It contained no explicit LICENSE, NOTICE or COPYRIGHT file, and the ROS
package manifests retained placeholder license declarations. Consequently,
its message definitions, predictor source, binary simulator and example
policy are excluded rather than assigned this project's license. This is
a record of the inspected archive, not a statement about future SDK releases
or a substitute for any terms provided separately by the manufacturer.

Only selected code, assets and evidence are published. Private machine
addresses, credentials, complete development logs, virtual environments,
build/install trees, unrelated motion experiments and vendor binaries are
outside the release. The release is a curated research snapshot, not a
complete archive of every training run or a claim of bitwise deterministic
retraining from the first random seed.
