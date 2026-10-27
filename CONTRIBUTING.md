# Contributing

Start with an issue describing the behavior to improve and a reproducible observation. For model or deployment reports, include the repository commit, robot/model version, task ID, gain recipe, model SHA-256, seed, command sequence and evaluation JSON. Remove credentials, private network details and personal paths before attaching logs.

Keep the architecture boundary intact: Python learning code belongs in `training/`; runtime code belongs in `deployment/`; the shared policy contract belongs in `docs/policy-contract.md` and ONNX metadata. An observation, action or joint-order change must update its tests and contract together.

Run the applicable checks before opening a pull request:

```bash
python tools/check_release.py
./deployment/scripts/test_core.sh
# With numpy + onnx installed:
python tools/check_release.py --onnx
```

Training changes should also include model checks and an appropriate short smoke run. Changes claiming improved behavior should include the complete relevant evaluation profile, not only a short demo. Hardware behavior claims should identify the exact tested adapter and firmware.

Do not commit training logs, build outputs, SDK archives, passwords, ROS bags or unrelated checkpoints. Preserve third-party attribution and license files. New contributions use the repository's Apache-2.0 code license unless explicitly documented otherwise.
