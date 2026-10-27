# v0.1.0 · experimental research release

Published source organization: **2026-10-27**. Archived model and simulation evaluation: **2026-09-09**.

- Separate `training/` Python/GPU and `deployment/` C++/ROS 2 packages.
- X2-only PPO task collection, curricula, domain randomization, History2 upgrade and export tools.
- Released moving/settling checkpoints and a single routed ONNX with checksums and model card.
- Eight archived simulation reports, a 35-second labeled video and a README highlight loop.
- CPU contract/artifact checks and independent deployment-core tests in GitHub Actions.
- Vendor SDK/message packages remain external; third-party robot-asset licenses are retained.

The public ROS 2 adapter is independently implemented for this release and has not completed robot regression. Historical hardware achieved standing and some forward motion, but turn-to-zero-command instability remains unresolved. Integration must verify the SDK sensor mapping against the training pelvis-frame IMU convention. See [hardware status](hardware-status.md) and the [policy contract](policy-contract.md).

The release does not bundle factory firmware, runtime libraries, proprietary deployment examples, training logs or credentials. Full source/artifact provenance is in [provenance](provenance.md).
