# Third-party notices

This repository uses more than one license. The root [LICENSE](LICENSE)
contains Apache-2.0 for the project's code and self-trained policy artifacts,
except where an original file or the asset exception below states otherwise.
It does not replace third-party licenses or grant rights in external SDKs.

| Included material | Source | License and retained text |
| --- | --- | --- |
| Training framework code adapted in `training/src/` and the generic training/play entry points | [Unitree RL Mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab), revision `1425b15f73bd4095f0df53709d7c389c3eb9e790` | Apache-2.0; full text in [LICENSE](LICENSE), attribution in [NOTICE](NOTICE) |
| `training/src/assets/robots/agibot_x2/xmls/x2_ultra.xml` and all 45 files in its `xmls/assets/` directory | [AgiBot X2 URDF](https://github.com/AgibotTech/agibot_x2_urdf), revision `77f43eb0904dae4c48ccd9154fee824f8ffd4d38`, X2 Ultra v1.3.0 | MulanPSL-2.0; complete original text in [OFFICIAL_MODEL_LICENSE](training/src/assets/robots/agibot_x2/OFFICIAL_MODEL_LICENSE) |

## Training source adaptations

The training package is an X2-focused adaptation, not the unmodified Unitree
distribution. Changes include the robot configuration, task registration,
velocity/gait rewards, command curricula, domain randomization, observation
history, metadata precision, checkpoint migration, expert export, and
simulation evaluation/recording. This release also makes installation and
entry points independent of the original development computers.

The retained upstream code and project changes are covered by Apache-2.0.
Existing attribution and license text must remain with redistributed copies.
The original upstream license file was named `LICENCE`; this repository
provides that Apache-2.0 text as the root `LICENSE`.

## Robot-description asset exception

The MulanPSL-2.0 scope covers the vendor-derived MJCF and STL meshes listed
above. The Python training configuration is project code; storing it beside
the model does not relicense the model as Apache-2.0.

Relative to the identified official v1.3.0 MJCF, the adapted file:

- changes the mesh directory from `meshes` to `assets`;
- adds `left_foot` and `right_foot` sites;
- renames the angular-velocity, linear-velocity and linear-acceleration sensors
  to the names consumed by the task; and
- adds the `root_angmom` subtree angular-momentum sensor.

The 45 supplied STL files are byte-identical to the corresponding files in
the official source revision. Geometry, inertial values and joint limits
originate from that robot description. Preserve the complete Mulan license
and the original copyright, patent, trademark and disclaimer statements when
redistributing the assets. Neither license grants manufacturer trademarks.

The simulation media render this same robot model. Their inclusion does not
change the underlying asset attribution or imply manufacturer endorsement.

## External dependencies, not bundled SDKs

Training uses [mjlab](https://github.com/mujocolab/mjlab),
[MuJoCo](https://github.com/google-deepmind/mujoco),
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp),
[RSL-RL](https://github.com/leggedrobotics/rsl_rl), PyTorch and other Python
packages installed through the package manager. Deployment uses ROS 2,
ONNX Runtime and the robot owner's separately obtained AimDK SDK. These
dependencies keep their own licenses and notices in their distributions;
the repository does not vendor their Python packages or compiled libraries.
Package dependencies are recorded in
[`training/pyproject.toml`](training/pyproject.toml) and
[`deployment/package.xml`](deployment/package.xml).

Official AimDK/X2 RL Deploy message definitions, predictor implementation,
simulator executables/libraries and sample dance-policy weights are excluded.
During release preparation, neither the historical vendor example nor the
official public X2 RL Deploy v1.1 archive contained an explicit redistribution
license; their ROS package manifests used placeholder license declarations.
Public download availability was not treated as permission to relicense them.
Obtain the appropriate SDK independently through the
[official SDK download instructions](https://x2-aimdk.agibot.com/zh-cn/latest/get_sdk/index.html).

The public deployment package implements its own policy/ROS integration and
consumes the official SDK as an external dependency. The project's license
does not grant redistribution rights to that external SDK. Provenance and
the release audit are recorded in [docs/provenance.md](docs/provenance.md).
