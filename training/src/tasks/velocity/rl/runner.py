import os

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


_PRECISE_NUMERIC_METADATA = (
  "joint_stiffness",
  "joint_damping",
  "default_joint_pos",
  "action_scale",
)


def _preserve_numeric_metadata_precision(metadata: dict) -> None:
  """Serialize deployment-critical arrays without the exporter 3-digit rounding."""
  for key in _PRECISE_NUMERIC_METADATA:
    value = metadata.get(key)
    if isinstance(value, list):
      metadata[key] = ",".join(format(float(item), ".9g") for item in value)


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def load(self, *args, **kwargs):
    if (
      os.environ.get("RESET_OPTIMIZER_ON_LOAD") == "1"
      and kwargs.get("load_cfg") is None
    ):
      kwargs["load_cfg"] = {
        "actor": True,
        "critic": True,
        "optimizer": False,
        "iteration": True,
        "rnd": False,
      }
    infos = super().load(*args, **kwargs)
    if os.environ.get("RESET_OPTIMIZER_ON_LOAD") == "1":
      print("[INFO] Optimizer state was reset for the changed fine-tuning objective.")
    if os.environ.get("FREEZE_OBS_NORMALIZATION") == "1":
      for model in (self.alg.actor, self.alg.critic):
        normalizer = getattr(model, "obs_normalizer", None)
        count = getattr(normalizer, "count", None)
        if count is not None:
          normalizer.until = int(count.item())
      print("[INFO] Observation normalization statistics are frozen.")
    return infos

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    _preserve_numeric_metadata_precision(metadata)
    history_length = self.env.unwrapped.cfg.observations["actor"].history_length
    history_length = int(history_length) if history_length else 1
    metadata.update(
      {
        "metadata_contract": (
          "agibot_x2_velocity_v2" if history_length > 1
          else "agibot_x2_velocity_v1"
        ),
        "policy_dt": "0.02",
        "phase_period": "0.6",
        "observation_dims": "3,3,3,2,31,31,31",
        "observation_history_length": str(history_length),
      }
    )
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
