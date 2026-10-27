"""Create a deployment ONNX with a strict, full-precision X2 policy contract."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path

import mujoco
import onnx

from src.assets.robots.agibot_x2.x2_ultra_constants import (
  HOME_KEYFRAME,
  X2_ULTRA_ACTION_SCALE,
  X2_ULTRA_ARTICULATION,
  X2_ULTRA_FACTORY_RECORDED_ACTION_SCALE,
  X2_ULTRA_FACTORY_RECORDED_ARTICULATION,
  X2_ULTRA_JOINT_NAMES,
  X2_ULTRA_KP090_KD080_ACTION_SCALE,
  X2_ULTRA_KP090_KD080_ARTICULATION,
  X2_ULTRA_XML,
)


GAIN_RECIPES = {
  "legacy": (X2_ULTRA_ARTICULATION, X2_ULTRA_ACTION_SCALE),
  "factory-recorded": (
    X2_ULTRA_FACTORY_RECORDED_ARTICULATION,
    X2_ULTRA_FACTORY_RECORDED_ACTION_SCALE,
  ),
  "kp090-kd080": (
    X2_ULTRA_KP090_KD080_ARTICULATION,
    X2_ULTRA_KP090_KD080_ACTION_SCALE,
  ),
}


def csv(values: list[float] | tuple[float, ...]) -> str:
  return ",".join(format(float(value), ".9g") for value in values)


def resolve_patterns(values: dict[str, float], *, default: float | None = None) -> list[float]:
  resolved: list[float] = []
  for joint_name in X2_ULTRA_JOINT_NAMES:
    matches = [value for pattern, value in values.items() if re.fullmatch(pattern, joint_name)]
    if not matches and default is not None:
      matches = [default]
    if len(matches) != 1:
      raise ValueError(f"Expected one value for {joint_name}, got {matches}")
    resolved.append(float(matches[0]))
  return resolved


def actuator_values(articulation, attribute: str) -> list[float]:
  by_pattern: dict[str, float] = {}
  for actuator in articulation.actuators:
    value = getattr(actuator, attribute)
    if value is None:
      raise ValueError(f"Missing actuator {attribute}: {actuator.target_names_expr}")
    for pattern in actuator.target_names_expr:
      by_pattern[pattern] = float(value)
  return resolve_patterns(by_pattern)


def select_gain_recipe(
  requested: str, metadata: dict[str, str]
) -> tuple[str, tuple]:
  if requested != "auto":
    return requested, GAIN_RECIPES[requested]

  for recipe_name, (articulation, action_scale) in GAIN_RECIPES.items():
    expected = {
      "joint_stiffness": actuator_values(articulation, "stiffness"),
      "joint_damping": actuator_values(articulation, "damping"),
    }
    matched = True
    for key, expected_values in expected.items():
      try:
        actual_values = [float(value) for value in metadata[key].split(",")]
      except (KeyError, ValueError):
        matched = False
        break
      if len(actual_values) != len(expected_values) or any(
        abs(actual - target) > 1.0e-6
        for actual, target in zip(actual_values, expected_values, strict=True)
      ):
        matched = False
        break
    if matched:
      return recipe_name, (articulation, action_scale)

  raise ValueError(
    "Cannot infer the gain recipe from source ONNX metadata; pass "
    "--gain-recipe=legacy, --gain-recipe=factory-recorded, or "
    "--gain-recipe=kp090-kd080 explicitly."
  )


def joint_limits() -> tuple[list[float], list[float]]:
  model = mujoco.MjModel.from_xml_path(str(X2_ULTRA_XML))
  ranges = {
    model.joint(index).name: tuple(float(value) for value in model.jnt_range[index])
    for index in range(model.njnt)
    if model.joint(index).name
  }
  lower: list[float] = []
  upper: list[float] = []
  for name in X2_ULTRA_JOINT_NAMES:
    if name not in ranges:
      raise ValueError(f"Missing joint limit for {name}")
    lo, hi = ranges[name]
    lower.append(lo)
    upper.append(hi)
  return lower, upper


def replace_metadata(model: onnx.ModelProto, values: dict[str, str]) -> None:
  retained = {item.key: item.value for item in model.metadata_props}
  retained.update(values)
  del model.metadata_props[:]
  for key in sorted(retained):
    entry = model.metadata_props.add()
    entry.key = key
    entry.value = retained[key]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("source", type=Path)
  parser.add_argument("destination", type=Path)
  parser.add_argument(
    "--gain-recipe",
    choices=("auto", *GAIN_RECIPES),
    default="auto",
    help="Gain metadata contract; auto preserves a recognized source recipe.",
  )
  args = parser.parse_args()

  source = args.source.expanduser().resolve()
  destination = args.destination.expanduser().resolve()
  if source == destination:
    raise ValueError("Destination must differ from source")
  if not source.is_file():
    raise FileNotFoundError(source)

  model = onnx.load(source, load_external_data=False)
  source_metadata = {item.key: item.value for item in model.metadata_props}
  recipe_name, (articulation, action_scale_by_pattern) = select_gain_recipe(
    args.gain_recipe, source_metadata
  )
  input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
  output_shape = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]
  if input_shape not in ([1, 104], [1, 208]) or output_shape != [1, 31]:
    raise ValueError(f"Unexpected policy shapes: input={input_shape}, output={output_shape}")
  history_length = input_shape[1] // 104

  default_joint_pos = resolve_patterns(HOME_KEYFRAME.joint_pos, default=0.0)
  action_scale = resolve_patterns(action_scale_by_pattern)
  lower, upper = joint_limits()
  source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
  replace_metadata(
    model,
    {
      "action_scale": csv(action_scale),
      "default_joint_pos": csv(default_joint_pos),
      "gain_recipe": recipe_name,
      "joint_damping": csv(actuator_values(articulation, "damping")),
      "joint_effort_limit": csv(actuator_values(articulation, "effort_limit")),
      "joint_limit_lower": csv(lower),
      "joint_limit_upper": csv(upper),
      "joint_names": ",".join(X2_ULTRA_JOINT_NAMES),
      "metadata_contract": (
        "agibot_x2_velocity_v2" if history_length > 1
        else "agibot_x2_velocity_v1"
      ),
      "observation_dims": "3,3,3,2,31,31,31",
      "observation_history_length": str(history_length),
      "observation_names": (
        "base_ang_vel,projected_gravity,command,phase,joint_pos,joint_vel,actions"
      ),
      "phase_period": "0.6",
      "policy_dt": "0.02",
      "source_policy_sha256": source_sha256,
      "training_model": "agibot_x2_ultra_v1.3.0",
      "joint_stiffness": csv(actuator_values(articulation, "stiffness")),
    },
  )

  destination.parent.mkdir(parents=True, exist_ok=True)
  temporary = destination.with_suffix(destination.suffix + ".tmp")
  onnx.save(model, temporary)
  onnx.checker.check_model(onnx.load(temporary, load_external_data=False))
  shutil.move(temporary, destination)
  print(destination)


if __name__ == "__main__":
  main()
