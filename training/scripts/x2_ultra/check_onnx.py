"""Validate the exported X2 locomotion ONNX and its deployment metadata."""

import argparse
import re
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

from src.assets.robots.agibot_x2.x2_ultra_constants import (
  X2_ULTRA_ACTION_SCALE,
  X2_ULTRA_ARTICULATION,
  X2_ULTRA_FACTORY_RECORDED_ACTION_SCALE,
  X2_ULTRA_FACTORY_RECORDED_ARTICULATION,
  X2_ULTRA_JOINT_NAMES,
  X2_ULTRA_KP090_KD080_ACTION_SCALE,
  X2_ULTRA_KP090_KD080_ARTICULATION,
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


def shape(value_info: onnx.ValueInfoProto) -> tuple[int, ...]:
  return tuple(dim.dim_value for dim in value_info.type.tensor_type.shape.dim)


def resolve_patterns(values: dict[str, float]) -> list[float]:
  resolved = []
  for joint_name in X2_ULTRA_JOINT_NAMES:
    matches = [value for pattern, value in values.items() if re.fullmatch(pattern, joint_name)]
    if len(matches) != 1:
      raise AssertionError(f"Expected one value for {joint_name}, got {matches}")
    resolved.append(matches[0])
  return resolved


def actuator_values(articulation, attribute: str) -> list[float]:
  by_pattern: dict[str, float] = {}
  for actuator in articulation.actuators:
    value = getattr(actuator, attribute)
    assert value is not None
    for pattern in actuator.target_names_expr:
      by_pattern[pattern] = float(value)
  return resolve_patterns(by_pattern)


def infer_gain_recipe(metadata: dict[str, str]) -> str:
  declared = metadata.get("gain_recipe")
  if declared in GAIN_RECIPES:
    return declared

  actual_stiffness = [
    float(value) for value in metadata["joint_stiffness"].split(",")
  ]
  for recipe_name, (articulation, _) in GAIN_RECIPES.items():
    try:
      np.testing.assert_allclose(
        actual_stiffness,
        actuator_values(articulation, "stiffness"),
        rtol=0.0,
        atol=1.0e-6,
      )
    except AssertionError:
      continue
    return recipe_name
  raise AssertionError("Unrecognized X2 gain recipe")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("onnx_file", type=Path)
  args = parser.parse_args()

  model = onnx.load(args.onnx_file, load_external_data=False)
  metadata = {item.key: item.value for item in model.metadata_props}

  assert len(model.graph.input) == 1
  assert model.graph.input[0].name == "obs"
  input_shape = shape(model.graph.input[0])
  assert input_shape in ((1, 104), (1, 208))
  history_length = input_shape[1] // 104
  assert len(model.graph.output) == 1
  assert model.graph.output[0].name == "actions"
  assert shape(model.graph.output[0]) == (1, 31)
  assert tuple(metadata["joint_names"].split(",")) == X2_ULTRA_JOINT_NAMES
  assert metadata["observation_names"].split(",") == [
    "base_ang_vel",
    "projected_gravity",
    "command",
    "phase",
    "joint_pos",
    "joint_vel",
    "actions",
  ]
  for key in ("action_scale", "default_joint_pos", "joint_stiffness", "joint_damping"):
    assert len(metadata[key].split(",")) == 31, key
  expected_contract = (
    "agibot_x2_velocity_v2" if history_length > 1
    else "agibot_x2_velocity_v1"
  )
  assert metadata["metadata_contract"] == expected_contract
  assert int(metadata["observation_history_length"]) == history_length
  assert float(metadata["policy_dt"]) == 0.02
  assert float(metadata["phase_period"]) == 0.6
  assert metadata["observation_dims"] == "3,3,3,2,31,31,31"
  recipe_name = infer_gain_recipe(metadata)
  articulation, action_scale = GAIN_RECIPES[recipe_name]
  actual_action_scale = [float(value) for value in metadata["action_scale"].split(",")]
  np.testing.assert_allclose(
    actual_action_scale,
    resolve_patterns(action_scale),
    rtol=0.0,
    # The direct training exporter serializes action scales from float32.
    atol=1.0e-8,
  )
  for metadata_key, actuator_attribute in (
    ("joint_stiffness", "stiffness"),
    ("joint_damping", "damping"),
  ):
    np.testing.assert_allclose(
      [float(value) for value in metadata[metadata_key].split(",")],
      actuator_values(articulation, actuator_attribute),
      rtol=0.0,
      atol=1.0e-6,
    )

  onnx.checker.check_model(model)
  evaluator = ReferenceEvaluator(model)
  for observation in (
    np.zeros(input_shape, dtype=np.float32),
    np.random.default_rng(42).normal(size=input_shape).astype(np.float32),
  ):
    actions = evaluator.run(None, {"obs": observation})[0]
    assert actions.shape == (1, 31)
    assert np.isfinite(actions).all()
  print("X2_ONNX_OK")
  print(f"file={args.onnx_file}")
  print(f"gain_recipe={recipe_name}")
  print(
    f"input=obs[1,{input_shape[1]}] history={history_length} "
    "output=actions[1,31] inference=finite"
  )


if __name__ == "__main__":
  main()
