"""Compile the X2 Ultra mjlab entity and verify the policy-facing contract."""

from mjlab.entity.entity import Entity

from src.assets.robots.agibot_x2.x2_ultra_constants import (
  X2_ULTRA_JOINT_NAMES,
  get_x2_ultra_robot_cfg,
)


def main() -> None:
  entity = Entity(get_x2_ultra_robot_cfg())
  model = entity.spec.compile()

  joint_names = tuple(model.joint(i).name for i in range(1, model.njnt))
  actuator_joint_names = tuple(
    model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)
  )
  sensor_names = tuple(model.sensor(i).name for i in range(model.nsensor))
  foot_geoms = tuple(
    model.geom(i).name
    for i in range(model.ngeom)
    if model.geom(i).name.startswith(("left_foot", "right_foot"))
  )

  assert model.nq == 38, model.nq
  assert model.nv == 37, model.nv
  assert model.nu == 31, model.nu
  assert joint_names == X2_ULTRA_JOINT_NAMES, joint_names
  assert set(actuator_joint_names) == set(X2_ULTRA_JOINT_NAMES)
  assert len(foot_geoms) == 24, foot_geoms
  assert {"imu_ang_vel", "imu_lin_vel", "imu_lin_acc", "root_angmom"} <= set(
    sensor_names
  )

  print("X2_MODEL_OK")
  print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")
  print("joint_order=" + ",".join(joint_names))
  print("actuator_order=" + ",".join(actuator_joint_names))


if __name__ == "__main__":
  main()
