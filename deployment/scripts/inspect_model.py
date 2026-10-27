#!/usr/bin/env python3
"""Verify the public ONNX boundary on CPU without ROS or a robot."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", nargs="?", type=Path,
        default=Path(__file__).resolve().parents[2] / "models/history2_hybrid_v1/policy.onnx",
    )
    args = parser.parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    assert len(inputs) == len(outputs) == 1, "expected one input/output"
    assert inputs[0].name == "obs" and outputs[0].name == "actions"
    assert inputs[0].type == outputs[0].type == "tensor(float)"
    metadata = session.get_modelmeta().custom_metadata_map
    assert metadata["metadata_contract"] in ("agibot_x2_velocity_v1", "agibot_x2_velocity_v2")
    history = 2 if metadata["metadata_contract"].endswith("v2") else 1
    if history == 2:
        assert metadata["observation_history_length"] == "2"
    assert inputs[0].shape == [1, 104 * history]
    assert outputs[0].shape == [1, 31]
    assert metadata["observation_names"] == "base_ang_vel,projected_gravity,command,phase,joint_pos,joint_vel,actions"
    assert metadata["observation_dims"] == "3,3,3,2,31,31,31"
    names = metadata["joint_names"].split(",")
    assert len(names) == len(set(names)) == 31
    for key in ("default_joint_pos", "action_scale", "joint_stiffness", "joint_damping", "joint_limit_lower", "joint_limit_upper"):
        values = np.asarray(metadata[key].split(","), dtype=np.float64)
        assert values.shape == (31,) and np.isfinite(values).all(), key
    assert float(metadata["policy_dt"]) > 0 and float(metadata["phase_period"]) > 0
    # An upright, zero-command observation. Joint positions are default-relative.
    observation = np.zeros(inputs[0].shape, dtype=np.float32)
    if history == 1:
        observation[0, 5] = -1.0
    else:
        observation[0, 8] = observation[0, 11] = -1.0
    actions = session.run(["actions"], {"obs": observation})[0]
    assert actions.shape == (1, 31) and np.isfinite(actions).all()
    print(f"sha256={hashlib.sha256(args.model.read_bytes()).hexdigest()}")
    print(f"contract={metadata['metadata_contract']} observation={inputs[0].shape} action={outputs[0].shape}")
    print(f"policy_dt={metadata['policy_dt']} phase_period={metadata['phase_period']} history={history}")
    print(f"upright_probe_action_min={actions.min():.6f} max={actions.max():.6f}")
    print("MODEL_CONTRACT_OK (inference smoke test, not a locomotion acceptance test)")


if __name__ == "__main__":
    main()
