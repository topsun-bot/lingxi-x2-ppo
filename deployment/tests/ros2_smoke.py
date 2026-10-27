#!/usr/bin/env python3
"""Localhost-only synthetic HAL integration; never run against robot feedback."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parents[2] / "models/history2_hybrid_v1/policy.onnx")
    parser.add_argument("--domain", type=int, default=191)
    args = parser.parse_args()
    if not 100 <= args.domain <= 200:
        parser.error("use an isolated test domain in 100..200")
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    os.environ["ROS_DOMAIN_ID"] = str(args.domain)
    import onnxruntime as ort
    import rclpy
    from aimdk_msgs.msg import JointCommandArray, JointState, JointStateArray
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu, Joy
    from std_srvs.srv import Trigger

    metadata = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
    names = metadata["joint_names"].split(",")
    defaults = [float(value) for value in metadata["default_joint_pos"].split(",")]
    rclpy.init()
    node = rclpy.create_node("x2_synthetic_hal_test")
    qos = qos_profile_sensor_data
    areas = ("leg", "waist", "arm", "head")
    states = {area: JointStateArray() for area in areas}
    for name, position in zip(names, defaults):
        area = "waist" if name.startswith("waist_") else "head" if name.startswith("head_") else (
            "leg" if any(token in name for token in ("hip_", "knee_", "ankle_")) else "arm"
        )
        joint = JointState()
        joint.name, joint.position = name, position
        states[area].joints.append(joint)
    publishers = {area: node.create_publisher(JointStateArray, f"/aima/hal/joint/{area}/state", qos) for area in areas}
    imu_publishers = [node.create_publisher(Imu, "/aima/hal/imu/torso/state", qos)]
    joy_publisher = node.create_publisher(Joy, "/joy", qos)
    imu = Imu()
    imu.header.frame_id = "pelvis"
    imu.orientation.w = 1.0
    received = []
    subscription = node.create_subscription(JointCommandArray, "/aima/hal/joint/leg/command", received.append, qos)
    estop = node.create_client(Trigger, "/x2/policy/estop")
    deployment = Path(__file__).resolve().parents[1]

    def pump(seconds: float, feed: bool = True) -> None:
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            if feed:
                for area in areas:
                    publishers[area].publish(states[area])
                for publisher in imu_publishers:
                    publisher.publish(imu)
            rclpy.spin_once(node, timeout_sec=0.005)
            time.sleep(0.005)

    def start(profile: str):
        profile = Path(profile)
        if not profile.is_absolute():
            profile = deployment / "config" / profile
        return subprocess.Popen([
            str(args.controller), "--config", str(profile),
            "--model", str(args.model),
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def stop(process):
        process.terminate()
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
        if process.returncode not in (0, -15):
            raise AssertionError(output)
        return output

    process = None
    try:
        process = start("dry_run.yaml")
        pump(3)
        assert process.poll() is None
        for area in areas:
            assert node.count_publishers(f"/aima/hal/joint/{area}/command") == 0
        assert estop.wait_for_service(timeout_sec=2)
        dry_estop = estop.call_async(Trigger.Request())
        until = time.monotonic() + 3
        while not dry_estop.done() and time.monotonic() < until:
            pump(0.02)
        assert dry_estop.done() and dry_estop.result().success
        pump(0.1)
        assert not received
        output = stop(process)
        process = None
        assert "DRY RUN: no motor command publishers exist" in output
        assert "Policy health:" in output, output
        assert not received

        process = start("simulation.yaml")
        pump(3)
        assert received and any(joint.stiffness > 0 for joint in received[-1].joints)
        assert estop.wait_for_service(timeout_sec=2)
        future = estop.call_async(Trigger.Request())
        until = time.monotonic() + 3
        while not future.done() and time.monotonic() < until:
            pump(0.02)
        assert future.done() and future.result().success
        pump(0.25)
        assert all(joint.stiffness == 0 for joint in received[-1].joints)
        # A fresh RL button edge must not undo a latched software stop.
        joy = Joy()
        joy.axes, joy.buttons = [0.0] * 8, [0] * 13
        joy_publisher.publish(joy)
        pump(0.1)
        joy.buttons[3] = joy.buttons[4] = 1
        joy_publisher.publish(joy)
        pump(0.25)
        assert all(joint.stiffness == 0 for joint in received[-1].joints)
        stop(process)
        process = None
        pump(0.1)
        received.clear()
        with tempfile.TemporaryDirectory(prefix="x2-joy-edge-test-") as directory:
            profile = Path(directory) / "disarmed.yaml"
            profile.write_text((deployment / "config/simulation.yaml").read_text().replace(
                "start_mode: rl", "start_mode: disarmed").replace(
                "require_deadman: false", "require_deadman: true"))
            process = start(str(profile))
            pump(0.75)
            joy.buttons[3] = joy.buttons[4] = 1
            for _ in range(20):
                joy_publisher.publish(joy)
                pump(0.01)
            assert not received, "held button at first observation unexpectedly armed output"
            joy.buttons[3] = 0
            for _ in range(5):
                joy_publisher.publish(joy)
                pump(0.01)
            joy.buttons[3] = 1
            for _ in range(15):
                joy_publisher.publish(joy)
                pump(0.01)
            assert received and any(joint.stiffness > 0 for joint in received[-1].joints)
            stop(process)
            process = None
        print("ROS2_SMOKE_OK: dry-run has no motor publishers; E-stop latches; held startup button does not arm; fresh edge arms")
    finally:
        if process is not None:
            stop(process)
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
