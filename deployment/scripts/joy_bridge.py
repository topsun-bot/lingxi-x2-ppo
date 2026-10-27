#!/usr/bin/env python3
"""Forward physical Linux joydev axes/buttons to ROS 2 without synthetic arming."""
import argparse
import os
import select
import struct
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/input/js0")
    parser.add_argument("--topic", default="/joy")
    parser.add_argument("--rate", type=float, default=50)
    parser.add_argument("--deadzone", type=float, default=0.05)
    parser.add_argument("--axes", type=int, default=8)
    parser.add_argument("--buttons", type=int, default=13)
    args = parser.parse_args()
    if not 10 <= args.rate <= 200 or not 0 <= args.deadzone < 0.5:
        parser.error("rate must be 10..200 Hz and deadzone 0..<0.5")
    if not 1 <= args.axes <= 64 or not 1 <= args.buttons <= 64:
        parser.error("axes and buttons must be 1..64")
    rclpy.init()
    node = rclpy.create_node("lingxi_x2_joy_bridge")
    publisher = node.create_publisher(Joy, args.topic, qos_profile_sensor_data)
    axes, buttons = [0.0] * args.axes, [0] * args.buttons
    device = os.open(args.device, os.O_RDONLY | os.O_NONBLOCK)
    deadline = time.monotonic()
    try:
        while rclpy.ok():
            while select.select([device], [], [], 0)[0]:
                data = os.read(device, 8 * 64)
                if not data:
                    raise RuntimeError("joystick disconnected")
                for offset in range(0, len(data) - 7, 8):
                    _, value, event, index = struct.unpack("IhBB", data[offset:offset + 8])
                    event &= 0x7F
                    if event == 1 and index < len(buttons):
                        buttons[index] = int(bool(value))
                    elif event == 2 and index < len(axes):
                        normalized = max(-1.0, min(1.0, value / 32767))
                        magnitude = abs(normalized)
                        axes[index] = 0.0 if magnitude < args.deadzone else (
                            (1 if normalized >= 0 else -1) * (magnitude - args.deadzone) / (1 - args.deadzone)
                        )
            message = Joy()
            message.header.stamp = node.get_clock().now().to_msg()
            message.header.frame_id = "physical_joydev"
            message.axes, message.buttons = axes, buttons
            publisher.publish(message)
            deadline += 1 / args.rate
            time.sleep(max(0, deadline - time.monotonic()))
            if time.monotonic() - deadline > 1 / args.rate:
                deadline = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        os.close(device)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
