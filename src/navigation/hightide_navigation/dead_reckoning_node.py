#!/usr/bin/env python3
"""
Dead Reckoning Node — Timed thrust maneuvers using FOG heading.

Tier 3 fallback: when visual tracking is lost, navigate by commanding
constant thrust for a calculated duration while the FOG maintains heading.
"""

import math
import time as pytime
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64
from hightide_interfaces.msg import ThrusterCommand
from hightide_navigation import PIDController, normalize_angle, quaternion_to_yaw


class DeadReckoningNode(Node):
    """Dead reckoning navigation using FOG heading + timed thrusts."""

    def __init__(self):
        super().__init__('dead_reckoning_node')

        self.declare_parameter('surge_speed', 0.4)
        self.declare_parameter('sway_speed', 0.3)
        self.declare_parameter('speed_to_mps', 0.5)
        self.declare_parameter('heading_kp', 2.0)
        self.declare_parameter('heading_ki', 0.05)
        self.declare_parameter('heading_kd', 0.5)

        self.surge_speed = self.get_parameter('surge_speed').value
        self.sway_speed = self.get_parameter('sway_speed').value
        self.speed_to_mps = self.get_parameter('speed_to_mps').value

        self.heading_pid = PIDController(
            self.get_parameter('heading_kp').value,
            self.get_parameter('heading_ki').value,
            self.get_parameter('heading_kd').value)

        self.current_heading = 0.0
        self.heading_received = False
        self.executing = False

        self.imu_sub = self.create_subscription(
            Imu, '/mavros/imu/data', self._imu_callback, 10)
        self.cmd_pub = self.create_publisher(ThrusterCommand, '/hightide/cmd_vel', 10)

        # Dead reckoning command subscriber:
        # Publish a Float64MultiArray [surge_m, sway_m] to trigger
        # Or use the programmatic API from behavior tree
        self.get_logger().info('Dead Reckoning Node started')

    def _imu_callback(self, msg: Imu):
        self.current_heading = quaternion_to_yaw(msg.orientation)
        self.heading_received = True

    def execute_dead_reckon(self, surge_m: float, sway_m: float,
                            target_heading: float = None):
        """
        Execute a dead reckoning maneuver.

        Args:
            surge_m: Distance to travel forward (meters, negative = backward)
            sway_m: Distance to travel laterally (meters, positive = right)
            target_heading: Heading to maintain (radians, None = current)
        """
        if not self.heading_received:
            self.get_logger().error('No heading data — cannot dead reckon')
            return False

        if target_heading is None:
            target_heading = self.current_heading

        self.executing = True
        self.heading_pid.reset()

        # Calculate durations
        surge_time = abs(surge_m) / (self.surge_speed * self.speed_to_mps) if surge_m != 0 else 0
        sway_time = abs(sway_m) / (self.sway_speed * self.speed_to_mps) if sway_m != 0 else 0

        surge_dir = 1.0 if surge_m >= 0 else -1.0
        sway_dir = 1.0 if sway_m >= 0 else -1.0

        self.get_logger().info(
            f'Dead reckon: surge={surge_m:.1f}m ({surge_time:.1f}s) '
            f'sway={sway_m:.1f}m ({sway_time:.1f}s) '
            f'heading={math.degrees(target_heading):.0f}°')

        # Execute surge phase
        if surge_time > 0:
            self._timed_thrust(
                surge=surge_dir * self.surge_speed, sway=0.0,
                target_heading=target_heading, duration=surge_time)

        # Execute sway phase
        if sway_time > 0:
            self._timed_thrust(
                surge=0.0, sway=sway_dir * self.sway_speed,
                target_heading=target_heading, duration=sway_time)

        # Stop
        stop = ThrusterCommand()
        self.cmd_pub.publish(stop)
        self.executing = False

        self.get_logger().info('Dead reckoning complete')
        return True

    def _timed_thrust(self, surge: float, sway: float,
                      target_heading: float, duration: float):
        """Command constant thrust while maintaining heading for duration."""
        start = pytime.time()
        rate = 20  # Hz
        period = 1.0 / rate
        last_t = pytime.time()

        while (pytime.time() - start) < duration:
            now = pytime.time()
            dt = now - last_t
            last_t = now

            # Heading correction
            yaw_error = normalize_angle(target_heading - self.current_heading)
            yaw_cmd = self.heading_pid.compute(yaw_error, dt)

            cmd = ThrusterCommand()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.surge = surge
            cmd.sway = sway
            cmd.yaw = yaw_cmd
            self.cmd_pub.publish(cmd)

            rclpy.spin_once(self, timeout_sec=0)
            pytime.sleep(max(0, period - (pytime.time() - now)))


def main(args=None):
    rclpy.init(args=args)
    node = DeadReckoningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
