#!/usr/bin/env python3
"""
Dead Reckoning Node — Timed thrust maneuvers using FOG heading.

Tier 3 fallback: when visual tracking is lost, navigate by commanding
constant thrust for a calculated duration while the FOG maintains heading.
"""

import math
import rclpy
from rclpy.node import Node
from enum import Enum
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64
from hightide_interfaces.msg import ThrusterCommand
from hightide_navigation import PIDController, normalize_angle, quaternion_to_yaw
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class State(Enum):
    IDLE = 0
    SURGE_PHASE = 1
    SWAY_PHASE = 2


class DeadReckoningNode(Node):
    """Dead reckoning navigation using an asynchronous state machine timer."""

    def __init__(self):
        super().__init__('dead_reckoning_node')

        # 1. Parameter Declarations (with 0.0 safety baselines)
        self.declare_parameter('surge_speed', 0.0)
        self.declare_parameter('sway_speed', 0.0)
        self.declare_parameter('speed_to_mps', 0.0)
        self.declare_parameter('heading_kp', 0.0)
        self.declare_parameter('heading_ki', 0.0)
        self.declare_parameter('heading_kd', 0.0)

        # 2. Parameter Retreival (Sourced on startup from params.yaml)
        self.surge_speed = self.get_parameter('surge_speed').value
        self.sway_speed = self.get_parameter('sway_speed').value
        self.speed_to_mps = self.get_parameter('speed_to_mps').value

        self.heading_pid = PIDController(
            self.get_parameter('heading_kp').value,
            self.get_parameter('heading_ki').value,
            self.get_parameter('heading_kd').value)

        # State Machine Variables
        self.state = State.IDLE
        self.current_heading = 0.0
        self.heading_received = False
        
        # Phase tracking variables
        self.target_heading = 0.0
        self.phase_start_time = None
        self.surge_duration = 0.0
        self.sway_duration = 0.0
        self.surge_commanded_thrust = 0.0
        self.sway_commanded_thrust = 0.0
        
        self.last_time = self.get_clock().now()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/mavros/imu/data', self._imu_callback, sensor_qos)
        self.cmd_pub = self.create_publisher(ThrusterCommand, '/hightide/cmd_vel', 10)

        self.control_timer = self.create_timer(1.0 / 20.0, self._control_loop)
        
        self.get_logger().info('Asynchronous Dead Reckoning Node initialized')

    def _imu_callback(self, msg: Imu):
        self.current_heading = quaternion_to_yaw(msg.orientation)
        self.heading_received = True

    def execute_dead_reckon(self, surge_m: float, sway_m: float, target_heading: float = None):
        """
        Public programmatic API called to trigger a dead reckoning maneuver.
        Completely non-blocking — configures state variables and yields immediately.
        """
        if self.state != State.IDLE:
            self.get_logger().warn('Maneuver rejected: Dead reckoning node is already busy.')
            return False

        if not self.heading_received:
            self.get_logger().error('No heading data — cannot dead reckon')
            return False


        self.target_heading = target_heading if target_heading is not None else self.current_heading
        self.heading_pid.reset()
        
        self.surge_duration = abs(surge_m) / (self.surge_speed * self.speed_to_mps) if surge_m != 0 else 0.0
        self.sway_duration = abs(sway_m) / (self.sway_speed * self.speed_to_mps) if sway_m != 0 else 0.0
        
        surge_dir = 1.0 if surge_m >= 0 else -1.0
        sway_dir = 1.0 if sway_m >= 0 else -1.0
        self.surge_commanded_thrust = surge_dir * self.surge_speed
        self.sway_commanded_thrust = sway_dir * self.sway_speed

        self.get_logger().info(
            f'Triggering state run: surge={surge_m:.1f}m ({self.surge_duration:.1f}s) '
            f'sway={sway_m:.1f}m ({self.sway_duration:.1f}s) '
            f'target={math.degrees(self.target_heading):.1f}°')

        self.phase_start_time = self.get_clock().now()
        
        if self.surge_duration > 0:
            self.state = State.SURGE_PHASE
        elif self.sway_duration > 0:
            self.state = State.SWAY_PHASE
        else:
            self.state = State.IDLE
            
        return True

    def _control_loop(self):
        """Cyclic non-blocking state machine loop. Executed purely by the global executor loop."""
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        if self.state == State.IDLE:
            return

        if dt <= 0:
            return

        # Compute elapsed milestone timeline metric
        elapsed = (now - self.phase_start_time).nanoseconds / 1e9
        
        cmd = ThrusterCommand()
        cmd.header.stamp = now.to_msg()

        # Compute shared heading orientation balance vector using D-filter layer
        yaw_error = normalize_angle(self.target_heading - self.current_heading)
        yaw_cmd = self.heading_pid.compute(yaw_error, dt)
        cmd.yaw = yaw_cmd

        if self.state == State.SURGE_PHASE:
            if elapsed < self.surge_duration:
                cmd.surge = self.surge_commanded_thrust
                cmd.sway = 0.0
                self.cmd_pub.publish(cmd)
            else:
                self.get_logger().info('Surge phase complete. Transitioning to Sway phase.')
                self.phase_start_time = now
                if self.sway_duration > 0:
                    self.state = State.SWAY_PHASE
                else:
                    self._stop_vehicle()

        elif self.state == State.SWAY_PHASE:
            if elapsed < self.sway_duration:
                cmd.surge = 0.0
                cmd.sway = self.sway_commanded_thrust
                self.cmd_pub.publish(cmd)
            else:
                # Maneuver completed entirely
                self.get_logger().info('Sway phase complete. Stopping dead reckoning.')
                self._stop_vehicle()

    def _stop_vehicle(self):
        """Puts vehicle state machine back to idle and stops thruster output."""
        stop = ThrusterCommand()
        stop.header.stamp = self.get_clock().now().to_msg()
        self.cmd_pub.publish(stop)
        self.state = State.IDLE
        self.get_logger().info('Dead reckoning sequence cleanly ended.')


def main(args=None):
    rclpy.init(args=args)
    node = DeadReckoningNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main() 