#!/usr/bin/env python3
"""
Depth Controller Node — PID depth hold via throttle channel.

In Alt Hold mode, throttle channel (ch3) is a depth rate command:
  1500 = hold current depth
  >1500 = ascend (shallower)
  <1500 = descend (deeper)

This node takes a depth setpoint and PID-controls the throttle to reach it.
Publishes PWM value to /hightide/depth_pwm which rc_override_node picks up.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Int32
from mavros_msgs.msg import Altitude
from hightide_interfaces.srv import SetDepth
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class DepthControllerNode(Node):
    """PID depth controller outputting throttle PWM for Alt Hold mode."""

    def __init__(self):
        super().__init__('depth_controller_node')

        # PID parameters
        self.declare_parameter('kp', 0.0)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.0)
        self.declare_parameter('max_output', 400)
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('depth_tolerance', 0.1)
        self.declare_parameter('integral_max', 100.0)
        # Which MAVROS topic actually carries usable depth on YOUR rig. On ArduSub
        # the standard is global_position/rel_alt (baro/pressure-derived), but if it
        # reads zero/garbage underwater, switch to 'altitude' (needs the `altitude`
        # plugin whitelisted in mavros.yaml). Swap live via params.yaml — no rebuild.
        #   'rel_alt'  -> std_msgs/Float64      /mavros/global_position/rel_alt
        #   'altitude' -> mavros_msgs/Altitude  /mavros/altitude   (uses .relative)
        self.declare_parameter('depth_source', 'rel_alt')

        self.kp = self.get_parameter('kp').value
        self.ki = self.get_parameter('ki').value
        self.kd = self.get_parameter('kd').value
        self.max_output = self.get_parameter('max_output').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.depth_tolerance = self.get_parameter('depth_tolerance').value
        self.integral_max = self.get_parameter('integral_max').value
        self.depth_source = self.get_parameter('depth_source').value

        # State
        self.target_depth = None  # None = no target, hold current
        self.current_depth = 0.0
        self.depth_received = False
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None
        self.current_depth = 0.0
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Depth feedback — pick the source that actually works on this vehicle.
        if self.depth_source == 'altitude':
            self.depth_sub = self.create_subscription(
                Altitude, '/mavros/altitude',
                self._altitude_callback, sensor_qos)
            self.get_logger().info('Depth source: /mavros/altitude (.relative)')
        else:
            self.depth_sub = self.create_subscription(
                Float64, '/mavros/global_position/rel_alt',
                self._depth_callback, sensor_qos)
            self.get_logger().info('Depth source: /mavros/global_position/rel_alt')

        self.target_sub = self.create_subscription(
            Float64, '/hightide/target_depth',
            self._target_depth_callback, 10)

        # Service to set depth
        self.set_depth_srv = self.create_service(
            SetDepth, '/hightide/set_depth', self._set_depth_service)

        # Publisher — PWM for throttle channel
        self.pwm_pub = self.create_publisher(Int32, '/hightide/depth_pwm', 10)

        # Timer
        period = 1.0 / self.publish_rate
        self.timer = self.create_timer(period, self._control_loop)

        self.get_logger().info(
            f'Depth Controller started — Kp={self.kp} Ki={self.ki} Kd={self.kd}')

    def _depth_callback(self, msg: Float64):
        """
        Receive current depth from MAVROS.
        rel_alt is relative altitude: negative = below surface for subs.
        We convert to positive-down convention: depth_m = -rel_alt.
        """
        self.current_depth = -msg.data  # Convert to positive = deeper
        self.depth_received = True

    def _altitude_callback(self, msg: Altitude):
        """Depth from mavros_msgs/Altitude. `relative` is relative to the arming
        altitude (negative below surface for a sub), so depth_m = -relative."""
        self.current_depth = -msg.relative  # Convert to positive = deeper
        self.depth_received = True

    def _target_depth_callback(self, msg: Float64):
        """Receive target depth (positive = deeper in meters)."""
        self.target_depth = msg.data
        self.integral = 0.0  # Reset integral on new target
        self.prev_error = 0.0
        self.get_logger().info(f'New depth target: {self.target_depth:.2f} m')

    def _set_depth_service(self, request, response):
        """Service handler for setting target depth."""
        self.target_depth = request.target_depth_m
        self.integral = 0.0
        self.prev_error = 0.0
        response.success = True
        response.message = f'Target depth set to {self.target_depth:.2f} m'
        self.get_logger().info(response.message)
        return response

    def _control_loop(self):
        """PID control loop — compute throttle PWM from depth error."""
        now = self.get_clock().now()
        msg = Int32()

        # No target or no depth reading: output neutral (hold current depth)
        if self.target_depth is None or not self.depth_received:
            msg.data = 1500
            self.pwm_pub.publish(msg)
            self.last_time = now
            return

        # Compute dt
        if self.last_time is None:
            self.last_time = now
            msg.data = 1500
            self.pwm_pub.publish(msg)
            return

        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 0.0:
            return
        self.last_time = now

        # Error: positive = need to go deeper
        error = self.target_depth - self.current_depth

        # Within tolerance: hold
        if abs(error) < self.depth_tolerance:
            msg.data = 1500
            self.integral = 0.0
            self.pwm_pub.publish(msg)
            return

        # PID
        self.integral += error * dt
        self.integral = max(-self.integral_max, min(self.integral_max, self.integral))

        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        # Clamp output
        output = max(-self.max_output, min(self.max_output, output))

        # Convert to PWM:
        # In ArduSub Alt Hold: PWM < 1500 = descend (go deeper), PWM > 1500 = ascend
        # Our error: positive = need to go deeper → we want PWM < 1500
        # So: pwm = 1500 - output (positive error → lower PWM → descend)
        pwm = 1500 - int(output)
        pwm = max(1100, min(1900, pwm))

        msg.data = pwm
        self.pwm_pub.publish(msg)

        self.get_logger().debug(
            f'Depth: target={self.target_depth:.2f} current={self.current_depth:.2f} '
            f'error={error:.2f} pwm={pwm}')


def main(args=None):
    rclpy.init(args=args)
    node = DepthControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()