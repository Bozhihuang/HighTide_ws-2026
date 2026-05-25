#!/usr/bin/env python3
"""
Vision Servo Node — The "Crab Walk" controller.

Keeps a target detection centered in the camera frame by commanding
sway (lateral) and surge thrusters. Heading is NEVER changed — the FOG
holds it locked. This is the core of the strafing strategy.

Error computation:
  lateral_error = (target_center_x / image_width) - 0.5  → sway PID
  vertical_error = (target_center_y / image_height) - 0.5 → depth adjustment
  range_error = target_depth - desired_distance → surge PID
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from hightide_interfaces.msg import ThrusterCommand, DetectionArray
from hightide_navigation import PIDController


class VisionServoNode(Node):
    """Vision servoing via lateral strafing — keeps target centered."""

    def __init__(self):
        super().__init__('vision_servo_node')

        # Parameters
        self.declare_parameter('target_class', '')
        self.declare_parameter('target_x_normalized', 0.5)
        self.declare_parameter('target_y_normalized', 0.5)
        self.declare_parameter('approach_distance_m', 1.0)
        self.declare_parameter('enabled', False)
        self.declare_parameter('lateral_kp', 1.5)
        self.declare_parameter('lateral_ki', 0.05)
        self.declare_parameter('lateral_kd', 0.3)
        self.declare_parameter('vertical_kp', 1.0)
        self.declare_parameter('vertical_ki', 0.05)
        self.declare_parameter('vertical_kd', 0.2)
        self.declare_parameter('range_kp', 0.8)
        self.declare_parameter('range_ki', 0.02)
        self.declare_parameter('range_kd', 0.15)
        self.declare_parameter('publish_rate', 20.0)

        self.target_class = self.get_parameter('target_class').value
        self.target_x = self.get_parameter('target_x_normalized').value
        self.target_y = self.get_parameter('target_y_normalized').value
        self.approach_dist = self.get_parameter('approach_distance_m').value
        self.enabled = self.get_parameter('enabled').value

        # PID controllers
        self.lateral_pid = PIDController(
            self.get_parameter('lateral_kp').value,
            self.get_parameter('lateral_ki').value,
            self.get_parameter('lateral_kd').value)
        self.vertical_pid = PIDController(
            self.get_parameter('vertical_kp').value,
            self.get_parameter('vertical_ki').value,
            self.get_parameter('vertical_kd').value)
        self.range_pid = PIDController(
            self.get_parameter('range_kp').value,
            self.get_parameter('range_ki').value,
            self.get_parameter('range_kd').value)

        self.last_time = self.get_clock().now()
        self.latest_detections = None

        # Subscribers
        self.det_sub = self.create_subscription(
            DetectionArray, '/hightide/tracked_targets',
            self._detection_callback, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(ThrusterCommand, '/hightide/cmd_vel', 10)
        self.depth_adj_pub = self.create_publisher(
            Float64, '/hightide/depth_adjustment', 10)

        # Timer
        period = 1.0 / self.get_parameter('publish_rate').value
        self.timer = self.create_timer(period, self._control_loop)

        # Dynamic parameter reconfigure
        self.add_on_set_parameters_callback(self._param_callback)

        self.get_logger().info('Vision Servo Node started (crab walk mode)')

    def _param_callback(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for param in params:
            if param.name == 'target_class':
                self.target_class = param.value
                self.lateral_pid.reset()
                self.range_pid.reset()
            elif param.name == 'enabled':
                self.enabled = param.value
                if not self.enabled:
                    self.lateral_pid.reset()
                    self.range_pid.reset()
            elif param.name == 'approach_distance_m':
                self.approach_dist = param.value
        return SetParametersResult(successful=True)

    def _detection_callback(self, msg: DetectionArray):
        self.latest_detections = msg

    def _find_target(self):
        """Find the best matching detection for our target class."""
        if not self.latest_detections or not self.target_class:
            return None

        best = None
        best_conf = 0.0
        for det in self.latest_detections.detections:
            if det.class_name == self.target_class and det.confidence > best_conf:
                best = det
                best_conf = det.confidence
        return best

    def _control_loop(self):
        """Compute servo commands based on target position in frame."""
        if not self.enabled:
            return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        if dt <= 0:
            return

        target = self._find_target()
        cmd = ThrusterCommand()
        cmd.header.stamp = now.to_msg()

        if target is None:
            # Target lost — hold position (zero commands)
            self.cmd_pub.publish(cmd)
            return

        img_w = self.latest_detections.image_width or 1280
        img_h = self.latest_detections.image_height or 720

        # Lateral error: how far target center_x is from desired position
        # Positive error = target is to the right → strafe right (positive sway)
        lateral_error = (target.center_x / img_w) - self.target_x
        cmd.sway = self.lateral_pid.compute(lateral_error, dt)

        # Vertical error: target center_y offset from desired
        # Positive error = target is below center → adjust depth deeper
        vertical_error = (target.center_y / img_h) - self.target_y
        depth_adj = self.vertical_pid.compute(vertical_error, dt)
        depth_msg = Float64()
        depth_msg.data = depth_adj * 0.1  # Scale to meters adjustment
        self.depth_adj_pub.publish(depth_msg)

        # Range error: how far we are from desired approach distance
        # Positive error = too far away → surge forward
        if target.depth_m > 0:
            range_error = target.depth_m - self.approach_dist
            cmd.surge = self.range_pid.compute(range_error, dt)
        else:
            cmd.surge = 0.0

        # YAW IS ALWAYS ZERO — FOG holds heading
        cmd.yaw = 0.0
        cmd.pitch = 0.0
        cmd.roll = 0.0

        self.cmd_pub.publish(cmd)

        self.get_logger().debug(
            f'Servo: lat_err={lateral_error:.3f} sway={cmd.sway:.2f} '
            f'range={target.depth_m:.1f}m surge={cmd.surge:.2f}')


def main(args=None):
    rclpy.init(args=args)
    node = VisionServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
