#!/usr/bin/env python3

import math
import time as pytime
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_srvs.srv import Trigger
from hightide_interfaces.msg import ThrusterCommand
from hightide_navigation import PIDController, normalize_angle, quaternion_to_yaw
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class YawControllerNode(Node):

    def __init__(self):
        super().__init__('yaw_controller_node')

        self.declare_parameter('yaw_kp', 0.0)
        self.declare_parameter('yaw_ki', 0.0)
        self.declare_parameter('yaw_kd', 0.0)
        self.declare_parameter('yaw_tolerance', 0.05)
        self.declare_parameter('spin_speed', 0.6)
        self.declare_parameter('spin_timeout', 30.0)

        self.yaw_pid = PIDController(
            self.get_parameter('yaw_kp').value,
            self.get_parameter('yaw_ki').value,
            self.get_parameter('yaw_kd').value)
        self.yaw_tol = self.get_parameter('yaw_tolerance').value
        self.spin_speed = self.get_parameter('spin_speed').value
        self.spin_timeout = self.get_parameter('spin_timeout').value

        self.current_heading = 0.0
        self.heading_received = False
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/mavros/imu/data', self._imu_callback, sensor_qos)
        self.cmd_pub = self.create_publisher(ThrusterCommand, '/hightide/cmd_vel', 10)

        # Style spin service (1 full 360° spin)
        self.spin_srv = self.create_service(
            Trigger, '/hightide/yaw_spin', self._yaw_spin_service)

        self.get_logger().info('Yaw Controller Node started')

    def _imu_callback(self, msg: Imu):
        self.current_heading = quaternion_to_yaw(msg.orientation)
        self.heading_received = True

    def rotate_to_heading(self, target_heading: float, timeout: float = 10.0) -> bool:
        """Rotate to a specific heading using PID. Returns True on success."""
        if not self.heading_received:
            return False

        self.yaw_pid.reset()
        start = pytime.time()
        last_t = pytime.time()
        at_target_since = None

        while (pytime.time() - start) < timeout:
            now_t = pytime.time()
            dt = now_t - last_t
            last_t = now_t

            error = normalize_angle(target_heading - self.current_heading)
            yaw_cmd = self.yaw_pid.compute(error, dt)

            cmd = ThrusterCommand()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.yaw = yaw_cmd
            self.cmd_pub.publish(cmd)

            if abs(error) < self.yaw_tol:
                if at_target_since is None:
                    at_target_since = now_t
                elif (now_t - at_target_since) > 0.5:
                    return True
            else:
                at_target_since = None

            rclpy.spin_once(self, timeout_sec=0)
            pytime.sleep(0.05)

        return False

    def execute_spin(self, num_spins: int = 1, clockwise: bool = True) -> bool:
        """Execute N full 360° spins. FOG ensures precision return to heading."""
        if not self.heading_received:
            return False

        original_heading = self.current_heading
        total_rotation_needed = num_spins * 2 * math.pi
        accumulated = 0.0
        prev_heading = self.current_heading
        direction = 1.0 if clockwise else -1.0

        self.get_logger().info(
            f'Executing {num_spins}x 360° spin '
            f'({"CW" if clockwise else "CCW"})')

        start = pytime.time()

        while accumulated < total_rotation_needed:
            if (pytime.time() - start) > self.spin_timeout:
                self.get_logger().warn('Spin timeout!')
                break

            cmd = ThrusterCommand()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.yaw = direction * self.spin_speed
            self.cmd_pub.publish(cmd)

            # Track accumulated rotation
            delta = normalize_angle(self.current_heading - prev_heading)
            accumulated += abs(delta)
            prev_heading = self.current_heading

            rclpy.spin_once(self, timeout_sec=0)
            pytime.sleep(0.05)

        # Return to original heading using PID
        self.get_logger().info('Spin complete — returning to original heading')
        success = self.rotate_to_heading(original_heading, timeout=5.0)

        # Stop
        self.cmd_pub.publish(ThrusterCommand())
        return success

    def _yaw_spin_service(self, request, response):
        """Service handler for a single 360° yaw spin."""
        success = self.execute_spin(num_spins=1, clockwise=True)
        response.success = success
        response.message = 'Yaw spin complete' if success else 'Yaw spin failed'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = YawControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()
