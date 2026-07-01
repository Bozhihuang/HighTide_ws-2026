Here is the corrected version of search_pattern_node.py that includes the _det_callback() function from your original code, while keeping the ZED position (/mavros/zed/odom) and IMU heading (/mavros/imu/data) updates intact:

Python
#!/usr/bin/env python3
"""
Search Pattern Node — Systematic visual search when no pingers available.

Supports expanding square and lawnmower patterns. Maintains heading
via FOG. Stops when target class detected.
"""

import math
import time as pytime
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from hightide_interfaces.msg import ThrusterCommand, DetectionArray
from hightide_navigation import PIDController, normalize_angle, quaternion_to_yaw 
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class SearchPatternNode(Node):
    """Executes systematic search patterns to find competition objects using ZED position and IMU heading."""

    def __init__(self):
        super().__init__('search_pattern_node')
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.declare_parameter('pattern_type', 'expanding_square')
        self.declare_parameter('leg_length_m', 0.0)
        self.declare_parameter('leg_increment_m', 0.0)
        self.declare_parameter('search_speed', 0.0)
        self.declare_parameter('search_class', '')
        self.declare_parameter('timeout_sec', 120.0)
        self.declare_parameter('search_kp', 0.0)
        self.declare_parameter('search_ki', 0.0)
        self.declare_parameter('search_kd', 0.0)
        

        self.pattern = self.get_parameter('pattern_type').value
        self.leg_length = self.get_parameter('leg_length_m').value
        self.leg_increment = self.get_parameter('leg_increment_m').value
        self.search_speed = self.get_parameter('search_speed').value
        self.search_class = self.get_parameter('search_class').value
        self.timeout = self.get_parameter('timeout_sec').value

        self.current_odom = None
        self.current_heading = None
        self.target_found = False
        self.searching = False

        self.heading_pid = PIDController(
            self.get_parameter('search_kp').value,
            self.get_parameter('search_ki').value,
            self.get_parameter('search_kd').value
        )

        # Switched from /mavros/local_position/odom to /mavros/zed/odom for tracking position
        self.odom_sub = self.create_subscription(
            Odometry, '/mavros/zed/odom', self._odom_callback, sensor_qos)
            
        # Added IMU subscription to explicitly track heading from /mavros/imu/data
        self.imu_sub = self.create_subscription(
            Imu, '/mavros/imu/data', self._imu_callback, sensor_qos)
            
        self.det_sub = self.create_subscription(
            DetectionArray, '/hightide/tracked_targets', self._det_callback, 10)
        self.cmd_pub = self.create_publisher(ThrusterCommand, '/hightide/cmd_vel', 10)

        self.add_on_set_parameters_callback(self._param_callback)
        self.get_logger().info('Search Pattern Node started')

    def _param_callback(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == 'search_class':
                self.search_class = p.value
                self.target_found = False
        return SetParametersResult(successful=True)

    def _odom_callback(self, msg):
        self.current_odom = msg

    def _imu_callback(self, msg):
        self.current_heading = quaternion_to_yaw(msg.orientation)

    def _det_callback(self, msg: DetectionArray):
        if self.search_class and self.searching:
            for det in msg.detections:
                if det.class_name == self.search_class and det.confidence > 0.5:
                    self.target_found = True
                    self.get_logger().info(
                        f'Target found: {self.search_class} '
                        f'(conf={det.confidence:.2f})')
                    break

    def execute_expanding_square(self) -> bool:
        """
        Expanding square search pattern.
        Move forward leg_length, turn 90°, repeat with increasing leg.
        """
        self.searching = True
        self.target_found = False
        current_leg = self.leg_length
        
        # Sourced initial heading from dedicated IMU tracking variable
        heading = self.current_heading if self.current_heading is not None else 0.0

        start = pytime.time()
        turn_count = 0

        self.get_logger().info(
            f'Starting expanding square search for "{self.search_class}"')

        while not self.target_found and (pytime.time() - start) < self.timeout:
            # Surge forward for current leg
            if not self._surge_distance(current_leg, heading):
                break

            if self.target_found:
                break

            # Turn 90° right
            turn_count += 1
            heading = normalize_angle(heading - math.pi / 2)

            # Every 2 turns, increase leg length
            if turn_count % 2 == 0:
                current_leg += self.leg_increment

        self.searching = False
        self.cmd_pub.publish(ThrusterCommand())  # Stop
        return self.target_found

    def _surge_distance(self, distance_m: float, target_heading: float) -> bool:
        """Surge forward a specific distance while maintaining heading."""
        if self.current_odom is None or self.current_heading is None:
            return False

        start_x = self.current_odom.pose.pose.position.x
        start_y = self.current_odom.pose.pose.position.y
        self.heading_pid.reset()
        last_t = pytime.time()

        while not self.target_found:
            if self.current_odom is None or self.current_heading is None:
                rclpy.spin_once(self, timeout_sec=0.05)
                continue

            now_t = pytime.time()
            dt = now_t - last_t
            last_t = now_t

            dx = self.current_odom.pose.pose.position.x - start_x
            dy = self.current_odom.pose.pose.position.y - start_y
            traveled = math.sqrt(dx * dx + dy * dy)

            if traveled >= distance_m:
                return True

            # Replaced odometry orientation extraction with dedicated IMU feedback variable
            yaw_error = normalize_angle(target_heading - self.current_heading)
            yaw_cmd = self.heading_pid.compute(yaw_error, dt)

            cmd = ThrusterCommand()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.surge = self.search_speed
            cmd.yaw = yaw_cmd
            self.cmd_pub.publish(cmd)

            rclpy.spin_once(self, timeout_sec=0)
            pytime.sleep(0.05)

        return True  # Target found


def main(args=None):
    rclpy.init(args=args)
    node = SearchPatternNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()