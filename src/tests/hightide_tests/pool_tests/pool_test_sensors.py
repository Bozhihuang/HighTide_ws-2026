#!/usr/bin/env python3
"""
Pool Test: Sensors (ZED + FOG)

Live monitors the EKF filtered odometry, raw ZED VIO, and FOG IMU.
Use this to manually push the sub around the pool and verify coordinate
tracking and heading stability.
"""

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class SensorsPoolTest(Node):
    def __init__(self):
        super().__init__('pool_test_sensors')
        sensor_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=10
        )
        self.create_subscription(Odometry, '/odometry/filtered', self._ekf_cb, 10)
        self.create_subscription(Imu, '/mavros/imu/data', self._imu_cb, sensor_qos)
        self.ekf_x = 0.0
        self.ekf_y = 0.0
        self.ekf_yaw = 0.0
        
        self.fog_yaw = 0.0
        
        self.timer = self.create_timer(1.0, self._print_status)
        self.get_logger().info('=== STARTING SENSORS POOL TEST ===')
        self.get_logger().info('Move the sub around the pool to verify tracking.')
        self.get_logger().info('Press Ctrl+C to exit.')
        

    def _ekf_cb(self, msg):
        self.ekf_x = msg.pose.pose.position.x
        self.ekf_y = msg.pose.pose.position.y
        self.ekf_yaw = self._quat_to_yaw(msg.pose.pose.orientation)

    def _imu_cb(self, msg):
        self.fog_yaw = self._quat_to_yaw(msg.orientation)

    def _quat_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _print_status(self):
        print(f"\n--- SENSOR STATUS ---")
        print(f"EKF Position : X = {self.ekf_x:+.2f}m | Y = {self.ekf_y:+.2f}m")
        print(f"EKF Heading  : {math.degrees(self.ekf_yaw):+.1f}°")
        print(f"FOG Heading  : {math.degrees(self.fog_yaw):+.1f}°")
        
        # In a perfect setup with FOG overriding ZED heading, EKF and FOG yaw should be nearly identical.
        diff = abs(math.degrees(self.ekf_yaw - self.fog_yaw))
        if diff > 1.0:
            print(f"⚠️ WARNING: EKF/FOG heading mismatch by {diff:.1f}°")


def main(args=None):
    rclpy.init(args=args)
    node = SensorsPoolTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
