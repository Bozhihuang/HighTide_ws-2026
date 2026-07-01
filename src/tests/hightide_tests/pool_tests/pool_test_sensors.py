#!/usr/bin/env python3
"""
Pool Test: Sensors (ZED + FOG) - FULL DIAGNOSTIC MODE

Live monitors the raw ZED odometry, acceleration, and raw FOG IMU.
Displays Position, Velocity, Acceleration, and converts heading to NED
(North=0, East=90, South=180, West=270).
"""

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import AccelWithCovarianceStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class SensorsPoolTest(Node):
    def __init__(self):
        super().__init__('pool_test_sensors')
        
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscriptions

        self.create_subscription(Odometry, '/mavros/zed/odom', self._zed_odom_cb, sensor_qos)
        self.create_subscription(AccelWithCovarianceStamped, '/accel/filtered', self._ekf_accel_cb, 10)
        self.create_subscription(Imu, '/mavros/imu/data', self._imu_cb, sensor_qos)
        
        # State Variables
        self.zed_x = 0.0
        self.zed_y = 0.0
        self.zed_z = 0.0
        
        self.zed_vx = 0.0
        self.zed_vy = 0.0
        self.zed_vz = 0.0
        self.zed_vyaw = 0.0
        
        self.ekf_ax = 0.0
        self.ekf_ay = 0.0
        self.ekf_az = 0.0
        
        self.fog_yaw_enu = 0.0
        
        self.timer = self.create_timer(1.0, self._print_status)
        self.get_logger().info('=== STARTING FULL SENSOR DIAGNOSTICS ===')
        self.get_logger().info('Push the sub to verify tracking. Check for lag or jitter.')
        self.get_logger().info('Press Ctrl+C to exit.')

    def _zed_odom_cb(self, msg):
        # Position from ZED
        self.zed_x = msg.pose.pose.position.x
        self.zed_y = msg.pose.pose.position.y
        self.zed_z = msg.pose.pose.position.z
        
        # Linear and Angular Velocity from ZED
        self.zed_vx = msg.twist.twist.linear.x
        self.zed_vy = msg.twist.twist.linear.y
        self.zed_vz = msg.twist.twist.linear.z
        self.zed_vyaw = msg.twist.twist.angular.z

    def _ekf_accel_cb(self, msg):
        # Acceleration from robot_localization
        self.ekf_ax = msg.accel.accel.linear.x
        self.ekf_ay = msg.accel.accel.linear.y
        self.ekf_az = msg.accel.accel.linear.z

    def _imu_cb(self, msg):
        # Sourcing heading entirely from the IMU topic
        self.fog_yaw_enu = self._quat_to_yaw(msg.orientation)

    def _quat_to_yaw(self, q):
        """Extracts standard ENU Yaw (radians) from a quaternion."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _enu_to_ned_deg(self, enu_yaw_rad):
        """Converts ENU radians to a standard NED Compass degree heading."""
        enu_deg = math.degrees(enu_yaw_rad)
        ned_deg = (90.0 - enu_deg) % 360.0
        return ned_deg

    def _print_status(self):
        # Convert the IMU/FOG heading to NED degrees
        fog_ned = self._enu_to_ned_deg(self.fog_yaw_enu)        
        print("\033[H\033[J", end="") # Clears the terminal screen for a clean live-dashboard
        print(f"m      : X={self.zed_x:+.4f} | Y={self.zed_y:+.4f} | Z={self.zed_z:+.4f}")
        print(f"m/s    : vX={self.zed_vx:+.4f} | vY={self.zed_vy:+.4f} | vZ={self.zed_vz:+.4f}")
        print(f"m/s^2  : aX={self.ekf_ax:+.4f} | aY={self.ekf_ay:+.4f} | aZ={self.ekf_az:+.4f}")
        print("---------------------------------------------------------")
        print(f"heading: {fog_ned:05.4f}°  |  turn rate: {math.degrees(self.zed_vyaw):+.4f}°/s")        
        

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