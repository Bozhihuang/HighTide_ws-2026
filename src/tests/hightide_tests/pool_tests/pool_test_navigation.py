#!/usr/bin/env python3
"""
Pool Test: Navigation (Waypoint)

Tests the waypoint navigator action server by commanding the sub to
move exactly 1 meter forward (surge) or 1 meter right (sway) from its
current position.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_srvs.srv import SetBool, Trigger
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from hightide_interfaces.action import NavigateToWaypoint
from hightide_navigation import quaternion_to_yaw


class NavigationPoolTest(Node):
    def __init__(self):
        super().__init__('pool_test_navigation')
        
        self.arm_cli = self.create_client(SetBool, '/hightide/arm')
        self.alt_hold_cli = self.create_client(Trigger, '/hightide/set_alt_hold')
        self.nav_client = ActionClient(self, NavigateToWaypoint, '/hightide/navigate_to_waypoint')
        
        self.create_subscription(Odometry, '/mavros/zed/odom', self._odom_cb, 10)
        
        # Added IMU subscriber to isolate heading tracking to /mavros/imu/data
        self.create_subscription(Imu, '/mavros/imu/data', self._imu_cb, 10)
        
        self.current_pose = None
        self.current_heading = None
        
        while not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for waypoint navigator action server...')

    def _odom_cb(self, msg):
        self.current_pose = msg.pose.pose

    def _imu_cb(self, msg):
        self.current_heading = quaternion_to_yaw(msg.orientation)

    def run_tests(self):
        self.get_logger().info('=== STARTING NAVIGATION POOL TEST ===')
        
        # Wait for both odometry (position) and IMU (heading) data
        while self.current_pose is None or self.current_heading is None:
            self.get_logger().info('Waiting for sensor streams (ZED odom and IMU)...')
            rclpy.spin_once(self, timeout_sec=1.0)
            
        input("Ensure sub is in water. Press Enter to ARM and set ALT HOLD...")
        req = SetBool.Request()
        req.data = True
        self.arm_cli.call_async(req)
        self.alt_hold_cli.call_async(Trigger.Request())
        self.get_logger().info('Sub is ARMED and in ALT HOLD.')
        
        input("Press Enter to move 1 METER FORWARD (Surge)...")
        self.send_waypoint(1.0, 0.0)
        
        input("Press Enter to move 1 METER RIGHT (Sway)...")
        self.send_waypoint(0.0, -1.0)
        
        self.get_logger().info('DISARMING...')
        req.data = False
        self.arm_cli.call_async(req)
        self.get_logger().info('=== NAVIGATION POOL TEST COMPLETE ===')

    def send_waypoint(self, relative_x, relative_y):
        if self.current_pose is None:
            return
            
        goal = NavigateToWaypoint.Goal()

        # The action Goal is a geometry_msgs/PoseStamped target_pose (+ tolerances /
        # timeout), NOT flat target_x/target_y fields. Build the pose explicitly.
        target = PoseStamped()
        target.header.frame_id = 'odom'
        target.header.stamp = self.get_clock().now().to_msg()

        # Transform relative to global (simplified, assumes yaw=0 for basic pool test)
        # In a real scenario, use TF2 or calculate based on the explicit self.current_heading
        target.pose.position.x = self.current_pose.position.x + relative_x
        target.pose.position.y = self.current_pose.position.y + relative_y
        target.pose.position.z = self.current_pose.position.z

        # Hold the CURRENT heading. The navigator derives its goal yaw from this
        # orientation; a default (all-zero) quaternion would command absolute yaw 0
        # and spin the sub. Encode current_heading as a yaw-only quaternion.
        half = (self.current_heading or 0.0) / 2.0
        target.pose.orientation.z = math.sin(half)
        target.pose.orientation.w = math.cos(half)

        goal.target_pose = target
        goal.position_tolerance = 0.2   # meters
        goal.yaw_tolerance = 0.15       # radians (~8.6 deg)
        goal.timeout_sec = 60.0         # abort if it can't reach the point in 60 s

        self.get_logger().info(
            f'Sending waypoint: X={target.pose.position.x:.2f}, '
            f'Y={target.pose.position.y:.2f}')
        
        future = self.nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by navigator')
            return
            
        self.get_logger().info('Goal accepted. Navigating...')
        
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        res = result_future.result().result
        self.get_logger().info(f'Navigation finished. Success: {res.success}')


def main(args=None):
    rclpy.init(args=args)
    node = NavigationPoolTest()
    try:
        node.run_tests()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()