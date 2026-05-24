#!/usr/bin/env python3
"""
Pool Test: Navigation (Waypoint)

Tests the waypoint navigator action server by commanding the sub to
move exactly 1 meter forward (surge) or 1 meter right (sway) from its
current position.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_srvs.srv import SetBool, Trigger
from nav_msgs.msg import Odometry
from HighTide_interfaces.action import NavigateToWaypoint


class NavigationPoolTest(Node):
    def __init__(self):
        super().__init__('pool_test_navigation')
        
        self.arm_cli = self.create_client(SetBool, '/HighTide/arm')
        self.alt_hold_cli = self.create_client(Trigger, '/HighTide/set_alt_hold')
        self.nav_client = ActionClient(self, NavigateToWaypoint, '/HighTide/navigate_to_waypoint')
        
        self.create_subscription(Odometry, '/HighTide/odometry/filtered', self._odom_cb, 10)
        self.current_pose = None
        
        while not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for waypoint navigator action server...')

    def _odom_cb(self, msg):
        self.current_pose = msg.pose.pose

    def run_tests(self):
        self.get_logger().info('=== STARTING NAVIGATION POOL TEST ===')
        
        # Wait for odometry
        while self.current_pose is None:
            self.get_logger().info('Waiting for odometry...')
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
        # Transform relative to global (simplified, assumes yaw=0 for basic pool test)
        # In a real scenario, use TF2 or calculate based on current yaw
        goal.target_x = self.current_pose.position.x + relative_x
        goal.target_y = self.current_pose.position.y + relative_y
        
        self.get_logger().info(f'Sending waypoint: X={goal.target_x:.2f}, Y={goal.target_y:.2f}')
        
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
