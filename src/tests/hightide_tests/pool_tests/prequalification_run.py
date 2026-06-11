#!/usr/bin/env python3
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_srvs.srv import SetBool, Trigger
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from hightide_interfaces.action import NavigateToWaypoint

class PrequalificationRun(Node):
    def __init__(self):
        super().__init__('prequalification_run')
        
        self.arm_cli = self.create_client(SetBool, '/hightide/arm')
        self.alt_hold_cli = self.create_client(Trigger, '/hightide/set_alt_hold')
        self.depth_pub = self.create_publisher(Float64, '/hightide/target_depth', 10)
        self.nav_client = ActionClient(self, NavigateToWaypoint, '/hightide/navigate_to_waypoint')
        
        self.create_subscription(Odometry, '/odometry/filtered', self._odom_cb, 10)
        self.current_pose = None
        
        self.get_logger().info('Connecting to flight systems...')
        self.arm_cli.wait_for_service()
        self.alt_hold_cli.wait_for_service()
        self.nav_client.wait_for_server()

    def _odom_cb(self, msg):
        self.current_pose = msg.pose.pose

    def _quaternion_to_yaw(self, q):
        """Converts spatial quaternion to Euler yaw angle."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def execute_mission(self):
        self.get_logger().info('=== STARTING PREQUALIFICATION SEQUENCE ===')
        
        while self.current_pose is None:
            self.get_logger().info('Waiting for EKF odometry lock...')
            rclpy.spin_once(self, timeout_sec=1.0)
            
        input("Position sub facing the gate. Press Enter to EXECUTE RUN...")
        
        self.get_logger().info('Arming vehicle...')
        req = SetBool.Request()
        req.data = True
        self.arm_cli.call_async(req)
        time.sleep(1.0)
        
        self.get_logger().info('Setting Alt Hold mode...')
        self.alt_hold_cli.call_async(Trigger.Request())
        time.sleep(1.0)
        
        self.get_logger().info('Diving to 1.0 meter...')
        depth_msg = Float64()
        depth_msg.data = 1.0
        self.depth_pub.publish(depth_msg)
        
        time.sleep(5.0) 
        

        start_x = self.current_pose.position.x
        start_y = self.current_pose.position.y
        current_yaw = self._quaternion_to_yaw(self.current_pose.orientation)
        
        distance_to_surge = 15.0 
        
        target_x = start_x + (distance_to_surge * math.cos(current_yaw))
        target_y = start_y + (distance_to_surge * math.sin(current_yaw))
        
        self.get_logger().info(f'Navigating forward 15m. Target: X={target_x:.2f}, Y={target_y:.2f}')
        
        goal = NavigateToWaypoint.Goal()
        goal.target_pose.pose.position.x = target_x
        goal.target_pose.pose.position.y = target_y
        goal.target_pose.pose.orientation = self.current_pose.orientation # Lock original heading
        goal.timeout_sec = 45.0  # Allow 45 seconds to cover the 15 meters
        

        future = self.nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Navigation goal was rejected!')
            return
            
        self.get_logger().info('Surging through Gate and to Marker...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        self.get_logger().info('Run complete. Surfacing...')
        depth_msg.data = 0.0
        self.depth_pub.publish(depth_msg)
        time.sleep(4.0)
        
        self.get_logger().info('Disarming...')
        req.data = False
        self.arm_cli.call_async(req)
        
        self.get_logger().info('=== PREQUALIFICATION COMPLETE ===')

def main(args=None):
    rclpy.init(args=args)
    node = PrequalificationRun()
    try:
        node.execute_mission()
    except KeyboardInterrupt:
        node.get_logger().info('Emergency Stop! Surfacing...')
        depth_msg = Float64()
        depth_msg.data = 0.0
        node.depth_pub.publish(depth_msg)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()