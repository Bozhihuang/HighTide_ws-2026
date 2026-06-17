#!/usr/bin/env python3
"""
RoboSub Streamlined U-Turn Prequalification Run (Rulebook Section 3.4.1.2)

Maneuver Steps:
  1. Submerge 3m behind the Gate down to 1.0 meter.
  2. Waypoint 1: Surge 13.5m straight forward (passing left of the marker).
  3. Waypoint 2A: STOP at the end line and pivot the nose 90 degrees Right in place.
  4. Waypoint 2B: Drive 3.0m straight across the back of the marker pole.
  5. Waypoint 3A: STOP on the right side and pivot another 90 degrees Right (facing home).
  6. Waypoint 3B: Surge straight home through the Gate to the absolute origin.
  7. Surface and disarm safely.
"""

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
        
        # Service & Action Clients
        self.arm_cli = self.create_client(SetBool, '/hightide/arm')
        self.alt_hold_cli = self.create_client(Trigger, '/hightide/set_alt_hold')
        self.depth_pub = self.create_publisher(Float64, '/hightide/target_depth', 10)
        self.nav_client = ActionClient(self, NavigateToWaypoint, '/hightide/navigate_to_waypoint')
        
        # Subscriptions
        self.create_subscription(Odometry, '/odometry/filtered', self._odom_cb, 10)
        self.current_pose = None
        
        self.get_logger().info('Connecting to flight control networks...')
        self.arm_cli.wait_for_service()
        self.alt_hold_cli.wait_for_service()
        self.nav_client.wait_for_server()

    def _odom_cb(self, msg):
        self.current_pose = msg.pose.pose

    def _quaternion_to_yaw(self, q):
        """Extracts Euler yaw from global state orientation quaternion."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _yaw_to_quaternion(self, yaw):
        """Converts an Euler angle heading target into a spatial Quaternion."""
        from geometry_msgs.msg import Quaternion
        return Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(yaw / 2.0),
            w=math.cos(yaw / 2.0)
        )

    def move_to(self, x, y, yaw_target):
        """Dispatches a waypoint goal to the action server."""
        goal = NavigateToWaypoint.Goal()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation = self._yaw_to_quaternion(yaw_target)
        
        self.get_logger().info(
            f'Sending Waypoint Target -> X: {x:.2f}, Y: {y:.2f}, Heading: {math.degrees(yaw_target):.1f}°'
        )
        
        future = self.nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error('Waypoint navigation goal rejected by server node!')
            return False
            
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return result_future.result().result.success

    def execute_mission(self):
        self.get_logger().info('=== STARTING HIGH-TIDE SQUARE PREQUALIFICATION ===')
        
        while self.current_pose is None:
            self.get_logger().info('Waiting for EKF localization lock...')
            rclpy.spin_once(self, timeout_sec=1.0)
            
        input("\n[!] Position vehicle 3m behind Gate, offset LEFT of Marker. Press Enter to DEPLOY...")
        
        # 1. Arm and Submerge below surface limits
        self.get_logger().info('Arming electronic speed controllers...')
        req = SetBool.Request()
        req.data = True
        self.arm_cli.call_async(req)
        time.sleep(1.0)
        
        self.get_logger().info('Engaging Alt-Hold mode...')
        self.alt_hold_cli.call_async(Trigger.Request())
        time.sleep(1.0)
        
        self.get_logger().info('Descending to 1.0 meter...')
        depth_msg = Float64()
        depth_msg.data = 1.0
        self.depth_pub.publish(depth_msg)
        time.sleep(6.0)  
        
        # 2. Establish spatial coordinate anchors based on deployment setup
        start_x = self.current_pose.position.x
        start_y = self.current_pose.position.y
        init_yaw = self._quaternion_to_yaw(self.current_pose.orientation)
        
        cos_f = math.cos(init_yaw)
        sin_f = math.sin(init_yaw)
        cos_r = math.cos(init_yaw - math.pi / 2.0)
        sin_r = math.sin(init_yaw - math.pi / 2.0)
        
        # Headings definitions
        heading_forward = init_yaw
        heading_right = normalize_angle(init_yaw - math.pi / 2.0)
        heading_home = normalize_angle(init_yaw + math.pi)

        # 3. WAYPOINT 1: Outbound Pass (13.5m straight out from starting line)
        self.get_logger().info('>>> PHASE 1: Surging Outbound Left of Marker...')
        wp1_x = start_x + (13.5 * cos_f)
        wp1_y = start_y + (13.5 * sin_f)
        self.move_to(wp1_x, wp1_y, heading_forward)
        
        # 4. WAYPOINT 2A: Pivot-in-Place Right
        # Holds current position, turns nose 90 degrees right to face parallel to gate
        self.get_logger().info('>>> PHASE 2A: Pivoting 90° Right in place...')
        self.move_to(wp1_x, wp1_y, heading_right)
        time.sleep(1.0) # Small buffer to allow heading to completely settle

        # wp2
        self.get_logger().info('go 2 metre straigt paralel to gate and behind marker')
        wp2_x = wp1_x + (2.0 * cos_r)
        wp2_y = wp1_y + (2.0 * sin_r)
        self.move_to(wp2_x, wp2_y, heading_right)
        
        # turn2
        self.get_logger().info('pivot 90deg')
        self.move_to(wp2_x, wp2_y, heading_home)
        time.sleep(1.0)

        # 7. wp3
        self.get_logger().info('>>> PHASE 3B: Surging Home Forward Through Gate...')
        self.move_to(start_x, start_y, heading_home)
        
        # 8. Safety Surface & Disarm sequence
        self.get_logger().info('Maneuver loop finished! Ascending to surface...')
        depth_msg.data = 0.0
        self.depth_pub.publish(depth_msg)
        time.sleep(5.0)
        
        self.get_logger().info('Disarming propulsion arrays...')
        req.data = False
        self.arm_cli.call_async(req)
        self.get_logger().info('=== PREQUALIFICATION MANEUVER EXECUTED SUCCESSFULLY ===')

def main(args=None):
    rclpy.init(args=args)
    node = PrequalificationRun()
    try:
        node.execute_mission()
    except KeyboardInterrupt:
        node.get_logger().warn('EMERGENCY MANUAL INTERRUPT! Forcing immediate float...')
        depth_msg = Float64()
        depth_msg.data = 0.0
        node.depth_pub.publish(depth_msg)
    finally:
        node.destroy_node()
        rclpy.shutdown()

def normalize_angle(angle: float) -> float:
    while angle > math.pi:  angle -= 2 * math.pi
    while angle < -math.pi: angle += 2 * math.pi
    return angle

if __name__ == '__main__':
    main()