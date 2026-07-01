#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from hightide_interfaces.msg import ThrusterCommand
from hightide_interfaces.action import NavigateToWaypoint
from hightide_navigation import PIDController, normalize_angle, quaternion_to_yaw
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class WaypointNavigatorNode(Node):
    """Navigate to waypoints by decomposing into body-frame surge/sway using ZED position and IMU heading."""

    def __init__(self):
        super().__init__('waypoint_navigator_node')
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.declare_parameter('surge_kp', 0.0)
        self.declare_parameter('surge_ki', 0.0)
        self.declare_parameter('surge_kd', 0.0)
        self.declare_parameter('sway_kp', 0.0)
        self.declare_parameter('sway_ki', 0.0)
        self.declare_parameter('sway_kd', 0.0)
        self.declare_parameter('yaw_kp', 0.0)
        self.declare_parameter('yaw_ki', 0.0)
        self.declare_parameter('yaw_kd', 0.0)
        self.declare_parameter('position_tolerance', 0.3)
        self.declare_parameter('yaw_tolerance', 0.1)
        self.declare_parameter('max_speed', 0.6)

        self.surge_pid = PIDController(
            self.get_parameter('surge_kp').value,
            self.get_parameter('surge_ki').value,
            self.get_parameter('surge_kd').value,
            output_max=self.get_parameter('max_speed').value)
        self.sway_pid = PIDController(
            self.get_parameter('sway_kp').value,
            self.get_parameter('sway_ki').value,
            self.get_parameter('sway_kd').value,
            output_max=self.get_parameter('max_speed').value)
        self.yaw_pid = PIDController(
            self.get_parameter('yaw_kp').value,
            self.get_parameter('yaw_ki').value,
            self.get_parameter('yaw_kd').value)

        self.pos_tol = self.get_parameter('position_tolerance').value
        self.yaw_tol = self.get_parameter('yaw_tolerance').value

        self.current_odom = None
        self.current_yaw = None
        self.last_time = self.get_clock().now()

        self.odom_sub = self.create_subscription(
            Odometry, '/mavros/zed/odom',
            self._odom_callback, sensor_qos)
            
        # Added IMU subscription to explicitly track vehicle heading
        self.imu_sub = self.create_subscription(
            Imu, '/mavros/imu/data',
            self._imu_callback, sensor_qos)
            
        self.cmd_pub = self.create_publisher(ThrusterCommand, '/hightide/cmd_vel', 10)

        self._action_server = ActionServer(
            self, NavigateToWaypoint, '/hightide/navigate_to_waypoint',
            self._execute_callback)

        self.get_logger().info('Waypoint Navigator Node started with ZED odom and IMU heading configurations')

    def _odom_callback(self, msg):
        self.current_odom = msg

    def _imu_callback(self, msg):
        self.current_yaw = quaternion_to_yaw(msg.orientation)

    def _execute_callback(self, goal_handle):
        """Execute navigation to waypoint."""
        goal = goal_handle.request
        start_time = self.get_clock().now()
        at_target_since = None

        self.get_logger().info(
            f'Navigating to ({goal.target_pose.pose.position.x:.1f}, '
            f'{goal.target_pose.pose.position.y:.1f})')

        feedback = NavigateToWaypoint.Feedback()
        result = NavigateToWaypoint.Result()

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.message = 'Cancelled'
                return result

            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed > goal.timeout_sec > 0:
                goal_handle.abort()
                result.success = False
                result.message = 'Timeout'
                return result

            # Wait until both sensor streams have provided initial data
            if self.current_odom is None or self.current_yaw is None:
                rclpy.spin_once(self, timeout_sec=0.05)
                continue

            now = self.get_clock().now()
            dt = (now - self.last_time).nanoseconds / 1e9
            self.last_time = now

            # Current position extracted from ZED, and heading extracted from IMU data
            pos = self.current_odom.pose.pose.position
            yaw = self.current_yaw

            # Goal pose
            gx = goal.target_pose.pose.position.x
            gy = goal.target_pose.pose.position.y
            goal_yaw = quaternion_to_yaw(goal.target_pose.pose.orientation)

            # World-frame error
            dx = gx - pos.x
            dy = gy - pos.y
            dist = math.sqrt(dx * dx + dy * dy)

            # Transform to body frame (crab walk decomposition)
            # surge = forward component, sway = lateral component
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            surge_error = dx * cos_yaw + dy * sin_yaw   # Forward
            sway_error = -dx * sin_yaw + dy * cos_yaw   # Lateral (right +)

            yaw_error = normalize_angle(goal_yaw - yaw)

            # PID
            cmd = ThrusterCommand()
            cmd.header.stamp = now.to_msg()
            cmd.surge = self.surge_pid.compute(surge_error, dt)
            cmd.sway = self.sway_pid.compute(sway_error, dt)
            cmd.yaw = self.yaw_pid.compute(yaw_error, dt)
            self.cmd_pub.publish(cmd)

            # Check if at target
            at_pos = dist < self.pos_tol
            at_yaw = abs(yaw_error) < self.yaw_tol

            if at_pos and at_yaw:
                if at_target_since is None:
                    at_target_since = now
                elif (now - at_target_since).nanoseconds / 1e9 > 1.0:
                    # Held position for 1 second — success
                    goal_handle.succeed()
                    result.success = True
                    result.final_distance_m = dist
                    result.message = 'Waypoint reached'
                    self.get_logger().info('Waypoint reached!')
                    return result
            else:
                at_target_since = None

            # Publish feedback
            feedback.distance_remaining_m = dist
            feedback.yaw_error_rad = yaw_error
            feedback.elapsed_sec = elapsed
            goal_handle.publish_feedback(feedback)

            rclpy.spin_once(self, timeout_sec=0.05)

        result.success = False
        result.message = 'Node shutdown'
        return result


def main(args=None):
    rclpy.init(args=args)
    node = WaypointNavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()