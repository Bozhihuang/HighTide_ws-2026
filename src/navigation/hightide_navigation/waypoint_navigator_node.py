#!/usr/bin/env python3

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
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
        # Yaw output sign. IMU yaw (ENU) is CCW-positive, but ThrusterCommand.yaw /
        # ArduSub ch4 are CW-positive, so the PID output must be negated to hold
        # heading instead of diverging. If the sub ever corrects the WRONG way at the
        # pool, flip this to +1.0 in params.yaml (no rebuild needed).
        self.declare_parameter('yaw_command_sign', -1.0)

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
        self.yaw_command_sign = self.get_parameter('yaw_command_sign').value

        self.current_odom = None
        self.current_yaw = None
        self.last_time = self.get_clock().now()

        # Reentrant group so the sensor callbacks AND the action cancel request can be
        # serviced concurrently while _execute_callback is running its control loop.
        # Without this (default group + single-thread executor) the blocking loop
        # starves cancel handling and sensor updates — cancels are ignored and yaw
        # goes stale, which is what made the sub spin and ignore Ctrl+C.
        self.callback_group = ReentrantCallbackGroup()

        self.odom_sub = self.create_subscription(
            Odometry, '/mavros/zed/odom',
            self._odom_callback, sensor_qos,
            callback_group=self.callback_group)

        # Added IMU subscription to explicitly track vehicle heading
        self.imu_sub = self.create_subscription(
            Imu, '/mavros/imu/data',
            self._imu_callback, sensor_qos,
            callback_group=self.callback_group)

        self.cmd_pub = self.create_publisher(ThrusterCommand, '/hightide/cmd_vel', 10)

        self._action_server = ActionServer(
            self, NavigateToWaypoint, '/hightide/navigate_to_waypoint',
            execute_callback=self._execute_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.callback_group)

        self.get_logger().info('Waypoint Navigator Node started with ZED odom and IMU heading configurations')

    def _cancel_callback(self, goal_handle):
        """Accept every cancel request so Ctrl+C / client shutdown can stop the sub."""
        self.get_logger().warn('Waypoint cancel requested — stopping vehicle.')
        return CancelResponse.ACCEPT

    def _odom_callback(self, msg):
        self.current_odom = msg

    def _imu_callback(self, msg):
        self.current_yaw = quaternion_to_yaw(msg.orientation)

    def _publish_stop(self):
        """Command zero thrust on all 6 DOF.

        A default ThrusterCommand is all zeros → neutral. Called on every exit path
        so the vehicle does not coast on the last PID output, and does not rely on
        rc_override_node's 0.5 s timeout to neutralize.
        """
        stop = ThrusterCommand()
        stop.header.stamp = self.get_clock().now().to_msg()
        self.cmd_pub.publish(stop)

    def _execute_callback(self, goal_handle):
        """Execute navigation to waypoint."""
        goal = goal_handle.request
        start_time = self.get_clock().now()
        at_target_since = None

        # Reset PID state at the START of every goal. The PIDs live for the node's
        # whole lifetime, so without this the integrator / derivative memory from a
        # previous (possibly aborted) waypoint carries into the next run and the
        # controller commands stale thrust before it even sees the new error.
        self.surge_pid.reset()
        self.sway_pid.reset()
        self.yaw_pid.reset()
        # Fresh dt baseline so the first iteration's derivative isn't a huge spike
        # from the time since the previous goal.
        self.last_time = self.get_clock().now()

        self.get_logger().info(
            f'Navigating to ({goal.target_pose.pose.position.x:.1f}, '
            f'{goal.target_pose.pose.position.y:.1f})')

        feedback = NavigateToWaypoint.Feedback()
        result = NavigateToWaypoint.Result()

        # try/finally GUARANTEES a zero-thrust command is published no matter how we
        # leave this callback — normal return, cancel, timeout, or an exception
        # (including KeyboardInterrupt unwinding through here on shutdown). rc_override
        # then holds neutral; its 0.5 s timeout is only the backstop, not the primary.
        try:
            while rclpy.ok():
                # Cancel accepted → status is CANCELING (is_active is already False),
                # so check this BEFORE anything that assumes an active goal.
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    result.message = 'Cancelled'
                    return result

                # Goal was aborted/preempted from elsewhere — stop working on it.
                if not goal_handle.is_active:
                    result.success = False
                    result.message = 'Preempted'
                    return result

                elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
                if elapsed > goal.timeout_sec > 0:
                    goal_handle.abort()
                    result.success = False
                    result.message = 'Timeout'
                    return result

                # Wait until both sensor streams have provided initial data. The
                # callbacks run in other executor threads now, so just yield.
                if self.current_odom is None or self.current_yaw is None:
                    time.sleep(0.05)
                    continue

                now = self.get_clock().now()
                dt = (now - self.last_time).nanoseconds / 1e9
                self.last_time = now
                if dt <= 0.0:
                    time.sleep(0.05)
                    continue

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
                # Negate (see yaw_command_sign): converts the ENU CCW-positive yaw
                # error into ArduSub's CW-positive yaw command so heading is held.
                cmd.yaw = self.yaw_command_sign * self.yaw_pid.compute(yaw_error, dt)
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

                time.sleep(0.05)

            # Loop exited because rclpy is shutting down or the goal went inactive.
            if goal_handle.is_active:
                goal_handle.abort()
            result.success = False
            result.message = 'Node shutdown'
            return result
        finally:
            self._publish_stop()


def main(args=None):
    rclpy.init(args=args)
    node = WaypointNavigatorNode()
    # MultiThreadedExecutor lets the blocking control loop, the sensor callbacks, and
    # the action cancel handler run at the same time — required for cancel + Ctrl+C to
    # actually stop the vehicle.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Best-effort neutral on the way down, in case a goal was mid-flight.
        try:
            node._publish_stop()
        except Exception:
            pass
        node.destroy_node()

if __name__ == '__main__':
    main()