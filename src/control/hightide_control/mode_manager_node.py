#!/usr/bin/env python3
"""
Mode Manager Node — Handles arming/disarming, flight mode changes, and barrel roll.

Flight modes (ArduSub):
  ALT_HOLD  — Depth hold + stabilized (primary mission mode)
  MANUAL    — No stabilization (used for barrel roll)
  STABILIZE — Roll/pitch stabilized, no depth hold

MAVROS set_mode uses custom_mode as a string: 'ALT_HOLD', 'MANUAL', 'STABILIZE'.
MAVROS cmd/arming uses CommandBool: value=true to arm, false to disarm.
System ID must be 255 for ArduSub to accept GCS commands.
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool, Trigger
from mavros_msgs.msg import State, OverrideRCIn
from mavros_msgs.srv import CommandBool, SetMode


class ModeManagerNode(Node):
    """Manages ArduSub flight modes, arming, and special maneuvers."""

    def __init__(self):
        super().__init__('mode_manager_node')

        # Parameters
        self.declare_parameter('barrel_roll_duration_sec', 3.0)
        self.declare_parameter('barrel_roll_pwm', 1900)

        self.barrel_roll_duration = self.get_parameter('barrel_roll_duration_sec').value
        self.barrel_roll_pwm = self.get_parameter('barrel_roll_pwm').value

        # State tracking
        self.armed = False
        self.current_mode = ''
        self.connected = False

        # MAVROS state subscriber
        self.state_sub = self.create_subscription(
            State, '/mavros/state', self._state_callback, 10)

        # MAVROS service clients
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # RC Override publisher (for barrel roll)
        self.rc_pub = self.create_publisher(OverrideRCIn, '/mavros/rc/override', 10)

        # Services we provide
        self.arm_srv = self.create_service(
            SetBool, '/hightide/arm', self._arm_service)
        self.set_alt_hold_srv = self.create_service(
            Trigger, '/hightide/set_alt_hold', self._set_alt_hold_service)
        self.set_manual_srv = self.create_service(
            Trigger, '/hightide/set_manual', self._set_manual_service)
        self.barrel_roll_srv = self.create_service(
            Trigger, '/hightide/barrel_roll', self._barrel_roll_service)

        self.get_logger().info('Mode Manager Node started')

    def _state_callback(self, msg: State):
        """Track current vehicle state."""
        prev_armed = self.armed
        prev_mode = self.current_mode

        self.armed = msg.armed
        self.current_mode = msg.mode
        self.connected = msg.connected

        if self.armed != prev_armed:
            self.get_logger().info(f'Armed: {self.armed}')
        if self.current_mode != prev_mode:
            self.get_logger().info(f'Mode: {self.current_mode}')

    def _call_arm(self, arm: bool) -> bool:
        """Call MAVROS arming service."""
        if not self.arm_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Arming service not available')
            return False

        req = CommandBool.Request()
        req.value = arm
        future = self.arm_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.result() is not None and future.result().success:
            self.get_logger().info(f'{"Armed" if arm else "Disarmed"} successfully')
            return True
        else:
            self.get_logger().error(f'Failed to {"arm" if arm else "disarm"}')
            return False

    def _call_set_mode(self, mode: str) -> bool:
        """Call MAVROS set_mode service."""
        if not self.mode_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Set mode service not available')
            return False

        req = SetMode.Request()
        req.base_mode = 0
        req.custom_mode = mode
        future = self.mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.result() is not None and future.result().mode_sent:
            self.get_logger().info(f'Mode set to {mode}')
            return True
        else:
            self.get_logger().error(f'Failed to set mode {mode}')
            return False

    def _arm_service(self, request, response):
        """Arm or disarm the vehicle."""
        success = self._call_arm(request.data)
        response.success = success
        response.message = (
            f'{"Armed" if request.data else "Disarmed"} '
            f'{"successfully" if success else "failed"}')
        return response

    def _set_alt_hold_service(self, request, response):
        """Switch to Alt Hold mode."""
        success = self._call_set_mode('ALT_HOLD')
        response.success = success
        response.message = f'Alt Hold {"set" if success else "failed"}'
        return response

    def _set_manual_service(self, request, response):
        """Switch to Manual mode."""
        success = self._call_set_mode('MANUAL')
        response.success = success
        response.message = f'Manual mode {"set" if success else "failed"}'
        return response

    def _barrel_roll_service(self, request, response):
        """
        Execute a barrel roll for style points.

        Switches to MANUAL mode, fires opposite vertical thrusters for
        barrel_roll_duration seconds, then stops.

        WARNING: This WILL destroy FOG heading reference. Only use as
        the final maneuver of the mission.
        """
        self.get_logger().warn('=== BARREL ROLL INITIATED ===')
        self.get_logger().warn('FOG heading will be lost after this maneuver!')

        # Switch to Manual mode
        if not self._call_set_mode('MANUAL'):
            response.success = False
            response.message = 'Failed to switch to Manual mode'
            return response

        # Brief pause to let mode change take effect
        import time
        time.sleep(0.5)

        # Command barrel roll: opposite roll command via RC Override
        # Channel 2 (Roll): full deflection
        roll_msg = OverrideRCIn()
        roll_channels = [1500] * 18
        roll_channels[1] = self.barrel_roll_pwm  # Full roll
        roll_msg.channels = roll_channels

        # Publish at 20Hz for the duration
        rate = 20
        iterations = int(self.barrel_roll_duration * rate)
        publish_period = 1.0 / rate

        self.get_logger().info(
            f'Rolling for {self.barrel_roll_duration}s at PWM {self.barrel_roll_pwm}')

        for _ in range(iterations):
            self.rc_pub.publish(roll_msg)
            time.sleep(publish_period)

        # Stop motors
        stop_msg = OverrideRCIn()
        stop_msg.channels = [1500] * 18
        self.rc_pub.publish(stop_msg)

        self.get_logger().info('Barrel roll complete — motors stopped')

        response.success = True
        response.message = 'Barrel roll executed successfully'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ModeManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
