#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import SetBool, Trigger
from mavros_msgs.msg import State, OverrideRCIn
from mavros_msgs.srv import CommandBool, SetMode


class ModeManagerNode(Node):
    """Manages ArduSub flight modes, arming, and special maneuvers with state feedback."""

    def __init__(self):
        super().__init__('mode_manager_node')

        # Reentrant Callback Group to allow concurrent execution under MultiThreadedExecutor
        self.callback_group = ReentrantCallbackGroup()

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
            State, 
            '/mavros/state', 
            self._state_callback, 
            10,
            callback_group=self.callback_group
        )

        # MAVROS service clients
        self.arm_client = self.create_client(
            CommandBool, 
            '/mavros/cmd/arming',
            callback_group=self.callback_group
        )
        self.mode_client = self.create_client(
            SetMode, 
            '/mavros/set_mode',
            callback_group=self.callback_group
        )

        # RC Override publisher (for barrel roll)
        self.rc_pub = self.create_publisher(
            OverrideRCIn, 
            '/mavros/rc/override', 
            10,
            callback_group=self.callback_group
        )

        # Services we provide
        self.arm_srv = self.create_service(
            SetBool, 
            '/hightide/arm', 
            self._arm_service,
            callback_group=self.callback_group
        )
        self.set_alt_hold_srv = self.create_service(
            Trigger, 
            '/hightide/set_alt_hold', 
            self._set_alt_hold_service,
            callback_group=self.callback_group
        )
        self.set_manual_srv = self.create_service(
            Trigger, 
            '/hightide/set_manual', 
            self._set_manual_service,
            callback_group=self.callback_group
        )
        self.barrel_roll_srv = self.create_service(
            Trigger, 
            '/hightide/barrel_roll', 
            self._barrel_roll_service,
            callback_group=self.callback_group
        )

        self.get_logger().info('Mode Manager Node initialized with Multithreading and Reentrant Groups.')

    def _state_callback(self, msg: State):
        """Track current vehicle state."""
        prev_armed = self.armed
        prev_mode = self.current_mode

        self.armed = msg.armed
        self.current_mode = msg.mode
        self.connected = msg.connected

        if self.armed != prev_armed:
            self.get_logger().info(f'FCU Armed State Update -> {self.armed}')
        if self.current_mode != prev_mode:
            self.get_logger().info(f'FCU Mode State Update -> {self.current_mode}')

    def _call_arm(self, arm: bool) -> bool:
        """Call MAVROS arming service and verify the state change actually registers on the FCU."""
        if not self.arm_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Arming service not available')
            return False

        req = CommandBool.Request()
        req.value = arm
        
        self.get_logger().info(f'Sending CommandBool Request -> value={arm}')
        future = self.arm_client.call_async(req)

        # Wait for service response asynchronously without deadlocking the executor
        start_time = self.get_clock().now()
        while rclpy.ok() and not future.done():
            elapsed = (self.get_clock().now() - start_time).nanoseconds * 1e-9
            if elapsed > 4.0:
                self.get_logger().error('Timeout waiting for arming service response!')
                return False
            time.sleep(0.05)

        res = future.result()
        if res is None or not res.success:
            self.get_logger().error(f'MAVROS rejected the {"arm" if arm else "disarm"} command.')
            return False

        # VERIFICATION LOOP: Poll the state subscriber updates to ensure the FCU accepted it
        self.get_logger().info('Command accepted by MAVROS. Verifying actual FCU state transition...')
        start_time = self.get_clock().now()
        while rclpy.ok():
            if self.armed == arm:
                self.get_logger().info(f'FCU successfully transitioned to {"Armed" if arm else "Disarmed"} state.')
                return True
            
            elapsed = (self.get_clock().now() - start_time).nanoseconds * 1e-9
            if elapsed > 4.0:
                self.get_logger().error(f'State verification TIMEOUT! FCU is still {"armed" if self.armed else "disarmed"}.')
                return False
            time.sleep(0.05)

        return False

    def _call_set_mode(self, mode: str) -> bool:
        """Call MAVROS set_mode service and verify the state change actually registers on the FCU."""
        if not self.mode_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Set mode service not available')
            return False

        req = SetMode.Request()
        req.base_mode = 0
        req.custom_mode = mode
        
        self.get_logger().info(f'Sending SetMode Request -> custom_mode={mode}')
        future = self.mode_client.call_async(req)

        # Wait for service response asynchronously
        start_time = self.get_clock().now()
        while rclpy.ok() and not future.done():
            elapsed = (self.get_clock().now() - start_time).nanoseconds * 1e-9
            if elapsed > 4.0:
                self.get_logger().error('Timeout waiting for set_mode service response!')
                return False
            time.sleep(0.05)

        res = future.result()
        if res is None or not res.mode_sent:
            self.get_logger().error(f'MAVROS set_mode message failed to send for mode: {mode}')
            return False

        # VERIFICATION LOOP: Poll the state subscriber updates to ensure the FCU accepted it
        self.get_logger().info(f'Mode switch sent. Verifying actual FCU transition to {mode}...')
        start_time = self.get_clock().now()
        while rclpy.ok():
            if self.current_mode == mode:
                self.get_logger().info(f'FCU successfully transitioned to mode: {self.current_mode}.')
                return True
            
            elapsed = (self.get_clock().now() - start_time).nanoseconds * 1e-9
            if elapsed > 4.0:
                self.get_logger().error(f'State verification TIMEOUT! FCU is still in mode: {self.current_mode}.')
                return False
            time.sleep(0.05)

        return False

    def _arm_service(self, request, response):
        """Arm or disarm the vehicle."""
        success = self._call_arm(request.data)
        response.success = success
        response.message = (
            f'{"Armed" if request.data else "Disarmed"} '
            f'{"successfully" if success else "failed"}'
        )
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

        # Brief pause to let mode change take effect on physical vehicle
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
            f'Rolling for {self.barrel_roll_duration}s at PWM {self.barrel_roll_pwm}'
        )

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
    
    # Use MultiThreadedExecutor to allow state callback to run while service is waiting
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()