#!/usr/bin/env python3
"""
Actuator Driver Node — Controls torpedoes and marker droppers via MAVROS relay commands.
"""

import time as pytime
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from hightide_interfaces.srv import FireTorpedo, DropMarker
from mavros_msgs.srv import CommandLong


class ActuatorDriverNode(Node):
    """Controls torpedoes and marker droppers via MAVLink relays."""

    def __init__(self):
        super().__init__('actuator_driver_node')

        self.declare_parameter('torpedo_1_relay', 0.0)
        self.declare_parameter('torpedo_2_relay', 1.0)
        self.declare_parameter('dropper_1_relay', 2.0)
        self.declare_parameter('dropper_2_relay', 3.0)
        self.declare_parameter('pulse_duration_ms', 500)

        self.relays = {
            'torpedo_1': float(self.get_parameter('torpedo_1_relay').value),
            'torpedo_2': float(self.get_parameter('torpedo_2_relay').value),
            'dropper_1': float(self.get_parameter('dropper_1_relay').value),
            'dropper_2': float(self.get_parameter('dropper_2_relay').value),
        }
        self.pulse_ms = self.get_parameter('pulse_duration_ms').value

        self.torpedoes_fired = {1: False, 2: False}
        self.markers_dropped = {1: False, 2: False}

        # FIX: Use a single Reentrant group. This allows the node to spin up 
        # a new thread for every incoming and outgoing request without deadlocking.
        self.cb_group = ReentrantCallbackGroup()

        # MAVROS command service
        self.cmd_client = self.create_client(
            CommandLong, 
            '/mavros/cmd/command',
            callback_group=self.cb_group
        )

        self.get_logger().info('Waiting for MAVROS command service...')

        # Services exposed to the Mission Behavior Tree
        self.torpedo_srv = self.create_service(
            FireTorpedo, 
            '/hightide/fire_torpedo', 
            self._fire_torpedo,
            callback_group=self.cb_group
        )
        self.dropper_srv = self.create_service(
            DropMarker, 
            '/hightide/drop_marker', 
            self._drop_marker,
            callback_group=self.cb_group
        )

        self.get_logger().info('Actuator Driver Node started (MAVLink Relay Mode)')

    def _set_relay(self, relay_pin: float, state: float) -> bool:
        """Synchronously send a MAVLink relay command."""
        if not self.cmd_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('MAVROS cmd service not available!')
            return False

        req = CommandLong.Request()
        req.command = 181  # MAV_CMD_DO_SET_RELAY
        req.broadcast = False
        req.confirmation = 0
        req.param1 = relay_pin  
        req.param2 = state      
        req.param3 = 0.0
        req.param4 = 0.0
        req.param5 = 0.0
        req.param6 = 0.0
        req.param7 = 0.0

        # FIX: Direct synchronous call. Because we are in a Reentrant group 
        # inside a MultiThreadedExecutor, this safely mimics the terminal CLI call.
        result = self.cmd_client.call(req)
        
        if result.success:
            return True
        else:
            # Decode the exact reason ArduSub rejected it
            reasons = {
                1: 'TEMPORARILY REJECTED',
                2: 'DENIED',
                3: 'UNSUPPORTED (Check QGroundControl RELAY_PIN params)',
                4: 'FAILED',
                5: 'TIMEOUT (Submarine did not acknowledge)'
            }
            error_msg = reasons.get(result.result, f'UNKNOWN CODE {result.result}')
            self.get_logger().error(f'MAVROS Rejected Command! Reason: {error_msg}')
            return False

    def _actuate_relay(self, name: str) -> bool:
        relay_pin = self.relays.get(name)
        if relay_pin is None:
            return False

        self.get_logger().info(f'Actuating {name} (Relay {relay_pin}) for {self.pulse_ms}ms')

        success_on = self._set_relay(relay_pin, 1.0)
        if not success_on:
            self.get_logger().error(f'Failed to turn ON {name}')
            return False

        pytime.sleep(self.pulse_ms / 1000.0)

        success_off = self._set_relay(relay_pin, 0.0)
        if not success_off:
            self.get_logger().error(f'Failed to turn OFF {name}')
            return False

        return True

    def _fire_torpedo(self, request, response):
        tube_id = request.tube_id

        if tube_id not in (1, 2):
            response.success = False
            response.message = f'Invalid tube_id: {tube_id} (must be 1 or 2)'
            return response

        if self.torpedoes_fired[tube_id]:
            response.success = False
            response.message = f'Torpedo {tube_id} already fired!'
            self.get_logger().warn(response.message)
            return response

        pin_name = f'torpedo_{tube_id}'
        success = self._actuate_relay(pin_name)
        
        if success:
            self.torpedoes_fired[tube_id] = True

        response.success = success
        response.message = f'Torpedo {tube_id} {"fired" if success else "failed"}!'
        self.get_logger().info(response.message)
        return response

    def _drop_marker(self, request, response):
        dropper_id = request.dropper_id

        if dropper_id not in (1, 2):
            response.success = False
            response.message = f'Invalid dropper_id: {dropper_id} (must be 1 or 2)'
            return response

        if self.markers_dropped[dropper_id]:
            response.success = False
            response.message = f'Marker {dropper_id} already dropped!'
            self.get_logger().warn(response.message)
            return response

        pin_name = f'dropper_{dropper_id}'
        success = self._actuate_relay(pin_name)
        
        if success:
            self.markers_dropped[dropper_id] = True

        response.success = success
        response.message = f'Marker {dropper_id} {"dropped" if success else "failed"}!'
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ActuatorDriverNode()
    
    # We increase the thread count to 4 to ensure there is always 
    # a thread available for the MAVROS response to arrive on.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()