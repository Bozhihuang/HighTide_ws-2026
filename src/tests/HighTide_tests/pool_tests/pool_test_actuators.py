#!/usr/bin/env python3
"""
Pool Test: Actuators

Verifies that the GPIO relays successfully trigger the torpedo solenoids
and marker droppers when the vehicle is in the water.
"""

import time
import rclpy
from rclpy.node import Node
from HighTide_interfaces.srv import FireTorpedo, DropMarker


class ActuatorPoolTest(Node):
    def __init__(self):
        super().__init__('pool_test_actuators')
        
        self.torpedo_cli = self.create_client(FireTorpedo, '/HighTide/fire_torpedo')
        self.dropper_cli = self.create_client(DropMarker, '/HighTide/drop_marker')
        
        while not self.torpedo_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for torpedo service...')
            
        while not self.dropper_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for dropper service...')

    def run_tests(self):
        self.get_logger().info('=== STARTING ACTUATOR POOL TEST ===')
        
        input("Press Enter to FIRE TORPEDO 1...")
        self.fire_torpedo(1)
        
        input("Press Enter to FIRE TORPEDO 2...")
        self.fire_torpedo(2)
        
        input("Press Enter to DROP MARKER 1...")
        self.drop_marker(1)
        
        input("Press Enter to DROP MARKER 2...")
        self.drop_marker(2)
        
        self.get_logger().info('=== ACTUATOR POOL TEST COMPLETE ===')

    def fire_torpedo(self, tube_id):
        req = FireTorpedo.Request()
        req.tube_id = tube_id
        future = self.torpedo_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        self.get_logger().info(f'Torpedo {tube_id} response: success={res.success}, msg="{res.message}"')

    def drop_marker(self, dropper_id):
        req = DropMarker.Request()
        req.dropper_id = dropper_id
        future = self.dropper_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        self.get_logger().info(f'Marker {dropper_id} response: success={res.success}, msg="{res.message}"')


def main(args=None):
    rclpy.init(args=args)
    node = ActuatorPoolTest()
    try:
        node.run_tests()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
