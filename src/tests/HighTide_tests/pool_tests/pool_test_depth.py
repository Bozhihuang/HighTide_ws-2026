#!/usr/bin/env python3
"""
Pool Test: Depth Controller (Alt Hold)

Tests the ability of the sub to arm, switch to Alt Hold, dive to a target
depth, hold that depth for 10 seconds, and surface safely.
"""

import time
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool, Trigger
from std_msgs.msg import Float64


class DepthPoolTest(Node):
    def __init__(self):
        super().__init__('pool_test_depth')
        
        self.arm_cli = self.create_client(SetBool, '/HighTide/arm')
        self.alt_hold_cli = self.create_client(Trigger, '/HighTide/set_alt_hold')
        self.depth_pub = self.create_publisher(Float64, '/HighTide/target_depth', 10)
        
        self.current_depth = 0.0
        self.create_subscription(Float64, '/mavros/global_position/rel_alt', self._depth_cb, 10)
        
        while not self.arm_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for arm service...')
        while not self.alt_hold_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for alt_hold service...')

    def _depth_cb(self, msg):
        self.current_depth = -msg.data  # Convert to positive = deeper

    def run_tests(self):
        self.get_logger().info('=== STARTING DEPTH POOL TEST ===')
        
        input("Ensure sub is in water. Press Enter to ARM...")
        req = SetBool.Request()
        req.data = True
        future = self.arm_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info('Vehicle ARMED.')
        
        input("Press Enter to SET ALT HOLD MODE...")
        future = self.alt_hold_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info('ALT HOLD SET.')
        
        input("Press Enter to DIVE TO 1.0 METER...")
        msg = Float64()
        msg.data = 1.0
        self.depth_pub.publish(msg)
        self.get_logger().info('Diving to 1.0m. Holding for 15 seconds...')
        
        # Hold and monitor
        start_time = time.time()
        while (time.time() - start_time) < 15.0:
            self.depth_pub.publish(msg)  # Keep publishing
            self.get_logger().info(f'Current depth: {self.current_depth:.2f}m')
            rclpy.spin_once(self, timeout_sec=0.5)
            
        input("Press Enter to SURFACE...")
        msg.data = 0.0
        self.depth_pub.publish(msg)
        self.get_logger().info('Surfacing...')
        
        start_time = time.time()
        while (time.time() - start_time) < 5.0:
            self.depth_pub.publish(msg)
            self.get_logger().info(f'Current depth: {self.current_depth:.2f}m')
            rclpy.spin_once(self, timeout_sec=0.5)
            
        self.get_logger().info('DISARMING...')
        req.data = False
        future = self.arm_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        self.get_logger().info('=== DEPTH POOL TEST COMPLETE ===')


def main(args=None):
    rclpy.init(args=args)
    node = DepthPoolTest()
    try:
        node.run_tests()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
