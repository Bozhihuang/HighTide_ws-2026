#!/usr/bin/env python3
"""
Pool Test: Thrusters

Tests individual normalized ThrusterCommands to verify motor mapping
through the RC Override node. Ensure the sub moves in the expected directions.
"""

import time
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from hightide_interfaces.msg import ThrusterCommand


class ThrusterPoolTest(Node):
    def __init__(self):
        super().__init__('pool_test_thrusters')
        
        self.arm_cli = self.create_client(SetBool, '/hightide/arm')
        self.cmd_pub = self.create_publisher(ThrusterCommand, '/hightide/cmd_vel', 10)
        
        while not self.arm_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for arm service...')

    def run_tests(self):
        self.get_logger().info('=== STARTING THRUSTER POOL TEST ===')
        
        input("Ensure sub is in water. Press Enter to ARM...")
        req = SetBool.Request()
        req.data = True
        future = self.arm_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info('Vehicle ARMED (Manual mode expected).')
        
        # Test sequences
        self._test_axis('SURGE (Forward)', surge=0.3)
        self._test_axis('SURGE (Reverse)', surge=-0.3)
        
        self._test_axis('SWAY (Right)', sway=0.3)
        self._test_axis('SWAY (Left)', sway=-0.3)
        
        self._test_axis('YAW (Clockwise)', yaw=0.3)
        self._test_axis('YAW (Counter-Clockwise)', yaw=-0.3)
        
        self._test_axis('HEAVE (Down)', heave=0.3)
        self._test_axis('HEAVE (Up)', heave=-0.3)
        
        self.get_logger().info('DISARMING...')
        req.data = False
        future = self.arm_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        self.get_logger().info('=== THRUSTER POOL TEST COMPLETE ===')

    def _test_axis(self, name, surge=0.0, sway=0.0, heave=0.0, yaw=0.0):
        input(f"Press Enter to test {name} for 3 seconds...")
        cmd = ThrusterCommand()
        cmd.surge = surge
        cmd.sway = sway
        cmd.heave = heave
        cmd.yaw = yaw
        
        self.get_logger().info(f'Applying {name} thrust...')
        start_time = time.time()
        while (time.time() - start_time) < 3.0:
            cmd.header.stamp = self.get_clock().now().to_msg()
            self.cmd_pub.publish(cmd)
            time.sleep(0.1)
            
        # Stop
        self.cmd_pub.publish(ThrusterCommand())
        self.get_logger().info('Stopped.')


def main(args=None):
    rclpy.init(args=args)
    node = ThrusterPoolTest()
    try:
        node.run_tests()
    except KeyboardInterrupt:
        pass
    finally:
        # Emergency stop
        node.cmd_pub.publish(ThrusterCommand())
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
