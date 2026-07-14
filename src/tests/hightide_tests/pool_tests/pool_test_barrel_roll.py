#!/usr/bin/env python3
"""
Pool Test: Barrel Roll (style points)

Arms, dives to a shallow test depth, holds it, then triggers the
/hightide/barrel_roll style maneuver in place — same service the mission's
finale calls. Standalone so you can dial in barrel_roll_pwm /
barrel_roll_duration_sec (mode_manager_node params) without running the
whole mission tree.

WARNING (same as the mission): the barrel roll switches to MANUAL mode and
destroys the FOG heading reference. This test disarms right after, so that's
fine here, but never run this before a heading-dependent maneuver.
"""

import time
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool, Trigger
from std_msgs.msg import Float64
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class BarrelRollPoolTest(Node):
    def __init__(self):
        super().__init__('pool_test_barrel_roll')

        self.arm_cli = self.create_client(SetBool, '/hightide/arm')
        self.alt_hold_cli = self.create_client(Trigger, '/hightide/set_alt_hold')
        self.barrel_roll_cli = self.create_client(Trigger, '/hightide/barrel_roll')
        self.depth_pub = self.create_publisher(Float64, '/hightide/target_depth', 10)

        self.current_depth = 0.0
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.create_subscription(Float64, '/mavros/global_position/rel_alt',
                                 self._depth_cb, sensor_qos)

        while not self.arm_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for arm service...')
        while not self.alt_hold_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for alt_hold service...')
        while not self.barrel_roll_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for barrel_roll service...')

    def _depth_cb(self, msg):
        self.current_depth = -msg.data  # positive = deeper

    def run_tests(self, test_depth_m=1.0, hold_sec=5.0):
        self.get_logger().info('=== STARTING BARREL ROLL POOL TEST ===')

        input("Ensure sub is in water with clearance to roll. Press Enter to ARM...")
        req = SetBool.Request()
        req.data = True
        future = self.arm_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info('Vehicle ARMED.')

        input("Press Enter to SET ALT HOLD MODE...")
        future = self.alt_hold_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info('ALT HOLD SET.')

        input(f"Press Enter to DIVE TO {test_depth_m:.1f}m...")
        msg = Float64()
        msg.data = test_depth_m
        start_time = time.time()
        while (time.time() - start_time) < hold_sec:
            self.depth_pub.publish(msg)
            self.get_logger().info(f'Current depth: {self.current_depth:.2f}m')
            rclpy.spin_once(self, timeout_sec=0.5)

        input("Sub should be holding depth in place. Press Enter to BARREL ROLL "
              "(this switches to MANUAL and WILL destroy the FOG heading reference)...")
        future = self.barrel_roll_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        self.get_logger().info(
            f'Barrel roll response: success={res.success}, msg="{res.message}"')

        self.get_logger().info('DISARMING...')
        req.data = False
        future = self.arm_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        self.get_logger().info('=== BARREL ROLL POOL TEST COMPLETE ===')


def main(args=None):
    rclpy.init(args=args)
    node = BarrelRollPoolTest()
    try:
        node.run_tests()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
