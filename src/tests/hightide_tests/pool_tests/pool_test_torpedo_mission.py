#!/usr/bin/env python3
"""
Pool Test: Torpedo Mission (standalone)

Runs Task 4 (Torpedoes) — hightide_mission.behaviors.torpedoes'
create_torpedoes_subtree() — by itself: find the board, approach, align on
the large hole and fire, align on the small hole and fire. Same behavior
tree the full mission ticks, just isolated so you can tune/debug it without
running the gate/slalom/octagon tasks first.

Requires MAVROS, the ZED wrapper, and the perception stack (yolo_pt_detector
+ target_tracker, so /hightide/tracked_targets is publishing) already
running — this script only arms/dives/ticks the torpedo subtree, it doesn't
bring up perception itself.
"""

import time
import rclpy
import py_trees
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_srvs.srv import SetBool, Trigger
from std_msgs.msg import Float64
from sensor_msgs.msg import Imu
from hightide_interfaces.msg import ThrusterCommand, DetectionArray

from hightide_mission.behaviors import blackboard_keys as bb
from hightide_mission.behaviors.torpedoes import create_torpedoes_subtree
from hightide_navigation import quaternion_to_yaw


class TorpedoMissionPoolTest(Node):
    def __init__(self):
        super().__init__('pool_test_torpedo_mission')

        # Same heading-hold gains the mission uses by default — tune with
        # `ros2 param set /pool_test_torpedo_mission yaw_hold_kp ...` if the
        # sub weaves while approaching/aligning.
        self.declare_parameter('yaw_hold_kp', 0.45)
        self.declare_parameter('yaw_hold_kd', 0.2)
        self.declare_parameter('yaw_hold_limit', 0.6)
        self.declare_parameter('yaw_hold_sign', -1.0)
        self.declare_parameter('test_depth_m', 1.0)
        self.declare_parameter('torpedoes_timeout_sec', 240.0)
        self.declare_parameter('tick_rate', 20.0)
        self.yaw_hold_kp = float(self.get_parameter('yaw_hold_kp').value)
        self.yaw_hold_kd = float(self.get_parameter('yaw_hold_kd').value)
        self.yaw_hold_limit = float(self.get_parameter('yaw_hold_limit').value)
        self.yaw_hold_sign = float(self.get_parameter('yaw_hold_sign').value)
        self.test_depth_m = float(self.get_parameter('test_depth_m').value)
        self.torpedoes_timeout = float(self.get_parameter('torpedoes_timeout_sec').value)
        self.tick_rate = float(self.get_parameter('tick_rate').value)

        self.arm_cli = self.create_client(SetBool, '/hightide/arm')
        self.alt_hold_cli = self.create_client(Trigger, '/hightide/set_alt_hold')
        self.cmd_pub = self.create_publisher(ThrusterCommand, '/hightide/cmd_vel', 10)
        self.depth_pub = self.create_publisher(Float64, '/hightide/target_depth', 10)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.current_depth = 0.0
        self.current_heading = None
        self.create_subscription(DetectionArray, '/hightide/tracked_targets',
                                 self._detections_cb, 10)
        self.create_subscription(Float64, '/mavros/global_position/rel_alt',
                                 self._depth_cb, sensor_qos)
        self.create_subscription(Imu, '/mavros/imu/data', self._imu_cb, sensor_qos)

        # Blackboard — same keys torpedoes.py's behaviors read/write.
        self.blackboard = py_trees.blackboard.Client(name='TorpedoPoolTest')
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.CURRENT_DEPTH, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.TARGET_DETECTION, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.TORPEDOES_FIRED, access=py_trees.common.Access.WRITE)
        self.blackboard.set(bb.ROS_NODE, self)
        self.blackboard.set(bb.DETECTIONS, None)
        self.blackboard.set(bb.CURRENT_DEPTH, 0.0)
        self.blackboard.set(bb.TORPEDOES_FIRED, 0)

        while not self.arm_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for arm service...')
        while not self.alt_hold_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for alt_hold service...')

    def _detections_cb(self, msg):
        self.blackboard.set(bb.DETECTIONS, msg)

    def _depth_cb(self, msg):
        depth = -msg.data
        self.current_depth = depth
        self.blackboard.set(bb.CURRENT_DEPTH, depth)

    def _imu_cb(self, msg):
        self.current_heading = quaternion_to_yaw(msg.orientation)

    def run_test(self):
        self.get_logger().info('=== STARTING TORPEDO MISSION POOL TEST ===')

        input("Ensure sub is in water, torpedo board in the general area. "
              "Press Enter to ARM...")
        req = SetBool.Request()
        req.data = True
        future = self.arm_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info('Vehicle ARMED.')

        input("Press Enter to SET ALT HOLD MODE...")
        future = self.alt_hold_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info('ALT HOLD SET.')

        input(f"Press Enter to DIVE TO {self.test_depth_m:.1f}m and start the "
              "torpedo task...")
        msg = Float64()
        msg.data = self.test_depth_m
        self.depth_pub.publish(msg)
        settle_start = time.time()
        while (time.time() - settle_start) < 3.0:
            self.depth_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(f'Holding {self.test_depth_m:.1f}m — depth now '
                               f'{self.current_depth:.2f}m. Ticking torpedo task...')

        tree = py_trees.trees.BehaviourTree(
            root=create_torpedoes_subtree(total_timeout=self.torpedoes_timeout))

        period = 1.0 / self.tick_rate
        start = time.time()
        while rclpy.ok():
            self.depth_pub.publish(msg)  # keep holding test depth throughout
            tree.tick()
            rclpy.spin_once(self, timeout_sec=0.0)
            status = tree.root.status
            if status in (py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE):
                self.get_logger().info(f'Torpedo task finished: {status}')
                break
            time.sleep(period)

        fired = self.blackboard.get(bb.TORPEDOES_FIRED)
        self.get_logger().info(
            f'Torpedoes fired: {fired}/2. Elapsed: {time.time() - start:.1f}s')

        self.cmd_pub.publish(ThrusterCommand())  # stop
        input("Press Enter to SURFACE and DISARM...")
        msg.data = 0.0
        surface_start = time.time()
        while (time.time() - surface_start) < 10.0:
            self.depth_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.5)

        req.data = False
        future = self.arm_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        self.get_logger().info('=== TORPEDO MISSION POOL TEST COMPLETE ===')


def main(args=None):
    rclpy.init(args=args)
    node = TorpedoMissionPoolTest()
    try:
        node.run_test()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.cmd_pub.publish(ThrusterCommand())
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
