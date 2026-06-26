#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class TestOdomSubscriber(Node):

    def __init__(self):
        super().__init__('test_odom_subscriber')
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Initialize dictionary to track min and max bounds for tracking fields
        self.bounds = {
            'pos_x': {'min': float('inf'), 'max': float('-inf')},
            'pos_y': {'min': float('inf'), 'max': float('-inf')},
            'pos_z': {'min': float('inf'), 'max': float('-inf')},
            'ori_x': {'min': float('inf'), 'max': float('-inf')},
            'ori_y': {'min': float('inf'), 'max': float('-inf')},
            'ori_z': {'min': float('inf'), 'max': float('-inf')},
            'ori_w': {'min': float('inf'), 'max': float('-inf')},
            'vel_x': {'min': float('inf'), 'max': float('-inf')},
            'vel_yaw': {'min': float('inf'), 'max': float('-inf')},
        }

        self.subscription = self.create_subscription(
            Odometry,
            '/mavros/local_position/odom',
            self.odom_callback,
            sensor_qos
        )
        self.get_logger().info("Test Odometry Tracking Subscriber Started. Listening on /mavros/local_position/odom...")

    def update_bounds(self, key, value):
        if value < self.bounds[key]['min']:
            self.bounds[key]['min'] = value
        if value > self.bounds[key]['max']:
            self.bounds[key]['max'] = value

    def odom_callback(self, msg: Odometry):
        # 1. Extract Current Values
        pos_x = msg.pose.pose.position.x
        pos_y = msg.pose.pose.position.y
        pos_z = msg.pose.pose.position.z

        ori_x = msg.pose.pose.orientation.x
        ori_y = msg.pose.pose.orientation.y
        ori_z = msg.pose.pose.orientation.z
        ori_w = msg.pose.pose.orientation.w

        vel_lin_x = msg.twist.twist.linear.x
        vel_ang_z = msg.twist.twist.angular.z

        # 2. Update Limits
        self.update_bounds('pos_x', pos_x)
        self.update_bounds('pos_y', pos_y)
        self.update_bounds('pos_z', pos_z)
        self.update_bounds('ori_x', ori_x)
        self.update_bounds('ori_y', ori_y)
        self.update_bounds('ori_z', ori_z)
        self.update_bounds('ori_w', ori_w)
        self.update_bounds('vel_x', vel_lin_x)
        self.update_bounds('vel_yaw', vel_ang_z)

        
        deltas = {k: self.bounds[k]['max'] - self.bounds[k]['min'] for k in self.bounds}

        # 3. Print values with oscillation delta (max - min)
        log_msg = (
            f"\n======================================================== TELEMETRY OSCILLATION DUMP ========================================================"
            f"\n[Metric]          |  Pos X    |  Pos Y    |  Pos Z    |  Ori X    |  Ori Y    |  Ori Z    |  Ori W    |  Lin VelX |  Ang VelZ |"
            f"\n--------------------------------------------------------------------------------------------------------------------------------------------"
            f"\nCurrent           | {pos_x:9.4f} | {pos_y:9.4f} | {pos_z:9.4f} | {ori_x:9.4f} | {ori_y:9.4f} | {ori_z:9.4f} | {ori_w:9.4f} | {vel_lin_x:9.4f} | {vel_ang_z:9.4f} |"
            f"\nMin               | {self.bounds['pos_x']['min']:9.4f} | {self.bounds['pos_y']['min']:9.4f} | {self.bounds['pos_z']['min']:9.4f} | {self.bounds['ori_x']['min']:9.4f} | {self.bounds['ori_y']['min']:9.4f} | {self.bounds['ori_z']['min']:9.4f} | {self.bounds['ori_w']['min']:9.4f} | {self.bounds['vel_x']['min']:9.4f} | {self.bounds['vel_yaw']['min']:9.4f} |"
            f"\nMax               | {self.bounds['pos_x']['max']:9.4f} | {self.bounds['pos_y']['max']:9.4f} | {self.bounds['pos_z']['max']:9.4f} | {self.bounds['ori_x']['max']:9.4f} | {self.bounds['ori_y']['max']:9.4f} | {self.bounds['ori_z']['max']:9.4f} | {self.bounds['ori_w']['max']:9.4f} | {self.bounds['vel_x']['max']:9.4f} | {self.bounds['vel_yaw']['max']:9.4f} |"
            f"\nOscillation Delta | {deltas['pos_x']:9.4f} | {deltas['pos_y']:9.4f} | {deltas['pos_z']:9.4f} | {deltas['ori_x']:9.4f} | {deltas['ori_y']:9.4f} | {deltas['ori_z']:9.4f} | {deltas['ori_w']:9.4f} | {deltas['vel_x']:9.4f} | {deltas['vel_yaw']:9.4f} |"
        )
        self.get_logger().info(log_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TestOdomSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down tracking script.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()