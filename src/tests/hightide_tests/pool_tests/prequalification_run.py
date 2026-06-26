#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from mavros_msgs.msg import OverrideRCIn
from mavros_msgs.srv import CommandBool, SetMode
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math
import time


class ArcReturnManeuver(Node):
    def __init__(self):
        super().__init__('arc_return_maneuver')
        self.get_logger().info("Waiting 30 seconds before arming and starting...")
        time.sleep(20)

        # MAVROS
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

        self.arm_client.wait_for_service()
        self.mode_client.wait_for_service()

        # IMU
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/mavros/imu/data',
            self.imu_cb,
            qos
        )

        # RC
        self.rc_pub = self.create_publisher(
            OverrideRCIn,
            '/mavros/rc/override',
            10
        )

        # state
        self.current_heading = 0.0
        self.initial_heading = None
        self.target_heading = None

        self.state = "FORWARD1"
        self.state_start_time = time.time()

        self.kp = 0.8

        self.set_mode("ALT_HOLD")
        self.arm(True)
        

        self.timer = self.create_timer(0.1, self.control_loop)

    # ---------------- MAVROS ----------------

    def set_mode(self, mode):
        req = SetMode.Request()
        req.custom_mode = mode
        future = self.mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

    def arm(self, value):
        req = CommandBool.Request()
        req.value = value
        future = self.arm_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

    # ---------------- IMU ----------------

    def imu_cb(self, msg):
        q = msg.orientation

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.current_heading = math.degrees(yaw) % 360.0

        if self.initial_heading is None:
            self.initial_heading = self.current_heading
            self.target_heading = self.current_heading

            self.get_logger().info(
                f"Initial heading: {self.initial_heading:.1f}°"
            )

    # ---------------- HELPERS ----------------

    def heading_error(self, target, current):
        err = target - current
        if err > 180:
            err -= 360
        if err < -180:
            err += 360
        return err

    # ---------------- CONTROL ----------------

    def control_loop(self):
        now = time.time()
        elapsed = now - self.state_start_time

        rc = OverrideRCIn()
        rc.channels = [65535] * 18

        rc.channels[0] = 1500
        rc.channels[1] = 1500
        rc.channels[2] = 1500
        rc.channels[5] = 1500

        yaw_pwm = 1500

        # heading hold
        if self.target_heading is not None:
            err = self.heading_error(self.target_heading, self.current_heading)
            yaw_pwm = int(1500 + self.kp * err)
            yaw_pwm = max(1400, min(1600, yaw_pwm))

        # ---------------- STATES ----------------

        if self.state == "FORWARD1":
            rc.channels[3] = yaw_pwm
            rc.channels[4] = 1600

            if elapsed >= 30.0:
                self.state = "TURN1A"
                self.state_start_time = now
                self.get_logger().info("TURN1A start")

        elif self.state == "TURN1A":
            rc.channels[3] = 1400      # CCW yaw
            rc.channels[4] = 1525      # slight surge

            if elapsed >= 2.6:
                self.state = "STRAIGHT_MID"
                self.state_start_time = now
                self.target_heading = self.current_heading

                self.get_logger().info(
                    f"TURN1A done, heading locked at "
                    f"{self.target_heading:.1f}°"
                )

        elif self.state == "STRAIGHT_MID":
            rc.channels[3] = yaw_pwm
            rc.channels[4] = 1600

            if elapsed >= 3.0:
                self.state = "TURN1B"
                self.state_start_time = now
                self.get_logger().info("TURN1B start")

        elif self.state == "TURN1B":
            rc.channels[3] = 1400      # CCW yaw
            rc.channels[4] = 1525      # slight surge

            if elapsed >= 2.9:
                self.state = "FORWARD2"
                self.state_start_time = now

                self.target_heading = self.current_heading

                self.get_logger().info(
                    f"TURN1B done, heading locked at "
                    f"{self.target_heading:.1f}°"
                )

        elif self.state == "FORWARD2":
            rc.channels[3] = yaw_pwm
            rc.channels[4] = 1600

            if elapsed >= 21.0:
                self.state = "TURN2"
                self.state_start_time = now

                self.get_logger().info("TURN2 start")
        elif self.state == "TURN2":
            # same CCW yaw as first turn
            rc.channels[3] = 1400
            rc.channels[4] = 1525

            if elapsed >= 1.35:
                self.state = "FORWARD3"
                self.state_start_time = now

                self.target_heading = self.current_heading

                self.get_logger().info(
                    f"TURN2 done, heading locked at "
                    f"{self.target_heading:.1f}°"
                )
        elif self.state == "FORWARD3":
            rc.channels[3] = yaw_pwm
            rc.channels[4] = 1600

            if elapsed >= 30.0:
                self.state = "STOP"

        else:
            rc.channels[3] = 1500
            rc.channels[4] = 1500

        self.rc_pub.publish(rc)


def main():
    rclpy.init()
    node = ArcReturnManeuver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
