#!/usr/bin/env python3
"""
RC Override Node — Translates ThrusterCommand into MAVROS RC Override PWM values.

ArduSub RC Channel Mapping (0-indexed in the array):
  Channel 1 (idx 0): Pitch
  Channel 2 (idx 1): Roll
  Channel 3 (idx 2): Throttle / Heave (depth rate in Alt Hold)
  Channel 4 (idx 3): Yaw
  Channel 5 (idx 4): Forward / Surge
  Channel 6 (idx 5): Lateral / Sway

PWM Conversion: pwm = neutral + int(normalized * scale)
  -1.0 → 1100   0.0 → 1500   +1.0 → 1900
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from mavros_msgs.msg import OverrideRCIn
from std_msgs.msg import Int32, Bool
from hightide_interfaces.msg import ThrusterCommand


# ArduSub channel indices (0-based)
CH_PITCH = 0
CH_ROLL = 1
CH_THROTTLE = 2
CH_YAW = 3
CH_FORWARD = 4
CH_LATERAL = 5

CHAN_NOCHANGE = 65535
NUM_CHANNELS = 18


class RCOverrideNode(Node):
    """Maps normalized 6-DOF ThrusterCommand to ArduSub RC Override PWM."""

    def __init__(self):
        super().__init__('rc_override_node')

        # Declare parameters
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('max_pwm', 1900)
        self.declare_parameter('min_pwm', 1100)
        self.declare_parameter('neutral_pwm', 1500)
        self.declare_parameter('deadzone', 0.05)
        self.declare_parameter('timeout_sec', 0.5)
        # Safety cap: how long a hold can suppress our publishing before we
        # auto-release, in case whoever set the hold (mode_manager's barrel
        # roll) crashes without clearing it. The barrel roll is only ~3 s.
        self.declare_parameter('hold_max_sec', 15.0)

        self.publish_rate = self.get_parameter('publish_rate').value
        self.max_pwm = self.get_parameter('max_pwm').value
        self.min_pwm = self.get_parameter('min_pwm').value
        self.neutral_pwm = self.get_parameter('neutral_pwm').value
        self.deadzone = self.get_parameter('deadzone').value
        self.timeout_sec = self.get_parameter('timeout_sec').value
        self.hold_max_sec = self.get_parameter('hold_max_sec').value

        self.pwm_scale = (self.max_pwm - self.neutral_pwm)  # 400

        # Latest command state
        self.last_cmd = ThrusterCommand()
        self.last_cmd_time = self.get_clock().now()
        self.depth_pwm = self.neutral_pwm  # From depth controller
        self.depth_pwm_received = False

        # When another node needs exclusive RC-override control (the barrel
        # roll fires the roll channel directly and can't share this topic with
        # us — we'd stomp it back to neutral every tick), it publishes True on
        # /hightide/rc_override_hold and we stop publishing until it clears it.
        self.hold = False
        self.hold_start = None

        # Subscribers
        self.cmd_sub = self.create_subscription(
            ThrusterCommand, '/hightide/cmd_vel',
            self._cmd_vel_callback, 10)

        self.depth_pwm_sub = self.create_subscription(
            Int32, '/hightide/depth_pwm',
            self._depth_pwm_callback, 10)

        self.hold_sub = self.create_subscription(
            Bool, '/hightide/rc_override_hold',
            self._hold_callback, 10)

        # Publisher — MAVROS RC Override
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.rc_pub = self.create_publisher(OverrideRCIn, '/mavros/rc/override', qos)

        # Timer for continuous publishing
        period = 1.0 / self.publish_rate
        self.timer = self.create_timer(period, self._publish_rc_override)

        self.get_logger().info(
            f'RC Override Node started — {self.publish_rate} Hz, '
            f'PWM range [{self.min_pwm}, {self.max_pwm}], neutral {self.neutral_pwm}')

    def _cmd_vel_callback(self, msg: ThrusterCommand):
        """Store latest thruster command."""
        self.last_cmd = msg
        self.last_cmd_time = self.get_clock().now()

    def _depth_pwm_callback(self, msg: Int32):
        """Receive depth controller PWM for throttle channel."""
        self.depth_pwm = msg.data
        self.depth_pwm_received = True

    def _hold_callback(self, msg: Bool):
        """Yield the RC-override topic to another node (e.g. the barrel roll)."""
        if msg.data and not self.hold:
            self.hold_start = self.get_clock().now()
            self.get_logger().info('RC override HELD — yielding topic (barrel roll?)')
        elif not msg.data and self.hold:
            self.get_logger().info('RC override released — resuming control')
        self.hold = msg.data

    def _normalize_to_pwm(self, value: float) -> int:
        """Convert normalized [-1.0, 1.0] value to PWM [min_pwm, max_pwm]."""
        if abs(value) < self.deadzone:
            return self.neutral_pwm
        clamped = max(-1.0, min(1.0, value))
        pwm = self.neutral_pwm + int(clamped * self.pwm_scale)
        return max(self.min_pwm, min(self.max_pwm, pwm))

    def _publish_rc_override(self):
        """Publish RC Override message at fixed rate."""
        now = self.get_clock().now()

        # Held: another node owns the RC-override topic right now (barrel roll).
        # Stay silent so we don't fight it — but auto-release after hold_max_sec
        # so a crashed holder can't leave the vehicle uncommandable.
        if self.hold:
            if self.hold_start is not None and \
                    (now - self.hold_start).nanoseconds / 1e9 > self.hold_max_sec:
                self.get_logger().warn(
                    f'RC override hold exceeded {self.hold_max_sec}s — force-releasing')
                self.hold = False
            else:
                return

        dt = (now - self.last_cmd_time).nanoseconds / 1e9

        # Safety: if no command received within timeout, send neutral
        if dt > self.timeout_sec:
            channels = [CHAN_NOCHANGE] * NUM_CHANNELS
            channels[CH_PITCH] = self.neutral_pwm
            channels[CH_ROLL] = self.neutral_pwm
            channels[CH_YAW] = self.neutral_pwm
            channels[CH_FORWARD] = self.neutral_pwm
            channels[CH_LATERAL] = self.neutral_pwm
            # Keep depth controller active even during timeout
            if self.depth_pwm_received:
                channels[CH_THROTTLE] = max(self.min_pwm,
                                            min(self.max_pwm, self.depth_pwm))
            else:
                channels[CH_THROTTLE] = self.neutral_pwm
        else:
            cmd = self.last_cmd
            channels = [CHAN_NOCHANGE] * NUM_CHANNELS
            channels[CH_PITCH] = self._normalize_to_pwm(cmd.pitch)
            channels[CH_ROLL] = self._normalize_to_pwm(cmd.roll)
            channels[CH_YAW] = self._normalize_to_pwm(cmd.yaw)
            channels[CH_FORWARD] = self._normalize_to_pwm(cmd.surge)
            channels[CH_LATERAL] = self._normalize_to_pwm(cmd.sway)

            # Throttle channel: use depth controller PWM if available,
            # otherwise use heave from ThrusterCommand
            if self.depth_pwm_received:
                channels[CH_THROTTLE] = max(self.min_pwm,
                                            min(self.max_pwm, self.depth_pwm))
            elif abs(cmd.heave) > self.deadzone:
                channels[CH_THROTTLE] = self._normalize_to_pwm(cmd.heave)
            else:
                channels[CH_THROTTLE] = self.neutral_pwm

        msg = OverrideRCIn()
        msg.channels = channels
        self.rc_pub.publish(msg)

        self.get_logger().debug(
            f'RC: P={channels[CH_PITCH]} R={channels[CH_ROLL]} '
            f'T={channels[CH_THROTTLE]} Y={channels[CH_YAW]} '
            f'F={channels[CH_FORWARD]} L={channels[CH_LATERAL]}')


def main(args=None):
    rclpy.init(args=args)
    node = RCOverrideNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
