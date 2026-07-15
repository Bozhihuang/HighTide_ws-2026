#!/usr/bin/env python3
"""
Depth Controller Node — PID depth hold via throttle channel.

In Alt Hold mode, throttle channel (ch3) is a depth rate command:
  1500 = hold current depth
  >1500 = ascend (shallower)
  <1500 = descend (deeper)

This node takes a depth setpoint and PID-controls the throttle to reach it.
Publishes PWM value to /hightide/depth_pwm which rc_override_node picks up.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Int32
from mavros_msgs.msg import Altitude
from hightide_interfaces.srv import SetDepth
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class DepthControllerNode(Node):
    """PID depth controller outputting throttle PWM for Alt Hold mode."""

    def __init__(self):
        super().__init__('depth_controller_node')

        # PID parameters
        self.declare_parameter('kp', 300.0)
        self.declare_parameter('ki', 35.0)
        self.declare_parameter('kd', 0.0)
        self.declare_parameter('max_output', 600)
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('depth_tolerance', 0.1)
        self.declare_parameter('integral_max', 100.0)
        # Which MAVROS topic actually carries usable depth on YOUR rig. On ArduSub
        # the standard is global_position/rel_alt (baro/pressure-derived), but if it
        # reads zero/garbage underwater, switch to 'altitude' (needs the `altitude`
        # plugin whitelisted in mavros.yaml). Swap live via params.yaml — no rebuild.
        #   'rel_alt'  -> std_msgs/Float64      /mavros/global_position/rel_alt
        #   'altitude' -> mavros_msgs/Altitude  /mavros/altitude   (uses .relative)
        self.declare_parameter('depth_source', 'rel_alt')
        # --- Surface tare + plausibility gate -------------------------------
        # rel_alt is EKF-altitude-minus-HOME-altitude, NOT water-pressure depth.
        # If the home/EKF origin carries a real-world elevation (BlueOS "Set EKF
        # origin" stores your city's altitude), rel_alt sits at a huge constant
        # offset (observed: ~-630 m) while BlueOS's own depth display — surface-
        # referenced water pressure — stays correct. auto_tare kills any such
        # constant offset: at boot (vehicle on deck ≈ surface) the median of the
        # first tare_duration_sec of readings becomes the surface reference and
        # depth = -(rel_alt - ref). If the node ever restarts while SUBMERGED,
        # set auto_tare false and provide surface_offset_m manually instead.
        self.declare_parameter('auto_tare', True)
        self.declare_parameter('tare_duration_sec', 5.0)
        self.declare_parameter('surface_offset_m', 0.0)   # used when auto_tare false
        # Readings that map outside (-1.5 .. max_plausible_depth_m) after taring
        # are REJECTED (not fed to the PID, not republished): a drifting internal
        # cabin baro feeding the EKF (enclosure heating = fake hundreds of m) is
        # an FCU config problem no ROS math can fix — reject loudly instead of
        # commanding full ascend against garbage. Jumps faster than
        # max_depth_rate_mps are rejected too (3 s of persistent rejects resyncs,
        # in case the EKF origin legitimately re-zeroed mid-run).
        self.declare_parameter('max_plausible_depth_m', 10.0)
        self.declare_parameter('max_depth_rate_mps', 2.0)

        self.kp = self.get_parameter('kp').value
        self.ki = self.get_parameter('ki').value
        self.kd = self.get_parameter('kd').value
        self.max_output = self.get_parameter('max_output').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.depth_tolerance = self.get_parameter('depth_tolerance').value
        self.integral_max = self.get_parameter('integral_max').value
        self.depth_source = self.get_parameter('depth_source').value
        self.auto_tare = bool(self.get_parameter('auto_tare').value)
        self.tare_duration = float(self.get_parameter('tare_duration_sec').value)
        self.surface_offset = float(self.get_parameter('surface_offset_m').value)
        self.max_plausible = float(self.get_parameter('max_plausible_depth_m').value)
        self.max_rate = float(self.get_parameter('max_depth_rate_mps').value)

        # State
        self.target_depth = None  # None = no target, hold current
        self.current_depth = 0.0
        self.depth_received = False
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None
        self.current_depth = 0.0
        self._diag_tick = 0        # throttles the "why am I holding 1500" log
        # Tare / gate state
        self._tare_samples = []
        self._tare_start = None
        self._tare_ref = None if self.auto_tare else self.surface_offset
        self._last_accept_t = None
        self._reject_since = None
        self._last_warn_t = 0.0

        # If gains never loaded (ran without --params-file / wrong node name), the PID
        # output is 0 and pwm sticks at 1500 — indistinguishable from "no feedback"
        # on the wire. Say so loudly at startup so it isn't mistaken for a sensor issue.
        if self.kp == 0.0 and self.ki == 0.0 and self.kd == 0.0:
            self.get_logger().error(
                'ALL depth gains are 0 (kp=ki=kd=0) — output will stay 1500 no matter '
                'what. params.yaml likely did not load. Check: '
                'ros2 param get /depth_controller_node kp')

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Depth feedback — pick the source that actually works on this vehicle.
        if self.depth_source == 'altitude':
            self.depth_sub = self.create_subscription(
                Altitude, '/mavros/altitude',
                self._altitude_callback, sensor_qos)
            self.get_logger().info('Depth source: /mavros/altitude (.relative)')
        else:
            self.depth_sub = self.create_subscription(
                Float64, '/mavros/global_position/rel_alt',
                self._depth_callback, sensor_qos)
            self.get_logger().info('Depth source: /mavros/global_position/rel_alt')

        self.target_sub = self.create_subscription(
            Float64, '/hightide/target_depth',
            self._target_depth_callback, 10)

        # Service to set depth
        self.set_depth_srv = self.create_service(
            SetDepth, '/hightide/set_depth', self._set_depth_service)

        # Publisher — PWM for throttle channel
        self.pwm_pub = self.create_publisher(Int32, '/hightide/depth_pwm', 10)
        # Calibrated depth (positive = deeper, surface-tared, garbage-gated).
        # THE depth the rest of the stack should consume — mission_node prefers
        # this over raw rel_alt, so the -630m home-offset class of garbage never
        # reaches mission logic.
        self.depth_out_pub = self.create_publisher(
            Float64, '/hightide/current_depth', 10)

        # Timer
        period = 1.0 / self.publish_rate
        self.timer = self.create_timer(period, self._control_loop)

        self.get_logger().info(
            f'Depth Controller started — Kp={self.kp} Ki={self.ki} Kd={self.kd}')

    def _depth_callback(self, msg: Float64):
        """rel_alt from MAVROS: negative = below the HOME reference (not the
        surface — see the tare comment in __init__). Tared + gated in _ingest."""
        self._ingest(msg.data)

    def _altitude_callback(self, msg: Altitude):
        """Depth from mavros_msgs/Altitude (.relative). Same tare + gate path."""
        self._ingest(msg.relative)

    def _warn_throttled(self, text):
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._last_warn_t >= 2.0:
            self._last_warn_t = now
            self.get_logger().warn(text)

    def _ingest(self, rel_alt):
        """Turn a raw home-relative altitude into trusted positive-down depth.

        Pipeline: (1) auto-tare — median of the first tare_duration_sec of
        readings becomes the surface reference, so any constant home/EKF-origin
        altitude offset (the -630 m class of garbage) cancels exactly;
        (2) plausibility gate — post-tare depths outside (-1.5, max_plausible)
        or jumping faster than max_rate are rejected and never reach the PID or
        /hightide/current_depth.

        Rate-gate self-healing: rate is measured against the LAST ACCEPTED
        sample, so a single glitch is rejected outright, while a SUSTAINED step
        is re-accepted once enough time has passed that the sub could
        physically have moved there (|jump| / max_rate seconds) — the gate
        converges to reality at the vehicle's own speed limit. The 3 s
        persistent-reject resync below is a backstop for pathological cases
        (oscillating garbage) so depth can never be locked out forever.
        """
        now = self.get_clock().now().nanoseconds / 1e9

        # --- Tare phase ---
        if self._tare_ref is None:
            if self._tare_start is None:
                self._tare_start = now
                self.get_logger().info(
                    f'Depth tare: sampling surface reference for '
                    f'{self.tare_duration:.0f}s (raw rel_alt={rel_alt:+.1f}m)')
            self._tare_samples.append(rel_alt)
            if (now - self._tare_start) < self.tare_duration:
                return  # not trusted yet — controller holds 1500 meanwhile
            s = sorted(self._tare_samples)
            self._tare_ref = s[len(s) // 2]
            if abs(self._tare_ref) > 2.0:
                self.get_logger().warn(
                    f'Depth tare: surface reference {self._tare_ref:+.1f}m — '
                    'rel_alt has a big home/EKF-origin offset. Tared out here, '
                    'but check BARO/EKF origin config on the FCU.')
            else:
                self.get_logger().info(
                    f'Depth tare complete: surface reference {self._tare_ref:+.2f}m')

        depth = -(rel_alt - self._tare_ref)   # positive = deeper

        # --- Plausibility gate ---
        if not (-1.5 <= depth <= self.max_plausible):
            self._warn_throttled(
                f'REJECTING depth {depth:+.1f}m (raw rel_alt {rel_alt:+.1f}m) — '
                f'outside (-1.5, {self.max_plausible:.0f})m. EKF altitude is '
                'garbage (wrong baro / origin reset?) — fix on the FCU side.')
            return
        if self._last_accept_t is not None:
            dt = max(now - self._last_accept_t, 0.02)
            rate = abs(depth - self.current_depth) / dt
            if rate > self.max_rate:
                if self._reject_since is None:
                    self._reject_since = now
                if (now - self._reject_since) < 3.0:
                    self._warn_throttled(
                        f'REJECTING depth {depth:+.2f}m — jumped '
                        f'{rate:.1f}m/s from {self.current_depth:+.2f}m')
                    return
                self.get_logger().warn(
                    f'Depth jump persisted 3s — resyncing to {depth:+.2f}m')
        self._reject_since = None
        self._last_accept_t = now

        self.current_depth = depth
        self.depth_received = True
        out = Float64()
        out.data = depth
        self.depth_out_pub.publish(out)

    def _target_depth_callback(self, msg: Float64):
        """Receive target depth (positive = deeper in meters)."""
        self.target_depth = msg.data
        self.integral = 0.0  # Reset integral on new target
        self.prev_error = 0.0
        self.get_logger().info(f'New depth target: {self.target_depth:.2f} m')

    def _set_depth_service(self, request, response):
        """Service handler for setting target depth."""
        self.target_depth = request.target_depth_m
        self.integral = 0.0
        self.prev_error = 0.0
        response.success = True
        response.message = f'Target depth set to {self.target_depth:.2f} m'
        self.get_logger().info(response.message)
        return response

    def _control_loop(self):
        """PID control loop — compute throttle PWM from depth error."""
        now = self.get_clock().now()
        msg = Int32()
        self._diag_tick += 1
        diag = (self._diag_tick % 40 == 0)   # ~2 s at 20 Hz

        # No target or no depth reading: output neutral (hold current depth).
        # This is the #1 reason the PWM "never changes" — surface exactly which
        # input is missing so it isn't mistaken for a dead controller.
        if self.target_depth is None or not self.depth_received:
            msg.data = 1500
            self.pwm_pub.publish(msg)
            self.last_time = now
            if diag:
                reasons = []
                if self.target_depth is None:
                    reasons.append('NO TARGET (publish /hightide/target_depth or call '
                                   '/hightide/set_depth)')
                if not self.depth_received:
                    reasons.append(f"NO DEPTH FEEDBACK on source '{self.depth_source}' "
                                   f"(check that topic is publishing)")
                self.get_logger().warn(f'Holding 1500 — {"; ".join(reasons)}')
            return

        # Compute dt
        if self.last_time is None:
            self.last_time = now
            msg.data = 1500
            self.pwm_pub.publish(msg)
            return

        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 0.0:
            return
        self.last_time = now

        # Error: positive = need to go deeper
        error = self.target_depth - self.current_depth

        # Within tolerance: hold
        if abs(error) < self.depth_tolerance:
            msg.data = 1500
            self.integral = 0.0
            self.pwm_pub.publish(msg)
            if diag:
                self.get_logger().info(
                    f'Holding 1500 — AT TARGET (target={self.target_depth:.2f} '
                    f'current={self.current_depth:.2f}, |err|<{self.depth_tolerance})')
            return

        # PID
        self.integral += error * dt
        self.integral = max(-self.integral_max, min(self.integral_max, self.integral))

        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        # Clamp output
        output = max(-self.max_output, min(self.max_output, output))

        # Convert to PWM:
        # In ArduSub Alt Hold: PWM < 1500 = descend (go deeper), PWM > 1500 = ascend
        # Our error: positive = need to go deeper → we want PWM < 1500
        # So: pwm = 1500 - output (positive error → lower PWM → descend)
        pwm = 1500 - int(output)
        pwm = max(1100, min(1900, pwm))

        msg.data = pwm
        self.pwm_pub.publish(msg)

        if diag:
            self.get_logger().info(
                f'ACTIVE: target={self.target_depth:.2f} current={self.current_depth:.2f} '
                f'error={error:+.2f} output={output:+.1f} -> pwm={pwm}')

        self.get_logger().debug(
            f'Depth: target={self.target_depth:.2f} current={self.current_depth:.2f} '
            f'error={error:.2f} pwm={pwm}')


def main(args=None):
    rclpy.init(args=args)
    node = DepthControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()