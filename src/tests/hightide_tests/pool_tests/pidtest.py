#!/usr/bin/env python3
"""
Single-axis PID tuning harness with companion heading-hold stability.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import SetParametersResult
from std_srvs.srv import SetBool, Trigger
from std_msgs.msg import Float64, Float64MultiArray, Empty
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from hightide_interfaces.msg import ThrusterCommand
from hightide_navigation import PIDController, normalize_angle, quaternion_to_yaw

VALID_AXES = ('surge', 'sway', 'yaw')


class PidTuner(Node):
    def __init__(self):
        super().__init__('pid_tuner')

        # ---- Parameters (all live-tunable except axis) ----
        self.declare_parameter('axis', 'surge')       # surge | sway | yaw
        self.declare_parameter('kp', 0.0)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.0)
        self.declare_parameter('max_output', 0.6)      # matches navigator max_speed
        self.declare_parameter('yaw_command_sign', -1.0)  # only used for yaw
        self.declare_parameter('auto_arm', True)       # arm + alt-hold on start, disarm on exit
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('heading_source', 'imu')  # 'imu' | 'zed'
        self.declare_parameter('frame_offset_deg', 0.0)
        self.declare_parameter('auto_calibrate', True)
        self.declare_parameter('calibrate_samples', 20)  # ~1 s at 20 Hz

        self.axis = self.get_parameter('axis').value
        if self.axis not in VALID_AXES:
            raise ValueError(f"axis must be one of {VALID_AXES}, got '{self.axis}'")
        self.max_output = self.get_parameter('max_output').value
        self.yaw_command_sign = self.get_parameter('yaw_command_sign').value
        self.auto_arm = self.get_parameter('auto_arm').value
        self.heading_source = self.get_parameter('heading_source').value
        self.frame_offset = math.radians(self.get_parameter('frame_offset_deg').value)
        self.auto_calibrate = self.get_parameter('auto_calibrate').value
        self._cal_samples = self.get_parameter('calibrate_samples').value
        rate = self.get_parameter('rate_hz').value

        self._needs_cal = self.auto_calibrate and self.heading_source == 'imu'
        self._calibrated = not self._needs_cal
        self._cal_sin = 0.0
        self._cal_cos = 0.0
        self._cal_n = 0

        # Primary tuning controller
        self.pid = PIDController(
            self.get_parameter('kp').value,
            self.get_parameter('ki').value,
            self.get_parameter('kd').value,
            output_min=-self.max_output, output_max=self.max_output)

        # ---- Companion Heading Hold Loop ----
        # Uses your customized stable yaw values to freeze yaw during surge/sway runs
        self.companion_yaw_pid = PIDController(kp=0.45, ki=0.0, kd=0.2)

        # React to `ros2 param set` for live tuning.
        self.add_on_set_parameters_callback(self._on_set_params)

        # ---- State ----
        self.pos = None            # (x, y) in the ZED odom frame
        self.zed_yaw = None        # heading in the ZED odom frame (same frame as pos)
        self.imu_yaw = None        # heading from IMU/FOG (true heading, different frame)
        self.have_target = False
        self.ref = (0.0, 0.0)      # capture point at step time
        self.ref_yaw = 0.0
        self.target_xy = (0.0, 0.0)
        self.target_yaw = 0.0
        self.step_val = 0.0
        self.armed_ok = not self.auto_arm  # if not auto-arming, assume operator armed
        self.last_time = self.get_clock().now()

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(Odometry, '/mavros/zed/odom', self._odom_cb, sensor_qos)
        self.create_subscription(Imu, '/mavros/imu/data', self._imu_cb, sensor_qos)
        self.create_subscription(Float64, '/pid_tune/step', self._step_cb, 10)
        self.create_subscription(Empty, '/pid_tune/recalibrate', self._recalibrate_cb, 10)

        self.cmd_pub = self.create_publisher(ThrusterCommand, '/hightide/cmd_vel', 10)
        self.telem_pub = self.create_publisher(Float64MultiArray, '/pid_tune/telemetry', 10)

        self.arm_cli = self.create_client(SetBool, '/hightide/arm')
        self.alt_hold_cli = self.create_client(Trigger, '/hightide/set_alt_hold')

        self.timer = self.create_timer(1.0 / rate, self._control_loop)

        self.get_logger().info(
            f"PID tuner ready | axis={self.axis} | "
            f"kp={self.pid.kp} ki={self.pid.ki} kd={self.pid.kd} | "
            f"Companion Heading Hold locked with your tuned yaw parameters!")

    # ---------- live parameter tuning ----------
    def _on_set_params(self, params):
        for p in params:
            if p.name == 'kp':
                self.pid.kp = p.value
            elif p.name == 'ki':
                self.pid.ki = p.value
                self.pid.integral_max = abs(self.max_output / p.value) if p.value else 100.0
            elif p.name == 'kd':
                self.pid.kd = p.value
            elif p.name == 'yaw_command_sign':
                self.yaw_command_sign = p.value
        self.get_logger().info(
            f"gains updated -> kp={self.pid.kp} ki={self.pid.ki} kd={self.pid.kd}")
        return SetParametersResult(successful=True)

    # ---------- feedback ----------
    def _odom_cb(self, msg):
        self.pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.zed_yaw = quaternion_to_yaw(msg.pose.pose.orientation)

    def _imu_cb(self, msg):
        self.imu_yaw = quaternion_to_yaw(msg.orientation)

    def _control_heading(self):
        return self.imu_yaw if self.heading_source == 'imu' else self.zed_yaw

    def _proj_heading(self):
        if self.heading_source == 'imu':
            return None if self.imu_yaw is None else normalize_angle(self.imu_yaw + self.frame_offset)
        return self.zed_yaw

    def _recalibrate_cb(self, msg):
        if not self._needs_cal:
            return
        self._calibrated = False
        self._cal_sin = self._cal_cos = 0.0
        self._cal_n = 0
        self.get_logger().info("Recalibration requested — hold the sub still.")

    def _try_calibrate(self):
        if self.zed_yaw is None or self.imu_yaw is None:
            return
        diff = normalize_angle(self.zed_yaw - self.imu_yaw)
        self._cal_sin += math.sin(diff)
        self._cal_cos += math.cos(diff)
        self._cal_n += 1
        if self._cal_n >= self._cal_samples:
            self.frame_offset = math.atan2(self._cal_sin, self._cal_cos)
            self._calibrated = True
            self.get_logger().info(f"Auto-calibrated offset = {math.degrees(self.frame_offset):+.1f} deg.")

    def _log_frame_offset(self):
        if self.zed_yaw is None or self.imu_yaw is None:
            return
        off = math.degrees(normalize_angle(self.zed_yaw - self.imu_yaw))
        self.get_logger().info(f"frame offset (zed-imu) = {off:+.1f} deg")

    def _step_cb(self, msg):
        yaw = self._control_heading() if self.axis == 'yaw' else self._proj_heading()
        if self.pos is None or yaw is None:
            self.get_logger().warn("Step ignored — no feedback yet.")
            return
        self.step_val = msg.data
        self.ref = self.pos
        
        # Lock current baseline heading for linear tests
        self.ref_yaw = self._control_heading() 
        
        x0, y0 = self.pos
        d = self.step_val
        if self.axis == 'surge':
            ux, uy = math.cos(yaw), math.sin(yaw)
            self.target_xy = (x0 + d * ux, y0 + d * uy)
        elif self.axis == 'sway':
            ux, uy = -math.sin(yaw), math.cos(yaw)
            self.target_xy = (x0 + d * ux, y0 + d * uy)
        else:
            self.target_yaw = normalize_angle(yaw + d)
            
        self.pid.reset()
        self.companion_yaw_pid.reset()
        self.last_time = self.get_clock().now()
        self.have_target = True
        self.get_logger().info(f"New step on {self.axis}: {d:+.3f}")

    # ---------- control ----------
    def _measure(self):
        if self.axis == 'yaw':
            yaw = self._control_heading()
            measured = normalize_angle(yaw - self.ref_yaw)
            error = normalize_angle(self.target_yaw - yaw)
            return self.step_val, measured, error
            
        yaw = self._proj_heading()
        dx = self.target_xy[0] - self.pos[0]
        dy = self.target_xy[1] - self.pos[1]
        cy, sy = math.cos(yaw), math.sin(yaw)
        if self.axis == 'surge':
            error = dx * cy + dy * sy
        else:
            error = -dx * sy + dy * cy
        measured = self.step_val - error
        return self.step_val, measured, error

    def _control_loop(self):
        cmd = ThrusterCommand()
        cmd.header.stamp = self.get_clock().now().to_msg()

        self._tick = getattr(self, '_tick', 0) + 1
        if self._tick % 40 == 0:
            self._log_frame_offset()

        if not self._calibrated:
            self._try_calibrate()
            self.cmd_pub.publish(cmd)
            return

        active_heading = self._control_heading() if self.axis == 'yaw' else self._proj_heading()

        if not (self.armed_ok and self.have_target and self.pos is not None and active_heading is not None):
            self.cmd_pub.publish(cmd)
            return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        if dt <= 0.0:
            return

        target, measured, error = self._measure()
        output = self.pid.compute(error, dt)

        if self.axis == 'surge':
            cmd.surge = output
            # Companion Stabilizer: Actively suppress yaw rotations away from self.ref_yaw
            yaw_error = normalize_angle(self.ref_yaw - self._control_heading())
            cmd.yaw = self.yaw_command_sign * self.companion_yaw_pid.compute(yaw_error, dt)
        elif self.axis == 'sway':
            cmd.sway = output
            # Companion Stabilizer: Actively suppress yaw rotations away from self.ref_yaw
            yaw_error = normalize_angle(self.ref_yaw - self._control_heading())
            cmd.yaw = self.yaw_command_sign * self.companion_yaw_pid.compute(yaw_error, dt)
        else:
            # Main axis tuning is yaw
            output = self.yaw_command_sign * output
            cmd.yaw = output
            
        self.cmd_pub.publish(cmd)

        telem = Float64MultiArray()
        telem.data = [float(target), float(measured), float(error), float(output)]
        self.telem_pub.publish(telem)

    # ---------- arming / teardown ----------
    def arm_and_hold(self):
        if not self.auto_arm:
            return
        if not self.arm_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("arm service missing.")
            return
        arm_future = self.arm_cli.call_async(_setbool(True))
        rclpy.spin_until_future_complete(self, arm_future, timeout_sec=5.0)
        if not (arm_future.done() and arm_future.result() and arm_future.result().success):
            return
        hold_future = self.alt_hold_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, hold_future, timeout_sec=5.0)
        self.armed_ok = True
        self.get_logger().info("Armed + Alt-Hold engaged.")

    def shutdown(self):
        try:
            self.cmd_pub.publish(ThrusterCommand())
        except Exception:
            pass
        if self.auto_arm and self.arm_cli.service_is_ready():
            fut = self.arm_cli.call_async(_setbool(False))
            rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)


def _setbool(val):
    req = SetBool.Request()
    req.data = val
    return req


def main(args=None):
    rclpy.init(args=args)
    node = PidTuner()
    try:
        node.arm_and_hold()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.shutdown()
        except Exception as e:
            node.get_logger().error(f"Shutdown failed: {e}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()