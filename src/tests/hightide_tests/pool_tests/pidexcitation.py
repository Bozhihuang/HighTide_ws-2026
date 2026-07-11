#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool, Trigger
from mavros_msgs.msg import OverrideRCIn
from std_msgs.msg import Float64MultiArray
import subprocess
import signal
from datetime import datetime

class SystemIDExciter(Node):
    def __init__(self):
        super().__init__('sys_id_exciter')
        
        # Publish purely to raw RC overrides to command the sub
        self.rc_pub = self.create_publisher(OverrideRCIn, '/mavros/rc/override', 10)
        
        # Publish standard array strictly so MATLAB can read the commands without compiling mavros_msgs
        self.matlab_pub = self.create_publisher(Float64MultiArray, '/sys_id_efforts', 10)
        
        self.arm_cli = self.create_client(SetBool, '/hightide/arm')
        self.alt_hold_cli = self.create_client(Trigger, '/hightide/set_alt_hold')

        self.get_logger().info("Waiting for Flight Controller services...")
        self.arm_cli.wait_for_service()
        self.alt_hold_cli.wait_for_service()
        
        self.bag_process = None
        self.dt = 0.05  # 20 Hz
        self.timer = self.create_timer(self.dt, self.timer_cb)
        
        self.state = 'INIT'
        self.tick_count = 0
        self.state_tick_start = 0
        self.future = None
        # Retry bookkeeping for arm / alt-hold / disarm service calls
        self.retry_count = 0
        self.max_retries = 3

        # The ENTIRE run stays in ALT_HOLD. depth_controller_node is an outer loop
        # whose throttle-offset output ArduSub's Alt-Hold turns into a climb/descent
        # rate, so the plant its PID controls is "throttle offset -> depth" measured
        # in ALT_HOLD. Every axis (heave included) is therefore excited in ALT_HOLD;
        # MANUAL would identify a different (raw open-loop) plant and mistune the PID.
        
        # True PRBS Setup (7-bit LFSR)
        self.lfsr = 0b1000000 
        self.prbs_clock_ticks = 5  
        self.current_prbs_sign = 1
        
        self.sequence = self.build_sequence()
        self.current_idx = 0

    def build_sequence(self):
        seq = []
        axes = ['surge', 'sway', 'yaw', 'heave']
        amps = [0.3, 0.6, 0.8]

        for axis in axes:
            for amp in amps:
                # Heave in ALT_HOLD commands a climb/descent RATE, so depth
                # integrates and the vehicle keeps moving the whole pulse. Keep
                # heave pulses SHORT (2.0s) so it cannot breach the surface or hit
                # the floor; horizontal axes settle to a velocity, so 5.0s is fine.
                # NOTE: verify 2.0s x amp stays within your pool depth given
                # ArduSub's PILOT_SPEED_UP / PILOT_SPEED_DN, and start each heave
                # test from mid-pool depth.
                ticks_on = 40 if axis == 'heave' else 100
                seq.append({'mode': 'step', 'axis': axis, 'amp': amp, 'ticks_on': ticks_on, 'ticks_off': 100, 'ticks_reset': 400})
                seq.append({'mode': 'step', 'axis': axis, 'amp': -amp, 'ticks_on': ticks_on, 'ticks_off': 100, 'ticks_reset': 400})

        seq.append({'mode': 'prbs', 'axis': 'surge', 'amp': 0.6, 'ticks_total': 300, 'ticks_reset': 400})
        seq.append({'mode': 'prbs', 'axis': 'sway',  'amp': 0.6, 'ticks_total': 300, 'ticks_reset': 400})
        seq.append({'mode': 'prbs', 'axis': 'yaw',   'amp': 0.6, 'ticks_total': 300, 'ticks_reset': 400})
        seq.append({'mode': 'prbs', 'axis': 'heave',   'amp': 0.6, 'ticks_total': 300, 'ticks_reset': 400})
        return seq

    def get_next_prbs_sign(self):
        bit = ((self.lfsr >> 6) ^ (self.lfsr >> 5)) & 1
        self.lfsr = ((self.lfsr << 1) | bit) & 0x7F
        return 1 if (self.lfsr & 1) else -1

    def send_thrust(self, axis=None, amp=0.0):
        """Sends raw RC Override to Pixhawk, and standard array to MATLAB."""
        rc_msg = OverrideRCIn()
        rc_msg.channels = [65535] * 18
        
        # Force the 6 core DOFs to neutral (1500)
        rc_msg.channels[0] = 1500 # Pitch
        rc_msg.channels[1] = 1500 # Roll
        rc_msg.channels[2] = 1500 # Heave
        rc_msg.channels[3] = 1500 # Yaw
        rc_msg.channels[4] = 1500 # Surge
        rc_msg.channels[5] = 1500 # Sway
        
        # Tracking variables for MATLAB logging
        s_eff, sw_eff, h_eff, y_eff = 0.0, 0.0, 0.0, 0.0
        
        if axis is not None:
            active_pwm = int(1500 + (amp * 400))
            active_pwm = max(1100, min(1900, active_pwm))  
            
            if axis == 'surge':   
                rc_msg.channels[4] = active_pwm
                s_eff = float(amp)
            elif axis == 'sway':  
                rc_msg.channels[5] = active_pwm
                sw_eff = float(amp)
            elif axis == 'heave': 
                rc_msg.channels[2] = active_pwm
                h_eff = float(amp)
            elif axis == 'yaw':   
                rc_msg.channels[3] = active_pwm
                y_eff = float(amp)

        # 1. Command the sub
        self.rc_pub.publish(rc_msg)
        
        # 2. Log exactly what we commanded in a native format for MATLAB
        matlab_msg = Float64MultiArray()
        matlab_msg.data = [s_eff, sw_eff, h_eff, y_eff]
        self.matlab_pub.publish(matlab_msg)

    def _service_succeeded(self, label):
        """Return True only if the pending future resolved with success=True."""
        try:
            result = self.future.result()
        except Exception as e:  # service died / call failed
            self.get_logger().error(f"{label} service call raised: {e}")
            return False
        if result is None or not getattr(result, 'success', False):
            msg = getattr(result, 'message', '') if result is not None else 'no response'
            self.get_logger().error(f"{label} rejected by FC: {msg}")
            return False
        return True

    def _retries_exhausted(self, label):
        """Increment retry counter; True once we give up (caller should ABORT)."""
        self.retry_count += 1
        if self.retry_count >= self.max_retries:
            self.get_logger().error(
                f"{label} FAILED after {self.max_retries} attempts. Aborting run for safety.")
            self.retry_count = 0
            return True
        self.get_logger().warn(f"Retrying {label} ({self.retry_count}/{self.max_retries})...")
        return False

    def timer_cb(self):
        self.tick_count += 1
        ticks_in_state = self.tick_count - self.state_tick_start
        
        if self.state == 'INIT':
            if ticks_in_state == 1:
                self.get_logger().info("DEPLOYMENT STARTING IN 5 SECONDS. Clear the pool...")
            elif ticks_in_state > 100:  
                self.state = 'ARMING'
                self.state_tick_start = self.tick_count
                
        elif self.state == 'ARMING':
            if ticks_in_state == 1:
                self.get_logger().info("Arming thrusters...")
                req = SetBool.Request()
                req.data = True
                self.future = self.arm_cli.call_async(req)
            elif self.future is not None and self.future.done():
                # Only advance if the FC actually confirmed the arm. Without this
                # a rejected arm silently runs the entire excitation on a dead sub.
                if self._service_succeeded('Arm'):
                    self.get_logger().info("Armed successfully.")
                    self.retry_count = 0
                    self.state = 'SET_ALT_HOLD'
                    self.state_tick_start = self.tick_count
                elif self._retries_exhausted('ARM'):
                    self.state = 'ABORT'
                    self.state_tick_start = self.tick_count
                else:
                    self.future = None
                    self.state_tick_start = self.tick_count  # re-enter -> re-send next tick

        elif self.state == 'SET_ALT_HOLD':
            if ticks_in_state == 1:
                self.get_logger().info("Engaging Alt-Hold mode...")
                self.future = self.alt_hold_cli.call_async(Trigger.Request())
            elif self.future is not None and self.future.done():
                if self._service_succeeded('Alt-Hold'):
                    self.get_logger().info("Alt-Hold engaged.")
                    self.retry_count = 0
                    self.state = 'START_BAGGING'
                    self.state_tick_start = self.tick_count
                elif self._retries_exhausted('ALT_HOLD'):
                    self.state = 'ABORT'
                    self.state_tick_start = self.tick_count
                else:
                    self.future = None
                    self.state_tick_start = self.tick_count
                
        elif self.state == 'START_BAGGING':
            if ticks_in_state == 1:
                bag_name = datetime.now().strftime("sys_id_bag_%Y%m%d_%H%M%S")
                self.get_logger().info(f"Starting automatic bag recording: {bag_name}")
                # We swapped rc/override for sys_id_efforts so MATLAB can natively read the inputs
                self.bag_process = subprocess.Popen(
                    ['ros2', 'bag', 'record', '-o', bag_name, 
                     '/mavros/zed/odom', 
                     '/mavros/imu/data', 
                     '/mavros/global_position/rel_alt',
                     '/sys_id_efforts']
                )
            elif ticks_in_state > 60: 
                self.state = 'NEXT_TEST'
                self.state_tick_start = self.tick_count

        elif self.state == 'NEXT_TEST':
            if self.current_idx >= len(self.sequence):
                self.send_thrust() 
                self.state = 'STOP_BAGGING'
                self.state_tick_start = self.tick_count
                return
                
            self.current_test = self.sequence[self.current_idx]
            self.state_tick_start = self.tick_count

            if self.current_test['mode'] == 'step':
                self.state = 'STEP_ON'
                self.get_logger().info(f"Running STEP | Axis: {self.current_test['axis']} | Amp: {self.current_test['amp']}")
            elif self.current_test['mode'] == 'prbs':
                self.state = 'PRBS_RUN'
                self.lfsr = 0b1000000
                self.current_prbs_sign = 1  # deterministic start; don't inherit prior block's sign
                self.get_logger().info(f"Running PRBS | Axis: {self.current_test['axis']}")
            else:
                self.get_logger().error(f"Unknown test mode: {self.current_test['mode']}, skipping.")
                self.current_idx += 1

        elif self.state == 'STEP_ON':
            self.send_thrust(self.current_test['axis'], self.current_test['amp'])
            if ticks_in_state >= self.current_test['ticks_on']:
                self.state = 'COAST'
                self.state_tick_start = self.tick_count
                
        elif self.state == 'COAST':
            self.send_thrust()
            if ticks_in_state >= self.current_test['ticks_off']:
                self.state = 'MANUAL_RESET'
                self.state_tick_start = self.tick_count
                
        elif self.state == 'MANUAL_RESET':
            self.send_thrust() 
            remaining_seconds = int((self.current_test['ticks_reset'] - ticks_in_state) * self.dt)
            
            if remaining_seconds > 10:
                if ticks_in_state % 100 == 0:  
                    self.get_logger().warn(f"[!] MANUAL RESET ACTIVE: {remaining_seconds}s remaining. Move sub back to starting position!")
            else:
                if ticks_in_state % 20 == 0 and remaining_seconds > 0:  
                    self.get_logger().warn(f"[!] LAUNCH COUNTDOWN: {remaining_seconds} seconds! RELEASE THE VEHICLE NOW!")
                
            if ticks_in_state >= self.current_test['ticks_reset']:
                self.current_idx += 1
                self.state = 'NEXT_TEST'
                self.state_tick_start = self.tick_count
                
        elif self.state == 'PRBS_RUN':
            if ticks_in_state % self.prbs_clock_ticks == 0:
                self.current_prbs_sign = self.get_next_prbs_sign()
                
            self.send_thrust(self.current_test['axis'], self.current_prbs_sign * self.current_test['amp'])
            
            if ticks_in_state >= self.current_test['ticks_total']:
                self.send_thrust()
                self.state = 'MANUAL_RESET'
                self.state_tick_start = self.tick_count

        elif self.state == 'STOP_BAGGING':
            if ticks_in_state == 1:
                self.get_logger().info("Tests complete. Stopping bag recording safely...")
                if self.bag_process:
                    self.bag_process.send_signal(signal.SIGINT)
            elif ticks_in_state > 60: 
                self.state = 'DISARMING'
                self.state_tick_start = self.tick_count
                
        elif self.state == 'DISARMING':
            if ticks_in_state == 1:
                self.get_logger().info("Disarming thrusters...")
                req = SetBool.Request()
                req.data = False
                self.future = self.arm_cli.call_async(req)
            elif self.future is not None and self.future.done():
                if self._service_succeeded('Disarm'):
                    self.get_logger().info("=== PIPELINE COMPLETE. SAFE TO RETRIEVE SUBMARINE ===")
                    self.retry_count = 0
                    self.state = 'DONE'
                elif self._retries_exhausted('DISARM'):
                    # Could not confirm disarm — loudest possible warning, keep neutral.
                    self.get_logger().error(
                        "!!! DISARM UNCONFIRMED — VEHICLE MAY STILL BE ARMED. "
                        "Cut power manually before handling. !!!")
                    self.send_thrust()
                    self.state = 'DONE'
                else:
                    self.future = None
                    self.state_tick_start = self.tick_count

        elif self.state == 'ABORT':
            # Safe teardown when a setup service failed: neutralize, close the bag,
            # then fall through to a disarm attempt.
            if ticks_in_state == 1:
                self.get_logger().error("=== ABORTING RUN: neutralizing thrusters and closing bag ===")
                self.send_thrust()
                if self.bag_process:
                    self.bag_process.send_signal(signal.SIGINT)
            else:
                self.send_thrust()  # keep commanding neutral while the bag closes
                if ticks_in_state > 60:
                    self.state = 'DISARMING'
                    self.state_tick_start = self.tick_count

    def cleanup(self):
        self.send_thrust()
        if self.bag_process and self.bag_process.poll() is None:
            self.get_logger().info("Emergency Stop: Closing bag safely...")
            self.bag_process.send_signal(signal.SIGINT)
            self.bag_process.wait()

def main(args=None):
    rclpy.init(args=args)
    node = SystemIDExciter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Manual Override Detected!")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()