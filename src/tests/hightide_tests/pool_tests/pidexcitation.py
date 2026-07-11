#!/usr/bin/env python3
"""
HighTide AUV - Fully Automated System ID Harvester (V5.2)
Features:
- Auto-Arming & Alt-Hold Engagement
- Spawns/terminates 'ros2 bag record' cleanly
- Monotonic Tick-Driven FSM with True LFSR PRBS Generation
- 20-Second MANUAL_RESET countdown state after every step and PRBS phase to pull the sub to safety
- Multi-interface mapping: ThrusterCommand for horizontal, RC Override for Heave
- Explicit telemetry recording for ZED Odometry, IMU Data, and Pixhawk Relative Altitude
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool, Trigger
from hightide_interfaces.msg import ThrusterCommand
from mavros_msgs.msg import OverrideRCIn
import subprocess
import signal
from datetime import datetime

class SystemIDExciter(Node):
    def __init__(self):
        super().__init__('sys_id_exciter')
        
        # Publishers and Service Clients
        self.pub = self.create_publisher(ThrusterCommand, '/hightide/cmd_vel', 10)
        self.rc_pub = self.create_publisher(OverrideRCIn, '/mavros/rc/override', 10)
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
        
        # True PRBS Setup (7-bit LFSR, period = 127)
        self.lfsr = 0b1000000 
        self.prbs_clock_ticks = 5  
        self.current_prbs_sign = 1
        
        self.sequence = self.build_sequence()
        self.current_idx = 0

    def build_sequence(self):
        """Pre-computes state durations in absolute tick counts."""
        seq = []
        axes = ['surge', 'sway', 'yaw', 'heave']
        amps = [0.3, 0.6, 0.8]
        
        for axis in axes:
            for amp in amps:
                # 100 ticks on (5s), 100 ticks coast (5s), 400 ticks manual reset (20s)
                seq.append({'mode': 'step', 'axis': axis, 'amp': amp, 'ticks_on': 100, 'ticks_off': 100, 'ticks_reset': 400})
                seq.append({'mode': 'step', 'axis': axis, 'amp': -amp, 'ticks_on': 100, 'ticks_off': 100, 'ticks_reset': 400})
                
        # PRBS run for 300 ticks (15s) with a 400 tick (20s) manual reset buffer at the end
        seq.append({'mode': 'prbs', 'axis': 'surge', 'amp': 0.6, 'ticks_total': 300, 'ticks_reset': 400})
        return seq

    def get_next_prbs_sign(self):
        bit = ((self.lfsr >> 6) ^ (self.lfsr >> 5)) & 1
        self.lfsr = ((self.lfsr << 1) | bit) & 0x7F
        return 1 if (self.lfsr & 1) else -1

    def send_thrust(self, axis=None, amp=0.0):
        """Sends ThrusterCommand for horizontal plane, and raw RC Override for heave."""
        msg = ThrusterCommand()
        if axis == 'surge':   msg.surge = float(amp)
        elif axis == 'sway':  msg.sway = float(amp)
        elif axis == 'yaw':   msg.yaw = float(amp)
        self.pub.publish(msg)

        if axis == 'heave' or axis is None:
            rc_msg = OverrideRCIn()
            rc_msg.channels = [65535] * 18
            if axis == 'heave':
                pwm = int(1500 + (amp * 400))
                rc_msg.channels[2] = max(1100, min(1900, pwm))  
            else:
                rc_msg.channels[2] = 1500  
            self.rc_pub.publish(rc_msg)

    def timer_cb(self):
        self.tick_count += 1
        ticks_in_state = self.tick_count - self.state_tick_start
        
        # ==========================================
        # AUTOMATED SETUP & BAGGING PHASE
        # ==========================================
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
            elif self.future.done():
                self.state = 'SET_ALT_HOLD'
                self.state_tick_start = self.tick_count
                
        elif self.state == 'SET_ALT_HOLD':
            if ticks_in_state == 1:
                self.get_logger().info("Engaging Alt-Hold mode...")
                self.future = self.alt_hold_cli.call_async(Trigger.Request())
            elif self.future.done():
                self.state = 'START_BAGGING'
                self.state_tick_start = self.tick_count
                
        elif self.state == 'START_BAGGING':
            if ticks_in_state == 1:
                bag_name = datetime.now().strftime("sys_id_bag_%Y%m%d_%H%M%S")
                self.get_logger().info(f"Starting automatic bag recording: {bag_name}")
                # Configured to record ZED visual-inertial odometry, Pixhawk relative altitude, and IMU data streams
                self.bag_process = subprocess.Popen(
                    ['ros2', 'bag', 'record', '-o', bag_name, 
                     '/mavros/zed/odom', 
                     '/mavros/imu/data', 
                     '/mavros/global_position/rel_alt',
                     '/hightide/cmd_vel', 
                     '/mavros/rc/override']
                )
            elif ticks_in_state > 60: 
                self.state = 'NEXT_TEST'
                self.state_tick_start = self.tick_count

        # ==========================================
        # EXCITATION / SYSTEM ID PHASE
        # ==========================================
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
                self.get_logger().info(f"Running PRBS | Axis: {self.current_test['axis']}")
                
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
            self.send_thrust() # Guarantee absolute neutral power
            remaining_seconds = int((self.current_test['ticks_reset'] - ticks_in_state) * self.dt)
            
            # Periodical warnings for the first 10 seconds, then a tight second-by-second countdown for the final 10 seconds
            if remaining_seconds > 10:
                if ticks_in_state % 100 == 0:  # Every 5 seconds (100 ticks)
                    self.get_logger().warn(f"[!] MANUAL RESET ACTIVE: {remaining_seconds}s remaining. Move sub back to starting position!")
            else:
                if ticks_in_state % 20 == 0 and remaining_seconds > 0:  # Every second (20 ticks)
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
                self.state = 'MANUAL_RESET' # Intercept to allow centering after broadband sweep
                self.state_tick_start = self.tick_count

        # ==========================================
        # AUTOMATED SHUTDOWN PHASE
        # ==========================================
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
            elif self.future.done():
                self.get_logger().info("=== PIPELINE COMPLETE. SAFE TO RETRIEVE SUBMARINE ===")
                self.state = 'DONE'

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