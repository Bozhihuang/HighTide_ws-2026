#!/usr/bin/env python3
"""
HighTide AUV - Fully Automated System ID Harvester (V5)
Features:
- Auto-Arming & Alt-Hold Engagement
- Auto-spawns and cleanly terminates 'ros2 bag record' subprocess
- Monotonic Tick-Driven FSM with True LFSR PRBS Generation
- Safe Auto-Disarm on completion
- Native integration with HighTide ThrusterCommand interface
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
        
        # Subprocess handler for automatic bag recording
        self.bag_process = None
        
        # Control Loop Timing (20 Hz matching rc_override_node)
        self.dt = 0.05  
        self.timer = self.create_timer(self.dt, self.timer_cb)
        
        # FSM Variables
        self.state = 'INIT'
        self.tick_count = 0
        self.state_tick_start = 0
        self.future = None # Tracks async service calls
        
        # True PRBS Setup (7-bit LFSR, period = 127)
        self.lfsr = 0b1000000 
        self.prbs_clock_ticks = 5  # Flip state every 5 ticks (0.25s)
        self.current_prbs_sign = 1
        
        self.sequence = self.build_sequence()
        self.current_idx = 0

    def build_sequence(self):
        """Pre-computes state durations in absolute tick counts."""
        seq = []
        axes = ['surge', 'sway', 'yaw', 'heave']
        amps = [0.3, 0.6, 0.8]
        
        # 1. Symmetric Steps (e.g., 100 ticks = 5.0s)
        for axis in axes:
            for amp in amps:
                seq.append({'mode': 'step', 'axis': axis, 'amp': amp, 'ticks_on': 100, 'ticks_off': 100})
                seq.append({'mode': 'step', 'axis': axis, 'amp': -amp, 'ticks_on': 100, 'ticks_off': 100})
                
        # 2. Broadband Excitation (PRBS run for 300 ticks = 15.0s)
        seq.append({'mode': 'prbs', 'axis': 'surge', 'amp': 0.6, 'ticks_total': 300})
        return seq

    def get_next_prbs_sign(self):
        """Calculates next LFSR state and returns a normalized multiplier (-1 or 1)."""
        bit = ((self.lfsr >> 6) ^ (self.lfsr >> 5)) & 1
        self.lfsr = ((self.lfsr << 1) | bit) & 0x7F
        return 1 if (self.lfsr & 1) else -1

    def send_thrust(self, axis=None, amp=0.0):
        """Sends ThrusterCommand for horizontal plane, and raw RC Override for heave."""
        # 1. Middleware Command (Surge, Sway, Yaw)
        msg = ThrusterCommand()
        if axis == 'surge':   msg.surge = float(amp)
        elif axis == 'sway':  msg.sway = float(amp)
        elif axis == 'yaw':   msg.yaw = float(amp)
        self.pub.publish(msg)

        # 2. Raw RC Override (Heave ONLY)
        if axis == 'heave' or axis is None:
            rc_msg = OverrideRCIn()
            rc_msg.channels = [65535] * 18
            if axis == 'heave':
                # Convert normalized [-1.0, 1.0] amplitude to [1100, 1900] PWM
                pwm = int(1500 + (amp * 400))
                rc_msg.channels[2] = max(1100, min(1900, pwm))  # Channel 3 is Heave
            else:
                # Explicit neutral stop when coasting
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
                # Log cmd_vel, MAVROS local odometry, and raw RC overrides for MATLAB
                self.bag_process = subprocess.Popen(
                    ['ros2', 'bag', 'record', '-o', bag_name, '/mavros/local_position/odom', '/hightide/cmd_vel', '/mavros/rc/override']
                )
            elif ticks_in_state > 60: # Give rosbag 3 seconds to fully initialize
                self.state = 'NEXT_TEST'
                self.state_tick_start = self.tick_count

        # ==========================================
        # EXCITATION / SYSTEM ID PHASE
        # ==========================================
        elif self.state == 'NEXT_TEST':
            if self.current_idx >= len(self.sequence):
                self.send_thrust() # Neutralize thrusters
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
                self.lfsr = 0b1000000 # Reset LFSR seed
                self.get_logger().info(f"Running PRBS | Axis: {self.current_test['axis']}")
                
        elif self.state == 'STEP_ON':
            self.send_thrust(self.current_test['axis'], self.current_test['amp'])
            if ticks_in_state >= self.current_test['ticks_on']:
                self.state = 'COAST'
                self.state_tick_start = self.tick_count
                
        elif self.state == 'COAST':
            self.send_thrust()
            if ticks_in_state >= self.current_test['ticks_off']:
                self.current_idx += 1
                self.state = 'NEXT_TEST'
                
        elif self.state == 'PRBS_RUN':
            if ticks_in_state % self.prbs_clock_ticks == 0:
                self.current_prbs_sign = self.get_next_prbs_sign()
                
            self.send_thrust(self.current_test['axis'], self.current_prbs_sign * self.current_test['amp'])
            
            if ticks_in_state >= self.current_test['ticks_total']:
                self.send_thrust()
                self.current_idx += 1
                self.state = 'NEXT_TEST'

        # ==========================================
        # AUTOMATED SHUTDOWN PHASE
        # ==========================================
        elif self.state == 'STOP_BAGGING':
            if ticks_in_state == 1:
                self.get_logger().info("Tests complete. Stopping bag recording safely...")
                if self.bag_process:
                    # SIGINT is required so ROS2 saves the database file cleanly without corruption
                    self.bag_process.send_signal(signal.SIGINT)
            elif ticks_in_state > 60: # Give rosbag 3 seconds to close out the database
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
        """Emergency catch to ensure bag doesn't corrupt on Ctrl+C"""
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