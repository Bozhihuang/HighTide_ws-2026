#!/usr/bin/env python3
"""
HighTide AUV - EKF Data Excitation Logger
Automates diverse, repeatable, and coupled discrete motions to strictly excite 
the Extended Kalman Filter (EKF) states for process noise (Q) identification.

THEORETICAL ALIGNMENT & SYSTEM AUTOMATION:
- Automatic Bag Recording: Dynamically spawns and kills isolated 'ros2 bag record' 
  subprocesses using POSIX process groups to eliminate manual operator timing errors.
- Continuous Background Publishing: Prevents RC override timeouts and auto-disarms 
  by broadcasting ThrusterCommands at 20 Hz continuously, even during blocking 
  service calls and baseline wait periods.
- Active Watchdog Heartbeat: Dynamically stamps the current ROS 2 clock time onto 
  the ThrusterCommand header to keep the RC watchdog and PX4 failsafes happy.
- Asymmetric Coupling: Replaces periodic square-wave zig-zags with stochastic-like, 
  asymmetric duration and amplitude steps to ensure true statistical identifiability.
- Actuator Saturation Limits: MANUAL mode pitch/roll commands are strictly bounded 
  to low amplitudes (0.2 - 0.3) to excite the plant dynamics without fighting 
  hidden stabilizing filters.
"""

import rclpy
from rclpy.node import Node
from hightide_interfaces.msg import ThrusterCommand
from std_srvs.srv import SetBool, Trigger
import time
import threading
import subprocess
import os
import signal

class EKFExcitationLogger(Node):
    def __init__(self):
        super().__init__('ekf_excitation_logger')
        
        # Initialize target command to neutral (all 0.0)
        self.current_cmd = ThrusterCommand()
        self.current_cmd.surge = 0.0
        self.current_cmd.sway = 0.0
        self.current_cmd.heave = 0.0
        self.current_cmd.roll = 0.0
        self.current_cmd.pitch = 0.0
        self.current_cmd.yaw = 0.0

        # Subprocess tracking for automatic rosbag recording
        self.bag_proc = None

        # Publishers and Clients
        self.control_pub = self.create_publisher(ThrusterCommand, '/hightide/cmd_vel', 10)
        self.arm_client = self.create_client(SetBool, '/hightide/arm')
        self.alt_hold_client = self.create_client(Trigger, '/hightide/set_alt_hold')
        self.manual_client = self.create_client(Trigger, '/hightide/set_manual')
        
        # =====================================================================
        # BAG 1: ALT_HOLD REGIME (Translational Dynamics & Yaw)
        # =====================================================================
        self.alt_hold_pass = [
            # Phase B: Pure Axis Excitation (Clean Isolation)
            (6.0,  0.6,  0.0,  0.0,  0.0,  0.0,  0.0, 'ALT_HOLD', "Sweep: Surge Forward"),
            (5.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 'ALT_HOLD', "Coast & Settle"),
            (6.0, -0.6,  0.0,  0.0,  0.0,  0.0,  0.0, 'ALT_HOLD', "Sweep: Surge Backward"),
            (5.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 'ALT_HOLD', "Coast & Settle"),
            
            (6.0,  0.0,  0.6,  0.0,  0.0,  0.0,  0.0, 'ALT_HOLD', "Sweep: Sway Right"),
            (5.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 'ALT_HOLD', "Coast & Settle"),
            (6.0,  0.0, -0.6,  0.0,  0.0,  0.0,  0.0, 'ALT_HOLD', "Sweep: Sway Left"),
            (5.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 'ALT_HOLD', "Coast & Settle"),
            
            (8.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.4, 'ALT_HOLD', "Sweep: Yaw Clockwise"),
            (5.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 'ALT_HOLD', "Coast & Settle"),
            (8.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.4, 'ALT_HOLD', "Sweep: Yaw Counter-Clockwise"),
            (5.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 'ALT_HOLD', "Coast & Settle"),
            
            (5.0,  0.0,  0.0, -0.6,  0.0,  0.0,  0.0, 'ALT_HOLD', "Sweep: Heave Descend"),
            (4.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 'ALT_HOLD', "Hold Depth (Settle)"),
            (5.0,  0.0,  0.0,  0.6,  0.0,  0.0,  0.0, 'ALT_HOLD', "Sweep: Heave Ascend"),
            (5.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 'ALT_HOLD', "Coast & Settle"),
            
            # Phase C: Coupled Asymmetric Excitation (Prevents Periodic Bias)
            (3.5,  0.6,  0.0,  0.0,  0.0,  0.0,  0.3, 'ALT_HOLD', "Stochastic Coupling: Fwd + Light Yaw CW"),
            (5.0,  0.4,  0.0,  0.0,  0.0,  0.0, -0.5, 'ALT_HOLD', "Stochastic Coupling: Fwd + Hard Yaw CCW"),
            (2.5,  0.7,  0.0,  0.0,  0.0,  0.0,  0.4, 'ALT_HOLD', "Sweep: Step Surge Accent"),
            (5.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 'ALT_HOLD', "Coast & Settle"),
            
            (4.5, -0.5,  0.5,  0.0,  0.0,  0.0,  0.0, 'ALT_HOLD', "Stochastic Coupling: Rev + Right"),
            (3.0,  0.0, -0.6,  0.0,  0.0,  0.0,  0.4, 'ALT_HOLD', "Stochastic Coupling: Left + Yaw CW"),
            (5.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 'ALT_HOLD', "Coast & Settle"),
        ]
        
        # =====================================================================
        # BAG 2: MANUAL REGIME (Direct Pitch/Roll Orientation Excitation)
        # =====================================================================
        self.manual_pass = [
            # Phase D: Orientation Dynamics (Strictly Bounded Amplitudes to Prevent Saturation)
            (4.0,  0.0,  0.0,  0.0,  0.25,  0.0,  0.0, 'MANUAL', "Sweep: Light Roll Right"),
            (3.0,  0.0,  0.0,  0.0,  0.0,   0.0,  0.0, 'MANUAL', "Coast & Level"),
            (4.5,  0.0,  0.0,  0.0, -0.3,   0.0,  0.0, 'MANUAL', "Sweep: Med Roll Left"),
            (3.0,  0.0,  0.0,  0.0,  0.0,   0.0,  0.0, 'MANUAL', "Coast & Level"),
            
            (3.5,  0.0,  0.0,  0.0,  0.0,   0.25, 0.0, 'MANUAL', "Sweep: Light Pitch Up"),
            (3.0,  0.0,  0.0,  0.0,  0.0,   0.0,  0.0, 'MANUAL', "Coast & Level"),
            (4.0,  0.0,  0.0,  0.0,  0.0,  -0.3,  0.0, 'MANUAL', "Sweep: Med Pitch Down"),
            (3.0,  0.0,  0.0,  0.0,  0.0,   0.0,  0.0, 'MANUAL', "Coast & Level"),
            
            # Asymmetric Coupled Orientation
            (2.5,  0.0,  0.0,  0.0,  0.2,   0.3,  0.0, 'MANUAL', "Asymmetric: Roll Right + Hard Pitch Up"),
            (4.0,  0.0,  0.0,  0.0, -0.3,  -0.2,  0.0, 'MANUAL', "Asymmetric: Hard Roll Left + Light Pitch Down"),
            (4.0,  0.0,  0.0,  0.0,  0.0,   0.0,  0.0, 'MANUAL', "Coast & Level"),
        ]

        # Start continuous 20 Hz publishing immediately to establish RC heartbeat
        self.pub_timer = self.create_timer(0.05, self.publish_current_command)

        # Spawns a background thread to handle linear sequential tasks synchronously
        self.worker_thread = threading.Thread(target=self.run)
        self.worker_thread.start()

    def start_bag_recording(self, bag_name):
        """Spawns an asynchronous ROS 2 bag record process in its own process group."""
        self.get_logger().info(f"--- STARTING ROS 2 BAG RECORDING: {bag_name} ---")
        topics = [
            "/zed/odom",
            "/mavros/imu/data",
            "/odometry/filtered",
            "/hightide/cmd_vel"
        ]
        cmd = ["ros2", "bag", "record", "-o", bag_name] + topics
        
        # We use preexec_fn=os.setsid to isolate the recording process group.
        # This prevents child database writing processes from orphaning on SIGINT.
        try:
            self.bag_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, # Corrected: Redirect error logs properly using standard DEVNULL
                preexec_fn=os.setsid
            )
        except Exception as e:
            self.get_logger().error(f"Failed to trigger automatic ROS 2 bag recording: {e}")

    def stop_bag_recording(self):
        """Gracefully terminates the current ROS 2 bag recording process."""
        if self.bag_proc is not None:
            self.get_logger().info("Stopping ROS 2 bag recorder...")
            try:
                # Send SIGINT to the entire process group to ensure clean database writes
                os.killpg(os.getpgid(self.bag_proc.pid), signal.SIGINT)
                self.bag_proc.wait(timeout=5.0)
            except Exception as e:
                self.get_logger().warn(f"Bag process termination warning: {e}")
            finally:
                self.bag_proc = None

    def publish_current_command(self):
        """Continuously publishes the current command target to maintain failsafes."""
        self.current_cmd.header.stamp = self.get_clock().now().to_msg()
        self.control_pub.publish(self.current_cmd)

    def call_service_synced(self, client, req):
        """Helper to synchronously wait for and execute service calls inside our worker thread."""
        while rclpy.ok() and not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for Mode Manager service: {client.srv_name}...")
        
        future = client.call_async(req)
        while rclpy.ok() and not future.done():
            time.sleep(0.05)
            
        try:
            return future.result()
        except Exception as e:
            self.get_logger().error(f"Mode Manager call to {client.srv_name} failed: {e}")
            return None

    def set_flight_mode(self, mode):
        """Triggers Alt Hold or Manual services provided by the Mode Manager."""
        self.get_logger().info(f"Requesting flight mode transition to: {mode}")
        req = Trigger.Request()
        if mode == 'ALT_HOLD':
            client = self.alt_hold_client
        elif mode == 'MANUAL':
            client = self.manual_client
        else:
            self.get_logger().error(f"Unmapped flight mode requested: {mode}")
            return False

        res = self.call_service_synced(client, req)
        if res is not None and res.success:
            self.get_logger().info(f"Flight mode {mode} established successfully!")
            return True
        else:
            msg = res.message if res else "Unknown service failure"
            self.get_logger().error(f"Mode Manager rejected {mode}: {msg}")
            return False

    def set_arm_state(self, arm_state):
        """Triggers Arm/Disarm services provided by the Mode Manager."""
        action = "ARM" if arm_state else "DISARM"
        self.get_logger().info(f"Requesting Mode Manager to {action} vehicle...")
        req = SetBool.Request()
        req.data = arm_state
        res = self.call_service_synced(self.arm_client, req)
        if res is not None and res.success:
            self.get_logger().info(f"Vehicle {action}ED successfully!")
            return True
        else:
            msg = res.message if res else "Unknown service failure"
            self.get_logger().error(f"Mode Manager rejected {action} request: {msg}")
            return False

    def execute_step(self, duration, surge, sway, heave, roll, pitch, yaw, label):
        """Executes a single step for duration seconds by setting the target values."""
        self.get_logger().info(f"Executing Phase step: {label}")
        
        # Set values for the continuous publisher thread to pick up
        self.current_cmd.surge = float(surge)
        self.current_cmd.sway = float(sway)
        self.current_cmd.heave = float(heave)
        self.current_cmd.roll = float(roll)
        self.current_cmd.pitch = float(pitch)
        self.current_cmd.yaw = float(yaw)
        
        time.sleep(duration)

    def run(self):
        """Pure linear execution sequence of the full test layout."""
        self.get_logger().info("=========================================================")
        self.get_logger().info(" EKF EXCITATION LOGGER NODE STARTED.")
        self.get_logger().info(" Waiting for HighTide Mode Manager services to connect...")
        self.get_logger().info("=========================================================")

        # Block until services are ready
        while rclpy.ok() and (not self.arm_client.wait_for_service(timeout_sec=1.0) or
                              not self.alt_hold_client.wait_for_service(timeout_sec=1.0) or
                              not self.manual_client.wait_for_service(timeout_sec=1.0)):
            time.sleep(1.0)

        # 1. Start Initial ALT_HOLD Mode (Timer is actively sending 0.0 at 20Hz)
        if not self.set_flight_mode('ALT_HOLD'):
            return

        # 2. Arm the vehicle safely
        if not self.set_arm_state(True):
            return

        # 3. Phase A Baseline (No Recording yet - just pure system stabilization check)
        self.execute_step(5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Preparing System...")

        # 4. Start automatic recording of Bag 1
        self.start_bag_recording("bag_alt_hold")
        self.execute_step(20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Phase A Baseline (Noise Floor)")

        # 5. Phase B & C Alt Hold sweep parameters
        for step in self.alt_hold_pass:
            duration, surge, sway, heave, roll, pitch, yaw, mode, label = step
            self.execute_step(duration, surge, sway, heave, roll, pitch, yaw, label)

        # 6. Gracefully stop recording of Bag 1
        self.stop_bag_recording()

        self.get_logger().warn("\n" + "!"*65)
        self.get_logger().warn(" AUTOMATIC SPLIT: Stop Bag 1 complete.")
        self.get_logger().warn(" Preparing transition to MANUAL Mode (Wait 15s)...")
        self.get_logger().warn("!"*65 + "\n")

        self.execute_step(15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Transition Settle Window")

        # 7. Establish MANUAL Mode for orientation parameters
        if not self.set_flight_mode('MANUAL'):
            return

        # 8. Start automatic recording of Bag 2
        self.start_bag_recording("bag_manual")
        self.execute_step(5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Bag 2 Baseline Orientation Capture")

        # 9. Phase D Manual sweeps
        for step in self.manual_pass:
            duration, surge, sway, heave, roll, pitch, yaw, mode, label = step
            self.execute_step(duration, surge, sway, heave, roll, pitch, yaw, label)

        # 10. Gracefully stop recording of Bag 2
        self.stop_bag_recording()

        # 11. Clean up and Disarm
        self.execute_step(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Stopping Thrusters")
        self.set_arm_state(False)

        self.get_logger().info("=========================================================")
        self.get_logger().info(" TEST COMPLETE. Safe to retrieve vehicle.")
        self.get_logger().info(" Bag 1 (bag_alt_hold) & Bag 2 (bag_manual) recorded successfully.")
        self.get_logger().info("=========================================================")

def main(args=None):
    rclpy.init(args=args)
    node = EKFExcitationLogger()

    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()

    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted! Emergency stop and automatic cleanup initiated...")
    finally:
        # Ensure we kill the recording process and disarm the sub safely during sudden stop
        node.stop_bag_recording()
        node.current_cmd.surge = 0.0
        node.current_cmd.sway = 0.0
        node.current_cmd.heave = 0.0
        node.current_cmd.roll = 0.0
        node.current_cmd.pitch = 0.0
        node.current_cmd.yaw = 0.0
        node.publish_current_command()
        
        # Synchronous direct disarm call during sudden termination
        req = SetBool.Request()
        req.data = False
        node.arm_client.call(req)
        
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()