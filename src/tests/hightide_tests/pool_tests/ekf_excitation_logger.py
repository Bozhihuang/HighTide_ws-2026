#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from interfaces.msg import ThrusterCommand
from mavros_msgs.srv import SetMode
import time

class EKFExcitationLogger(Node):
    def __init__(self):
        super().__init__('ekf_excitation_logger')
        
        self.control_pub = self.create_publisher(ThrusterCommand, '/hightide/control', 10)
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.current_mode = ""
        
        # Run at stable 20 Hz
        self.timer = self.create_timer(0.05, self.timer_callback)
        
        # =====================================================================
        # BAG 1: ALT_HOLD REGIME (Translational Dynamics & Yaw)
        # =====================================================================
        alt_hold_pass = [
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
            (2.5,  0.7,  0.0,  0.0,  0.0,  0.0,  0.2, 'ALT_HOLD', "Stochastic Coupling: Hard Fwd + Light Yaw CW"),
            (4.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 'ALT_HOLD', "Coast & Settle"),
            
            (4.5, -0.5,  0.5,  0.0,  0.0,  0.0,  0.0, 'ALT_HOLD', "Stochastic Coupling: Rev + Right"),
            (3.0,  0.0, -0.6,  0.0,  0.0,  0.0,  0.4, 'ALT_HOLD', "Stochastic Coupling: Left + Yaw CW"),
            (5.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 'ALT_HOLD', "Coast & Settle"),
        ]
        
        # =====================================================================
        # BAG 2: MANUAL REGIME (Direct Pitch/Roll Orientation Excitation)
        # =====================================================================
        manual_pass = [
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
        
        # =====================================================================
        # FULL PIPELINE ASSEMBLY
        # Single Independent Pass per execution for EM stationarity
        # =====================================================================
        self.sequence = [
            (25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'ALT_HOLD', "START BAG 1 NOW: Phase A Baseline (Noise Floor)")
        ] + alt_hold_pass + [
            (20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'MANUAL',   "SPLIT LOGS NOW! Stop Bag 1. Start Bag 2. (Waiting 20s...)")
        ] + manual_pass + [
            (2.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'MANUAL',   "TEST COMPLETE. RUN SCRIPT AGAIN LATER FOR INDEPENDENT STATS.")
        ]
        
        self.current_step = 0
        self.step_start_time = time.time()
        
        self.get_logger().info("=========================================================")
        self.get_logger().info(" AWAITING DEPLOYMENT. RECORD BAG 1 (ALT_HOLD) NOW.")
        self.get_logger().info("=========================================================")
        self.get_logger().info(f"Phase: {self.sequence[0][8]}")
        
        self.set_flight_mode(self.sequence[0][7])

    def set_flight_mode(self, target_mode):
        if self.current_mode == target_mode:
            return
            
        if not self.mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f"MAVROS mode service not available. Cannot switch to {target_mode}!")
            return
            
        req = SetMode.Request()
        req.custom_mode = target_mode
        self.mode_client.call_async(req)
        
        self.current_mode = target_mode
        self.get_logger().info(f">>> BOUNDARY CONDITION ENFORCED: {target_mode} <<<")

    def timer_callback(self):
        if self.current_step >= len(self.sequence):
            self.send_thrust(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return
            
        duration, surge, sway, heave, roll, pitch, yaw, target_mode, phase_name = self.sequence[self.current_step]
        elapsed_time = time.time() - self.step_start_time
        
        if elapsed_time >= duration:
            self.current_step += 1
            if self.current_step < len(self.sequence):
                self.step_start_time = time.time()
                next_mode = self.sequence[self.current_step][7]
                next_phase = self.sequence[self.current_step][8]
                
                if "SPLIT LOGS NOW" in next_phase:
                    self.get_logger().warn("\n" + "!"*60)
                    self.get_logger().warn(" STOP RECORDING BAG 1 (Translational Stats collected).")
                    self.get_logger().warn(" START RECORDING BAG 2 (Rotational Stats) IMMEDIATELY.")
                    self.get_logger().warn("!"*60 + "\n")
                
                self.set_flight_mode(next_mode)
                self.get_logger().info(f"Transitioning to: {next_phase}")
            else:
                self.set_flight_mode('MANUAL')
                self.get_logger().info("EKF Excitation Sequence Complete. Stop Bag 2.")
            return
            
        self.send_thrust(surge, sway, heave, roll, pitch, yaw)

    def send_thrust(self, surge, sway, heave, roll, pitch, yaw):
        msg = ThrusterCommand()
        msg.surge = float(surge)
        msg.sway  = float(sway)
        msg.heave = float(heave)
        msg.roll  = float(roll)
        msg.pitch = float(pitch)
        msg.yaw   = float(yaw)
        self.control_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = EKFExcitationLogger()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted! Stopping thrusters...")
        node.send_thrust(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        node.set_flight_mode('MANUAL')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()