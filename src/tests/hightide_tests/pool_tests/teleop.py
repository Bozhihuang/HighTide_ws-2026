#!/usr/bin/env python3
"""
Continuous Hold/Release Keyboard Teleoperation for HighTide Sub

Acts like a game controller: holding a key maintains thrust, 
and letting go immediately brings that axis back to zero.
"""

import sys
import select
import termios
import tty
import time
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from hightide_interfaces.msg import ThrusterCommand

msg = """
HighTide Continuous Teleop (Controller-style)
-----------------------------------------------
Hold keys to move; let go to stop automatically!

Moving around:
   Up Arrow / w : Surge Forward
 Down Arrow / s : Surge Reverse
 Left Arrow / a : Yaw Counter-Clockwise (Turn Left)
Right Arrow / d : Yaw Clockwise (Turn Right)

Strafing & Depth Control:
   j / l : Sway Left / Sway Right
   i / k : Heave Up (Ascend) / Heave Down (Descend)

CTRL-C   : Quit and Disarm
"""

# Key codes for arrows
ARROW_UP = '\x1b[A'
ARROW_DOWN = '\x1b[B'
ARROW_RIGHT = '\x1b[C'
ARROW_LEFT = '\x1b[D'


def getKey(settings, timeout=0.05):
    """Non-blocking read of keyboard input with a small timeout."""
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        key = sys.stdin.read(1)
        if key == '\x1b':  # Handle multi-byte arrow key escape sequences
            key += sys.stdin.read(2)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


class SubTeleopNode(Node):
    def __init__(self):
        super().__init__('sub_teleop_node')
        
        self.arm_cli = self.create_client(SetBool, '/hightide/arm')
        self.cmd_pub = self.create_publisher(ThrusterCommand, '/hightide/cmd_vel', 10)
        
        while not self.arm_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for arm service...')
            
        self.speed = 0.3
        
    def arm_vehicle(self, state: bool):
        req = SetBool.Request()
        req.data = state
        future = self.arm_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if state:
            self.get_logger().info('Vehicle ARMED!')
        else:
            self.get_logger().info('Vehicle DISARMED.')


def main(args=None):
    rclpy.init(args=args)
    node = SubTeleopNode()
    
    settings = termios.tcgetattr(sys.stdin)
    
    try:
        print(msg)
        input("Ensure sub is in water. Press Enter to ARM and start teleop...")
        node.arm_vehicle(True)
        
        print("\nControl ready! Hold keys down to move...")
        
        # Track when the last valid command key was received
        last_key_time = time.time()
        # Timeout threshold: if no key repeat is seen for 150ms, assume key was released
        release_timeout = 0.15 
        
        surge = 0.0
        sway = 0.0
        heave = 0.0
        yaw = 0.0
        
        while rclpy.ok():
            # Check for a keypress quickly
            key = getKey(settings, timeout=0.03)
            current_time = time.time()
            
            if key != '':
                # Map active keys to directions
                if key in [ARROW_UP, 'w']:
                    surge = node.speed
                    last_key_time = current_time
                elif key in [ARROW_DOWN, 's']:
                    surge = -node.speed
                    last_key_time = current_time
                elif key in [ARROW_LEFT, 'a']:
                    yaw = -node.speed
                    last_key_time = current_time
                elif key in [ARROW_RIGHT, 'd']:
                    yaw = node.speed
                    last_key_time = current_time
                elif key == 'j':
                    sway = -node.speed
                    last_key_time = current_time
                elif key == 'l':
                    sway = node.speed
                    last_key_time = current_time
                elif key == 'i':
                    heave = -node.speed
                    last_key_time = current_time
                elif key == 'k':
                    heave = node.speed
                    last_key_time = current_time
                elif key == '\x03':  # CTRL-C
                    break
            else:
                # If no key has been pressed/repeated within the timeout window, clear everything
                if current_time - last_key_time > release_timeout:
                    surge = 0.0
                    sway = 0.0
                    heave = 0.0
                    yaw = 0.0
            
            # Continuously publish the state to keep the sub's watchdog active
            cmd = ThrusterCommand()
            cmd.header.stamp = node.get_clock().now().to_msg()
            cmd.surge = surge
            cmd.sway = sway
            cmd.heave = heave
            cmd.yaw = yaw
            
            node.cmd_pub.publish(cmd)
            
            # Print feedback status on a single line
            sys.stdout.write(f"\rActive Command -> Surge: {surge:+.1f} | Sway: {sway:+.1f} | Heave: {heave:+.1f} | Yaw: {yaw:+.1f}  ")
            sys.stdout.flush()
            
            # Small rest to keep loop frequency predictable (~20-30Hz publishing)
            time.sleep(0.02)

    except Exception as e:
        print(f"\nError encountered: {e}")
    finally:
        print("\nStopping thrusters and disarming vehicle safely...")
        node.cmd_pub.publish(ThrusterCommand())
        node.arm_vehicle(False)
        
        # Reset terminal attributes so your bash prompt goes back to normal
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()