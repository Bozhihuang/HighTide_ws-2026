#!/usr/bin/env python3
"""
Direct Yellow Box RC Override Node.

Subscribes to ZED camera stream, tracks a yellow box via HSV color space,
and drives standard ArduSub RC channels directly via /mavros/rc/override PWM outputs.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from mavros_msgs.msg import OverrideRCIn


class SimplePID:
    """Inline PID controller to keep the script fully self-contained."""
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        if dt <= 0:
            return 0.0
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)


class DirectYellowBoxRCNode(Node):
    """ROS2 node combining OpenCV vision and direct MAVROS RC Override generation."""

    def __init__(self):
        super().__init__('direct_yellow_box_rc_node')

        # Configuration Parameters
        self.declare_parameter('target_x_normalized', 0.5)
        self.declare_parameter('target_y_normalized', 0.5)
        self.declare_parameter('approach_distance_m', 1.5)
        self.declare_parameter('enabled', True)

        # PID Gains (Tunable for direct PWM scale outputs)
        self.declare_parameter('lateral_kp', 400.0)   # Controls Lateral/Sway PWM
        self.declare_parameter('lateral_kd', 40.0)
        self.declare_parameter('vertical_kp', 350.0)  # Controls Throttle/Heave PWM
        self.declare_parameter('vertical_kd', 20.0)
        self.declare_parameter('range_kp', 150.0)     # Controls Forward/Surge PWM
        self.declare_parameter('range_kd', 10.0)

        self.target_x = self.get_parameter('target_x_normalized').value
        self.target_y = self.get_parameter('target_y_normalized').value
        self.approach_dist = self.get_parameter('approach_distance_m').value
        self.enabled = self.get_parameter('enabled').value

        # Initialize PIDs
        self.lateral_pid = SimplePID(self.get_parameter('lateral_kp').value, 0.0, self.get_parameter('lateral_kd').value)
        self.vertical_pid = SimplePID(self.get_parameter('vertical_kp').value, 0.0, self.get_parameter('vertical_kd').value)
        self.range_pid = SimplePID(self.get_parameter('range_kp').value, 0.0, self.get_parameter('range_kd').value)

        # Yellow Color Tuning Configuration
        self.hue_low, self.hue_high = 20, 35
        self.sat_low, self.val_low = 100, 100
        self.min_area = 200

        self.bridge = CvBridge()
        self.last_time = self.get_clock().now()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/mavros/zed/rgb/color/rect/image',
            self._image_callback, 5)

        # Direct MAVROS Override Publisher
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.rc_pub = self.create_publisher(OverrideRCIn, '/mavros/rc/override', qos)
        
        # Diagnostics visualization topic
        self.viz_pub = self.create_publisher(Image, '/hightide/yellow_box_viz', 5)

        self.get_logger().info('Direct RC Override Yellow Box Tracking Node Started')

    def _find_yellow_bbox(self, image):
        """Processes BGR image to isolate and bound the largest yellow box segment."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([self.hue_low, self.sat_low, self.val_low])
        upper_yellow = np.array([self.hue_high, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_area:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        return {"x": x, "y": y, "w": w, "h": h}

    def _image_callback(self, msg: Image):
        """Processes camera feed, applies control logic, and sends direct RC overrides."""
        if not self.enabled:
            return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        if dt <= 0:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        img_h, img_w = cv_image.shape[:2]
        box = self._find_yellow_bbox(cv_image)

        # Prepare direct 18-channel RC message (65535 means no override change)
        channels = [65535] * 18

        # If box is lost, output safe neutral PWM values to stop moving
        if box is None:
            channels[0] = 1500  # Pitch
            channels[1] = 1500  # Roll
            channels[2] = 1500  # Throttle/Heave
            channels[3] = 1500  # Yaw
            channels[4] = 1500  # Forward/Surge
            channels[5] = 1500  # Lateral/Sway
            
            out_msg = OverrideRCIn()
            out_msg.channels = channels
            self.rc_pub.publish(out_msg)
            return

        # Core Centroid Properties
        center_x = box["x"] + (box["w"] / 2.0)
        center_y = box["y"] + (box["h"] / 2.0)
        distance_m = 1800.0 / box["w"]

        # Calculate Normalized Frame Errors
        lateral_error = (center_x / img_w) - self.target_x
        vertical_error = (center_y / img_h) - self.target_y
        range_error = distance_m - self.approach_dist

        # Compute PID Outputs mapped straight to PWM adjustments around neutral 1500
        sway_output = self.lateral_pid.compute(lateral_error, dt)
        heave_output = self.vertical_pid.compute(vertical_error, dt)
        surge_output = self.range_pid.compute(range_error, dt)

        # Apply standard ArduSub direction sign mappings:
        # Channel 3 (Throttle): positive error -> target below center -> lower PWM to descend
        # Channel 5 (Forward): positive error -> too far away -> higher PWM to surge forward
        # Channel 6 (Lateral): positive error -> target right -> higher PWM to strafe right
        
        pwm_pitch = 1500
        pwm_roll = 1500
        pwm_yaw = 1500
        pwm_throttle = 1500 - int(heave_output)  # Decreases PWM to dive
        pwm_forward = 1500 + int(surge_output)   # Increases PWM to move forward
        pwm_lateral = 1500 + int(sway_output)    # Increases PWM to strafe right

        # Load and clamp values within safe RC boundaries [1100, 1900]
        channels[0] = max(1100, min(1900, pwm_pitch))     # Pitch (Ch 1)
        channels[1] = max(1100, min(1900, pwm_roll))      # Roll (Ch 2)
        channels[2] = max(1100, min(1900, pwm_throttle))  # Throttle/Heave (Ch 3)
        channels[3] = max(1100, min(1900, pwm_yaw))       # Yaw (Ch 4)
        channels[4] = max(1100, min(1900, pwm_forward))   # Forward/Surge (Ch 5)
        channels[5] = max(1100, min(1900, pwm_lateral))   # Lateral/Sway (Ch 6)

        # Publish the final RC override command message
        rc_msg = OverrideRCIn()
        rc_msg.channels = channels
        self.rc_pub.publish(rc_msg)

        # Publish visual confirmation if anyone is listening
        if self.viz_pub.get_subscription_count() > 0:
            viz_img = cv_image.copy()
            cv2.rectangle(viz_img, (box["x"], box["y"]), 
                          (box["x"] + box["w"], box["y"] + box["h"]), (0, 255, 0), 2)
            cv2.circle(viz_img, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
            try:
                self.viz_pub.publish(self.bridge.cv2_to_imgmsg(viz_img, 'bgr8'))
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = DirectYellowBoxRCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()