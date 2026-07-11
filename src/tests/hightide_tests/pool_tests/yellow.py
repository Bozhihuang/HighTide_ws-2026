#!/usr/bin/env python3
"""
Pool Test: Yellow Box Tracking via OpenCV & Waypoint Navigation (No cv_bridge)

Uses raw OpenCV HSV color filtering on ZED images to detect a yellow box top,
samples ZED depth to find its distance, and iteratively sends goals to the
waypoint navigator action server until the sub is exactly 0.5m in front of it.
"""

import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_srvs.srv import SetBool, Trigger
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from hightide_interfaces.action import NavigateToWaypoint
from hightide_navigation import quaternion_to_yaw


# --- Native NumPy Image Converters (Bypasses cv_bridge & NumPy 2.x crash) ---
def imgmsg_to_cv2_bgr8(msg: Image) -> np.ndarray:
    """Converts a ROS sensor_msgs/Image (bgr8) to an OpenCV NumPy array."""
    img = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.step > msg.width * 3:
        img = img.reshape((msg.height, msg.step))[:, :msg.width * 3]
    return img.reshape((msg.height, msg.width, 3))

def imgmsg_to_cv2_32FC1(msg: Image) -> np.ndarray:
    """Converts a ROS sensor_msgs/Image (32FC1 float depth) to a NumPy array."""
    img = np.frombuffer(msg.data, dtype=np.float32)
    if msg.step > msg.width * 4:
        stride = msg.step // 4
        img = img.reshape((msg.height, stride))[:, :msg.width]
    else:
        img = img.reshape((msg.height, msg.width))
    return img
# -----------------------------------------------------------------------------


class YellowBoxOpenCVNavTest(Node):
    def __init__(self):
        super().__init__('pool_test_yellow_box_opencv')
        
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Service & Action Clients
        self.arm_cli = self.create_client(SetBool, '/hightide/arm')
        self.alt_hold_cli = self.create_client(Trigger, '/hightide/set_alt_hold')
        self.nav_client = ActionClient(self, NavigateToWaypoint, '/hightide/navigate_to_waypoint')
        
        # Sensor Subscribers
        self.create_subscription(Image, '/mavros/zed/left/image_rect_color', self._img_cb, 10)
        self.create_subscription(Image, '/mavros/zed/depth/depth_registered', self._depth_cb, sensor_qos)
        self.create_subscription(Odometry, '/mavros/zed/odom', self._odom_cb, sensor_qos)
        self.create_subscription(Imu, '/mavros/imu/data', self._imu_cb, sensor_qos)
        
        # State Variables
        self.current_pose = None
        self.current_yaw = None
        self.depth_image = None
        self.latest_box_target = None  
        
        # HSV Color Thresholds for Yellow Box Top
        self.hsv_lower = np.array([20, 100, 100])
        self.hsv_upper = np.array([35, 255, 255])
        
        # Wait for Action Server
        while not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for /hightide/navigate_to_waypoint server...')

    def _odom_cb(self, msg):
        self.current_pose = msg.pose.pose

    def _imu_cb(self, msg):
        self.current_yaw = quaternion_to_yaw(msg.orientation)

    def _depth_cb(self, msg):
        try:
            self.depth_image = imgmsg_to_cv2_32FC1(msg)
        except Exception as e:
            self.get_logger().warn(f'Depth conversion failed: {e}')

    def _img_cb(self, msg):
        try:
            cv_img = imgmsg_to_cv2_bgr8(msg)
        except Exception as e:
            return

        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_c)
            
            if area > 500:  
                M = cv2.moments(largest_c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    self.latest_box_target = {'cx': cx, 'cy': cy, 'area': area, 'width': cv_img.shape[1]}
                    return
                    
        self.latest_box_target = None

    def _get_target_depth(self, cx, cy, radius=5):
        if self.depth_image is None:
            return -1.0
        h, w = self.depth_image.shape[:2]
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            return -1.0
            
        patch = self.depth_image[max(0, cy-radius):min(h, cy+radius+1), 
                                 max(0, cx-radius):min(w, cx+radius+1)]
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        return float(np.median(valid)) if len(valid) > 0 else -1.0

    def send_waypoint_step(self, rel_surge, rel_sway):
        if self.current_pose is None or self.current_yaw is None:
            return False
            
        goal = NavigateToWaypoint.Goal()
        
        cos_y = math.cos(self.current_yaw)
        sin_y = math.sin(self.current_yaw)
        
        goal.target_pose.pose.position.x = self.current_pose.position.x + (rel_surge * cos_y - rel_sway * sin_y)
        goal.target_pose.pose.position.y = self.current_pose.position.y + (rel_surge * sin_y + rel_sway * cos_y)
        goal.target_pose.pose.position.z = self.current_pose.position.z
        goal.target_pose.pose.orientation = self.current_pose.orientation  
        
        goal.position_tolerance = 0.15
        goal.yaw_tolerance = 0.1
        goal.timeout_sec = 4.0  
        
        future = self.nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            return False
            
        res_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, res_future)
        return res_future.result().result.success

    def run_test(self):
        self.get_logger().info('=== STARTING YELLOW BOX OPENCV NAVIGATION TEST ===')
        
        while self.current_pose is None or self.current_yaw is None or self.depth_image is None:
            self.get_logger().info('Waiting for ZED video, depth, and IMU sensor streams...')
            rclpy.spin_once(self, timeout_sec=1.0)
            
        input("Sub in water facing general direction of yellow box? Press Enter to ARM & ALT HOLD...")
        
        req = SetBool.Request()
        req.data = True
        self.arm_cli.call_async(req)
        self.alt_hold_cli.call_async(Trigger.Request())
        self.get_logger().info('Sub ARMED. Starting autonomous visual navigation...')
        
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)
                
                if self.latest_box_target is None:
                    self.get_logger().warn('Yellow box not detected by OpenCV. Waiting...')
                    continue
                    
                cx = self.latest_box_target['cx']
                cy = self.latest_box_target['cy']
                img_w = self.latest_box_target['width']
                
                depth_m = self._get_target_depth(cx, cy)
                if depth_m <= 0.0:
                    self.get_logger().warn('Box detected in RGB, but depth is invalid (-1.0). Approaching blindly...')
                    depth_m = 1.5  
                    
                normalized_x_err = (cx / img_w) - 0.5
                lateral_offset_m = depth_m * math.tan(normalized_x_err * math.radians(110.0 / 2.0))
                
                self.get_logger().info(
                    f"Box Locked | Depth: {depth_m:.2f}m | Lateral Offset: {lateral_offset_m:.2f}m"
                )
                
                if 0.40 <= depth_m <= 0.60 and abs(lateral_offset_m) < 0.15:
                    self.get_logger().info('SUCCESS: Subpositioned exactly ~0.5m in front of the yellow box!')
                    break
                    
                forward_step = max(-0.5, min(1.0, depth_m - 0.5))
                sway_step = max(-0.5, min(0.5, lateral_offset_m))
                
                self.get_logger().info(f"Stepping -> Surge: {forward_step:.2f}m, Sway: {sway_step:.2f}m")
                self.send_waypoint_step(forward_step, sway_step)
                
        except KeyboardInterrupt:
            pass
            
        self.get_logger().info('Test finished. DISARMING...')
        req.data = False
        self.arm_cli.call_async(req)
        self.get_logger().info('=== TEST COMPLETE ===')

def main(args=None):
    rclpy.init(args=args)
    node = YellowBoxOpenCVNavTest()
    try:
        node.run_test()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()