#!/usr/bin/env python3
"""
Detection Visualization Node — Draws bounding boxes and crosshair on camera feed.
"""

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from hightide_interfaces.msg import DetectionArray

# Color mapping by class category
CATEGORY_COLORS = {
    'gate': (0, 255, 0), 'gate_divider': (0, 255, 0),
    'symbol_compass': (0, 255, 0), 'symbol_pickaxe': (0, 255, 0),
    'symbol_lifering': (0, 255, 0), 'symbol_sos': (0, 255, 0),
    'pipe_red': (0, 255, 255), 'pipe_white': (0, 255, 255),
    'bin': (255, 200, 0), 'symbol_fire': (0, 100, 255),
    'symbol_blood': (0, 0, 255),
    'torpedo_board': (0, 0, 255), 'torpedo_hole_large': (0, 0, 255),
    'torpedo_hole_small': (0, 0, 255),
    'octagon': (255, 0, 255), 'table': (255, 0, 255),
    'basket': (255, 0, 255),
    'path_marker': (0, 165, 255),
}
DEFAULT_COLOR = (200, 200, 200)


class DetectionVizNode(Node):
    """Overlays tracked detections and crosshair on camera image."""

    def __init__(self):
        super().__init__('detection_viz_node')
        self.bridge = CvBridge()
        self.latest_detections = None

        self.image_sub = self.create_subscription(
            Image, '/mavros/zed/rgb/color/rect/image',
            self._image_callback, 5)
        self.det_sub = self.create_subscription(
            DetectionArray, '/hightide/tracked_targets',
            self._det_callback, 10)
        self.viz_pub = self.create_publisher(
            Image, '/hightide/viz/detection_overlay', 5)

        self.get_logger().info('Detection Viz Node started')

    def _det_callback(self, msg):
        self.latest_detections = msg

    def _image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return

        h, w = frame.shape[:2]

        # Draw crosshair at image center
        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 1)

        if self.latest_detections:
            for det in self.latest_detections.detections:
                color = CATEGORY_COLORS.get(det.class_name, DEFAULT_COLOR)
                x1, y1 = int(det.x_min), int(det.y_min)
                x2, y2 = int(det.x_max), int(det.y_max)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f'{det.class_name} {det.confidence:.2f}'
                if det.depth_m > 0:
                    label += f' {det.depth_m:.1f}m'
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            cv2.putText(frame,
                        f'Detections: {len(self.latest_detections.detections)}',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        try:
            self.viz_pub.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = DetectionVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()
