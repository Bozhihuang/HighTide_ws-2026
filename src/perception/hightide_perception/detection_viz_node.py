#!/usr/bin/env python3

# Detection Visualization Node — Draws bounding boxes and crosshair on camera feed.


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
        self.bridge = CvBridge() # bridge that will convert img format
        self.latest_detections = None
        #empties any detecion data left over

        self.image_sub = self.create_subscription(
            #subscriber for raw camera feed
            Image, '/mavros/zed/rgb/color/rect/image', 
            self._image_callback, 5)
        
        self.det_sub = self.create_subscription(
            # subscriber to channel that publishes detection data in form of list 
            # with class name, confidence, bounding box coordinates, and depth
            DetectionArray, '/hightide/tracked_targets',
            self._det_callback, 10)
        
        # publisher for the annotated image with bounding boxes and crosshair
        self.viz_pub = self.create_publisher(
            Image, '/hightide/viz/detection_overlay', 5)

        self.get_logger().info('Detection Viz Node started')

    def _det_callback(self, msg):
        # stores latest detection info
        self.latest_detections = msg

    def _image_callback(self, msg):
        try:
            # convert the image to a bgr8 format so we can draw on it
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            # cv2 expects this type
        except Exception:
            return
        
        # get img dimensions
        h, w = frame.shape[:2]

        # Draw crosshair at image center
        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 1)

        if self.latest_detections:
            for det in self.latest_detections.detections:
    
                # for each detection(which follows detection.msg format),
                # set color based on class name, default to gray if not found
                color = CATEGORY_COLORS.get(det.class_name, DEFAULT_COLOR)
                # draw bounding box
                x1, y1 = int(det.x_min), int(det.y_min)
                x2, y2 = int(det.x_max), int(det.y_max)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # draw label above bounding box with class name, confidence, and depth if more than 0
                label = f'{det.class_name} {det.confidence:.2f}'
                if det.depth_m > 0:
                    label += f' {det.depth_m:.1f}m'
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            # draw number of detections on top left corner of image
            cv2.putText(frame,
                        f'Detections: {len(self.latest_detections.detections)}',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        #publish the annotated image to /hightide/viz/detection_overlay
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
