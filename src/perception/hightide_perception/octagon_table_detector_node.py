#!/usr/bin/env python3
"""
Octagon Table Detector — PLACEHOLDER.

The Restore/octagon task has a large patterned board (the "capability-matrix"
table) on the floor INSIDE the octagon. Because it sits physically inside, it is
a better localization cue than the buoy border (which is visible from outside):
the octagon behavior (NavigateIntoOctagon) already prefers a detection with
class_name 'octagon_table' over 'buoy' and centers/stops on it.

The table is NOT a class in the trained ffc YOLO model, so it is published as
the sentinel class 'octagon_table' (id 9, see hightide_perception
SENTINEL_CLASS_NAMES) by this auxiliary detector — mirroring how the white
slalom pipe is detected classically.

STATUS: SKELETON ONLY. The image pipeline, topics, params, and message
formatting are wired, but `_detect_table()` returns no detections yet — fill in
the actual detection (e.g. classical grid/checkerboard finding, or a small
segmentation model) once we have in-water footage of the table.

REMAINING WIRING (one step) to make the cue reach the mission behavior:
  the mission behavior reads /hightide/tracked_targets (via the target tracker,
  which currently only ingests /hightide/detections). Either
    (a) have the target_tracker_node also subscribe to /hightide/octagon_table
        and merge those detections into the tracked stream, or
    (b) have this node publish onto /hightide/detections (NOT recommended — the
        tracker treats each message as the full frame's detection set).
  Option (a) is the clean path.
"""

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from hightide_interfaces.msg import Detection, DetectionArray

TABLE_CLASS_ID = 9        # sentinel — see hightide_perception.SENTINEL_CLASS_NAMES
TABLE_CLASS_NAME = 'octagon_table'


class OctagonTableDetectorNode(Node):
    """Detect the under-octagon capability-matrix table and publish it as a
    sentinel 'octagon_table' detection. PLACEHOLDER detection logic."""

    def __init__(self):
        super().__init__('octagon_table_detector_node')

        self.declare_parameter('image_topic', '/mavros/zed/rgb/color/rect/image')
        self.declare_parameter('output_topic', '/hightide/octagon_table')
        self.declare_parameter('publish_viz', True)
        # Reserved knobs for the real detector (grid/board finding). Declared now
        # so launch files / tuning don't need to change once it's implemented.
        self.declare_parameter('min_area_frac', 0.02)   # board must fill >= this frac of frame
        self.declare_parameter('min_confidence', 0.4)

        self.image_topic = self.get_parameter('image_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.publish_viz = bool(self.get_parameter('publish_viz').value)
        self.min_area_frac = float(self.get_parameter('min_area_frac').value)
        self.min_confidence = float(self.get_parameter('min_confidence').value)

        self.bridge = CvBridge()

        # ZED image topics are typically BEST_EFFORT; a BEST_EFFORT subscriber is
        # compatible with both, so it never QoS-mismatches into silence.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5)
        self.sub = self.create_subscription(
            Image, self.image_topic, self._image_cb, sensor_qos)

        self.pub = self.create_publisher(DetectionArray, self.output_topic, 10)
        self.viz_pub = None
        if self.publish_viz:
            self.viz_pub = self.create_publisher(
                Image, self.output_topic + '_image', 5)

        self.get_logger().warn(
            'Octagon table detector up (PLACEHOLDER — _detect_table() returns '
            f'nothing yet). Listening on {self.image_topic}, publishing '
            f'{self.output_topic}.')

    def _detect_table(self, img):
        """Return a list of table detections (dicts) for this frame.

        PLACEHOLDER: returns []. Implement real detection here — e.g. find the
        large high-contrast patterned rectangle on the floor and report its
        bounding box + a confidence. Each dict should carry:
            center_x, center_y, width, height, confidence
        """
        # TODO(perception): implement capability-matrix table detection.
        return []

    def _image_cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        h, w = img.shape[:2]
        tables = self._detect_table(img)

        det_array = DetectionArray()
        det_array.header = msg.header
        det_array.image_width = w
        det_array.image_height = h

        viz = img.copy() if self.viz_pub is not None else None

        for tdet in tables:
            bw, bh = tdet['width'], tdet['height']
            x = tdet['center_x'] - bw / 2.0
            y = tdet['center_y'] - bh / 2.0
            det = Detection()
            det.header = msg.header
            det.class_id = TABLE_CLASS_ID
            det.class_name = TABLE_CLASS_NAME
            det.confidence = float(tdet.get('confidence', 1.0))
            det.x_min = float(x)
            det.y_min = float(y)
            det.x_max = float(x + bw)
            det.y_max = float(y + bh)
            det.center_x = float(tdet['center_x'])
            det.center_y = float(tdet['center_y'])
            det.width = float(bw)
            det.height = float(bh)
            det.depth_m = -1.0  # RGB-only placeholder — no range yet
            det_array.detections.append(det)

            if viz is not None:
                cv2.rectangle(viz, (int(x), int(y)),
                              (int(x + bw), int(y + bh)), (0, 255, 255), 2)
                cv2.putText(viz, TABLE_CLASS_NAME, (int(x), max(0, int(y) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        self.pub.publish(det_array)
        if viz is not None:
            try:
                self.viz_pub.publish(self.bridge.cv2_to_imgmsg(viz, 'bgr8'))
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = OctagonTableDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
