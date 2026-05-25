#!/usr/bin/env python3
"""
Target Tracker Node — IoU-based tracking with ZED depth augmentation.

Matches detections across frames using IoU, adds depth from ZED depth map,
and maintains smoothed bounding boxes with temporal filtering.
"""

import numpy as np
import time as pytime
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from hightide_interfaces.msg import Detection, DetectionArray
from hightide_perception import CLASS_NAMES


class TrackedTarget:
    """A single tracked target across frames."""

    def __init__(self, detection: Detection, track_id: int):
        self.track_id = track_id
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.confidence = detection.confidence
        self.bbox = [detection.x_min, detection.y_min,
                     detection.x_max, detection.y_max]
        self.center_x = detection.center_x
        self.center_y = detection.center_y
        self.depth_m = -1.0
        self.last_seen = pytime.time()
        self.hit_count = 1

    def update(self, detection: Detection, alpha=0.3):
        """Update tracked target with new detection using EMA smoothing."""
        self.confidence = detection.confidence
        # Exponential moving average on bbox
        self.bbox[0] = alpha * detection.x_min + (1 - alpha) * self.bbox[0]
        self.bbox[1] = alpha * detection.y_min + (1 - alpha) * self.bbox[1]
        self.bbox[2] = alpha * detection.x_max + (1 - alpha) * self.bbox[2]
        self.bbox[3] = alpha * detection.y_max + (1 - alpha) * self.bbox[3]
        self.center_x = (self.bbox[0] + self.bbox[2]) / 2.0
        self.center_y = (self.bbox[1] + self.bbox[3]) / 2.0
        self.last_seen = pytime.time()
        self.hit_count += 1


class TargetTrackerNode(Node):
    """IoU-based multi-object tracker with depth augmentation from ZED."""

    def __init__(self):
        super().__init__('target_tracker_node')

        self.declare_parameter('max_tracking_age', 1.0)
        self.declare_parameter('iou_threshold', 0.3)
        self.declare_parameter('depth_sample_radius', 5)

        self.max_age = self.get_parameter('max_tracking_age').value
        self.iou_thresh = self.get_parameter('iou_threshold').value
        self.depth_radius = self.get_parameter('depth_sample_radius').value

        self.bridge = CvBridge()
        self.tracks: dict[int, TrackedTarget] = {}
        self.next_track_id = 0
        self.depth_image = None

        # Subscribers
        self.det_sub = self.create_subscription(
            DetectionArray, '/hightide/detections',
            self._detection_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/zed/zed_node/depth/depth_registered',
            self._depth_callback, 5)

        # Publisher
        self.tracked_pub = self.create_publisher(
            DetectionArray, '/hightide/tracked_targets', 10)

        self.get_logger().info('Target Tracker Node started')

    @staticmethod
    def _compute_iou(box_a, box_b):
        """Compute IoU between two [x1,y1,x2,y2] boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    def _sample_depth(self, cx: float, cy: float) -> float:
        """Sample depth from ZED depth map at given pixel coordinates."""
        if self.depth_image is None:
            return -1.0

        h, w = self.depth_image.shape[:2]
        cx_int, cy_int = int(cx), int(cy)

        if cx_int < 0 or cx_int >= w or cy_int < 0 or cy_int >= h:
            return -1.0

        r = self.depth_radius
        y_start = max(0, cy_int - r)
        y_end = min(h, cy_int + r + 1)
        x_start = max(0, cx_int - r)
        x_end = min(w, cx_int + r + 1)

        patch = self.depth_image[y_start:y_end, x_start:x_end]
        valid = patch[np.isfinite(patch) & (patch > 0.0)]

        if len(valid) == 0:
            return -1.0

        return float(np.median(valid))

    def _depth_callback(self, msg: Image):
        """Store latest depth image."""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        except Exception as e:
            self.get_logger().warn(f'Depth conversion error: {e}')

    def _detection_callback(self, msg: DetectionArray):
        """Match detections to existing tracks, update, and publish."""
        now = pytime.time()

        # Remove stale tracks
        stale_ids = [tid for tid, t in self.tracks.items()
                     if (now - t.last_seen) > self.max_age]
        for tid in stale_ids:
            del self.tracks[tid]

        # Match new detections to existing tracks
        new_dets = list(msg.detections)
        matched_track_ids = set()
        matched_det_ids = set()

        for det_idx, det in enumerate(new_dets):
            det_box = [det.x_min, det.y_min, det.x_max, det.y_max]
            best_iou = 0.0
            best_tid = None

            for tid, track in self.tracks.items():
                if tid in matched_track_ids:
                    continue
                if track.class_id != det.class_id:
                    continue
                iou = self._compute_iou(det_box, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_iou >= self.iou_thresh and best_tid is not None:
                self.tracks[best_tid].update(det)
                matched_track_ids.add(best_tid)
                matched_det_ids.add(det_idx)

        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(new_dets):
            if det_idx not in matched_det_ids:
                track = TrackedTarget(det, self.next_track_id)
                self.tracks[self.next_track_id] = track
                self.next_track_id += 1

        # Add depth to all active tracks
        for track in self.tracks.values():
            depth = self._sample_depth(track.center_x, track.center_y)
            if depth > 0:
                if track.depth_m > 0:
                    track.depth_m = 0.7 * track.depth_m + 0.3 * depth  # EMA
                else:
                    track.depth_m = depth

        # Publish tracked targets
        out_msg = DetectionArray()
        out_msg.header = msg.header
        out_msg.image_width = msg.image_width
        out_msg.image_height = msg.image_height

        for track in self.tracks.values():
            det = Detection()
            det.header = msg.header
            det.class_id = track.class_id
            det.class_name = track.class_name
            det.confidence = track.confidence
            det.x_min = track.bbox[0]
            det.y_min = track.bbox[1]
            det.x_max = track.bbox[2]
            det.y_max = track.bbox[3]
            det.center_x = track.center_x
            det.center_y = track.center_y
            det.depth_m = track.depth_m
            det.width = track.bbox[2] - track.bbox[0]
            det.height = track.bbox[3] - track.bbox[1]
            out_msg.detections.append(det)

        self.tracked_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TargetTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
