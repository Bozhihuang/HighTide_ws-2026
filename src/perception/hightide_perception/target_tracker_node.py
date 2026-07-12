#!/usr/bin/env python3




# IoU: Intersection over Union: Way to measure overlap between two bounding boxes.
#   Divides the area of intersection by the area of union, giving a value between 0 and 1.
#   larger IoU = more overlap.

# IoU threshold: Minimum IoU required to consider two detections the same object across frames.

# Temporal filtering: Smooths bounding box coordinates and confidence scores over time to reduce jitter.
#   Works by doing a weighted average of the current detection andprevious detections of the same object.
#   This makes it so that the bounding box doesn't jump between frames and smoothly moves to new position.

import numpy as np
import time as pytime
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from hightide_interfaces.msg import Detection, DetectionArray
from hightide_perception import CLASS_NAMES


class TrackedTarget:
# this is a tracked target object that stores all the raw detection info but cleaner(the bbox is smoothed),
# with the addition of a track_id and last_seen time, and a hit_count 
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
        # Segmentation extras (0.0 / [] for detect-only models). Carried through
        # so mission/nav get the mask centroid, area, and contour, not just bbox.
        self.mask_area = detection.mask_area
        self.mask_polygon = list(detection.mask_polygon)
        self.last_seen = pytime.time()
        self.hit_count = 1 # number of frames this target has been detected as something


    # gets called when we are sure we are getting a new detection of same target
    def update(self, detection: Detection, alpha=0.3): 
        """Update tracked target with new detection using EMA smoothing."""
        self.confidence = detection.confidence
        # Exponential moving average on bbox
        # Alpha is the smoothing factor,
        # higher alpha means more weight to new detection when calculating the average
        self.bbox[0] = alpha * detection.x_min + (1 - alpha) * self.bbox[0]
        self.bbox[1] = alpha * detection.y_min + (1 - alpha) * self.bbox[1]
        self.bbox[2] = alpha * detection.x_max + (1 - alpha) * self.bbox[2]
        self.bbox[3] = alpha * detection.y_max + (1 - alpha) * self.bbox[3]
        # EMA the detection-provided center (which is the MASK centroid for seg
        # models, bbox center for detect models) rather than recomputing from
        # the bbox — this preserves the more-accurate segmentation centroid.
        self.center_x = alpha * detection.center_x + (1 - alpha) * self.center_x
        self.center_y = alpha * detection.center_y + (1 - alpha) * self.center_y
        # Mask area smooths; the polygon just tracks the latest frame.
        self.mask_area = alpha * detection.mask_area + (1 - alpha) * self.mask_area
        self.mask_polygon = list(detection.mask_polygon)
        # update last seen time and hit count
        self.last_seen = pytime.time()
        self.hit_count += 1


class TargetTrackerNode(Node):
    # target tracker node essentially cleans raw detections info tracked targets, smooths depth and bbox, removes bad data
    def __init__(self):
        super().__init__('target_tracker_node')


        # Default parameters
        self.declare_parameter('max_tracking_age', 1.0)
        self.declare_parameter('iou_threshold', 0.3)
        self.declare_parameter('depth_sample_radius', 5)
        # Minimum detections before a track is published. The mission commits
        # decisions (gate side, bin choice) on a single published detection,
        # so a one-frame YOLO false positive must not reach it. At ~15-30 fps
        # this delays real objects by only ~0.1-0.2 s.
        self.declare_parameter('min_hits', 3)

        # override parameters with whats in launch file if specially defined
        self.max_age = self.get_parameter('max_tracking_age').value
        self.iou_thresh = self.get_parameter('iou_threshold').value
        self.depth_radius = self.get_parameter('depth_sample_radius').value
        self.min_hits = self.get_parameter('min_hits').value

        self.bridge = CvBridge() # bridge that will convert depth img format
        # initialize empty dictionary assigning track_id int to TrackedTarget object
        self.tracks: dict[int, TrackedTarget] = {} 
        self.next_track_id = 0
        self.depth_image = None

        # Subscribers to get raw detections and depth 
        self.det_sub = self.create_subscription(
            DetectionArray, '/hightide/detections',
            self._detection_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/mavros/zed/depth/depth_registered',
            self._depth_callback, 5)

        # Publisher for polished detections data with depth and tracking info
        self.tracked_pub = self.create_publisher(
            DetectionArray, '/hightide/tracked_targets', 10)

        self.get_logger().info('Target Tracker Node started')

    @staticmethod # static method because it doesn't need any instance variables
    def _compute_iou(box_a, box_b):
        # Compute IoU between two bounding boxes
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1) # computing area of intersection
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter # computing area of union

        return inter / union if union > 0 else 0.0

    def _sample_depth(self, cx: float, cy: float) -> float:
        # Sample depth from ZED depth map at given pixel coordinates.
        # Any errors return -1.0, otherwise return depth in meters.

        if self.depth_image is None:
            return -1.0 # if we get a detection that is outside the image, return -1.0

        h, w = self.depth_image.shape[:2]
        cx_int, cy_int = int(cx), int(cy) # center x and y values as int

        if cx_int < 0 or cx_int >= w or cy_int < 0 or cy_int >= h:
            return -1.0 # if we get a detection that is outside the image, return -1.0


        r = self.depth_radius
        y_start = max(0, cy_int - r)
        y_end = min(h, cy_int + r + 1)
        x_start = max(0, cx_int - r)
        x_end = min(w, cx_int + r + 1)
        # makes a square patch around the center of the bounding box,
        # with square size having radius r determined by depth_radius parameter

        patch = self.depth_image[y_start:y_end, x_start:x_end]   
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        # samples depth values in the square patch,
        # and filters for only valid depth values (finite and greater than 0.0)


        if len(valid) == 0: 
            return -1.0
        # if there are no valid depth values in the patch, return -1.0

        return float(np.median(valid)) 
        # returns the median of the valid depth values in the patch,

    def _depth_callback(self, msg: Image):
        try:
            # convert depth image to a 2d array of float32 values in meters and store it
            # (distance from camera to each pixel in og image)
            # C1 means 1 channel (color range), this makes it like a grayscale image
            # (also has 1 channel but instead of brightness, it is based on depth)
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        except Exception as e:
            self.get_logger().warn(f'Depth conversion error: {e}')

    def _detection_callback(self, msg: DetectionArray):
        now = pytime.time()

        # identify stale tracks (things were tracking that haven't been seen for a while)
        stale_ids = [tid for tid, t in self.tracks.items()
                     if (now - t.last_seen) > self.max_age]
        
        #delete stale tracks from the dictionary of tracks
        for tid in stale_ids: 
            del self.tracks[tid]

        # Match new detections to existing tracks
        new_dets = list(msg.detections)
        # set that will contain the track ids that have been matched to a new detection (not being in the list means the track is stale)
        matched_track_ids = set()
        # set that will contain index of new detections that were matched to existing tracks (not being in the list means it is a new object being tracked)
        matched_det_ids = set()

        for det_idx, det in enumerate(new_dets):
            det_box = [det.x_min, det.y_min, det.x_max, det.y_max]
            # these 2 are used to track the best match for this new detection(if we find one) to an existing track
            best_iou = 0.0
            best_tid = None
            # for every new detection check all of the tracks to see if it is the same object as one of them,
            # this is based on IoU and class_id, if it is the same object, update the track with the new detection info
            for tid, track in self.tracks.items():
                # if not the same class or we already matched this track to a new detection skip it
                if tid in matched_track_ids:
                    continue
                if track.class_id != det.class_id:
                    continue
                iou = self._compute_iou(det_box, track.bbox)
                # take the track with most overlap with the new detection,
                
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            # if the best match has an IoU above the IoU threshold,
            #  we will update that track with the new detection info
            if best_iou >= self.iou_thresh and best_tid is not None:
                self.tracks[best_tid].update(det)
                # add track id and detection index to the matched sets
                matched_track_ids.add(best_tid)
                matched_det_ids.add(det_idx)

        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(new_dets):
            if det_idx not in matched_det_ids:
                # init new track with the new detection and assign it a new track id
                track = TrackedTarget(det, self.next_track_id)
                # add the new track to the dictionary
                self.tracks[self.next_track_id] = track
                # increment the next track id for the next new track
                self.next_track_id += 1

        # Add depth to all active tracks
        for track in self.tracks.values():
            depth = self._sample_depth(track.center_x, track.center_y)
            if depth > 0:
                if track.depth_m > 0:
                    # EMA: similar weighted blending to temporal filtering,
                    # but weights are different (0.7 to old depth, 0.3 to new depth)
                    track.depth_m = 0.7 * track.depth_m + 0.3 * depth  
                    # this prevent the depth value from jumping around too much
                else:
                    track.depth_m = depth

        # Publish tracked targets, follows DetectionArray.msg format,
        # but detection[] is a list of tracked targets instead of raw detections
        out_msg = DetectionArray()
        out_msg.header = msg.header
        out_msg.image_width = msg.image_width
        out_msg.image_height = msg.image_height

        # fill out the detection[] list with the tracked targets,
        # look at detection.msg for more on the fields of each detection
        for track in self.tracks.values():
            # suppress tracks that haven't been confirmed by enough frames yet
            # (single-frame false positives never get published)
            if track.hit_count < self.min_hits:
                continue
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
            det.mask_area = track.mask_area
            det.mask_polygon = track.mask_polygon
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

if __name__ == '__main__':
    main()
