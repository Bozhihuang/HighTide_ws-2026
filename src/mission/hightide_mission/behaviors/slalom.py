#!/usr/bin/env python3
"""
Slalom Pole Detector — classical OpenCV on the ZED RGB image (no depth, no YOLO).

The ffc YOLO model only knows the RED slalom poles ('slalom' class) and is blind
to the WHITE pipes, so the sub can drive into a white pipe it literally cannot
see. This node finds BOTH by STRUCTURE instead of by a trained class: it looks
for tall, narrow, high-contrast vertical bars in the grayscale image (a pole is
a pole whether it's red or white, and this survives underwater red attenuation),
then TAGS each pole red vs white from its color for the divider-side bonus.

Publishes a hightide_interfaces/DetectionArray on /hightide/slalom_poles:
  - class_name 'red_pole'  (class_id 6, mirrors the model's 'slalom')
  - class_name 'white_pole'(class_id 8, sentinel — not a model class)
with center_x / width / bbox populated so the slalom behavior can drive the gap
between poles while keeping the red divider on the correct side.

RGB-only. Every threshold is a ROS parameter — tune on in-water footage
(dataset: slalom_woollett, slalom_airbnb) via `ros2 param set` while watching
/hightide/slalom_poles_image.
"""

from collections import deque
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from hightide_interfaces.msg import Detection, DetectionArray

RED_CLASS_ID = 6     # mirrors CLASS_NAMES 'slalom'
WHITE_CLASS_ID = 8   # sentinel — white pipe is not a trained class


class SlalomPoleDetectorNode(Node):
    """Detect red + white slalom poles from RGB structure and publish them."""

    def __init__(self):
        super().__init__('slalom_pole_detector_node')

        self.declare_parameter('image_topic', '/mavros/zed/rgb/color/rect/image')
        # Edge / shape gates — a pole is a TALL, NARROW, high-contrast bar.
        self.declare_parameter('canny_low', 40)
        self.declare_parameter('canny_high', 120)
        self.declare_parameter('min_height_frac', 0.30)   # pole spans >= this frac of frame height
        self.declare_parameter('max_width_frac', 0.20)    # pole narrower than this frac of frame width
        self.declare_parameter('min_aspect', 2.5)         # height / width >= this
        self.declare_parameter('close_kernel_h', 25)      # vertical morphology to join a pole's two edges
        # Colour tagging (HSV). Underwater red is weak, so anything not clearly
        # red is treated as a white/neutral pole rather than dropped.
        self.declare_parameter('red_sat_min', 70)
        self.declare_parameter('red_val_min', 40)
        # Temporal M-of-N filter: only publish a pole seen in >= min_hits of the
        # last `window` frames, matched across frames by x-position. Kills
        # one-frame edge artifacts so steering doesn't jerk.
        self.declare_parameter('temporal_window', 5)
        self.declare_parameter('temporal_min_hits', 3)
        self.declare_parameter('cluster_x_tol_frac', 0.06)  # same-pole x match tolerance
        self.declare_parameter('publish_viz', True)

        self.image_topic = self.get_parameter('image_topic').value
        self.canny_low = int(self.get_parameter('canny_low').value)
        self.canny_high = int(self.get_parameter('canny_high').value)
        self.min_height_frac = float(self.get_parameter('min_height_frac').value)
        self.max_width_frac = float(self.get_parameter('max_width_frac').value)
        self.min_aspect = float(self.get_parameter('min_aspect').value)
        self.close_kernel_h = int(self.get_parameter('close_kernel_h').value)
        self.red_sat_min = int(self.get_parameter('red_sat_min').value)
        self.red_val_min = int(self.get_parameter('red_val_min').value)
        self.temporal_window = max(1, int(self.get_parameter('temporal_window').value))
        self.temporal_min_hits = max(1, int(self.get_parameter('temporal_min_hits').value))
        self.cluster_x_tol_frac = float(self.get_parameter('cluster_x_tol_frac').value)
        self.publish_viz = bool(self.get_parameter('publish_viz').value)

        self.bridge = CvBridge()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # Rolling history of per-frame raw pole lists for the M-of-N filter.
        self._frames = deque(maxlen=self.temporal_window)

        # ZED image topics are typically BEST_EFFORT; a BEST_EFFORT subscriber is
        # compatible with both, so it never QoS-mismatches into silence.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5)
        self.sub = self.create_subscription(
            Image, self.image_topic, self._image_cb, sensor_qos)

        self.pub = self.create_publisher(DetectionArray, '/hightide/slalom_poles', 10)
        self.viz_pub = None
        if self.publish_viz:
            self.viz_pub = self.create_publisher(Image, '/hightide/slalom_poles_image', 5)

        self.get_logger().info(
            f'Slalom pole detector up — RGB-only on {self.image_topic}. '
            'Tune thresholds against /hightide/slalom_poles_image.')

    def _classify_color(self, hsv_roi):
        """'red_pole' if the ROI is dominantly red, else 'white_pole'."""
        if hsv_roi.size == 0:
            return 'white_pole'
        h = hsv_roi[:, :, 0].astype(np.int32)
        s = hsv_roi[:, :, 1]
        v = hsv_roi[:, :, 2]
        # Red hue wraps around 0/180 in OpenCV's 0-179 hue space.
        red_hue = ((h <= 10) | (h >= 170))
        strong = (s >= self.red_sat_min) & (v >= self.red_val_min)
        red_frac = float(np.count_nonzero(red_hue & strong)) / h.size
        return 'red_pole' if red_frac > 0.25 else 'white_pole'

    def _detect_raw(self, img):
        """Raw per-frame pole detections (before temporal filtering).

        Returns a list of dicts with pixel geometry + color, one per bar that
        passes the shape gates."""
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = self._clahe.apply(gray)                      # boost underwater contrast
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        # Join each pole's left+right edges into one filled vertical bar, then
        # bridge small gaps, so a pole becomes a single tall contour.
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(3, self.close_kernel_h)))
        bars = cv2.dilate(edges, vk, iterations=1)
        bars = cv2.morphologyEx(bars, cv2.MORPH_CLOSE, vk, iterations=1)

        contours, _ = cv2.findContours(bars, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        raw = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw <= 0:
                continue
            # Shape gates: tall, narrow, spans enough of the frame to be a pole.
            if bh < self.min_height_frac * h:
                continue
            if bw > self.max_width_frac * w:
                continue
            if (bh / float(bw)) < self.min_aspect:
                continue
            raw.append({
                'center_x': x + bw / 2.0, 'center_y': y + bh / 2.0,
                'width': float(bw), 'height': float(bh),
                'color': self._classify_color(hsv[y:y + bh, x:x + bw]),
            })
        return raw

    def _temporal_stable(self, img_w):
        """M-of-N filter: cluster poles across the frame window by x-position and
        keep only clusters present in >= temporal_min_hits distinct frames.
        Returns averaged, de-jittered poles (majority-vote color)."""
        entries = []  # (frame_index, pole_dict)
        for fi, frame in enumerate(self._frames):
            for p in frame:
                entries.append((fi, p))
        if not entries:
            return []

        entries.sort(key=lambda e: e[1]['center_x'])
        x_tol = self.cluster_x_tol_frac * img_w

        # Greedy 1-D clustering on center_x (poles hold roughly still frame-to-
        # frame, so nearby x across frames == the same physical pole).
        clusters, cur = [], []
        for fi, p in entries:
            if cur and abs(p['center_x'] - (sum(q['center_x'] for _, q in cur) / len(cur))) > x_tol:
                clusters.append(cur)
                cur = []
            cur.append((fi, p))
        if cur:
            clusters.append(cur)

        stable = []
        for cl in clusters:
            if len({fi for fi, _ in cl}) < self.temporal_min_hits:
                continue  # not seen in enough distinct frames — likely an artifact
            n = len(cl)
            reds = sum(1 for _, p in cl if p['color'] == 'red_pole')
            stable.append({
                'center_x': sum(p['center_x'] for _, p in cl) / n,
                'center_y': sum(p['center_y'] for _, p in cl) / n,
                'width': sum(p['width'] for _, p in cl) / n,
                'height': sum(p['height'] for _, p in cl) / n,
                'color': 'red_pole' if reds * 2 >= n else 'white_pole',
            })
        return stable

    def _image_cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        h, w = img.shape[:2]
        raw = self._detect_raw(img)
        self._frames.append(raw)
        stable = self._temporal_stable(w)

        det_array = DetectionArray()
        det_array.header = msg.header
        det_array.image_width = w
        det_array.image_height = h

        viz = img.copy() if self.viz_pub is not None else None

        # Draw the raw (pre-filter) detections faintly so tuning is still visible.
        if viz is not None:
            for p in raw:
                rx = int(p['center_x'] - p['width'] / 2.0)
                ry = int(p['center_y'] - p['height'] / 2.0)
                cv2.rectangle(viz, (rx, ry),
                              (int(rx + p['width']), int(ry + p['height'])),
                              (120, 120, 120), 1)

        for p in stable:
            bw, bh = p['width'], p['height']
            x = p['center_x'] - bw / 2.0
            y = p['center_y'] - bh / 2.0
            color = p['color']
            det = Detection()
            det.header = msg.header
            det.class_id = RED_CLASS_ID if color == 'red_pole' else WHITE_CLASS_ID
            det.class_name = color
            det.confidence = 1.0
            det.x_min = float(x)
            det.y_min = float(y)
            det.x_max = float(x + bw)
            det.y_max = float(y + bh)
            det.center_x = float(p['center_x'])
            det.center_y = float(p['center_y'])
            det.width = float(bw)
            det.height = float(bh)
            det.depth_m = -1.0  # RGB-only — no range; slalom uses height as proximity
            det_array.detections.append(det)

            if viz is not None:
                col = (0, 0, 255) if color == 'red_pole' else (255, 255, 255)
                cv2.rectangle(viz, (int(x), int(y)),
                              (int(x + bw), int(y + bh)), col, 2)
                cv2.putText(viz, color, (int(x), max(0, int(y) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        self.pub.publish(det_array)
        if viz is not None:
            try:
                self.viz_pub.publish(self.bridge.cv2_to_imgmsg(viz, 'bgr8'))
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = SlalomPoleDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()