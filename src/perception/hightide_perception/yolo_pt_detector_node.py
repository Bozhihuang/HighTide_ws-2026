#!/usr/bin/env python3
"""
YOLO .pt Detector Node — runs an Ultralytics checkpoint directly (no TensorRT build).

Drop-in alternative to yolo_detector_node: subscribes to the ZED RGB image and
publishes the SAME hightide_interfaces/DetectionArray on /hightide/detections, so
target_tracker_node and the mission tree consume it unchanged.

Why this exists
---------------
Building a TensorRT engine on the ZED box can fail (GPU OOM during the build) and,
for end-to-end / NMS-baked exports, the raw-tensor parser in yolo_detector_node
doesn't match the engine's output. This node hands the frame to the Ultralytics
`YOLO` API, which does its own letterbox / NMS / mask decode and returns parsed
results — so it works with:
  * a .pt checkpoint (PyTorch/CUDA — slower, but zero build steps), OR
  * an Ultralytics-exported .engine (fast) — same code, just change model_path.

Trade-off: a .pt runs ~2-4x slower than a hand-tuned TensorRT engine and uses more
GPU memory. Fine for bring-up / pool testing; switch model_path to a .engine later
for competition FPS without touching this file.

Masks: when the model is a -seg model, the mask centroid (steadier than the bbox
center on thin/rotated/occluded shapes), true mask area, and contour polygon are
populated exactly like the TensorRT node, so downstream alignment is identical.
"""

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from hightide_interfaces.msg import Detection, DetectionArray
from hightide_perception import CLASS_NAMES

try:
    from ultralytics import YOLO
    ULTRA_AVAILABLE = True
except ImportError:
    ULTRA_AVAILABLE = False


class YoloPtDetectorNode(Node):
    """Ultralytics-API detector (.pt or .engine) publishing DetectionArray."""

    def __init__(self):
        super().__init__('yolo_pt_detector_node')

        self.declare_parameter('model_path', '')            # .pt or .engine
        self.declare_parameter('image_topic', '/mavros/zed/rgb/color/rect/image')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)       # Ultralytics NMS IoU
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('half', True)                # FP16 on GPU
        self.declare_parameter('device', '0')               # CUDA index, or 'cpu'
        self.declare_parameter('publish_viz', True)

        self.model_path = self.get_parameter('model_path').value
        self.image_topic = self.get_parameter('image_topic').value
        self.conf_thresh = float(self.get_parameter('confidence_threshold').value)
        self.iou_thresh = float(self.get_parameter('iou_threshold').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.half = bool(self.get_parameter('half').value)
        self.device = self.get_parameter('device').value
        self.publish_viz = bool(self.get_parameter('publish_viz').value)

        self.bridge = CvBridge()
        self.model = None

        if ULTRA_AVAILABLE and self.model_path:
            self._load_model()
        elif not ULTRA_AVAILABLE:
            self.get_logger().warn(
                'ultralytics not installed — running in MOCK MODE (no detections). '
                'Install with: pip install ultralytics')
        else:
            self.get_logger().warn(
                'No model_path set — running in MOCK MODE (no detections).')

        # ZED image topics are BEST_EFFORT; match them or receive nothing.
        # depth=1 so we always process the freshest frame and drop stale ones
        # (inference is slower than the camera, so a deep queue just adds lag).
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self._image_callback, sensor_qos)

        self.det_pub = self.create_publisher(DetectionArray, '/hightide/detections', 10)
        self.viz_pub = None
        if self.publish_viz:
            self.viz_pub = self.create_publisher(Image, '/hightide/detection_image', 5)

        self.get_logger().info('YOLO .pt Detector Node started')

    def _load_model(self):
        """Load the checkpoint/engine and warm it up so the first real frame is fast."""
        try:
            self.model = YOLO(self.model_path)
            names = list(self.model.names.values())
            self.get_logger().info(
                f'Loaded {self.model_path} (task={getattr(self.model, "task", "?")}, '
                f'classes={names})')
            # Warn if the checkpoint's class order disagrees with the contract the
            # rest of the stack keys off (mission matches class_name strings).
            expected = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
            if names != expected:
                self.get_logger().warn(
                    f'Model class list != hightide_perception.CLASS_NAMES.\n'
                    f'  model    : {names}\n  expected : {expected}\n'
                    f'  Detections use the MODEL names; update CLASS_NAMES if they '
                    f'differ or mission behaviors will look for the wrong strings.')
            # Warm-up inference on a dummy frame (first call JITs/allocates).
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(dummy, imgsz=self.imgsz, half=self.half,
                               device=self.device, verbose=False)
            self.get_logger().info('Model warm-up complete')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            self.model = None

    def _image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        orig_h, orig_w = cv_image.shape[:2]
        det_array = DetectionArray()
        det_array.header = msg.header
        det_array.image_width = orig_w
        det_array.image_height = orig_h

        if self.model is None:
            self.det_pub.publish(det_array)   # mock mode: empty, but keep the topic alive
            return

        try:
            # Ultralytics handles letterbox, NMS, and mask decode internally, and
            # returns boxes/masks already scaled to the original image.
            results = self.model.predict(
                cv_image, imgsz=self.imgsz, conf=self.conf_thresh,
                iou=self.iou_thresh, half=self.half, device=self.device,
                verbose=False)
        except Exception as e:
            self.get_logger().error(f'Inference error: {e}')
            self.det_pub.publish(det_array)
            return

        r = results[0]
        names = r.names
        boxes = r.boxes
        # masks.xy is a list of Nx2 polygons in original-image pixel coords, one per
        # detection, index-aligned with boxes. None for a detect-only model.
        polys = r.masks.xy if getattr(r, 'masks', None) is not None else None

        n = 0 if boxes is None else len(boxes)
        for i in range(n):
            xyxy = boxes.xyxy[i].tolist()
            x1, y1, x2, y2 = xyxy
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())

            det = Detection()
            det.header = msg.header
            det.class_id = cls_id
            det.class_name = names.get(cls_id, f'class_{cls_id}')
            det.confidence = conf
            det.x_min, det.y_min, det.x_max, det.y_max = (
                float(x1), float(y1), float(x2), float(y2))
            det.width = float(x2 - x1)
            det.height = float(y2 - y1)
            det.depth_m = -1.0        # range added by target_tracker_node from ZED depth
            det.center_x = float((x1 + x2) / 2.0)   # bbox center (mask centroid below)
            det.center_y = float((y1 + y2) / 2.0)
            det.mask_area = 0.0
            det.mask_polygon = []

            # Segmentation extras: prefer the mask centroid + true area, like the
            # TensorRT node, so thin/rotated props align on the segmented shape.
            if polys is not None and i < len(polys) and len(polys[i]) >= 3:
                poly = np.asarray(polys[i], dtype=np.float32)
                cnt = poly.reshape(-1, 1, 2)
                area = float(cv2.contourArea(cnt))
                if area > 0:
                    m = cv2.moments(cnt)
                    if m['m00'] != 0:
                        det.center_x = float(m['m10'] / m['m00'])
                        det.center_y = float(m['m01'] / m['m00'])
                    det.mask_area = area
                    det.mask_polygon = poly.flatten().tolist()

            det_array.detections.append(det)

        self.det_pub.publish(det_array)

        if self.viz_pub is not None and n > 0:
            try:
                # r.plot() returns an annotated BGR image (boxes + masks + labels).
                self.viz_pub.publish(self.bridge.cv2_to_imgmsg(r.plot(), 'bgr8'))
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = YoloPtDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
