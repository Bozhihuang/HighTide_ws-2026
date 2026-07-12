#!/usr/bin/env python3
"""
YOLO Detector Node — TensorRT inference on ZED camera images.

Subscribes to ZED rectified RGB, runs YOLOv8 TensorRT inference,
publishes DetectionArray. Falls back to mock mode if TensorRT unavailable.
"""

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from hightide_interfaces.msg import Detection, DetectionArray
from hightide_perception import CLASS_NAMES

# Try importing TensorRT — gracefully degrade if unavailable
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


class YoloDetectorNode(Node):
    """YOLOv8 TensorRT detector node for hightide competition objects."""

    def __init__(self):
        super().__init__('yolo_detector_node')

        self.declare_parameter('engine_path', '')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('input_width', 640)
        self.declare_parameter('input_height', 640)
        self.declare_parameter('publish_viz', True)

        self.engine_path = self.get_parameter('engine_path').value
        self.conf_thresh = self.get_parameter('confidence_threshold').value
        self.nms_thresh = self.get_parameter('nms_threshold').value
        self.input_w = self.get_parameter('input_width').value
        self.input_h = self.get_parameter('input_height').value
        self.publish_viz = self.get_parameter('publish_viz').value

        self.bridge = CvBridge()
        self.engine = None
        self.trt_context = None
        self.stream = None
        self.d_input = None
        self.h_input = None
        self.use_v3 = False        # TensorRT 10 tensor-name API vs legacy bindings
        self.input_name = None
        # Engines can expose more than one output tensor (e.g. an NMS/end2end
        # export). Keep ALL of them so every address gets set before enqueue —
        # missing one triggers "no address set for output tensor" and leaves the
        # detection buffer unwritten. Each entry: {name, shape, host, device}.
        self.outputs = []
        self.num_classes = len(CLASS_NAMES)

        # Load TensorRT engine
        if TRT_AVAILABLE and self.engine_path:
            self._load_engine()
        elif not TRT_AVAILABLE:
            self.get_logger().warn(
                'TensorRT not available — running in MOCK MODE (no detections)')
        else:
            self.get_logger().warn(
                'No engine_path set — running in MOCK MODE (no detections)')

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/mavros/zed/rgb/color/rect/image',
            self._image_callback, 5)

        # Publishers
        self.det_pub = self.create_publisher(DetectionArray, '/hightide/detections', 10)
        if self.publish_viz:
            self.viz_pub = self.create_publisher(Image, '/hightide/detection_image', 5)

        self.get_logger().info('YOLO Detector Node started')

    def _load_engine(self):
        """Load TensorRT engine and allocate buffers.

        Supports both the legacy binding API (TRT 8.x) and the tensor-name
        API — the binding API (num_bindings / get_binding_shape /
        execute_async_v2) was removed in TensorRT 10, which is what JetPack 6
        ships. self.use_v3 records which execution path to take.
        """
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            with open(self.engine_path, 'rb') as f:
                runtime = trt.Runtime(logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.trt_context = self.engine.create_execution_context()
            self.use_v3 = not hasattr(self.engine, 'num_bindings')

            if self.use_v3:
                # TensorRT 10+ tensor-name API
                for i in range(self.engine.num_io_tensors):
                    name = self.engine.get_tensor_name(i)
                    shape = self.engine.get_tensor_shape(name)
                    size = trt.volume(shape)
                    dtype = trt.nptype(self.engine.get_tensor_dtype(name))

                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)

                    if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                        self.input_name = name
                        self.h_input = host_mem
                        self.d_input = device_mem
                    else:
                        self.outputs.append({
                            'name': name,
                            'shape': tuple(shape),
                            'host': host_mem,
                            'device': device_mem,
                        })
            else:
                # Legacy binding API (TensorRT 8.x)
                for binding in range(self.engine.num_bindings):
                    shape = self.engine.get_binding_shape(binding)
                    size = trt.volume(shape)
                    dtype = trt.nptype(self.engine.get_binding_dtype(binding))

                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)

                    if self.engine.binding_is_input(binding):
                        self.h_input = host_mem
                        self.d_input = device_mem
                    else:
                        self.outputs.append({
                            'name': self.engine.get_binding_name(binding),
                            'shape': tuple(shape),
                            'host': host_mem,
                            'device': device_mem,
                        })

            self.stream = cuda.Stream()
            self.get_logger().info(
                f'TensorRT engine loaded: {self.engine_path} '
                f'(API: {"v3/tensor-name" if self.use_v3 else "v2/bindings"}, '
                f'outputs: {[(o["name"], o["shape"]) for o in self.outputs]})')

            # Verify the engine's class count matches this node's CLASS_NAMES.
            # A mismatch (e.g. an 80-class COCO engine vs the 8-class ffc model)
            # is the #1 cause of impossible class IDs like class_28/class_30.
            det = self._detection_output()
            nc4 = self.num_classes + 4
            if det is not None:
                dims = [d for d in det['shape'] if d != 1]
                if nc4 not in dims:
                    self.get_logger().error(
                        f'ENGINE/CLASS MISMATCH: detection output {det["shape"]} does not '
                        f'contain the expected {nc4} features (4 + {self.num_classes} classes). '
                        f'This engine was NOT built for the {self.num_classes}-class model — '
                        f're-export from the correct .pt. Detections will be dropped.')

        except Exception as e:
            self.get_logger().error(f'Failed to load TensorRT engine: {e}')
            self.engine = None

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Letterbox resize, normalize, transpose for YOLO input."""
        h, w = img.shape[:2]
        scale = min(self.input_w / w, self.input_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Letterbox padding
        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        dx = (self.input_w - new_w) // 2
        dy = (self.input_h - new_h) // 2
        canvas[dy:dy + new_h, dx:dx + new_w] = resized

        # BGR to RGB, normalize, CHW, add batch dim
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # HWC → CHW
        blob = np.expand_dims(blob, axis=0)  # Add batch
        return blob, scale, dx, dy

    def _detection_output(self):
        """Pick the raw YOLO detection tensor from the engine's outputs.

        A plain detect export has one output; NMS/end2end exports have several
        (boxes, scores, ...). We want the raw (1, 4+nc, N) / (1, N, 4+nc)
        tensor: prefer a tensor named 'output0', else the one that actually has
        a dim equal to 4+num_classes, else fall back to the largest tensor."""
        if not self.outputs:
            return None
        nc4 = self.num_classes + 4
        for o in self.outputs:
            if o['name'] == 'output0':
                return o
        for o in self.outputs:
            if nc4 in tuple(o['shape']):
                return o
        return max(self.outputs, key=lambda o: int(np.prod(o['shape'])))

    def _postprocess(self, output: np.ndarray, scale, dx, dy, orig_w, orig_h):
        """Parse YOLO output, apply NMS, scale boxes to original image."""
        # YOLOv8 raw output is (1, num_classes+4, num_detections). Some exports
        # emit the transposed (1, num_detections, num_classes+4) instead, so
        # orient by whichever axis matches 4+num_classes rather than assuming.
        arr = np.squeeze(np.asarray(output))
        if arr.ndim != 2:
            self.get_logger().warn(
                f'Unexpected detection output shape {np.asarray(output).shape} — skipping frame')
            return []
        nc4 = self.num_classes + 4
        if arr.shape[0] == nc4:          # (4+nc, N) → (N, 4+nc)
            predictions = arr.T
        elif arr.shape[1] == nc4:        # already (N, 4+nc)
            predictions = arr
        else:                            # ambiguous: features are the shorter axis
            predictions = arr.T if arr.shape[0] < arr.shape[1] else arr

        # Sanity check: the feature width MUST be 4 + num_classes. If it isn't,
        # the engine was built for a different number of classes than this node
        # expects (e.g. an 80-class COCO export instead of the 8-class ffc
        # model) — argmax would then emit impossible class IDs (28, 30, ...).
        # Bail loudly rather than publish garbage detections.
        feat = predictions.shape[1]
        if feat != nc4:
            self._diag_tick += 1
            if self._diag_tick % 40 == 1:
                self.get_logger().error(
                    f'Engine output has {feat} features (={feat - 4} classes) but this '
                    f'node expects {self.num_classes} classes ({nc4} features). WRONG '
                    f'ENGINE — re-export from the {self.num_classes}-class .pt. '
                    f'Dropping all detections.')
            return []

        detections = []
        for pred in predictions:
            x_c, y_c, w, h = pred[:4]
            scores = pred[4:]
            class_id = int(np.argmax(scores))
            # Defensive: never accept an out-of-range class (mismatched engine,
            # corrupt buffer). CLASS_NAMES only defines 0..num_classes-1.
            if class_id >= self.num_classes:
                continue
            confidence = float(scores[class_id])

            if confidence < self.conf_thresh:
                continue

            # Undo letterbox
            x_c = (x_c - dx) / scale
            y_c = (y_c - dy) / scale
            w = w / scale
            h = h / scale

            x1 = max(0, x_c - w / 2)
            y1 = max(0, y_c - h / 2)
            x2 = min(orig_w, x_c + w / 2)
            y2 = min(orig_h, y_c + h / 2)

            detections.append({
                'class_id': class_id,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2],
            })

        # Per-class NMS. Two things matter here:
        #  1. cv2.dnn.NMSBoxes expects [x, y, WIDTH, HEIGHT] — feeding it the
        #     corner format we store would compute IoU on garbage geometry.
        #  2. NMS must run per class: on the torpedo board the 'circle' holes
        #     overlap the 'fire'/'blood' symbol box, and a single class-
        #     agnostic pass would suppress one with the other.
        if not detections:
            return []

        kept = []
        class_ids = {d['class_id'] for d in detections}
        for cid in class_ids:
            cls_dets = [d for d in detections if d['class_id'] == cid]
            boxes_xywh = [
                [d['bbox'][0], d['bbox'][1],
                 d['bbox'][2] - d['bbox'][0], d['bbox'][3] - d['bbox'][1]]
                for d in cls_dets]
            scores = [d['confidence'] for d in cls_dets]
            indices = cv2.dnn.NMSBoxes(
                boxes_xywh, scores, self.conf_thresh, self.nms_thresh)
            if len(indices) > 0:
                kept.extend(cls_dets[i] for i in np.array(indices).flatten())

        return kept

    def _image_callback(self, msg: Image):
        """Process incoming image: inference + publish detections."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        orig_h, orig_w = cv_image.shape[:2]
        det_list = []

        # Run inference if engine available
        if self.engine is not None and self.trt_context is not None:
            blob, scale, dx, dy = self._preprocess(cv_image)
            np.copyto(self.h_input, blob.ravel())
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            if self.use_v3:
                self.trt_context.set_tensor_address(self.input_name, int(self.d_input))
                # Every output tensor needs an address, not just the one we
                # care about — enqueueV3 refuses to run otherwise.
                for o in self.outputs:
                    self.trt_context.set_tensor_address(o['name'], int(o['device']))
                self.trt_context.execute_async_v3(stream_handle=self.stream.handle)
            else:
                self.trt_context.execute_async_v2(
                    bindings=[int(self.d_input)] + [int(o['device']) for o in self.outputs],
                    stream_handle=self.stream.handle)
            for o in self.outputs:
                cuda.memcpy_dtoh_async(o['host'], o['device'], self.stream)
            self.stream.synchronize()

            det_out = self._detection_output()
            if det_out is not None:
                output = det_out['host'].reshape(det_out['shape'])
                det_list = self._postprocess(output, scale, dx, dy, orig_w, orig_h)

        # Build DetectionArray message
        det_array = DetectionArray()
        det_array.header = msg.header
        det_array.image_width = orig_w
        det_array.image_height = orig_h

        for d in det_list:
            det = Detection()
            det.header = msg.header
            det.class_id = d['class_id']
            det.class_name = CLASS_NAMES.get(d['class_id'], f'class_{d["class_id"]}')
            det.confidence = d['confidence']
            det.x_min = float(d['bbox'][0])
            det.y_min = float(d['bbox'][1])
            det.x_max = float(d['bbox'][2])
            det.y_max = float(d['bbox'][3])
            det.center_x = (det.x_min + det.x_max) / 2.0
            det.center_y = (det.y_min + det.y_max) / 2.0
            det.width = det.x_max - det.x_min
            det.height = det.y_max - det.y_min
            det.depth_m = -1.0  # Depth added by tracker node
            det_array.detections.append(det)

        self.det_pub.publish(det_array)

        # Publish viz image
        if self.publish_viz and det_list:
            viz_img = cv_image.copy()
            for d in det_list:
                x1, y1, x2, y2 = [int(v) for v in d['bbox']]
                label = f'{CLASS_NAMES.get(d["class_id"], "?")} {d["confidence"]:.2f}'
                cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(viz_img, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            try:
                self.viz_pub.publish(self.bridge.cv2_to_imgmsg(viz_img, 'bgr8'))
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()