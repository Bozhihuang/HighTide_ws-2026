#!/usr/bin/env python3
"""
YOLO Detector Node — TensorRT inference on ZED camera images.

Subscribes to ZED rectified RGB, runs YOLO-seg TensorRT inference, publishes
DetectionArray. Falls back to mock mode if TensorRT unavailable.

Works with both the ffc instance-SEGMENTATION model (YOLO*-seg — two output
tensors: detections+mask-coeffs and mask protos) and a plain detection model
(single output). When masks are present it uses them to its advantage:
  - center_x/center_y become the MASK centroid, not the bbox center — more
    stable for thin/rotated/occluded shapes (slalom poles, angled symbols);
  - mask_area (true segmented pixels) is a cleaner size cue than bbox area;
  - mask_polygon carries the contour for precise downstream alignment/viz.
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
    """YOLO-seg TensorRT detector node for hightide competition objects."""

    def __init__(self):
        super().__init__('yolo_detector_node')

        self.declare_parameter('engine_path', '')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('input_width', 640)
        self.declare_parameter('input_height', 640)
        self.declare_parameter('mask_threshold', 0.5)   # sigmoid cutoff for seg masks
        self.declare_parameter('publish_viz', True)

        self.engine_path = self.get_parameter('engine_path').value
        self.conf_thresh = self.get_parameter('confidence_threshold').value
        self.nms_thresh = self.get_parameter('nms_threshold').value
        self.input_w = self.get_parameter('input_width').value
        self.input_h = self.get_parameter('input_height').value
        self.mask_thresh = self.get_parameter('mask_threshold').value
        self.publish_viz = self.get_parameter('publish_viz').value

        self.bridge = CvBridge()
        self.engine = None
        self.trt_context = None
        self.stream = None
        self.d_input = None
        self.h_input = None
        self.use_v3 = False        # TensorRT 10 tensor-name API vs legacy bindings
        self.input_name = None
        # Engines can expose more than one output tensor. A seg export has two:
        # the detection tensor (boxes + scores + mask coeffs) AND the mask proto
        # tensor. Keep ALL of them so every address gets set before enqueue —
        # missing one triggers "no address set for output tensor" and leaves the
        # detection buffer unwritten. Each entry: {name, shape, host, device}.
        self.outputs = []
        self.num_classes = len(CLASS_NAMES)
        self.num_masks = 0         # >0 once a seg engine's proto tensor is seen
        self._diag_tick = 0        # rate-limits the wrong-engine error spam

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

            # Record how many mask prototypes the engine carries (0 = detect-only).
            proto = self._proto_output()
            self.num_masks = self._proto_channels(proto) if proto is not None else 0

            self.get_logger().info(
                f'TensorRT engine loaded: {self.engine_path} '
                f'(API: {"v3/tensor-name" if self.use_v3 else "v2/bindings"}, '
                f'mode: {"segmentation" if self.num_masks else "detect-only"}, '
                f'masks: {self.num_masks}, '
                f'outputs: {[(o["name"], o["shape"]) for o in self.outputs]})')

            # Verify the engine's feature count matches this node's expectation.
            # A mismatch (e.g. an 80-class COCO engine, or a detect engine where
            # a seg one is expected) is the #1 cause of dropped/garbage
            # detections. Expected detection feature width is 4 + num_classes
            # (+ num_masks for a seg engine).
            det = self._detection_output()
            expected = self.num_classes + 4 + self.num_masks
            if det is not None:
                dims = [d for d in det['shape'] if d != 1]
                if expected not in dims:
                    self.get_logger().error(
                        f'ENGINE/CLASS MISMATCH: detection output {det["shape"]} does not '
                        f'contain the expected {expected} features (4 + {self.num_classes} '
                        f'classes + {self.num_masks} mask coeffs). This engine was NOT built '
                        f'for the {self.num_classes}-class model — re-export from the correct '
                        f'.pt (see scripts/export_yolo_seg_engine.py). Detections will drop.')

        except Exception as e:
            self.get_logger().error(f'Failed to load TensorRT engine: {e}')
            self.engine = None

    def _preprocess(self, img: np.ndarray):
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
        return blob, scale, dx, dy, new_w, new_h

    def _detection_output(self):
        """Pick the raw YOLO detection tensor from the engine's outputs.

        A plain detect export has one output; seg/NMS/end2end exports have
        several. We want the raw (1, 4+nc(+nm), N) / (1, N, 4+nc(+nm)) tensor:
        prefer a tensor named 'output0', else the one that actually has a dim
        equal to 4+num_classes(+num_masks), else the largest 2D-ish tensor
        (never the 4D proto)."""
        if not self.outputs:
            return None
        for o in self.outputs:
            if o['name'] == 'output0':
                return o
        for extra in (self.num_masks, 0):
            target = self.num_classes + 4 + extra
            for o in self.outputs:
                if target in tuple(o['shape']):
                    return o
        non_proto = [o for o in self.outputs if len(o['shape']) != 4]
        pool = non_proto or self.outputs
        return max(pool, key=lambda o: int(np.prod(o['shape'])))

    def _proto_output(self):
        """Pick the mask-prototype tensor from a seg engine's outputs.

        Protos are the 4D (1, num_masks, mh, mw) tensor — distinct from the
        detection tensor. Returns None for a detect-only engine."""
        det = self._detection_output()
        det_name = det['name'] if det is not None else None
        for o in self.outputs:
            if o['name'] == det_name:
                continue
            if len(o['shape']) == 4:
                return o
        return None

    @staticmethod
    def _proto_channels(proto):
        """Number of mask prototypes = the small (channel) dim of the proto tensor."""
        dims = [d for d in proto['shape'] if d != 1]
        return int(min(dims)) if dims else 0

    @staticmethod
    def _prep_proto(arr):
        """Squeeze proto to (num_masks, mh, mw) with the channel axis first."""
        arr = np.squeeze(np.asarray(arr))
        if arr.ndim != 3:
            return None
        c_axis = int(np.argmin(arr.shape))   # 32 << 160, so smallest dim is channels
        return np.ascontiguousarray(np.moveaxis(arr, c_axis, 0))

    def _postprocess(self, output, proto, scale, dx, dy, new_w, new_h, orig_w, orig_h):
        """Parse YOLO output, apply per-class NMS, decode masks, scale to image."""
        # YOLO raw output is (1, 4+nc(+nm), N). Some exports emit the transposed
        # (1, N, 4+nc(+nm)) instead, so orient by whichever axis matches the
        # expected feature width rather than assuming.
        arr = np.squeeze(np.asarray(output))
        if arr.ndim != 2:
            self.get_logger().warn(
                f'Unexpected detection output shape {np.asarray(output).shape} — skipping frame')
            return []

        nc = self.num_classes
        nm = self._proto_channels_arr(proto)   # mask coeffs available this frame
        expected = nc + 4 + nm
        if arr.shape[0] == expected:            # (feat, N) → (N, feat)
            predictions = arr.T
        elif arr.shape[1] == expected:          # already (N, feat)
            predictions = arr
        else:                                   # ambiguous: features are shorter axis
            predictions = arr.T if arr.shape[0] < arr.shape[1] else arr

        # Sanity check: feature width MUST be 4 + num_classes (+ num_masks). If
        # it isn't, the engine was built for a different config than this node
        # expects (e.g. an 80-class COCO export) — argmax would emit impossible
        # class IDs. Bail loudly rather than publish garbage.
        feat = predictions.shape[1]
        if feat != expected:
            self._diag_tick += 1
            if self._diag_tick % 40 == 1:
                self.get_logger().error(
                    f'Engine output has {feat} features but this node expects {expected} '
                    f'(4 + {nc} classes + {nm} mask coeffs). WRONG ENGINE — re-export from '
                    f'the {nc}-class .pt (scripts/export_yolo_seg_engine.py). Dropping all '
                    f'detections.')
            return []

        detections = []
        for pred in predictions:
            x_c, y_c, w, h = pred[:4]
            scores = pred[4:4 + nc]
            class_id = int(np.argmax(scores))
            # Defensive: never accept an out-of-range class (mismatched engine,
            # corrupt buffer). CLASS_NAMES only defines 0..num_classes-1.
            if class_id >= nc:
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

            det = {
                'class_id': class_id,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2],
                'mask_area': 0.0,
                'mask_polygon': [],
                'center': ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            }
            if nm > 0:
                det['coeffs'] = np.asarray(pred[4 + nc:4 + nc + nm], dtype=np.float32)
            detections.append(det)

        if not detections:
            return []

        # Per-class NMS. Two things matter here:
        #  1. cv2.dnn.NMSBoxes expects [x, y, WIDTH, HEIGHT] — feeding it the
        #     corner format we store would compute IoU on garbage geometry.
        #  2. NMS must run per class: on the torpedo board the 'circle' holes
        #     overlap the 'fire'/'blood' symbol box, and a single class-
        #     agnostic pass would suppress one with the other.
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

        # Decode masks only for the survivors (cheap — a handful of objects).
        if nm > 0 and proto is not None:
            for d in kept:
                self._apply_mask(d, proto, scale, dx, dy, new_w, new_h, orig_w, orig_h)

        return kept

    @staticmethod
    def _proto_channels_arr(proto):
        return 0 if proto is None else int(proto.shape[0])

    def _apply_mask(self, det, proto, scale, dx, dy, new_w, new_h, orig_w, orig_h):
        """Turn a detection's mask coefficients into a full-image mask, then set
        its mask_area, mask_polygon, and refine center to the mask centroid."""
        nm, mh, mw = proto.shape
        # Linear-combine the prototypes, squash with sigmoid → soft mask in the
        # proto grid (which corresponds to the letterboxed input downscaled).
        m = det['coeffs'] @ proto.reshape(nm, -1)
        m = 1.0 / (1.0 + np.exp(-m))
        m = m.reshape(mh, mw)

        # proto grid → letterboxed input (input_w x input_h) → strip padding →
        # original image size, so the mask lines up with the bbox coords.
        m = cv2.resize(m, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        m = m[dy:dy + new_h, dx:dx + new_w]
        m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        binm = (m >= self.mask_thresh).astype(np.uint8)

        # Confine the mask to the detection's bbox so a neighbouring object's
        # pixels can't leak into this instance's contour/centroid.
        x1, y1, x2, y2 = det['bbox']
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(orig_w, int(x2)), min(orig_h, int(y2))
        roi = np.zeros_like(binm)
        if x2i > x1i and y2i > y1i:
            roi[y1i:y2i, x1i:x2i] = binm[y1i:y2i, x1i:x2i]

        area = float(roi.sum())
        det['mask_area'] = area
        if area <= 0:
            return   # keep bbox-center fallback already stored

        moments = cv2.moments(roi, binaryImage=True)
        if moments['m00'] > 0:
            det['center'] = (moments['m10'] / moments['m00'],
                             moments['m01'] / moments['m00'])

        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            eps = 0.01 * cv2.arcLength(largest, True)
            poly = cv2.approxPolyDP(largest, eps, True).reshape(-1, 2)
            det['mask_polygon'] = poly.astype(np.float32).flatten().tolist()

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
            blob, scale, dx, dy, new_w, new_h = self._preprocess(cv_image)
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
            proto_out = self._proto_output()
            proto_arr = None
            if proto_out is not None:
                proto_arr = self._prep_proto(
                    proto_out['host'].reshape(proto_out['shape']))
            if det_out is not None:
                output = det_out['host'].reshape(det_out['shape'])
                det_list = self._postprocess(
                    output, proto_arr, scale, dx, dy, new_w, new_h, orig_w, orig_h)

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
            # center is the mask centroid when a mask was decoded, else bbox center
            det.center_x = float(d['center'][0])
            det.center_y = float(d['center'][1])
            det.width = det.x_max - det.x_min
            det.height = det.y_max - det.y_min
            det.depth_m = -1.0  # Depth added by tracker node
            det.mask_area = float(d['mask_area'])
            det.mask_polygon = d['mask_polygon']
            det_array.detections.append(det)

        self.det_pub.publish(det_array)

        # Publish viz image
        if self.publish_viz and det_list:
            viz_img = cv_image.copy()
            overlay = viz_img.copy()
            for d in det_list:
                x1, y1, x2, y2 = [int(v) for v in d['bbox']]
                # Shade the segmentation mask when present (nice at-a-glance cue
                # that seg is actually running); fall back to just the box.
                if d['mask_polygon']:
                    pts = np.array(d['mask_polygon'], dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(overlay, [pts], (0, 200, 0))
                    cv2.polylines(viz_img, [pts], True, (0, 255, 0), 2)
                label = f'{CLASS_NAMES.get(d["class_id"], "?")} {d["confidence"]:.2f}'
                cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(viz_img, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.addWeighted(overlay, 0.35, viz_img, 0.65, 0, viz_img)
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
