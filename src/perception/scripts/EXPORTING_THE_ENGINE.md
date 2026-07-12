# Exporting the ffc `.pt` → TensorRT engine (from scratch, on the ZED Box)

This is the full, zero-assumptions procedure to turn the trained **ffc
instance-segmentation** checkpoint (`ffc_seg.pt`) into the TensorRT `.engine`
file the `yolo_detector_node` loads. Do the whole thing **on the ZED Box Mini
itself** — a TensorRT engine is not portable across TensorRT/GPU/JetPack
versions, so an engine built on a laptop will fail on the robot.

> **Key facts this guide is built on**
> - Workspace: `~/HighTide_ws-2026` (ROS 2 **Humble**)
> - Export script: `src/perception/scripts/export_yolo_seg_engine.py`
> - Engine path the stack expects: **`/home/user/models/ffc_seg.engine`**
>   (params.yaml `yolo_detector_node.engine_path` + launch arg `yolo_engine`)
> - Class contract: `hightide_perception/__init__.py` → `CLASS_NAMES`
>   (order: `blood, buoy, compass, circle, fire, hammer_and_wrench, slalom, sos`)
> - The `.pt` **must** be `task: segment` and exported with **`nms=False`**
>   (built-in NMS strips the mask coefficients → segmentation can't be decoded).

---

## 0. Before you start — what you need

- The trained checkpoint `ffc_seg.pt` (on your laptop or a USB stick).
- SSH/terminal access to the ZED Box.
- ~15 min (the export itself takes several minutes of TensorRT building).

---

## 1. Get onto the box and confirm the toolchain

SSH in (or open a terminal on the box), then confirm CUDA + TensorRT are present
(they ship with JetPack — you should NOT need to install them):

```bash
# On the ZED Box
nvcc --version                 # CUDA present?
python3 -c "import tensorrt as trt; print('TensorRT', trt.__version__)"
```

- If `import tensorrt` fails, install the JetPack Python bindings:
  ```bash
  sudo apt update && sudo apt install -y python3-libnvinfer python3-libnvinfer-dev
  ```
- If `nvcc` isn't found, CUDA isn't on PATH — add it (adjust the version to what's
  installed under `/usr/local/`):
  ```bash
  echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
  source ~/.bashrc
  ```

## 2. Install the two Python packages the exporter needs

TensorRT does the building; **ultralytics** drives the export and **pycuda** is
what `yolo_detector_node` uses at runtime.

```bash
python3 -m pip install --upgrade pip
python3 -m pip install ultralytics
python3 -m pip install pycuda        # needs CUDA on PATH (step 1) or it won't build
```

Sanity check:

```bash
python3 -c "import ultralytics, pycuda; print('ultralytics + pycuda OK')"
```

## 3. Put the checkpoint on the box

From your **laptop** (not the box), copy the `.pt` over, and make the models
directory the stack expects:

```bash
# On the box: create the target dir (matches params.yaml engine_path)
mkdir -p /home/user/models

# On your laptop: copy the checkpoint to the box
scp ffc_seg.pt user@<zed-box-ip>:/home/user/models/ffc_seg.pt
```

> If the box's username/home is **not** `user`, either put the model where you
> like and adjust `engine_path` in `params.yaml`, or override at launch with
> `yolo_engine:=/your/path/ffc_seg.engine` (step 7).

## 4. Run the export

The export script sanity-checks the checkpoint (task + class list) **before**
spending minutes building, forces `nms=False`, and prints the engine's I/O so you
can confirm it's really a seg engine with the right classes.

```bash
cd ~/HighTide_ws-2026
python3 src/perception/scripts/export_yolo_seg_engine.py \
    --pt  /home/user/models/ffc_seg.pt \
    --out /home/user/models/ffc_seg.engine \
    --imgsz 640 \
    --half
```

- `--imgsz 640` **must** match `input_width` / `input_height` in `params.yaml`.
- `--half` = FP16: ~2× faster, half the memory, negligible accuracy loss — the
  recommended competition default on Jetson. Drop it for FP32 only if you're
  chasing an accuracy problem. (Do **not** attempt INT8 without a calibration set.)

## 5. Read the verification output

At the top you should see the checkpoint check:

```
Loaded /home/user/models/ffc_seg.pt
  task    : segment
  classes : ['blood', 'buoy', 'compass', 'circle', 'fire', 'hammer_and_wrench', 'slalom', 'sos']
```

- If `task` is not `segment`, or the class list doesn't match the order above,
  **stop** — fix the checkpoint (or update `CLASS_NAMES`) before continuing, or
  the detector will produce no masks / mislabel objects.

At the bottom, the engine I/O report should say:

```
Engine I/O tensors:
  [IN ] images: (1, 3, 640, 640)
  [OUT] output0: (1, 44, 8400)          # 4 + 8 classes + 32 mask coeffs
  [OUT] output1: (1, 32, 160, 160)      # 32 mask prototypes

  OK: segmentation engine detected — 8 classes, 32 mask protos. Detection tensor feature width = 44.
```

`OK: segmentation engine detected` is the green light. If instead you see
`WARNING: no output tensor has 44 (seg) or 12 (detect) features`, the `.pt` was
built for a different class count — re-train/re-export from the correct model.

## 6. Point the config at the engine

If you exported to the default path `/home/user/models/ffc_seg.engine`, nothing
to change — `params.yaml` already points there:

```yaml
# src/launch/config/params.yaml
yolo_detector_node:
  ros__parameters:
    engine_path: '/home/user/models/ffc_seg.engine'
    input_width: 640
    input_height: 640
    ...
```

Only edit `engine_path` (and the launch `yolo_engine` default) if you put the
engine somewhere else.

## 7. Rebuild the workspace and source it

The segmentation change **added fields to `hightide_interfaces/Detection.msg`**
(`mask_area`, `mask_polygon`), so the interfaces package must be rebuilt or the
nodes will deserialize the message wrong.

```bash
cd ~/HighTide_ws-2026
rosdep install --from-paths src --ignore-src -r -y   # first time only
colcon build --symlink-install
source /opt/ros/humble/setup.bash
source ~/HighTide_ws-2026/install/setup.bash
```

## 8. Launch and confirm it's actually segmenting

```bash
ros2 launch hightide_launch full_system.launch.py
# (or override the engine path if you used a custom location:)
# ros2 launch hightide_launch full_system.launch.py yolo_engine:=/your/path/ffc_seg.engine
```

In the detector's log, look for:

```
TensorRT engine loaded: /home/user/models/ffc_seg.engine (API: v3/tensor-name, mode: segmentation, masks: 32, ...)
```

`mode: segmentation, masks: 32` means it's live. Then verify from another
terminal (source the workspace first):

```bash
# Are detections flowing, and do they carry a real mask?
ros2 topic echo /hightide/detections --once
#   -> mask_area should be > 0.0 and mask_polygon non-empty for a seen object

# Watch the annotated image (shaded masks + centroid dots) in rqt:
ros2 run rqt_image_view rqt_image_view /hightide/detection_image
```

If `mask_area` is `0.0` and `mask_polygon` is empty on every detection, the
engine loaded in detect-only mode — re-check step 5 (`nms=False`, `task: segment`).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Log says `MOCK MODE (no detections)` | `engine_path` empty or file missing | Check the path exists on the box; confirm `params.yaml` / `yolo_engine` |
| `ENGINE/CLASS MISMATCH ... expected 44 features` | Engine built for a different class count (e.g. 80-class COCO) or a detect model where seg is expected | Re-export from the correct 8-class **seg** `.pt` |
| Detections appear but `mask_area` always `0.0` | Exported with `nms=True` (coeffs stripped) or the `.pt` is `task: detect` | Re-export with the script (it forces `nms=False`); confirm `task: segment` |
| `import tensorrt` / `pycuda` fails at runtime | Bindings not installed, or CUDA not on PATH | Steps 1–2 |
| Engine loads on laptop but fails on box | Engine isn't portable across TRT/GPU versions | Always export **on the box** |
| Detector runs but slow | FP32 engine | Re-export with `--half` (FP16) |

## Re-exporting later

Any time the model is retrained, or you change `imgsz`, just re-run **step 4**
(the script overwrites the engine). No code changes needed — the node adapts to
the engine's class/mask/precision config automatically. You only need to rebuild
the workspace (step 7) again if `Detection.msg` or other interfaces changed.
