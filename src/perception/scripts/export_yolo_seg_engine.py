#!/usr/bin/env python3
"""
Export a YOLO(v8/11/26)-seg .pt checkpoint to a TensorRT engine for the ffc
detector, and verify it matches what yolo_detector_node expects.

Run this ON THE ZED BOX MINI (it needs the same TensorRT that runs inference —
an engine is not portable across TensorRT/GPU versions, so exporting on your
laptop will NOT work on the robot). Requires `ultralytics` and `tensorrt`.

    python3 src/perception/scripts/export_yolo_seg_engine.py \
        --pt /home/user/models/ffc_seg.pt \
        --out /home/user/models/ffc_seg.engine \
        --imgsz 640 --half

Notes
-----
* The export COMMAND is the same for detect vs seg models — ultralytics reads
  the task from the checkpoint. What differs is the engine's outputs: a seg
  engine emits TWO tensors (output0 = boxes+scores+mask-coeffs, output1 = mask
  protos). yolo_detector_node consumes both.
* Do NOT pass nms=True. Built-in NMS strips the raw mask coefficients, which
  makes segmentation impossible to decode downstream. This script forces it off.
* imgsz must match the node's input_width/input_height in params.yaml (640).
"""
import argparse
import sys

# Must line up with hightide_perception.CLASS_NAMES (the ffc 8-class model).
EXPECTED_CLASSES = [
    'blood', 'buoy', 'compass', 'circle',
    'fire', 'hammer_and_wrench', 'slalom', 'sos',
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pt', required=True, help='path to the -seg .pt checkpoint')
    ap.add_argument('--out', default='', help='engine output path (default: alongside .pt)')
    ap.add_argument('--imgsz', type=int, default=640, help='inference size (must match params.yaml)')
    ap.add_argument('--half', action='store_true', help='FP16 — recommended on Jetson/ZED Box')
    ap.add_argument('--workspace', type=int, default=4, help='TensorRT workspace size (GiB)')
    ap.add_argument('--device', default='0', help='CUDA device index')
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit('ultralytics not installed. Run:  pip install ultralytics')

    model = YOLO(args.pt)

    # --- Sanity-check the checkpoint before spending minutes on the export ---
    task = getattr(model, 'task', None)
    names = list(model.names.values()) if hasattr(model, 'names') else []
    print(f'Loaded {args.pt}')
    print(f'  task    : {task}')
    print(f'  classes : {names}')

    if task != 'segment':
        print(f'  WARNING: task is {task!r}, not \'segment\'. This exporter is for the '
              f'seg model; a detect model still works but produces no masks.')
    if names != EXPECTED_CLASSES:
        print(f'  WARNING: class list does not match the expected ffc order:\n'
              f'           expected {EXPECTED_CLASSES}\n'
              f'           got      {names}\n'
              f'           Update hightide_perception/__init__.py CLASS_NAMES to match, '
              f'or the detector will mislabel objects.')

    # --- Export. nms=False is the important bit for seg. ---
    print('Exporting to TensorRT engine (this can take several minutes)...')
    engine_path = model.export(
        format='engine',
        imgsz=args.imgsz,
        half=args.half,
        nms=False,            # keep raw mask coefficients — do NOT change
        workspace=args.workspace,
        device=args.device,
    )
    print(f'Engine written: {engine_path}')

    if args.out and args.out != engine_path:
        import shutil
        shutil.move(engine_path, args.out)
        engine_path = args.out
        print(f'Moved to: {engine_path}')

    _report_engine_io(engine_path, num_classes=len(EXPECTED_CLASSES))

    print('\nNext steps:')
    print(f'  1. Point params.yaml yolo_detector_node.engine_path at {engine_path}')
    print('  2. colcon build --symlink-install   # rebuild hightide_interfaces (new mask fields)')
    print('  3. ros2 launch hightide_launch full_system.launch.py')


def _report_engine_io(engine_path, num_classes):
    """Print the engine's I/O tensor shapes so you can confirm it's a seg engine
    with the right class count before ever running the robot."""
    try:
        import tensorrt as trt
    except ImportError:
        print('\n(tensorrt not importable here — skipping engine I/O check)')
        return

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    print('\nEngine I/O tensors:')
    outputs = []
    # TensorRT 10 tensor-name API (JetPack 6). Fall back to bindings on TRT 8.
    if not hasattr(engine, 'num_bindings'):
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = tuple(engine.get_tensor_shape(name))
            mode = engine.get_tensor_mode(name)
            io = 'IN ' if mode == trt.TensorIOMode.INPUT else 'OUT'
            print(f'  [{io}] {name}: {shape}')
            if mode != trt.TensorIOMode.INPUT:
                outputs.append(shape)
    else:
        for b in range(engine.num_bindings):
            name = engine.get_binding_name(b)
            shape = tuple(engine.get_binding_shape(b))
            io = 'IN ' if engine.binding_is_input(b) else 'OUT'
            print(f'  [{io}] {name}: {shape}')
            if not engine.binding_is_input(b):
                outputs.append(shape)

    nc4 = num_classes + 4
    det = next((s for s in outputs if any(d == nc4 for d in s)), None)
    proto = next((s for s in outputs if len(s) == 4), None)
    if det is not None and proto is not None:
        masks = proto[1]
        print(f'\n  OK: segmentation engine detected — {num_classes} classes, '
              f'{masks} mask protos. Detection tensor feature width = {nc4 + masks}.')
    elif det is not None:
        print(f'\n  Detect-only engine ({num_classes} classes, no mask protos). '
              f'The node runs but produces boxes only.')
    else:
        print(f'\n  WARNING: no output tensor has {nc4} features. This engine was NOT '
              f'built for the {num_classes}-class model — re-export from the correct .pt.')


if __name__ == '__main__':
    main()
