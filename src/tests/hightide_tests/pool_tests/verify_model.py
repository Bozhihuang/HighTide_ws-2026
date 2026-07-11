#!/usr/bin/env python3
"""
Model verifier — cross-check a YOLO .pt (or .engine) against CLASS_NAMES.

Run this BEFORE exporting a new front-facing-camera model to TensorRT, to catch
the two things that silently break perception:

  1. Training image size (imgsz) — must match input_width/input_height in
     params.yaml (currently 640). Export the engine at this same size.
  2. Class id -> name map — must match hightide_perception.CLASS_NAMES exactly,
     since the detector reads names from that dict by the model's class id. A
     mismatch means every mission behavior keys off the wrong label.

This is a plain CLI utility, not a ROS node — it only needs `ultralytics`
installed (`pip install ultralytics`). It does not require the ROS graph to be
running, so you can run it on a laptop or on the ZED Box.

Usage:
    python3 -m hightide_tests.pool_tests.verify_model /path/to/ffc.pt
    verify_model /path/to/ffc.pt --imgsz 640      # if installed as a script

Exit code is 0 when everything matches, 1 otherwise — so it can gate a build.
"""

import argparse
import sys


def _load_class_names():
    """Return hightide_perception.CLASS_NAMES, or None if not importable."""
    try:
        from hightide_perception import CLASS_NAMES
        return dict(CLASS_NAMES)
    except Exception as e:  # noqa: BLE001 — surfaced to the user, not swallowed
        print(f'  ! could not import hightide_perception.CLASS_NAMES: {e}')
        print('    (source the workspace install/setup.bash first)')
        return None


def _model_names_to_dict(names):
    """Ultralytics exposes model.names as a dict {id: name} or a list; normalize."""
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def inspect_model(model_path):
    """Load a .pt/.engine with ultralytics and pull imgsz + class names."""
    from ultralytics import YOLO
    model = YOLO(model_path)

    names = _model_names_to_dict(model.names)

    # imgsz lives in the checkpoint's train args; may be an int or [w, h].
    imgsz = None
    try:
        args = getattr(model.model, 'args', None)
        if isinstance(args, dict):
            imgsz = args.get('imgsz')
        elif args is not None:
            imgsz = getattr(args, 'imgsz', None)
    except Exception:  # noqa: BLE001
        pass

    return imgsz, names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', help='Path to the .pt (or .engine) model file')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Expected input size from params.yaml (default: 640)')
    args = parser.parse_args(argv)

    print(f'== Verifying model: {args.model} ==')

    try:
        imgsz, model_names = inspect_model(args.model)
    except ImportError:
        print('  ! ultralytics not installed — run: pip install ultralytics')
        return 1
    except Exception as e:  # noqa: BLE001
        print(f'  ! failed to load model: {e}')
        return 1

    ok = True

    # --- imgsz check ---
    print(f'\n[imgsz] model trained at: {imgsz}   expected (params.yaml): {args.imgsz}')
    if imgsz is None:
        print('  ~ could not read imgsz from the model; verify manually.')
    else:
        # imgsz may be a scalar or [w, h]; compare the governing dimension.
        model_imgsz = imgsz if isinstance(imgsz, int) else max(imgsz)
        if model_imgsz != args.imgsz:
            print(f'  ! MISMATCH — export the engine at imgsz={model_imgsz} '
                  f'AND set input_width/height to {model_imgsz} in params.yaml.')
            ok = False
        else:
            print('  OK')

    # --- class map check ---
    print('\n[classes] model class map:')
    for cid in sorted(model_names):
        print(f'    {cid}: {model_names[cid]}')

    class_names = _load_class_names()
    if class_names is None:
        print('  ~ skipping class-map comparison (CLASS_NAMES unavailable).')
    else:
        if model_names == class_names:
            print('  OK — matches hightide_perception.CLASS_NAMES exactly.')
        else:
            ok = False
            print('  ! MISMATCH vs hightide_perception.CLASS_NAMES:')
            all_ids = sorted(set(model_names) | set(class_names))
            for cid in all_ids:
                m = model_names.get(cid, '<missing>')
                c = class_names.get(cid, '<missing>')
                flag = '' if m == c else '   <-- differs'
                print(f'    id {cid}: model={m!r}  CLASS_NAMES={c!r}{flag}')
            print('\n    Fix: edit CLASS_NAMES in '
                  'src/perception/hightide_perception/__init__.py to match the model.')

    print(f'\n== {"PASS" if ok else "FAIL"} ==')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
