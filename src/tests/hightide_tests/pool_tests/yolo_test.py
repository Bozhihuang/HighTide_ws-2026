#!/usr/bin/env python3
"""
Pool Test: YOLO Class Coverage — interactive "show me each class" checker.

Verifies the live YOLO detector can actually recognize EVERY class in
hightide_perception.CLASS_NAMES. It walks you through the classes one at a
time: it names the class, you hold that object in front of the front-facing
camera, and it tells you whether the detector fired for it (and at what
confidence). Use it after exporting a new .engine to confirm no class is dead.

This listens to the RAW detector output (/hightide/detections), not the tracker,
so what you see here is exactly what YOLO produced — no tracking/depth filtering
in the way.

Prereqs (run one at a time, this is interactive):
  - yolo_detector_node must be running with a real TensorRT engine loaded
    (in MOCK MODE it never publishes detections, so every class will FAIL).
  - the ZED RGB stream feeding the detector must be up.

Usage:
    ros2 run hightide_tests pool_test_yolo_classes
    ros2 run hightide_tests pool_test_yolo_classes --ros-args \
        -p watch_seconds:=6.0 -p min_hits:=5 -p topic:=/hightide/tracked_targets

At each prompt:
    <Enter>  start watching for the named class
    s        skip this class (marked SKIPPED in the summary)
    a        auto-run every remaining class back-to-back (no more prompts)
    q        quit early and print the summary so far
"""

import sys
import time
import threading

import rclpy
from rclpy.node import Node

from hightide_interfaces.msg import DetectionArray
from hightide_perception import CLASS_NAMES


class YoloClassChecker(Node):
    """Collects raw detections and scores them per-class for a watch window."""

    def __init__(self):
        super().__init__('pool_test_yolo_classes')

        self.declare_parameter('topic', '/hightide/detections')
        self.declare_parameter('watch_seconds', 5.0)
        self.declare_parameter('min_hits', 3)

        self.topic = self.get_parameter('topic').value
        self.watch_seconds = self.get_parameter('watch_seconds').value
        self.min_hits = self.get_parameter('min_hits').value

        # Shared window state, guarded by _lock (callback writes, main reads).
        # Keyed by class_id: hits this window, peak confidence, last-seen conf.
        self._lock = threading.Lock()
        self._window: dict[int, dict] = {}

        self.create_subscription(
            DetectionArray, self.topic, self._detection_cb, 10)

    def _detection_cb(self, msg: DetectionArray):
        with self._lock:
            for det in msg.detections:
                slot = self._window.get(det.class_id)
                if slot is None:
                    slot = {'hits': 0, 'max_conf': 0.0, 'last_conf': 0.0}
                    self._window[det.class_id] = slot
                slot['hits'] += 1
                slot['last_conf'] = det.confidence
                if det.confidence > slot['max_conf']:
                    slot['max_conf'] = det.confidence

    def reset_window(self):
        """Clear all per-class tallies before starting a new watch window."""
        with self._lock:
            self._window.clear()

    def snapshot(self) -> dict[int, dict]:
        """Copy the current per-class tallies for reading on the main thread."""
        with self._lock:
            return {cid: dict(s) for cid, s in self._window.items()}


def _watch_class(node: YoloClassChecker, class_id: int, class_name: str):
    """Watch for one class over the configured window; return a result dict."""
    node.reset_window()
    deadline = time.time() + node.watch_seconds

    while time.time() < deadline:
        remaining = deadline - time.time()
        snap = node.snapshot()
        target = snap.get(class_id, {'hits': 0, 'max_conf': 0.0, 'last_conf': 0.0})

        # Anything the detector is currently seeing OTHER than the target — a
        # heads-up that it's confusing the object for a different class.
        others = sorted(
            (CLASS_NAMES.get(cid, f'class_{cid}'), s['hits'])
            for cid, s in snap.items() if cid != class_id
        )
        others_str = ', '.join(f'{n}x{h}' for n, h in others) if others else '—'

        print(
            f'\r  watching {class_name!r:<20} '
            f'{remaining:4.1f}s left | hits={target["hits"]:3d} '
            f'peak={target["max_conf"]:.2f} last={target["last_conf"]:.2f} '
            f'| other: {others_str}      ',
            end='', flush=True)
        time.sleep(0.15)

    snap = node.snapshot()
    target = snap.get(class_id, {'hits': 0, 'max_conf': 0.0})
    detected = target['hits'] >= node.min_hits
    print()  # end the live line
    return {
        'detected': detected,
        'hits': target['hits'],
        'max_conf': target['max_conf'],
        'confused_with': [CLASS_NAMES.get(cid, f'class_{cid}')
                          for cid in snap if cid != class_id],
    }


def _print_summary(results: dict):
    print('\n' + '=' * 60)
    print(' YOLO CLASS COVERAGE SUMMARY')
    print('=' * 60)
    n_pass = n_fail = n_skip = 0
    for cid in sorted(CLASS_NAMES):
        name = CLASS_NAMES[cid]
        res = results.get(cid)
        if res is None or res.get('skipped'):
            status, detail = 'SKIP', ''
            n_skip += 1
        elif res['detected']:
            status = 'PASS'
            detail = f"hits={res['hits']}  peak_conf={res['max_conf']:.2f}"
            n_pass += 1
        else:
            status = 'FAIL'
            detail = f"hits={res['hits']}  peak_conf={res['max_conf']:.2f}"
            if res['confused_with']:
                detail += f"  (saw instead: {', '.join(res['confused_with'])})"
            n_fail += 1
        print(f'  [{status}] {cid}: {name:<20} {detail}')
    print('-' * 60)
    print(f'  {n_pass} passed, {n_fail} failed, {n_skip} skipped '
          f'of {len(CLASS_NAMES)} classes')
    print('=' * 60)


def _run(node: YoloClassChecker):
    """Main interactive loop — runs on its own thread while rclpy spins."""
    print('\n=== YOLO Class Coverage Test ===')
    print(f'Listening on: {node.topic}')
    print(f'A class PASSES if it gets >= {node.min_hits} detections within '
          f'a {node.watch_seconds:.0f}s window.\n')
    print('Show each named object to the front camera when prompted.')
    print('  <Enter>=test   s=skip   a=auto-run rest   q=quit\n')

    results: dict[int, dict] = {}
    auto = False

    for cid in sorted(CLASS_NAMES):
        name = CLASS_NAMES[cid]
        if not auto:
            try:
                choice = input(
                    f'--> Show class {cid}: {name!r}  [Enter/s/a/q]: '
                ).strip().lower()
            except EOFError:
                choice = 'q'
            if choice == 'q':
                break
            if choice == 's':
                results[cid] = {'skipped': True}
                print(f'  skipped {name!r}\n')
                continue
            if choice == 'a':
                auto = True

        results[cid] = _watch_class(node, cid, name)
        verdict = 'DETECTED' if results[cid]['detected'] else 'NOT detected'
        print(f'  => {name!r}: {verdict} '
              f'(hits={results[cid]["hits"]}, '
              f'peak={results[cid]["max_conf"]:.2f})\n')

    _print_summary(results)


def main(args=None):
    rclpy.init(args=args)
    node = YoloClassChecker()

    # Spin ROS in the background so the main thread can do blocking input().
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        _run(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main() or 0)
