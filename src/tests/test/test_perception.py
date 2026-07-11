"""
Tests for hightide_perception — YOLO class map and detection logic.

Covers:
  - CLASS_NAMES completeness and validity
  - TrackedTarget EMA smoothing
  - IoU computation
  - Detection → TrackedTarget creation
"""

import pytest
import math


class TestClassNames:
    """Verify the YOLO class name mapping is complete and consistent."""

    def test_class_names_exist(self):
        from hightide_perception import CLASS_NAMES
        assert isinstance(CLASS_NAMES, dict)
        assert len(CLASS_NAMES) > 0

    def test_class_names_count(self):
        from hightide_perception import CLASS_NAMES
        assert len(CLASS_NAMES) == 8, f'Expected 8 classes, got {len(CLASS_NAMES)}'

    def test_class_ids_sequential(self):
        from hightide_perception import CLASS_NAMES
        expected_ids = list(range(8))
        actual_ids = sorted(CLASS_NAMES.keys())
        assert actual_ids == expected_ids, f'Non-sequential IDs: {actual_ids}'

    def test_class_names_are_strings(self):
        from hightide_perception import CLASS_NAMES
        for cid, name in CLASS_NAMES.items():
            assert isinstance(name, str), f'Class {cid} name is not a string'
            assert len(name) > 0, f'Class {cid} has empty name'

    def test_class_names_unique(self):
        from hightide_perception import CLASS_NAMES
        names = list(CLASS_NAMES.values())
        assert len(names) == len(set(names)), 'Duplicate class names found'

    def test_class_map_matches_ffc_model(self):
        """CLASS_NAMES must match the front-facing-camera model's data.yaml
        exactly — id order is the contract with the trained weights."""
        from hightide_perception import CLASS_NAMES
        expected = {
            0: 'blood',
            1: 'buoy',
            2: 'compass',
            3: 'circle',
            4: 'fire',
            5: 'hammer_and_wrench',
            6: 'slalom',
            7: 'sos',
        }
        assert CLASS_NAMES == expected

    def test_critical_classes_present(self):
        """Competition-critical classes (structural cues) must exist."""
        from hightide_perception import CLASS_NAMES
        required = [
            'slalom',   # red slalom poles
            'circle',   # torpedo holes
            'buoy',     # octagon cue
        ]
        names = set(CLASS_NAMES.values())
        for cls in required:
            assert cls in names, f'Required class "{cls}" missing'

    def test_role_symbols_present(self):
        """Both role symbol sets must exist."""
        from hightide_perception import CLASS_NAMES
        names = set(CLASS_NAMES.values())
        # Survey & Repair
        assert 'compass' in names
        assert 'hammer_and_wrench' in names
        # Search & Rescue
        assert 'sos' in names

    def test_bin_symbols_present(self):
        from hightide_perception import CLASS_NAMES
        names = set(CLASS_NAMES.values())
        assert 'fire' in names
        assert 'blood' in names


class TestTrackedTarget:
    """Tests for the TrackedTarget class in target_tracker_node."""

    def _make_detection(self, x_min=100, y_min=100, x_max=200, y_max=200,
                        class_id=0, class_name='compass', confidence=0.9):
        from hightide_interfaces.msg import Detection
        det = Detection()
        det.class_id = class_id
        det.class_name = class_name
        det.confidence = confidence
        det.x_min = float(x_min)
        det.y_min = float(y_min)
        det.x_max = float(x_max)
        det.y_max = float(y_max)
        det.center_x = (x_min + x_max) / 2.0
        det.center_y = (y_min + y_max) / 2.0
        det.depth_m = -1.0
        return det

    def test_creation(self):
        from hightide_perception.target_tracker_node import TrackedTarget
        det = self._make_detection()
        track = TrackedTarget(det, track_id=0)
        assert track.track_id == 0
        assert track.class_id == 0
        assert track.class_name == 'compass'
        assert track.hit_count == 1
        assert track.depth_m == -1.0

    def test_center_calculation(self):
        from hightide_perception.target_tracker_node import TrackedTarget
        det = self._make_detection(x_min=100, y_min=200, x_max=300, y_max=400)
        track = TrackedTarget(det, track_id=1)
        assert abs(track.center_x - 200.0) < 1e-6
        assert abs(track.center_y - 300.0) < 1e-6

    def test_ema_update_moves_bbox(self):
        from hightide_perception.target_tracker_node import TrackedTarget
        det1 = self._make_detection(x_min=100, y_min=100, x_max=200, y_max=200)
        track = TrackedTarget(det1, track_id=0)

        det2 = self._make_detection(x_min=200, y_min=200, x_max=300, y_max=300)
        track.update(det2, alpha=0.3)

        # After EMA: bbox[0] = 0.3*200 + 0.7*100 = 130
        assert abs(track.bbox[0] - 130.0) < 1e-6
        assert track.hit_count == 2

    def test_ema_alpha_1_snaps_to_new(self):
        from hightide_perception.target_tracker_node import TrackedTarget
        det1 = self._make_detection(x_min=0, y_min=0, x_max=100, y_max=100)
        track = TrackedTarget(det1, track_id=0)

        det2 = self._make_detection(x_min=500, y_min=500, x_max=600, y_max=600)
        track.update(det2, alpha=1.0)

        assert abs(track.bbox[0] - 500.0) < 1e-6
        assert abs(track.bbox[2] - 600.0) < 1e-6

    def test_ema_alpha_0_keeps_old(self):
        from hightide_perception.target_tracker_node import TrackedTarget
        det1 = self._make_detection(x_min=100, y_min=100, x_max=200, y_max=200)
        track = TrackedTarget(det1, track_id=0)

        det2 = self._make_detection(x_min=500, y_min=500, x_max=600, y_max=600)
        track.update(det2, alpha=0.0)

        assert abs(track.bbox[0] - 100.0) < 1e-6


class TestIoUComputation:
    """Tests for IoU computation in the tracker."""

    def _compute_iou(self, box_a, box_b):
        from hightide_perception.target_tracker_node import TargetTrackerNode
        return TargetTrackerNode._compute_iou(box_a, box_b)

    def test_identical_boxes(self):
        iou = self._compute_iou([0, 0, 100, 100], [0, 0, 100, 100])
        assert abs(iou - 1.0) < 1e-6

    def test_no_overlap(self):
        iou = self._compute_iou([0, 0, 50, 50], [100, 100, 200, 200])
        assert abs(iou - 0.0) < 1e-6

    def test_partial_overlap(self):
        iou = self._compute_iou([0, 0, 100, 100], [50, 50, 150, 150])
        # Intersection: 50x50 = 2500. Union: 2*10000 - 2500 = 17500
        expected = 2500.0 / 17500.0
        assert abs(iou - expected) < 1e-4

    def test_contained_box(self):
        iou = self._compute_iou([0, 0, 200, 200], [50, 50, 150, 150])
        # Intersection = 100x100 = 10000. Union = 40000 + 10000 - 10000 = 40000
        expected = 10000.0 / 40000.0
        assert abs(iou - expected) < 1e-4

    def test_zero_area_box(self):
        iou = self._compute_iou([0, 0, 0, 0], [0, 0, 100, 100])
        assert iou == 0.0

    def test_touching_edges(self):
        iou = self._compute_iou([0, 0, 100, 100], [100, 0, 200, 100])
        assert abs(iou) < 1e-6

    def test_symmetric(self):
        iou_ab = self._compute_iou([10, 10, 80, 80], [40, 40, 120, 120])
        iou_ba = self._compute_iou([40, 40, 120, 120], [10, 10, 80, 80])
        assert abs(iou_ab - iou_ba) < 1e-6
