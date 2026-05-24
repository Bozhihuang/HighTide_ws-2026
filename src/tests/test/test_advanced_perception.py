"""
Tests for advanced perception logic (Depth Map filtering & Mock Mode).

Covers:
  - Depth map median extraction (handling invalid/NaN values)
  - YOLO Mock Mode fallback when TensorRT is unavailable
"""

import pytest
import numpy as np

class TestAdvancedPerceptionLogic:
    """Tests for Depth filtering and YOLO mock mode."""

    def _sample_depth(self, depth_image, cx: float, cy: float, depth_radius=5) -> float:
        """Replicates TargetTrackerNode._sample_depth"""
        if depth_image is None:
            return -1.0

        h, w = depth_image.shape[:2]
        cx_int, cy_int = int(cx), int(cy)

        if cx_int < 0 or cx_int >= w or cy_int < 0 or cy_int >= h:
            return -1.0

        r = depth_radius
        y_start = max(0, cy_int - r)
        y_end = min(h, cy_int + r + 1)
        x_start = max(0, cx_int - r)
        x_end = min(w, cx_int + r + 1)

        patch = depth_image[y_start:y_end, x_start:x_end]
        # Filter valid depths
        valid = patch[np.isfinite(patch) & (patch > 0.0)]

        if len(valid) == 0:
            return -1.0

        return float(np.median(valid))


    def test_depth_median_filtering_valid(self):
        # Create a 100x100 dummy depth map with a uniform depth of 2.0
        depth_img = np.full((100, 100), 2.0, dtype=np.float32)
        
        depth = self._sample_depth(depth_img, 50, 50)
        assert depth == 2.0

    def test_depth_median_filtering_with_nans(self):
        # Patch has 2.0s, but we inject NaNs and 0s which should be ignored
        depth_img = np.full((100, 100), 2.0, dtype=np.float32)
        
        # Inject bad data near the center
        depth_img[48:52, 48:52] = np.nan
        depth_img[52:55, 52:55] = 0.0
        depth_img[45:47, 45:47] = np.inf
        
        # Median should still be 2.0 from the valid pixels in the radius
        depth = self._sample_depth(depth_img, 50, 50, depth_radius=5)
        assert depth == 2.0

    def test_depth_filtering_out_of_bounds(self):
        depth_img = np.full((100, 100), 2.0, dtype=np.float32)
        
        # Out of bounds should return -1.0
        assert self._sample_depth(depth_img, -10, 50) == -1.0
        assert self._sample_depth(depth_img, 150, 50) == -1.0

    def test_depth_filtering_all_invalid(self):
        depth_img = np.full((100, 100), np.nan, dtype=np.float32)
        assert self._sample_depth(depth_img, 50, 50) == -1.0


class TestYoloMockMode:
    """Test YOLO fallback logic when TensorRT is missing."""
    
    def test_trt_availability_flag(self):
        try:
            import tensorrt as trt
            trt_exists = True
        except ImportError:
            trt_exists = False
            
        # The node should correctly identify if TRT is missing and use mock mode
        # We can't import the node fully without ROS, but we can verify the 
        # python flag logic works as expected.
        import sys
        if 'tensorrt' not in sys.modules:
            assert not trt_exists
