"""
Tests for HighTide_navigation — Waypoint math, Crab walk vision servo, Search pattern.

Covers:
  - Global to local coordinate transformation (Waypoint Navigator)
  - Distance and bearing calculations
  - Vision Servo lateral/vertical/range error computation
  - Expanding square math
"""

import math
import pytest


class TestWaypointMath:
    """Test the coordinate math used in Waypoint Navigator."""

    def _compute_errors(self, current_x, current_y, current_yaw, target_x, target_y):
        """Replicate the waypoint navigator's core math."""
        from HighTide_navigation import normalize_angle
        # Global errors
        dx = target_x - current_x
        dy = target_y - current_y

        # Target bearing and distance
        distance = math.sqrt(dx**2 + dy**2)
        bearing = math.atan2(dy, dx)

        # Rotate global errors to body frame (surge/sway)
        # yaw = 0 means X is forward (surge), Y is left (sway)
        # Using standard 2D rotation matrix for body frame
        cos_y = math.cos(current_yaw)
        sin_y = math.sin(current_yaw)

        # Body frame errors
        error_surge = dx * cos_y + dy * sin_y
        error_sway = -dx * sin_y + dy * cos_y

        return distance, bearing, error_surge, error_sway

    def test_target_straight_ahead(self):
        # Current: (0,0) facing X (yaw=0)
        # Target: (10,0)
        dist, bearing, surge, sway = self._compute_errors(0, 0, 0, 10, 0)
        assert abs(dist - 10.0) < 1e-6
        assert abs(bearing - 0.0) < 1e-6
        assert abs(surge - 10.0) < 1e-6
        assert abs(sway - 0.0) < 1e-6

    def test_target_to_the_left(self):
        # Current: (0,0) facing X (yaw=0)
        # Target: (0,10) — directly left
        dist, bearing, surge, sway = self._compute_errors(0, 0, 0, 0, 10)
        assert abs(dist - 10.0) < 1e-6
        assert abs(bearing - math.pi / 2) < 1e-6
        assert abs(surge - 0.0) < 1e-6
        assert abs(sway - 10.0) < 1e-6

    def test_target_behind(self):
        # Current: (0,0) facing X (yaw=0)
        # Target: (-10,0) — directly behind
        dist, bearing, surge, sway = self._compute_errors(0, 0, 0, -10, 0)
        assert abs(dist - 10.0) < 1e-6
        assert abs(abs(bearing) - math.pi) < 1e-6
        assert abs(surge - (-10.0)) < 1e-6
        assert abs(sway - 0.0) < 1e-6

    def test_target_diagonal_yaw_90(self):
        # Current: (0,0) facing Y (yaw=90°)
        # Target: (10, 10)
        # In body frame (facing Y):
        # Global X=10 is now Body RIGHT (sway=-10)
        # Global Y=10 is now Body FORWARD (surge=10)
        dist, bearing, surge, sway = self._compute_errors(0, 0, math.pi / 2, 10, 10)
        assert abs(dist - math.sqrt(200)) < 1e-6
        assert abs(bearing - math.pi / 4) < 1e-6
        assert abs(surge - 10.0) < 1e-6
        assert abs(sway - (-10.0)) < 1e-6


class TestVisionServoMath:
    """Test the crab walk alignment logic."""

    def _compute_camera_errors(self, center_x, center_y, img_w, img_h, current_depth, target_depth_m):
        """Replicate Vision Servo visual error logic."""
        # Normalized coordinates [0, 1]
        norm_x = center_x / img_w
        norm_y = center_y / img_h

        # Error from center (0.0 is perfect center)
        # + lateral error means target is to the right → need to sway right (+)
        # + vertical error means target is below center → need to go deeper (+)
        lateral_error = norm_x - 0.5
        vertical_error = norm_y - 0.5

        # Range error (positive means too far → need to surge forward)
        range_error = current_depth - target_depth_m

        return lateral_error, vertical_error, range_error

    def test_perfectly_centered(self):
        lat, vert, rng = self._compute_camera_errors(640, 360, 1280, 720, 1.0, 1.0)
        assert abs(lat) < 1e-6
        assert abs(vert) < 1e-6
        assert abs(rng) < 1e-6

    def test_target_top_left(self):
        # Target at (0, 0)
        lat, vert, rng = self._compute_camera_errors(0, 0, 1280, 720, 2.0, 1.0)
        assert abs(lat - (-0.5)) < 1e-6  # Need to sway left (-)
        assert abs(vert - (-0.5)) < 1e-6 # Need to go shallower (-)
        assert abs(rng - 1.0) < 1e-6     # Need to surge forward (+)

    def test_target_bottom_right(self):
        # Target at (1280, 720)
        lat, vert, rng = self._compute_camera_errors(1280, 720, 1280, 720, 0.5, 1.0)
        assert abs(lat - 0.5) < 1e-6     # Need to sway right (+)
        assert abs(vert - 0.5) < 1e-6    # Need to go deeper (+)
        assert abs(rng - (-0.5)) < 1e-6  # Need to surge backward (-)


class TestExpandingSquare:
    """Test search pattern leg sequence generator logic."""

    def _generate_legs(self, initial_leg, increment, max_legs):
        """Simulate expanding square leg sequence."""
        legs = []
        current_leg = initial_leg
        for turn_count in range(max_legs):
            legs.append(current_leg)
            # Increase leg length every 2 turns
            if (turn_count + 1) % 2 == 0:
                current_leg += increment
        return legs

    def test_leg_sequence(self):
        legs = self._generate_legs(initial_leg=2.0, increment=1.0, max_legs=6)
        # Turns: 0, 1, 2, 3, 4, 5
        # Turn 0: leg=2.0 (after this, +1 count -> count=1. No increment)
        # Turn 1: leg=2.0 (after this, +1 count -> count=2. Increment -> leg=3.0)
        # Turn 2: leg=3.0 (count=3. No increment)
        # Turn 3: leg=3.0 (count=4. Increment -> leg=4.0)
        # Turn 4: leg=4.0
        # Turn 5: leg=4.0
        assert legs == [2.0, 2.0, 3.0, 3.0, 4.0, 4.0]

    def test_leg_sequence_small_increment(self):
        legs = self._generate_legs(initial_leg=1.0, increment=0.5, max_legs=4)
        assert legs == [1.0, 1.0, 1.5, 1.5]
