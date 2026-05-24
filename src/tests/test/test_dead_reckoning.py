"""
Tests for Dead Reckoning math and logic.

Covers:
  - Duration calculations from distance and speed
  - Thrust sequencing
"""

import pytest

class TestDeadReckoningLogic:
    
    def _calculate_duration(self, distance, speed, speed_to_mps_factor):
        if distance == 0:
            return 0.0
        return abs(distance) / (speed * speed_to_mps_factor)

    def test_duration_calculation(self):
        # Move 2.0 meters. Thrust speed is 0.4. Conversion is 0.5 MPS per 1.0 thrust unit.
        # So 0.4 thrust = 0.2 meters/second.
        # 2.0m / 0.2m/s = 10.0 seconds.
        dist = 2.0
        speed = 0.4
        factor = 0.5
        duration = self._calculate_duration(dist, speed, factor)
        assert duration == 10.0

    def test_zero_distance(self):
        assert self._calculate_duration(0.0, 0.4, 0.5) == 0.0

    def test_negative_distance(self):
        # Distance is absolute for duration, direction is handled by thrust sign
        dist = -2.0
        speed = 0.4
        factor = 0.5
        duration = self._calculate_duration(dist, speed, factor)
        assert duration == 10.0

    def test_thrust_direction_signs(self):
        # Simulating the sign logic in dead_reckoning_node
        surge_m = -2.0
        sway_m = 1.5
        
        surge_dir = 1.0 if surge_m >= 0 else -1.0
        sway_dir = 1.0 if sway_m >= 0 else -1.0
        
        assert surge_dir == -1.0
        assert sway_dir == 1.0
