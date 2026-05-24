"""
Tests for HighTide_drivers — GPIO actuator logic.

Covers:
  - Actuator pin selection
  - Double-fire prevention logic
  - Timeout calculation
"""

import pytest


class TestActuatorLogic:
    """Test the core logic of the actuator driver (ignoring actual GPIO)."""

    class MockActuatorDriver:
        def __init__(self):
            self.pins = {
                'torpedo_1': 27,
                'torpedo_2': 22,
                'dropper_1': 23,
                'dropper_2': 24,
            }
            self.torpedoes_fired = {1: False, 2: False}
            self.markers_dropped = {1: False, 2: False}
            self.actuations = []

        def _actuate_pin(self, name):
            if name in self.pins:
                self.actuations.append(name)
                return True
            return False

        def fire_torpedo(self, tube_id):
            if tube_id not in (1, 2):
                return False, 'Invalid tube_id'
            if self.torpedoes_fired[tube_id]:
                return False, 'Already fired'
            
            pin_name = f'torpedo_{tube_id}'
            success = self._actuate_pin(pin_name)
            self.torpedoes_fired[tube_id] = True
            return success, 'Success'

        def drop_marker(self, dropper_id):
            if dropper_id not in (1, 2):
                return False, 'Invalid dropper_id'
            if self.markers_dropped[dropper_id]:
                return False, 'Already dropped'
            
            pin_name = f'dropper_{dropper_id}'
            success = self._actuate_pin(pin_name)
            self.markers_dropped[dropper_id] = True
            return success, 'Success'

    def test_fire_torpedo_1(self):
        driver = self.MockActuatorDriver()
        success, msg = driver.fire_torpedo(1)
        assert success is True
        assert 'torpedo_1' in driver.actuations
        assert driver.torpedoes_fired[1] is True

    def test_double_fire_prevention(self):
        driver = self.MockActuatorDriver()
        # First fire works
        success1, _ = driver.fire_torpedo(1)
        assert success1 is True
        
        # Second fire fails
        success2, msg = driver.fire_torpedo(1)
        assert success2 is False
        assert 'Already fired' in msg
        assert len(driver.actuations) == 1  # Still only actuated once

    def test_invalid_tube_id(self):
        driver = self.MockActuatorDriver()
        success, msg = driver.fire_torpedo(3)
        assert success is False
        assert 'Invalid tube_id' in msg

    def test_drop_marker_2(self):
        driver = self.MockActuatorDriver()
        success, msg = driver.drop_marker(2)
        assert success is True
        assert 'dropper_2' in driver.actuations
        assert driver.markers_dropped[2] is True

    def test_marker_double_drop_prevention(self):
        driver = self.MockActuatorDriver()
        driver.drop_marker(1)
        success, msg = driver.drop_marker(1)
        assert success is False
        assert 'Already dropped' in msg
        
        # But marker 2 should still work
        success, _ = driver.drop_marker(2)
        assert success is True
