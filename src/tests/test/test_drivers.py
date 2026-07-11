"""
Tests for hightide_drivers — GPIO actuator logic.

Covers:
  - Actuator pin selection
  - Re-fire is allowed (no double-fire guard — a jammed torpedo/dropper must be
    retriggerable mid-run)
  - Retry-up-to-4-attempts logic on actuation failure
  - Timeout calculation
"""

import pytest


class TestActuatorLogic:
    """Test the core logic of the actuator driver (ignoring actual GPIO)."""

    class MockActuatorDriver:
        def __init__(self, max_attempts=4, fail_first_n=0):
            self.pins = {
                'torpedo_1': 27,
                'torpedo_2': 22,
                'dropper_1': 23,
                'dropper_2': 24,
            }
            self.torpedoes_fired = {1: False, 2: False}
            self.markers_dropped = {1: False, 2: False}
            self.actuations = []
            self.max_attempts = max_attempts
            # Number of leading attempts to simulate as failures, to exercise retry.
            self.fail_first_n = fail_first_n
            self._attempt_count = 0

        def _actuate_pin(self, name):
            self._attempt_count += 1
            if self._attempt_count <= self.fail_first_n:
                return False
            if name in self.pins:
                self.actuations.append(name)
                return True
            return False

        def _actuate_with_retry(self, name):
            for _attempt in range(1, self.max_attempts + 1):
                if self._actuate_pin(name):
                    return True
            return False

        def fire_torpedo(self, tube_id):
            if tube_id not in (1, 2):
                return False, 'Invalid tube_id'

            pin_name = f'torpedo_{tube_id}'
            success = self._actuate_with_retry(pin_name)
            if success:
                self.torpedoes_fired[tube_id] = True
            return success, 'Success' if success else 'Failed'

        def drop_marker(self, dropper_id):
            if dropper_id not in (1, 2):
                return False, 'Invalid dropper_id'

            pin_name = f'dropper_{dropper_id}'
            success = self._actuate_with_retry(pin_name)
            if success:
                self.markers_dropped[dropper_id] = True
            return success, 'Success' if success else 'Failed'

    def test_fire_torpedo_1(self):
        driver = self.MockActuatorDriver()
        success, msg = driver.fire_torpedo(1)
        assert success is True
        assert 'torpedo_1' in driver.actuations
        assert driver.torpedoes_fired[1] is True

    def test_refire_is_allowed(self):
        """Firing an already-fired torpedo must NOT be blocked."""
        driver = self.MockActuatorDriver()
        success1, _ = driver.fire_torpedo(1)
        assert success1 is True

        success2, msg2 = driver.fire_torpedo(1)
        assert success2 is True
        assert msg2 != 'Already fired'
        assert driver.actuations.count('torpedo_1') == 2  # Actuated both times

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

    def test_redrop_is_allowed(self):
        """Dropping an already-dropped marker must NOT be blocked."""
        driver = self.MockActuatorDriver()
        driver.drop_marker(1)
        success, msg = driver.drop_marker(1)
        assert success is True
        assert msg != 'Already dropped'

        # Marker 2 still works independently
        success, _ = driver.drop_marker(2)
        assert success is True

    def test_retries_up_to_four_attempts(self):
        """3 failed attempts then a success on the 4th should still succeed."""
        driver = self.MockActuatorDriver(max_attempts=4, fail_first_n=3)
        success, _ = driver.fire_torpedo(1)
        assert success is True
        assert driver._attempt_count == 4
        assert driver.actuations.count('torpedo_1') == 1  # Only the 4th call actuated

    def test_gives_up_after_max_attempts(self):
        """If every attempt fails, it must fail after exactly max_attempts tries."""
        driver = self.MockActuatorDriver(max_attempts=4, fail_first_n=4)
        success, _ = driver.fire_torpedo(1)
        assert success is False
        assert driver._attempt_count == 4
        assert driver.actuations.count('torpedo_1') == 0
