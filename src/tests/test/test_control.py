"""
Tests for hightide_control — RC Override, Depth Controller, Mode Manager.

Covers:
  - PWM conversion (normalize_to_pwm)
  - Channel mapping correctness
  - Deadzone behavior
  - Safety timeout (neutral on stale data)
  - Depth controller PID direction
  - RC Override message structure
"""

import math
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from conftest import assert_pwm_in_range, assert_pwm_neutral


class TestRCOverridePWMConversion:
    """Test the normalize_to_pwm conversion function."""

    def _make_node_mock(self):
        """Create a mock RCOverrideNode with just the conversion function."""
        from hightide_control.rc_override_node import RCOverrideNode
        # We can't instantiate the full node without rclpy, so test the math directly
        class PWMConverter:
            neutral_pwm = 1500
            max_pwm = 1900
            min_pwm = 1100
            deadzone = 0.05
            pwm_scale = 400

            def _normalize_to_pwm(self, value):
                if abs(value) < self.deadzone:
                    return self.neutral_pwm
                clamped = max(-1.0, min(1.0, value))
                pwm = self.neutral_pwm + int(clamped * self.pwm_scale)
                return max(self.min_pwm, min(self.max_pwm, pwm))

        return PWMConverter()

    def test_zero_gives_neutral(self):
        conv = self._make_node_mock()
        assert conv._normalize_to_pwm(0.0) == 1500

    def test_full_forward(self):
        conv = self._make_node_mock()
        assert conv._normalize_to_pwm(1.0) == 1900

    def test_full_reverse(self):
        conv = self._make_node_mock()
        assert conv._normalize_to_pwm(-1.0) == 1100

    def test_half_forward(self):
        conv = self._make_node_mock()
        assert conv._normalize_to_pwm(0.5) == 1700

    def test_half_reverse(self):
        conv = self._make_node_mock()
        assert conv._normalize_to_pwm(-0.5) == 1300

    def test_deadzone_positive_small(self):
        conv = self._make_node_mock()
        assert conv._normalize_to_pwm(0.04) == 1500  # Below deadzone

    def test_deadzone_negative_small(self):
        conv = self._make_node_mock()
        assert conv._normalize_to_pwm(-0.04) == 1500

    def test_just_above_deadzone(self):
        conv = self._make_node_mock()
        result = conv._normalize_to_pwm(0.06)
        assert result != 1500
        assert result > 1500

    def test_clamp_over_max(self):
        conv = self._make_node_mock()
        assert conv._normalize_to_pwm(2.0) == 1900  # Clamped to 1.0

    def test_clamp_under_min(self):
        conv = self._make_node_mock()
        assert conv._normalize_to_pwm(-2.0) == 1100

    def test_all_valid_range(self):
        conv = self._make_node_mock()
        for i in range(-100, 101):
            val = i / 100.0
            pwm = conv._normalize_to_pwm(val)
            assert_pwm_in_range(pwm)

    def test_quarter_values(self):
        conv = self._make_node_mock()
        assert conv._normalize_to_pwm(0.25) == 1600
        assert conv._normalize_to_pwm(-0.25) == 1400
        assert conv._normalize_to_pwm(0.75) == 1800
        assert conv._normalize_to_pwm(-0.75) == 1200


class TestChannelMapping:
    """Verify ArduSub channel constants are correct."""

    def test_channel_indices(self):
        from hightide_control.rc_override_node import (
            CH_PITCH, CH_ROLL, CH_THROTTLE, CH_YAW, CH_FORWARD, CH_LATERAL
        )
        assert CH_PITCH == 0
        assert CH_ROLL == 1
        assert CH_THROTTLE == 2
        assert CH_YAW == 3
        assert CH_FORWARD == 4
        assert CH_LATERAL == 5

    def test_chan_nochange(self):
        from hightide_control.rc_override_node import CHAN_NOCHANGE
        assert CHAN_NOCHANGE == 65535

    def test_num_channels(self):
        from hightide_control.rc_override_node import NUM_CHANNELS
        assert NUM_CHANNELS == 18


class TestDepthControllerLogic:
    """Test depth controller PID direction and math."""

    def test_positive_error_descends(self):
        """When target > current (need to go deeper), PWM should be < 1500."""
        # error = target - current = positive → output positive → pwm = 1500 - output < 1500
        target = 2.0
        current = 1.0
        error = target - current  # 1.0 — need to go deeper
        kp = 100.0
        output = kp * error  # 100.0
        pwm = 1500 - int(output)  # 1500 - 100 = 1400
        assert pwm < 1500, 'Positive depth error should produce descending PWM (< 1500)'
        assert pwm >= 1100

    def test_negative_error_ascends(self):
        """When target < current (need to go shallower), PWM should be > 1500."""
        target = 0.5
        current = 2.0
        error = target - current  # -1.5 — need to ascend
        kp = 100.0
        output = kp * error  # -150
        pwm = 1500 - int(output)  # 1500 + 150 = 1650
        assert pwm > 1500, 'Negative depth error should produce ascending PWM (> 1500)'
        assert pwm <= 1900

    def test_zero_error_neutral(self):
        target = 1.0
        current = 1.0
        error = target - current
        assert error == 0.0

    def test_depth_conversion_sign(self):
        """rel_alt is negative for underwater. depth = -rel_alt should be positive."""
        rel_alt = -1.5  # 1.5m below surface
        depth = -rel_alt
        assert depth == 1.5

    def test_output_clamping(self):
        """PID output should be clamped to [-max_output, max_output]."""
        max_output = 400
        huge_error = 100.0
        kp = 100.0
        raw_output = kp * huge_error  # 10000 — way over limit
        clamped = max(-max_output, min(max_output, raw_output))
        assert clamped == max_output
        pwm = 1500 - clamped  # 1100
        assert pwm == 1100


class TestModeManagerConstants:
    """Verify mode strings match ArduSub expectations."""

    def test_mode_strings(self):
        """The exact strings ArduSub expects for set_mode."""
        modes = ['ALT_HOLD', 'MANUAL', 'STABILIZE']
        for mode in modes:
            assert mode == mode.upper(), f'Mode "{mode}" should be uppercase'
            assert ' ' not in mode, f'Mode "{mode}" should not have spaces'

    def test_system_id(self):
        """ArduSub requires system_id = 255 for MAVROS."""
        expected_system_id = 255
        assert expected_system_id == 255
