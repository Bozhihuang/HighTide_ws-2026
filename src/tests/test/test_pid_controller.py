"""
Tests for the PID controller utility in HighTide_navigation.

Covers:
  - Zero error produces zero output
  - Proportional response
  - Integral accumulation
  - Derivative kick
  - Output clamping
  - Anti-windup
  - Reset
  - normalize_angle
  - quaternion_to_yaw
"""

import math
import pytest
from HighTide_navigation import PIDController, normalize_angle, quaternion_to_yaw


class TestPIDController:
    """Tests for the shared PID controller class."""

    def test_zero_error_zero_output(self):
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
        assert pid.compute(0.0, 0.1) == 0.0

    def test_zero_dt_zero_output(self):
        pid = PIDController(kp=1.0, ki=0.5, kd=0.5)
        assert pid.compute(1.0, 0.0) == 0.0

    def test_negative_dt_zero_output(self):
        pid = PIDController(kp=1.0)
        assert pid.compute(1.0, -0.1) == 0.0

    def test_proportional_only(self):
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0)
        out = pid.compute(0.5, 0.1)
        assert abs(out - 1.0) < 1e-6, f'Expected 1.0, got {out}'

    def test_proportional_negative_error(self):
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0)
        out = pid.compute(-0.5, 0.1)
        assert abs(out - (-1.0)) < 1e-6

    def test_integral_accumulation(self):
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0)
        out1 = pid.compute(1.0, 0.1)  # integral = 0.1
        out2 = pid.compute(1.0, 0.1)  # integral = 0.2
        assert abs(out1 - 0.1) < 1e-6
        assert abs(out2 - 0.2) < 1e-6

    def test_integral_windup_clamping(self):
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0, integral_max=0.5)
        for _ in range(100):
            pid.compute(1.0, 0.1)
        # Integral should be clamped at integral_max
        out = pid.compute(0.0, 0.1)  # error=0 this tick, but integral still there
        assert out <= 0.5 + 1e-6

    def test_derivative_response(self):
        pid = PIDController(kp=0.0, ki=0.0, kd=1.0)
        pid.compute(0.0, 0.1)  # First call — establishes prev_error
        out = pid.compute(1.0, 0.1)  # Error jumped from 0 to 1, derivative = 10
        # With low-pass filter (alpha=0.7): derivative ≈ 0.7 * 10 + 0.3 * 0 = 7.0
        assert out > 0.0, 'Derivative should produce positive output on rising error'

    def test_output_clamping_max(self):
        pid = PIDController(kp=100.0, output_max=0.8)
        out = pid.compute(1.0, 0.1)
        assert out <= 0.8

    def test_output_clamping_min(self):
        pid = PIDController(kp=100.0, output_min=-0.8)
        out = pid.compute(-1.0, 0.1)
        assert out >= -0.8

    def test_reset_clears_state(self):
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0)
        for _ in range(10):
            pid.compute(1.0, 0.1)
        pid.reset()
        assert pid.integral == 0.0
        assert pid.prev_error == 0.0
        assert pid.prev_derivative == 0.0

    def test_symmetric_output(self):
        pid_pos = PIDController(kp=1.0, ki=0.5, kd=0.2)
        pid_neg = PIDController(kp=1.0, ki=0.5, kd=0.2)
        out_pos = pid_pos.compute(1.0, 0.1)
        out_neg = pid_neg.compute(-1.0, 0.1)
        assert abs(out_pos + out_neg) < 1e-6, 'PID should be symmetric'

    def test_custom_output_range(self):
        pid = PIDController(kp=10.0, output_min=-0.5, output_max=0.5)
        assert pid.compute(1.0, 0.1) == 0.5
        assert pid.compute(-1.0, 0.1) >= -0.5


class TestNormalizeAngle:
    """Tests for the angle normalization utility."""

    def test_zero(self):
        assert normalize_angle(0.0) == 0.0

    def test_pi(self):
        assert abs(normalize_angle(math.pi) - math.pi) < 1e-6

    def test_negative_pi(self):
        assert abs(normalize_angle(-math.pi) - (-math.pi)) < 1e-6

    def test_two_pi_wraps_to_zero(self):
        assert abs(normalize_angle(2 * math.pi)) < 1e-6

    def test_negative_two_pi_wraps_to_zero(self):
        assert abs(normalize_angle(-2 * math.pi)) < 1e-6

    def test_three_pi_wraps_to_negative_pi(self):
        result = normalize_angle(3 * math.pi)
        assert abs(result - math.pi) < 1e-6 or abs(result + math.pi) < 1e-6

    def test_large_positive(self):
        result = normalize_angle(10 * math.pi + 0.5)
        assert -math.pi <= result <= math.pi

    def test_large_negative(self):
        result = normalize_angle(-10 * math.pi - 0.5)
        assert -math.pi <= result <= math.pi

    def test_quarter_pi(self):
        assert abs(normalize_angle(math.pi / 4) - math.pi / 4) < 1e-6

    def test_three_quarter_pi(self):
        result = normalize_angle(3 * math.pi / 4)
        assert abs(result - 3 * math.pi / 4) < 1e-6


class TestQuaternionToYaw:
    """Tests for quaternion → yaw extraction."""

    def test_identity_quaternion_is_zero_yaw(self):
        q = type('Q', (), {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0})()
        assert abs(quaternion_to_yaw(q)) < 1e-6

    def test_90_degree_yaw(self):
        yaw = math.pi / 2
        q = type('Q', (), {
            'x': 0.0, 'y': 0.0,
            'z': math.sin(yaw / 2), 'w': math.cos(yaw / 2)
        })()
        assert abs(quaternion_to_yaw(q) - yaw) < 1e-6

    def test_negative_90_yaw(self):
        yaw = -math.pi / 2
        q = type('Q', (), {
            'x': 0.0, 'y': 0.0,
            'z': math.sin(yaw / 2), 'w': math.cos(yaw / 2)
        })()
        assert abs(quaternion_to_yaw(q) - yaw) < 1e-6

    def test_180_degree_yaw(self):
        yaw = math.pi
        q = type('Q', (), {
            'x': 0.0, 'y': 0.0,
            'z': math.sin(yaw / 2), 'w': math.cos(yaw / 2)
        })()
        result = quaternion_to_yaw(q)
        assert abs(result - math.pi) < 1e-4 or abs(result + math.pi) < 1e-4

    def test_45_degree_yaw(self):
        yaw = math.pi / 4
        q = type('Q', (), {
            'x': 0.0, 'y': 0.0,
            'z': math.sin(yaw / 2), 'w': math.cos(yaw / 2)
        })()
        assert abs(quaternion_to_yaw(q) - yaw) < 1e-6

    def test_roundtrip_various_angles(self):
        for deg in range(-180, 181, 15):
            yaw = math.radians(deg)
            q = type('Q', (), {
                'x': 0.0, 'y': 0.0,
                'z': math.sin(yaw / 2), 'w': math.cos(yaw / 2)
            })()
            result = quaternion_to_yaw(q)
            diff = abs(normalize_angle(result - yaw))
            assert diff < 1e-4, f'Roundtrip failed at {deg}°: got {math.degrees(result):.1f}°'
