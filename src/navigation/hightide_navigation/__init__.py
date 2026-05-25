# hightide_navigation package

import math


class PIDController:
    """Simple PID controller with integral windup protection."""

    def __init__(self, kp=1.0, ki=0.0, kd=0.0,
                 output_min=-1.0, output_max=1.0, integral_max=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_max = integral_max or abs(output_max / ki) if ki != 0 else 100.0

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0

    def compute(self, error: float, dt: float) -> float:
        """Compute PID output given error and time step."""
        if dt <= 0:
            return 0.0

        # Proportional
        p = self.kp * error

        # Integral with anti-windup
        self.integral += error * dt
        self.integral = max(-self.integral_max,
                            min(self.integral_max, self.integral))
        i = self.ki * self.integral

        # Derivative with low-pass filter (alpha=0.7)
        raw_derivative = (error - self.prev_error) / dt
        derivative = 0.7 * raw_derivative + 0.3 * self.prev_derivative
        self.prev_derivative = derivative
        self.prev_error = error
        d = self.kd * derivative

        output = p + i + d
        return max(self.output_min, min(self.output_max, output))

    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def quaternion_to_yaw(q) -> float:
    """Extract yaw from a quaternion (x, y, z, w)."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)
