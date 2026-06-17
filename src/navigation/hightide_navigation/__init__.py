# hightide_navigation package

import math


class PIDController:
    """Simple PID controller with integral windup protection."""

    def __init__(self, kp=1.0, ki=0.0, kd=0.0,
                 output_min=-1.0, output_max=1.0, integral_max=None, fc=5.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        if integral_max is not None:
            self.integral_max = integral_max
        elif ki != 0:
            self.integral_max = abs(output_max / ki)
        else:
            self.integral_max = 100.0

        self.fc = fc

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

        raw_derivative = (error - self.prev_error) / dt
        # rc = 1.0 / (2.0 * math.pi * self.fc)
        # dynamic_alpha = dt / (rc + dt)
        # derivative = dynamic_alpha * raw_derivative + (1.0 - dynamic_alpha) * self.prev_derivative
        alpha = 0.7
        derivative = alpha * raw_derivative + (1.0 - alpha) * self.prev_derivative
        
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
