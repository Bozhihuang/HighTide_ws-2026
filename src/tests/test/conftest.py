"""
Shared test fixtures and helpers for HighTide unit tests.

Provides:
  - rclpy_init/shutdown management
  - Mock ROS2 node factory
  - Message builders for common types
  - Assertion helpers
"""

import time
import math
import threading
from unittest.mock import MagicMock, patch
from contextlib import contextmanager

import pytest
import rclpy
from rclpy.node import Node


# ==================== rclpy lifecycle ====================

_rclpy_initialized = False
_rclpy_lock = threading.Lock()


@pytest.fixture(scope='session', autouse=True)
def rclpy_session():
    """Initialize rclpy once for the entire test session."""
    global _rclpy_initialized
    with _rclpy_lock:
        if not _rclpy_initialized:
            rclpy.init()
            _rclpy_initialized = True
    yield
    with _rclpy_lock:
        if _rclpy_initialized:
            rclpy.shutdown()
            _rclpy_initialized = False


@pytest.fixture
def ros_node():
    """Create a temporary ROS2 node for testing. Destroyed after test."""
    node = Node(f'test_node_{int(time.time() * 1000) % 100000}')
    yield node
    node.destroy_node()


# ==================== Message Builders ====================

def make_thruster_command(surge=0.0, sway=0.0, heave=0.0,
                          yaw=0.0, pitch=0.0, roll=0.0):
    """Build a ThrusterCommand message."""
    from HighTide_interfaces.msg import ThrusterCommand
    msg = ThrusterCommand()
    msg.surge = float(surge)
    msg.sway = float(sway)
    msg.heave = float(heave)
    msg.yaw = float(yaw)
    msg.pitch = float(pitch)
    msg.roll = float(roll)
    return msg


def make_detection(class_id=0, class_name='gate', confidence=0.9,
                   x_min=100.0, y_min=100.0, x_max=200.0, y_max=200.0,
                   depth_m=-1.0):
    """Build a Detection message."""
    from HighTide_interfaces.msg import Detection
    det = Detection()
    det.class_id = class_id
    det.class_name = class_name
    det.confidence = float(confidence)
    det.x_min = float(x_min)
    det.y_min = float(y_min)
    det.x_max = float(x_max)
    det.y_max = float(y_max)
    det.center_x = (x_min + x_max) / 2.0
    det.center_y = (y_min + y_max) / 2.0
    det.width = x_max - x_min
    det.height = y_max - y_min
    det.depth_m = float(depth_m)
    return det


def make_detection_array(detections=None, image_width=1280, image_height=720):
    """Build a DetectionArray message."""
    from HighTide_interfaces.msg import DetectionArray
    msg = DetectionArray()
    msg.image_width = image_width
    msg.image_height = image_height
    if detections:
        msg.detections = detections
    return msg


def make_odometry(x=0.0, y=0.0, z=0.0, yaw=0.0):
    """Build a nav_msgs/Odometry with position and yaw."""
    from nav_msgs.msg import Odometry
    msg = Odometry()
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.position.z = z
    # Convert yaw to quaternion
    msg.pose.pose.orientation.x = 0.0
    msg.pose.pose.orientation.y = 0.0
    msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
    msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
    return msg


def make_imu(yaw=0.0, pitch=0.0, roll=0.0):
    """Build a sensor_msgs/Imu with orientation."""
    from sensor_msgs.msg import Imu
    msg = Imu()
    # Simplified — only yaw for testing
    msg.orientation.x = 0.0
    msg.orientation.y = 0.0
    msg.orientation.z = math.sin(yaw / 2.0)
    msg.orientation.w = math.cos(yaw / 2.0)
    return msg


def make_mavros_state(armed=False, mode='ALT_HOLD', connected=True):
    """Build a mavros_msgs/State."""
    from mavros_msgs.msg import State
    msg = State()
    msg.armed = armed
    msg.mode = mode
    msg.connected = connected
    return msg


# ==================== Assertion Helpers ====================

def assert_pwm_in_range(pwm, min_pwm=1100, max_pwm=1900):
    """Assert a PWM value is within valid range."""
    assert min_pwm <= pwm <= max_pwm, \
        f'PWM {pwm} out of range [{min_pwm}, {max_pwm}]'


def assert_pwm_neutral(pwm, neutral=1500, tolerance=5):
    """Assert a PWM value is approximately neutral."""
    assert abs(pwm - neutral) <= tolerance, \
        f'PWM {pwm} not near neutral {neutral} (tolerance {tolerance})'


def assert_angle_close(angle_a, angle_b, tolerance=0.1):
    """Assert two angles are close (handles wraparound)."""
    diff = abs(angle_a - angle_b)
    diff = min(diff, 2 * math.pi - diff)
    assert diff <= tolerance, \
        f'Angles {angle_a:.3f} and {angle_b:.3f} differ by {diff:.3f} (tolerance {tolerance})'


# ==================== Spin Helpers ====================

def spin_for(node, duration_sec=0.1, rate_hz=100):
    """Spin a node for a given duration to process callbacks."""
    end = time.time() + duration_sec
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=1.0 / rate_hz)
