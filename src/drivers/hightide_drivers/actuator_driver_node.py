#!/usr/bin/env python3
"""
Actuator Driver Node — Controls torpedoes and marker droppers via GPIO.

Uses GPIO pins on Blue Robotics Navigator to trigger relays powering solenoids.
Gracefully degrades to simulation mode if GPIO library is unavailable.
"""

import time as pytime
import rclpy
from rclpy.node import Node
from hightide_interfaces.srv import FireTorpedo, DropMarker

# Try importing GPIO library — falls back to simulation if unavailable
try:
    import gpiod
    GPIO_AVAILABLE = True
except ImportError:
    try:
        import RPi.GPIO as GPIO
        GPIO_AVAILABLE = True
    except ImportError:
        GPIO_AVAILABLE = False


class ActuatorDriverNode(Node):
    """Controls torpedoes and marker droppers via GPIO pins."""

    def __init__(self):
        super().__init__('actuator_driver_node')

        # GPIO pin mapping
        self.declare_parameter('torpedo_1_pin', 27)
        self.declare_parameter('torpedo_2_pin', 22)
        self.declare_parameter('dropper_1_pin', 23)
        self.declare_parameter('dropper_2_pin', 24)
        self.declare_parameter('pulse_duration_ms', 500)
        self.declare_parameter('gpio_chip', '/dev/gpiochip0')

        self.pins = {
            'torpedo_1': self.get_parameter('torpedo_1_pin').value,
            'torpedo_2': self.get_parameter('torpedo_2_pin').value,
            'dropper_1': self.get_parameter('dropper_1_pin').value,
            'dropper_2': self.get_parameter('dropper_2_pin').value,
        }
        self.pulse_ms = self.get_parameter('pulse_duration_ms').value

        # Track fired state (prevent double-fire)
        self.torpedoes_fired = {1: False, 2: False}
        self.markers_dropped = {1: False, 2: False}

        # Initialize GPIO
        self.gpio_lines = {}
        if GPIO_AVAILABLE:
            try:
                self._setup_gpio()
            except Exception as e:
                self.get_logger().warn(f'GPIO setup failed: {e} — running in SIM mode')
        else:
            self.get_logger().warn('GPIO not available — running in SIMULATION mode')

        # Services
        self.torpedo_srv = self.create_service(
            FireTorpedo, '/hightide/fire_torpedo', self._fire_torpedo)
        self.dropper_srv = self.create_service(
            DropMarker, '/hightide/drop_marker', self._drop_marker)

        self.get_logger().info('Actuator Driver Node started')

    def _setup_gpio(self):
        """Initialize GPIO pins as outputs, set LOW."""
        if 'gpiod' in dir():
            chip = gpiod.Chip(self.get_parameter('gpio_chip').value)
            for name, pin in self.pins.items():
                line = chip.get_line(pin)
                line.request(consumer='hightide_actuator',
                             type=gpiod.LINE_REQ_DIR_OUT)
                line.set_value(0)
                self.gpio_lines[name] = line
            self.get_logger().info('GPIO initialized via gpiod')
        else:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for name, pin in self.pins.items():
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            self.get_logger().info('GPIO initialized via RPi.GPIO')

    def _actuate_pin(self, name: str) -> bool:
        """Pulse a GPIO pin HIGH for pulse_duration_ms."""
        pin = self.pins.get(name)
        if pin is None:
            return False

        self.get_logger().info(f'Actuating {name} (pin {pin}) for {self.pulse_ms}ms')

        if name in self.gpio_lines:
            self.gpio_lines[name].set_value(1)
            pytime.sleep(self.pulse_ms / 1000.0)
            self.gpio_lines[name].set_value(0)
        elif GPIO_AVAILABLE:
            GPIO.output(pin, GPIO.HIGH)
            pytime.sleep(self.pulse_ms / 1000.0)
            GPIO.output(pin, GPIO.LOW)
        else:
            # Simulation mode
            self.get_logger().info(f'[SIM] {name} actuated (pin {pin})')
            pytime.sleep(self.pulse_ms / 1000.0)

        return True

    def _fire_torpedo(self, request, response):
        """Fire torpedo from specified tube."""
        tube_id = request.tube_id

        if tube_id not in (1, 2):
            response.success = False
            response.message = f'Invalid tube_id: {tube_id} (must be 1 or 2)'
            return response

        if self.torpedoes_fired[tube_id]:
            response.success = False
            response.message = f'Torpedo {tube_id} already fired!'
            self.get_logger().warn(response.message)
            return response

        pin_name = f'torpedo_{tube_id}'
        success = self._actuate_pin(pin_name)
        self.torpedoes_fired[tube_id] = True

        response.success = success
        response.message = f'Torpedo {tube_id} {"fired" if success else "failed"}!'
        self.get_logger().info(response.message)
        return response

    def _drop_marker(self, request, response):
        """Drop marker from specified dropper."""
        dropper_id = request.dropper_id

        if dropper_id not in (1, 2):
            response.success = False
            response.message = f'Invalid dropper_id: {dropper_id}'
            return response

        if self.markers_dropped[dropper_id]:
            response.success = False
            response.message = f'Marker {dropper_id} already dropped!'
            self.get_logger().warn(response.message)
            return response

        pin_name = f'dropper_{dropper_id}'
        success = self._actuate_pin(pin_name)
        self.markers_dropped[dropper_id] = True

        response.success = success
        response.message = f'Marker {dropper_id} {"dropped" if success else "failed"}!'
        self.get_logger().info(response.message)
        return response

    def destroy_node(self):
        """Clean up GPIO on shutdown."""
        for line in self.gpio_lines.values():
            try:
                line.set_value(0)
                line.release()
            except Exception:
                pass
        if GPIO_AVAILABLE and not self.gpio_lines:
            try:
                GPIO.cleanup()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ActuatorDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
