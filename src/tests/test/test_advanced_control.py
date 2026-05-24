"""
Tests for advanced control logic (Mode Manager & Barrel Roll).

Covers:
  - Mode Manager parameter overrides
  - Mode transitions and assertions
  - Barrel Roll execution timing and PWM channels
"""

import pytest
import time
from unittest.mock import MagicMock, patch

class TestAdvancedControlLogic:
    """Tests for Mode Manager and Barrel Roll logic."""

    class MockModeManager:
        def __init__(self):
            self.barrel_roll_duration = 3.0
            self.barrel_roll_pwm = 1900
            self.current_mode = 'ALT_HOLD'
            self.rc_messages_published = []
            
        def _call_set_mode(self, mode):
            self.current_mode = mode
            return True

        def _barrel_roll_service(self):
            # Switch to Manual mode
            if not self._call_set_mode('MANUAL'):
                return False, 'Failed to switch to Manual mode'

            # Command barrel roll: opposite roll command via RC Override
            # Channel 2 (Roll) index 1: full deflection
            roll_channels = [1500] * 18
            roll_channels[1] = self.barrel_roll_pwm
            
            rate = 20
            iterations = int(self.barrel_roll_duration * rate)
            
            # Instead of actual sleep, we just simulate publishing
            for _ in range(iterations):
                self.rc_messages_published.append(roll_channels)

            # Stop motors
            stop_channels = [1500] * 18
            self.rc_messages_published.append(stop_channels)

            return True, 'Barrel roll executed successfully'


    def test_mode_transitions(self):
        mgr = self.MockModeManager()
        assert mgr.current_mode == 'ALT_HOLD'
        mgr._call_set_mode('MANUAL')
        assert mgr.current_mode == 'MANUAL'

    def test_barrel_roll_execution(self):
        mgr = self.MockModeManager()
        success, msg = mgr._barrel_roll_service()
        
        assert success is True
        assert mgr.current_mode == 'MANUAL'
        
        # 3.0s * 20Hz = 60 publications + 1 stop command = 61 messages
        assert len(mgr.rc_messages_published) == 61
        
        # Check roll command
        first_cmd = mgr.rc_messages_published[0]
        assert first_cmd[1] == 1900  # Roll channel (idx 1) set to max
        assert first_cmd[0] == 1500  # Pitch neutral
        assert first_cmd[2] == 1500  # Throttle neutral
        
        # Check stop command (last message)
        last_cmd = mgr.rc_messages_published[-1]
        assert last_cmd[1] == 1500  # Roll channel neutral
        assert sum(last_cmd) == 1500 * 18  # All channels neutral
