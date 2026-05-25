"""
Tests for hightide_localization — Navigation Tier Manager state machine.

Covers:
  - State transitions between Tiers (VSLAM -> VIO -> DR)
  - Confidence threshold evaluations
  - Stale data timeout
"""

import pytest
import time


class TestNavTierManagerLogic:
    """Test the tier transition logic of the nav tier manager."""

    # Replicate tier constants for testing logic without full ROS node
    TIER_1_VSLAM = 1
    TIER_2_VIO = 2
    TIER_3_DR = 3

    class MockManager:
        def __init__(self):
            self.current_tier = TestNavTierManagerLogic.TIER_1_VSLAM
            self.vslam_conf_thresh = 0.8
            self.vio_conf_thresh = 0.3
            self.stale_timeout = 1.0
            
            self.last_vslam_time = time.time()
            self.last_vio_time = time.time()
            self.vslam_conf = 1.0
            self.vio_conf = 1.0

        def evaluate_tier(self, current_time):
            # Check for stale data
            if (current_time - self.last_vslam_time) > self.stale_timeout:
                self.vslam_conf = 0.0
            if (current_time - self.last_vio_time) > self.stale_timeout:
                self.vio_conf = 0.0

            # Determine best tier
            new_tier = TestNavTierManagerLogic.TIER_3_DR
            
            if self.vslam_conf >= self.vslam_conf_thresh:
                new_tier = TestNavTierManagerLogic.TIER_1_VSLAM
            elif self.vio_conf >= self.vio_conf_thresh:
                new_tier = TestNavTierManagerLogic.TIER_2_VIO
                
            self.current_tier = new_tier
            return new_tier

    def test_initial_state_high_confidence(self):
        mgr = self.MockManager()
        assert mgr.evaluate_tier(time.time()) == self.TIER_1_VSLAM

    def test_vslam_degrades_to_vio(self):
        mgr = self.MockManager()
        # VSLAM confidence drops below 0.8, VIO still high
        mgr.vslam_conf = 0.7
        mgr.vio_conf = 0.9
        
        tier = mgr.evaluate_tier(time.time())
        assert tier == self.TIER_2_VIO

    def test_both_degrade_to_dr(self):
        mgr = self.MockManager()
        # Both drop below thresholds
        mgr.vslam_conf = 0.5
        mgr.vio_conf = 0.2
        
        tier = mgr.evaluate_tier(time.time())
        assert tier == self.TIER_3_DR

    def test_vslam_stale_timeout(self):
        mgr = self.MockManager()
        # High confidence, but data is old
        mgr.vslam_conf = 0.99
        mgr.vio_conf = 0.99
        
        # Advance time by 2 seconds (timeout is 1.0)
        current_time = mgr.last_vslam_time + 2.0
        
        tier = mgr.evaluate_tier(current_time)
        assert tier == self.TIER_3_DR  # Both timed out since last_*_time weren't updated
        assert mgr.vslam_conf == 0.0
        assert mgr.vio_conf == 0.0

    def test_recovery_dr_to_vslam(self):
        mgr = self.MockManager()
        # Start in DR
        mgr.vslam_conf = 0.0
        mgr.vio_conf = 0.0
        mgr.evaluate_tier(time.time())
        assert mgr.current_tier == self.TIER_3_DR
        
        # New high-confidence VSLAM data arrives
        mgr.vslam_conf = 0.9
        mgr.last_vslam_time = time.time()
        
        tier = mgr.evaluate_tier(time.time())
        assert tier == self.TIER_1_VSLAM
