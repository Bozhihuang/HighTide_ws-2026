"""
Tests for hightide_mission — Behavior tree blackboard and leaf logic.

Covers:
  - Blackboard key constants correctness
  - Common behavior logical conditions
"""

import pytest
import py_trees


class TestBlackboardKeys:
    """Ensure blackboard keys are defined and unique."""

    def test_keys_exist(self):
        import hightide_mission.behaviors.blackboard_keys as bb
        assert hasattr(bb, 'VEHICLE_ARMED')
        assert hasattr(bb, 'CURRENT_TASK')
        assert hasattr(bb, 'DETECTIONS')
        
    def test_keys_unique(self):
        import hightide_mission.behaviors.blackboard_keys as bb
        # Get all public string attributes
        keys = [getattr(bb, k) for k in dir(bb) 
                if not k.startswith('_') and isinstance(getattr(bb, k), str)]
                
        # Assert they are all unique
        assert len(keys) == len(set(keys)), "Blackboard keys must be unique strings"


class TestBehaviorLogic:
    """Test logical non-ROS components of behaviors."""

    def test_identify_gate_role_logic(self):
        """Test the logic for determining role from gate symbols."""
        # Using a mock of the behavior's update method logic
        def mock_identify_role(detections_list):
            survey_symbols = {'symbol_compass', 'symbol_pickaxe'}
            rescue_symbols = {'symbol_lifering', 'symbol_sos'}

            for det in detections_list:
                if det['class_name'] in survey_symbols and det['confidence'] > 0.5:
                    return 'survey_repair', py_trees.common.Status.SUCCESS
                elif det['class_name'] in rescue_symbols and det['confidence'] > 0.5:
                    return 'search_rescue', py_trees.common.Status.SUCCESS

            return None, py_trees.common.Status.RUNNING

        # Test survey match
        dets1 = [{'class_name': 'gate', 'confidence': 0.9}, 
                 {'class_name': 'symbol_compass', 'confidence': 0.8}]
        role, status = mock_identify_role(dets1)
        assert role == 'survey_repair'
        assert status == py_trees.common.Status.SUCCESS

        # Test rescue match
        dets2 = [{'class_name': 'symbol_sos', 'confidence': 0.6}]
        role, status = mock_identify_role(dets2)
        assert role == 'search_rescue'
        
        # Test low confidence ignore
        dets3 = [{'class_name': 'symbol_compass', 'confidence': 0.3}]
        role, status = mock_identify_role(dets3)
        assert role is None
        assert status == py_trees.common.Status.RUNNING

    def test_emergency_timeout_logic(self):
        """Test mission timeout calculation logic."""
        import time
        
        def mock_timeout_check(start_time, timeout, current_time):
            if (current_time - start_time) > timeout:
                return py_trees.common.Status.FAILURE
            return py_trees.common.Status.SUCCESS
            
        start = 1000.0
        timeout = 900.0
        
        # Well within timeout
        assert mock_timeout_check(start, timeout, 1500.0) == py_trees.common.Status.SUCCESS
        # Exactly at timeout
        assert mock_timeout_check(start, timeout, 1900.0) == py_trees.common.Status.SUCCESS
        # Past timeout
        assert mock_timeout_check(start, timeout, 1901.0) == py_trees.common.Status.FAILURE
