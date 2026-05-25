"""
Tests for advanced mission logic (Behavior Tree & Emergency Surface).

Covers:
  - Behavior Tree generation and structure
  - Emergency Surface execution
"""

import pytest
import py_trees
from unittest.mock import MagicMock

# Import the subtree generators
from hightide_mission.behaviors.gate import create_gate_subtree
from hightide_mission.behaviors.slalom import create_slalom_subtree
from hightide_mission.behaviors.bins import create_bins_subtree
from hightide_mission.behaviors.torpedoes import create_torpedoes_subtree
from hightide_mission.behaviors.octagon import create_octagon_subtree
from hightide_mission.behaviors.return_home import create_return_home_subtree
from hightide_mission.behaviors.emergency import EmergencySurface


class TestBehaviorTreeStructure:
    
    def test_gate_subtree(self):
        tree = create_gate_subtree()
        assert isinstance(tree, py_trees.composites.Sequence)
        assert tree.name == 'Task1_Gate'
        assert len(tree.children) >= 8  # Ensure multiple behaviors exist

    def test_slalom_subtree(self):
        tree = create_slalom_subtree()
        assert isinstance(tree, py_trees.composites.Sequence)
        assert tree.name == 'Task2_Slalom'
        assert len(tree.children) >= 5

    def test_bins_subtree(self):
        tree = create_bins_subtree()
        assert isinstance(tree, py_trees.composites.Sequence)
        assert tree.name == 'Task3_Bins'

    def test_torpedoes_subtree(self):
        tree = create_torpedoes_subtree()
        assert isinstance(tree, py_trees.composites.Sequence)
        assert tree.name == 'Task4_Torpedoes'
        
    def test_octagon_subtree(self):
        tree = create_octagon_subtree()
        assert isinstance(tree, py_trees.composites.Sequence)
        assert tree.name == 'Task5_Octagon'

    def test_return_home_subtree(self):
        tree = create_return_home_subtree()
        assert isinstance(tree, py_trees.composites.Sequence)
        assert tree.name == 'Task6_ReturnHome'


class TestEmergencySurface:

    def test_emergency_surface_sets_depth_zero(self):
        # Create mock ROS node to inject into blackboard
        mock_node = MagicMock()
        mock_node.get_logger = MagicMock()
        mock_node.depth_pub = MagicMock()
        
        # Set up py_trees blackboard
        py_trees.blackboard.Blackboard.clear()
        bb = py_trees.blackboard.Client(name='TestEmergency')
        bb.register_key(key='ros_node', access=py_trees.common.Access.WRITE)
        bb.set('ros_node', mock_node)
        
        # Initialize and tick behavior
        behavior = EmergencySurface(name="Emergency")
        behavior.setup(timeout=1.0)
        behavior.initialise()
        
        status = behavior.update()
        
        # Verify status is SUCCESS (it's a one-shot action)
        assert status == py_trees.common.Status.SUCCESS
        
        # Verify it published 0.0 to depth publisher
        assert mock_node.depth_pub.publish.called
        published_msg = mock_node.depth_pub.publish.call_args[0][0]
        assert published_msg.data == 0.0
        
        # Verify error log was printed
        assert mock_node.get_logger().error.called
        log_msg = mock_node.get_logger().error.call_args[0][0]
        assert 'EMERGENCY SURFACE' in log_msg
