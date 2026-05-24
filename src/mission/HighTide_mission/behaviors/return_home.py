"""
Task 6: Return Home — Pass back through the start gate underwater.
"""

import py_trees
from .common import (WaitForDetection, WaitForDuration,
                     LogBehavior, StopMotion, PublishDepthSetpoint)
from .gate import SurgeThrough
from . import blackboard_keys as bb


def create_return_home_subtree() -> py_trees.behaviour.Behaviour:
    """Build the Task 6 (Return Home) behavior subtree."""
    return py_trees.composites.Sequence(
        name='Task6_ReturnHome',
        memory=True,
        children=[
            LogBehavior('ReturnHome_Start', 'Starting Task 6: Return Home'),
            PublishDepthSetpoint('SubmergeForReturn', depth_m=1.0),
            WaitForDuration('WaitSubmerge', duration_sec=5.0),
            # Search for gate — expand search pattern
            WaitForDetection('FindGateReturn', 'gate', timeout=120.0),
            SurgeThrough('ApproachGateReturn', duration=3.0, speed=0.3),
            SurgeThrough('PassThroughGateReturn', duration=5.0, speed=0.5),
            StopMotion('StopAfterReturn'),
            LogBehavior('ReturnHome_Done', 'Task 6 Return Home COMPLETE'),
        ],
    )
