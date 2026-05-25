"""
Emergency/safety behaviors — checked every tick of the behavior tree.
"""

import py_trees
from . import blackboard_keys as bb
from std_msgs.msg import Float64


class CheckMissionTimeout(py_trees.behaviour.Behaviour):
    """Condition: fails if mission timeout exceeded."""

    def __init__(self, name='CheckMissionTimeout'):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.MISSION_START_TIME, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.MISSION_TIMEOUT, access=py_trees.common.Access.READ)

    def update(self):
        import time
        try:
            start = self.blackboard.get(bb.MISSION_START_TIME)
            timeout = self.blackboard.get(bb.MISSION_TIMEOUT)
            if (time.time() - start) > timeout:
                return py_trees.common.Status.FAILURE
        except KeyError:
            pass
        return py_trees.common.Status.SUCCESS


class EmergencySurface(py_trees.behaviour.Behaviour):
    """Emergency surface: set depth to 0 and disarm."""

    def __init__(self, name='EmergencySurface'):
        super().__init__(name)
        self.surfacing = False
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        self.surfacing = True
        node = self.blackboard.get(bb.ROS_NODE)
        node.get_logger().error('=== EMERGENCY SURFACE ===')
        msg = Float64()
        msg.data = 0.0
        node.depth_pub.publish(msg)

    def update(self):
        return py_trees.common.Status.SUCCESS
