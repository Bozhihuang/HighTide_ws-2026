"""
Emergency/safety behaviors.

NOTE: mission_node no longer wires these into the tree — timeout supervision
and the surface-then-disarm sequence run at the node level in _tick(), where
the tree can actually be halted (a BT parallel cannot stop the mission branch
from publishing commands that fight the emergency maneuver). These behaviors
are kept for tests and for standalone/scripted use.
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
    """Emergency surface: zero motion, set depth target to 0, and disarm."""

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
        # Stop lateral/forward motion and disarm (fire-and-forget — this is a
        # last-resort behavior, there is nothing useful to do if it fails).
        try:
            from std_srvs.srv import SetBool
            from hightide_interfaces.msg import ThrusterCommand
            node.cmd_pub.publish(ThrusterCommand())
            client = node.create_client(SetBool, '/hightide/arm')
            req = SetBool.Request()
            req.data = False
            client.call_async(req)
        except Exception:
            pass

    def update(self):
        return py_trees.common.Status.SUCCESS
