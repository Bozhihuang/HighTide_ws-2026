"""
Common behavior tree leaf nodes shared across all tasks.

Each behavior is a py_trees.behaviour.Behaviour subclass that accesses
ROS2 through the blackboard-stored node reference.
"""

import time as pytime
import py_trees
from std_msgs.msg import Float64
from hightide_interfaces.msg import ThrusterCommand, DetectionArray
from . import blackboard_keys as bb


class PublishThrusterCommand(py_trees.behaviour.Behaviour):
    """Publish a single ThrusterCommand and succeed."""

    def __init__(self, name, surge=0.0, sway=0.0, heave=0.0,
                 yaw=0.0, pitch=0.0, roll=0.0):
        super().__init__(name)
        self.cmd = ThrusterCommand()
        self.cmd.surge = surge
        self.cmd.sway = sway
        self.cmd.heave = heave
        self.cmd.yaw = yaw
        self.cmd.pitch = pitch
        self.cmd.roll = roll
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def update(self):
        node = self.blackboard.get(bb.ROS_NODE)
        pub = node.cmd_pub
        self.cmd.header.stamp = node.get_clock().now().to_msg()
        pub.publish(self.cmd)
        return py_trees.common.Status.SUCCESS


class PublishDepthSetpoint(py_trees.behaviour.Behaviour):
    """Publish a target depth and succeed."""

    def __init__(self, name, depth_m=1.0):
        super().__init__(name)
        self.depth_m = depth_m
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def update(self):
        node = self.blackboard.get(bb.ROS_NODE)
        msg = Float64()
        msg.data = self.depth_m
        node.depth_pub.publish(msg)
        return py_trees.common.Status.SUCCESS


class WaitForDuration(py_trees.behaviour.Behaviour):
    """Tick RUNNING for duration_sec, then SUCCESS."""

    def __init__(self, name, duration_sec=3.0):
        super().__init__(name)
        self.duration = duration_sec
        self.start_time = None

    def initialise(self):
        self.start_time = pytime.time()

    def update(self):
        elapsed = pytime.time() - self.start_time
        if elapsed >= self.duration:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class WaitForDetection(py_trees.behaviour.Behaviour):
    """Tick RUNNING until a specific class is detected. Writes detection to blackboard."""

    def __init__(self, name, target_class, confidence_threshold=0.5, timeout=30.0):
        super().__init__(name)
        self.target_class = target_class
        self.conf_thresh = confidence_threshold
        self.timeout = timeout
        self.start_time = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.TARGET_DETECTION, access=py_trees.common.Access.WRITE)

    def initialise(self):
        self.start_time = pytime.time()

    def update(self):
        if (pytime.time() - self.start_time) > self.timeout:
            return py_trees.common.Status.FAILURE

        try:
            detections = self.blackboard.get(bb.DETECTIONS)
        except KeyError:
            return py_trees.common.Status.RUNNING

        if detections is None:
            return py_trees.common.Status.RUNNING

        for det in detections.detections:
            if (det.class_name == self.target_class and
                    det.confidence >= self.conf_thresh):
                self.blackboard.set(bb.TARGET_DETECTION, det)
                return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class CheckDetectionVisible(py_trees.behaviour.Behaviour):
    """Condition: checks if a class is currently visible in detections."""

    def __init__(self, name, target_class, min_confidence=0.4):
        super().__init__(name)
        self.target_class = target_class
        self.min_conf = min_confidence
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)

    def update(self):
        try:
            detections = self.blackboard.get(bb.DETECTIONS)
        except KeyError:
            return py_trees.common.Status.FAILURE

        if detections:
            for det in detections.detections:
                if (det.class_name == self.target_class and
                        det.confidence >= self.min_conf):
                    return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class SetBlackboardValue(py_trees.behaviour.Behaviour):
    """Write a value to the blackboard and succeed."""

    def __init__(self, name, key, value):
        super().__init__(name)
        self.key = key
        self.value = value
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)

    def update(self):
        self.blackboard.set(self.key, self.value)
        return py_trees.common.Status.SUCCESS


class LogBehavior(py_trees.behaviour.Behaviour):
    """Log a message and succeed."""

    def __init__(self, name, message=''):
        super().__init__(name)
        self.message = message
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def update(self):
        node = self.blackboard.get(bb.ROS_NODE)
        node.get_logger().info(f'[BT] {self.message}')
        return py_trees.common.Status.SUCCESS


class StopMotion(py_trees.behaviour.Behaviour):
    """Publish zero command to stop all motion."""

    def __init__(self, name='StopMotion'):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def update(self):
        node = self.blackboard.get(bb.ROS_NODE)
        node.cmd_pub.publish(ThrusterCommand())
        return py_trees.common.Status.SUCCESS
