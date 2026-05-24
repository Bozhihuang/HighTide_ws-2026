"""
Task 5: Resupply (Octagon) — Navigate into octagon, face correct image, surface.

Strategy: Find octagon via visual search. Navigate inside. Identify correct
image based on role and items collected. Yaw spin for style points.
Surface inside octagon.
"""

import py_trees
from .common import (WaitForDetection, WaitForDuration,
                     LogBehavior, StopMotion, PublishDepthSetpoint)
from . import blackboard_keys as bb


class NavigateIntoOctagon(py_trees.behaviour.Behaviour):
    """Surge into the octagon area."""

    def __init__(self, name='NavigateIntoOctagon', timeout=30.0):
        super().__init__(name)
        self.timeout = timeout
        self.start_time = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        import time
        self.start_time = time.time()

    def update(self):
        import time
        from HighTide_interfaces.msg import ThrusterCommand
        node = self.blackboard.get(bb.ROS_NODE)

        if (time.time() - self.start_time) > self.timeout:
            return py_trees.common.Status.SUCCESS

        # Surge forward toward octagon
        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.surge = 0.3
        node.cmd_pub.publish(cmd)

        try:
            detections = self.blackboard.get(bb.DETECTIONS)
        except KeyError:
            return py_trees.common.Status.RUNNING

        # If octagon is very large in frame, we're inside it
        if detections:
            for det in detections.detections:
                if det.class_name == 'octagon':
                    # If bounding box fills most of frame, we're under it
                    img_w = detections.image_width or 1280
                    if det.width > img_w * 0.7:
                        return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class FaceCorrectImage(py_trees.behaviour.Behaviour):
    """
    Face the correct image on the octagon based on role and items collected.
    1 item → face 🧭 (survey) or 🛟 (rescue)
    2+ items → face ⚒️ (survey) or 🆘 (rescue)
    """

    def __init__(self, name='FaceCorrectImage', timeout=20.0):
        super().__init__(name)
        self.timeout = timeout
        self.start_time = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.CHOSEN_ROLE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.OBJECTS_COLLECTED, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        import time
        self.start_time = time.time()

    def update(self):
        import time
        from HighTide_interfaces.msg import ThrusterCommand
        node = self.blackboard.get(bb.ROS_NODE)

        if (time.time() - self.start_time) > self.timeout:
            return py_trees.common.Status.SUCCESS

        role = self.blackboard.get(bb.CHOSEN_ROLE)
        try:
            items = self.blackboard.get(bb.OBJECTS_COLLECTED)
        except KeyError:
            items = 0

        # Determine target symbol
        if role == 'survey_repair':
            target = 'symbol_compass' if items <= 1 else 'symbol_pickaxe'
        else:
            target = 'symbol_lifering' if items <= 1 else 'symbol_sos'

        try:
            detections = self.blackboard.get(bb.DETECTIONS)
        except KeyError:
            return py_trees.common.Status.RUNNING

        if detections:
            for det in detections.detections:
                if det.class_name == target:
                    img_w = detections.image_width or 1280
                    # Yaw to center the symbol
                    lateral_err = (det.center_x / img_w) - 0.5
                    cmd = ThrusterCommand()
                    cmd.header.stamp = node.get_clock().now().to_msg()
                    cmd.yaw = lateral_err * 2.0  # Use yaw here to face the image
                    node.cmd_pub.publish(cmd)

                    if abs(lateral_err) < 0.1:
                        node.get_logger().info(f'Facing correct image: {target}')
                        return py_trees.common.Status.SUCCESS

        # Slowly rotate to find the symbol
        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.yaw = 0.2  # Slow rotation
        node.cmd_pub.publish(cmd)

        return py_trees.common.Status.RUNNING


class SurfaceInOctagon(py_trees.behaviour.Behaviour):
    """Surface inside the octagon by setting depth to 0."""

    def __init__(self, name='SurfaceInOctagon'):
        super().__init__(name)
        self.start_time = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.CURRENT_DEPTH, access=py_trees.common.Access.READ)

    def initialise(self):
        import time
        from std_msgs.msg import Float64
        self.start_time = time.time()
        node = self.blackboard.get(bb.ROS_NODE)
        msg = Float64()
        msg.data = 0.0  # Surface
        node.depth_pub.publish(msg)

    def update(self):
        import time
        if (time.time() - self.start_time) > 30.0:
            return py_trees.common.Status.SUCCESS

        try:
            depth = self.blackboard.get(bb.CURRENT_DEPTH)
            if depth < 0.3:  # Close to surface
                return py_trees.common.Status.SUCCESS
        except KeyError:
            pass

        return py_trees.common.Status.RUNNING


def create_octagon_subtree() -> py_trees.behaviour.Behaviour:
    """Build the Task 5 (Octagon) behavior subtree."""
    return py_trees.composites.Sequence(
        name='Task5_Octagon',
        memory=True,
        children=[
            LogBehavior('Oct_Start', 'Starting Task 5: Octagon'),
            WaitForDetection('FindOctagon', 'octagon', timeout=120.0),
            NavigateIntoOctagon('EnterOctagon'),
            StopMotion('StopInOctagon'),
            FaceCorrectImage('FaceImage'),
            StopMotion('StopFacing'),
            WaitForDuration('StyleSpinDelay', duration_sec=5.0),
            SurfaceInOctagon('Surface'),
            LogBehavior('Oct_Done', 'Task 5 Octagon COMPLETE'),
        ],
    )
