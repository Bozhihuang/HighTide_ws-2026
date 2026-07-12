"""
Task 5: Restore (Octagon) — surface inside the octagon.

Hardware reality for hightide: we have NO manipulator/claw, so the resupply
portion (collecting nut/bolt/plug/pill/bandage and placing them in baskets, and
the inventory-based facing/rotation bonuses) is out of scope. We also have NO
hydrophone, so we cannot home on the acoustic pinger. This task therefore does
the achievable core: get to the octagon (drive toward it visually if the camera
picks it up, otherwise advance a fixed distance using ZED odometry) and surface
inside it.
"""

import math
import py_trees
from .common import (WaitForDetection, WaitForDuration,
                     LogBehavior, StopMotion, PublishDepthSetpoint)
from . import blackboard_keys as bb


class NavigateIntoOctagon(py_trees.behaviour.Behaviour):
    """
    Drive forward into the octagon. If the octagon is detected we center on it
    and stop once it fills the frame (we're underneath it). Otherwise we advance
    a fixed real-world distance using ZED odometry (CURRENT_POSE) as the
    measuring stick — not a timed guess — then stop, on the assumption the
    octagon is roughly `advance_distance_m` ahead. A timeout is kept only as a
    safety fallback in case odometry never arrives.
    """

    def __init__(self, name='NavigateIntoOctagon', advance_distance_m=3.0,
                 surge=0.3, timeout=40.0):
        super().__init__(name)
        self.advance_distance_m = advance_distance_m
        self.surge = surge
        self.timeout = timeout
        self.start_time = None
        self.start_pos = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.CURRENT_POSE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        import time
        self.start_time = time.time()
        self.start_pos = None
        try:
            pose = self.blackboard.get(bb.CURRENT_POSE)
            if pose is not None:
                self.start_pos = (pose.pose.pose.position.x, pose.pose.pose.position.y)
        except KeyError:
            pass

    def _distance_traveled(self):
        """Straight-line distance from where this behavior started, via ZED odom."""
        if self.start_pos is None:
            return None
        try:
            pose = self.blackboard.get(bb.CURRENT_POSE)
        except KeyError:
            return None
        if pose is None:
            return None
        pos = pose.pose.pose.position
        return math.hypot(pos.x - self.start_pos[0], pos.y - self.start_pos[1])

    def update(self):
        import time
        from hightide_interfaces.msg import ThrusterCommand
        node = self.blackboard.get(bb.ROS_NODE)

        if (time.time() - self.start_time) > self.timeout:
            node.get_logger().warn('Octagon approach timed out (no odom / never advanced)')
            node.cmd_pub.publish(ThrusterCommand())
            return py_trees.common.Status.SUCCESS

        try:
            detections = self.blackboard.get(bb.DETECTIONS)
        except KeyError:
            detections = None

        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.surge = self.surge

        # The ffc model has no 'octagon' box — the octagon's buoy is the visual
        # cue. If we see the 'buoy', center on it and treat "we're inside" as its
        # box filling most of the frame. Takes priority over the odometry advance.
        if detections:
            for det in detections.detections:
                if det.class_name == 'buoy' and det.confidence > 0.4:
                    img_w = detections.image_width or 1280
                    cmd.sway = max(-0.3, min(0.3,
                                   ((det.center_x / img_w) - 0.5) * 1.5))
                    if det.width > img_w * 0.7:
                        node.cmd_pub.publish(ThrusterCommand())
                        node.get_logger().info('Reached octagon interior (buoy visual)')
                        return py_trees.common.Status.SUCCESS
                    node.cmd_pub.publish(cmd)
                    return py_trees.common.Status.RUNNING

        # Octagon not (yet) visible — advance a fixed distance via ZED odometry.
        traveled = self._distance_traveled()
        if traveled is not None and traveled >= self.advance_distance_m:
            node.cmd_pub.publish(ThrusterCommand())
            node.get_logger().info(
                f'Advanced {traveled:.2f}m toward octagon (odometry) — stopping')
            return py_trees.common.Status.SUCCESS

        node.cmd_pub.publish(cmd)
        return py_trees.common.Status.RUNNING


class SurfaceInOctagon(py_trees.behaviour.Behaviour):
    """Surface inside the octagon by commanding depth setpoint to 0."""

    def __init__(self, name='SurfaceInOctagon', timeout=30.0):
        super().__init__(name)
        self.timeout = timeout
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
        node.get_logger().info('Surfacing in octagon')

    def update(self):
        import time
        from std_msgs.msg import Float64
        node = self.blackboard.get(bb.ROS_NODE)

        if (time.time() - self.start_time) > self.timeout:
            return py_trees.common.Status.SUCCESS

        # Keep commanding surface in case the setpoint is missed.
        msg = Float64()
        msg.data = 0.0
        node.depth_pub.publish(msg)

        try:
            depth = self.blackboard.get(bb.CURRENT_DEPTH)
            if depth is not None and depth < 0.3:  # Close to surface
                return py_trees.common.Status.SUCCESS
        except KeyError:
            pass

        return py_trees.common.Status.RUNNING


def create_octagon_subtree() -> py_trees.behaviour.Behaviour:
    """Build the Task 5 (Octagon) behavior subtree — surface only (no claw)."""
    return py_trees.composites.Sequence(
        name='Task5_Octagon',
        memory=True,
        children=[
            LogBehavior('Oct_Start', 'Starting Task 5: Octagon (surface only)'),
            NavigateIntoOctagon('EnterOctagon'),
            StopMotion('StopInOctagon'),
            WaitForDuration('SettleInOctagon', duration_sec=2.0),
            SurfaceInOctagon('Surface'),
            LogBehavior('Oct_Done', 'Task 5 Octagon COMPLETE'),
        ],
    )
