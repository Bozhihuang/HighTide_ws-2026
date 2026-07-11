"""
Task 6: Return Home — navigate back to the recorded gate pose and pass through.

We have no acoustic pinger, so homing is odometry-based: at the start of the run
the gate task recorded the vehicle's ZED-odometry pose into GATE_POSITION. Here
we dead-reckon back to that (x, y) using body-frame surge/sway (same crab-walk
decomposition as the waypoint navigator), then confirm the gate visually and
surge through. Odometry drift over a full run makes this approximate, so the
visual pass-through is what actually closes the task.
"""

import math
import py_trees
from .common import (WaitForDetection, WaitForAnyDetection, WaitForDuration,
                     LogBehavior, StopMotion, PublishDepthSetpoint)
from .gate import SurgeThrough, GATE_SYMBOLS
from . import blackboard_keys as bb


class NavigateToRecordedPose(py_trees.behaviour.Behaviour):
    """
    Drive back toward a previously recorded odometry pose (GATE_POSITION) using
    body-frame surge/sway with heading held by the FOG. Succeeds when within
    position tolerance or on timeout (best-effort so it can hand off to the
    visual search even if odometry has drifted).
    """

    def __init__(self, name='NavigateToRecordedPose', pos_tol=0.5,
                 max_speed=0.4, timeout=90.0):
        super().__init__(name)
        self.pos_tol = pos_tol
        self.max_speed = max_speed
        self.timeout = timeout
        self.start_time = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.GATE_POSITION, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.CURRENT_POSE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.CURRENT_HEADING, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        import time
        self.start_time = time.time()

    def update(self):
        import time
        from hightide_interfaces.msg import ThrusterCommand
        node = self.blackboard.get(bb.ROS_NODE)

        if (time.time() - self.start_time) > self.timeout:
            node.get_logger().warn('Return-home dead-reckon timed out')
            node.cmd_pub.publish(ThrusterCommand())
            return py_trees.common.Status.SUCCESS

        try:
            target = self.blackboard.get(bb.GATE_POSITION)
            current = self.blackboard.get(bb.CURRENT_POSE)
            yaw = self.blackboard.get(bb.CURRENT_HEADING)
        except KeyError:
            target = current = yaw = None

        # Without a recorded pose or live odometry we can't home — skip to the
        # visual search rather than driving blind.
        if target is None or current is None or yaw is None:
            node.get_logger().warn('No recorded pose / odometry — skipping dead-reckon')
            return py_trees.common.Status.SUCCESS

        gx = target.pose.pose.position.x
        gy = target.pose.pose.position.y
        pos = current.pose.pose.position

        dx = gx - pos.x
        dy = gy - pos.y
        dist = math.hypot(dx, dy)

        if dist < self.pos_tol:
            node.get_logger().info('Back at recorded gate position')
            node.cmd_pub.publish(ThrusterCommand())
            return py_trees.common.Status.SUCCESS

        # World-frame error -> body frame (right +sway), matching the waypoint
        # navigator's convention.
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        surge = dx * cos_yaw + dy * sin_yaw
        sway = -dx * sin_yaw + dy * cos_yaw

        # Proportional command, clamped to max_speed.
        norm = max(dist, 1e-3)
        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.surge = max(-self.max_speed, min(self.max_speed, self.max_speed * surge / norm))
        cmd.sway = max(-self.max_speed, min(self.max_speed, self.max_speed * sway / norm))
        node.cmd_pub.publish(cmd)
        return py_trees.common.Status.RUNNING


def create_return_home_subtree() -> py_trees.behaviour.Behaviour:
    """Build the Task 6 (Return Home) behavior subtree."""
    return py_trees.composites.Sequence(
        name='Task6_ReturnHome',
        memory=True,
        children=[
            LogBehavior('ReturnHome_Start', 'Starting Task 6: Return Home'),
            PublishDepthSetpoint('SubmergeForReturn', depth_m=1.0),
            WaitForDuration('WaitSubmerge', duration_sec=5.0),
            # Dead-reckon back toward where the gate was recorded at the start.
            NavigateToRecordedPose('DeadReckonToGate'),
            # Then confirm the gate visually (via its role symbols) and pass through.
            WaitForAnyDetection('FindGateReturn', GATE_SYMBOLS, timeout=60.0),
            SurgeThrough('ApproachGateReturn', duration=3.0, speed=0.3),
            SurgeThrough('PassThroughGateReturn', duration=5.0, speed=0.5),
            StopMotion('StopAfterReturn'),
            LogBehavior('ReturnHome_Done', 'Task 6 Return Home COMPLETE'),
        ],
    )
