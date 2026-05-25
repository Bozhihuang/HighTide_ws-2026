"""
Task 4: Deploy (Torpedoes) — Fire torpedoes through target board openings.

Strategy: Find torpedo board via expanding square search. Approach to ~1m.
Use Alt Hold to dial in Z-axis depth matching torpedo launcher to hole.
Vision servo to center hole in frame. Fire.
"""

import py_trees
from .common import (WaitForDetection, WaitForDuration,
                     LogBehavior, StopMotion)
from . import blackboard_keys as bb


class AlignTorpedo(py_trees.behaviour.Behaviour):
    """
    Align torpedo launcher with target hole using vision servo.
    Adjusts depth to match launcher Z with hole Z, strafes to center.
    """

    def __init__(self, name='AlignTorpedo', target_class='torpedo_hole_large',
                 timeout=30.0):
        super().__init__(name)
        self.target_class = target_class
        self.timeout = timeout
        self.start_time = None
        self.aligned_since = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        import time
        self.start_time = time.time()
        self.aligned_since = None

    def update(self):
        import time
        from hightide_interfaces.msg import ThrusterCommand
        from std_msgs.msg import Float64
        node = self.blackboard.get(bb.ROS_NODE)

        if (time.time() - self.start_time) > self.timeout:
            return py_trees.common.Status.SUCCESS  # Best effort

        try:
            detections = self.blackboard.get(bb.DETECTIONS)
        except KeyError:
            return py_trees.common.Status.RUNNING

        target = None
        if detections:
            for det in detections.detections:
                if det.class_name == self.target_class and det.confidence > 0.4:
                    target = det
                    break

        if target is None:
            return py_trees.common.Status.RUNNING

        img_w = detections.image_width or 1280
        img_h = detections.image_height or 720

        # Lateral alignment (strafe to center target horizontally)
        lateral_error = (target.center_x / img_w) - 0.5
        # Vertical alignment (adjust depth to center target vertically)
        vertical_error = (target.center_y / img_h) - 0.5

        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.sway = max(-0.4, min(0.4, lateral_error * 2.0))
        cmd.yaw = 0.0  # FOG locked

        # Depth adjustment via depth controller
        depth_adj = Float64()
        depth_adj.data = vertical_error * 0.05  # Small increments
        node.depth_pub.publish(depth_adj)

        node.cmd_pub.publish(cmd)

        # Check alignment
        if abs(lateral_error) < 0.08 and abs(vertical_error) < 0.08:
            if self.aligned_since is None:
                self.aligned_since = time.time()
            elif (time.time() - self.aligned_since) > 1.0:
                node.get_logger().info(
                    f'Aligned with {self.target_class}!')
                return py_trees.common.Status.SUCCESS
        else:
            self.aligned_since = None

        return py_trees.common.Status.RUNNING


class FireTorpedo(py_trees.behaviour.Behaviour):
    """Fire a torpedo via the actuator service."""

    def __init__(self, name='FireTorpedo', tube_id=1):
        super().__init__(name)
        self.tube_id = tube_id
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.TORPEDOES_FIRED, access=py_trees.common.Access.WRITE)
        self.client = None
        self.future = None

    def initialise(self):
        from hightide_interfaces.srv import FireTorpedo as FireTorpedoSrv
        node = self.blackboard.get(bb.ROS_NODE)
        self.client = node.create_client(FireTorpedoSrv, '/hightide/fire_torpedo')
        self.future = None

    def update(self):
        from hightide_interfaces.srv import FireTorpedo as FireTorpedoSrv
        node = self.blackboard.get(bb.ROS_NODE)

        if not self.client.wait_for_service(timeout_sec=1.0):
            return py_trees.common.Status.RUNNING

        if self.future is None:
            req = FireTorpedoSrv.Request()
            req.tube_id = self.tube_id
            self.future = self.client.call_async(req)
            return py_trees.common.Status.RUNNING

        if not self.future.done():
            return py_trees.common.Status.RUNNING

        result = self.future.result()
        if result and result.success:
            try:
                fired = self.blackboard.get(bb.TORPEDOES_FIRED)
            except KeyError:
                fired = 0
            self.blackboard.set(bb.TORPEDOES_FIRED, fired + 1)
            node.get_logger().info(f'Torpedo {self.tube_id} FIRED!')
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class ApproachBoard(py_trees.behaviour.Behaviour):
    """Surge toward torpedo board until at desired distance."""

    def __init__(self, name='ApproachBoard', target_distance_m=1.0, timeout=30.0):
        super().__init__(name)
        self.target_dist = target_distance_m
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
        from hightide_interfaces.msg import ThrusterCommand
        node = self.blackboard.get(bb.ROS_NODE)

        if (time.time() - self.start_time) > self.timeout:
            return py_trees.common.Status.SUCCESS

        try:
            detections = self.blackboard.get(bb.DETECTIONS)
        except KeyError:
            return py_trees.common.Status.RUNNING

        board = None
        if detections:
            for det in detections.detections:
                if det.class_name == 'torpedo_board':
                    board = det
                    break

        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()

        if board and board.depth_m > 0:
            if board.depth_m <= self.target_dist:
                return py_trees.common.Status.SUCCESS
            cmd.surge = min(0.4, (board.depth_m - self.target_dist) * 0.5)
            # Strafe to keep centered
            img_w = detections.image_width or 1280
            lateral_err = (board.center_x / img_w) - 0.5
            cmd.sway = lateral_err * 1.5
        else:
            cmd.surge = 0.2  # Slow approach

        node.cmd_pub.publish(cmd)
        return py_trees.common.Status.RUNNING


def create_torpedoes_subtree() -> py_trees.behaviour.Behaviour:
    """Build the Task 4 (Torpedoes) behavior subtree."""
    return py_trees.composites.Sequence(
        name='Task4_Torpedoes',
        memory=True,
        children=[
            LogBehavior('Torp_Start', 'Starting Task 4: Torpedoes'),
            WaitForDetection('FindTorpedoBoard', 'torpedo_board', timeout=90.0),
            ApproachBoard('ApproachBoard', target_distance_m=1.0),
            StopMotion('StopBeforeAlign1'),
            AlignTorpedo('AlignLargeHole', 'torpedo_hole_large'),
            WaitForDuration('SteadyAim1', duration_sec=2.0),
            FireTorpedo('FireTorpedo1', tube_id=1),
            WaitForDuration('WaitAfterFire1', duration_sec=2.0),
            AlignTorpedo('AlignSmallHole', 'torpedo_hole_small'),
            WaitForDuration('SteadyAim2', duration_sec=2.0),
            FireTorpedo('FireTorpedo2', tube_id=2),
            StopMotion('StopAfterTorpedoes'),
            LogBehavior('Torp_Done', 'Task 4 Torpedoes COMPLETE'),
        ],
    )
