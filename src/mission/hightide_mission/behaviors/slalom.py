"""
Task 2: Avoid Debris (Slalom) — Navigate around RED and WHITE pipes.

Strategy: Lock heading via FOG. Do NOT snake. Strafe laterally past each
pipe set while maintaining forward heading. Stay on correct side based
on which side the red divider was at the gate.
"""

import py_trees
from .common import (WaitForDetection, WaitForDuration,
                     LogBehavior, StopMotion)
from . import blackboard_keys as bb


class SlalomPipe(py_trees.behaviour.Behaviour):
    """
    Navigate around a single pipe set: surge until close,
    strafe to correct side, surge past.
    """

    def __init__(self, name='SlalomPipe', pipe_number=1):
        super().__init__(name)
        self.pipe_number = pipe_number
        self.phase = 'approach'  # approach → strafe → pass
        self.start_time = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.CHOSEN_ROLE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        import time
        self.start_time = time.time()
        self.phase = 'approach'

    def update(self):
        import time
        from hightide_interfaces.msg import ThrusterCommand
        node = self.blackboard.get(bb.ROS_NODE)
        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()

        # Timeout safety
        if (time.time() - self.start_time) > 30.0:
            return py_trees.common.Status.SUCCESS

        try:
            detections = self.blackboard.get(bb.DETECTIONS)
        except KeyError:
            detections = None

        if self.phase == 'approach':
            # Surge forward until we see a pipe close enough
            cmd.surge = 0.3
            pipe_det = None
            if detections:
                for det in detections.detections:
                    if det.class_name in ('pipe_red', 'pipe_white'):
                        pipe_det = det
                        break

            if pipe_det and pipe_det.depth_m > 0 and pipe_det.depth_m < 1.5:
                self.phase = 'strafe'
                node.get_logger().info(
                    f'Pipe {self.pipe_number} at {pipe_det.depth_m:.1f}m — strafing')
            node.cmd_pub.publish(cmd)
            return py_trees.common.Status.RUNNING

        elif self.phase == 'strafe':
            # Determine correct side based on role/gate choice
            # If survey_repair → red divider was on our chosen side
            role = self.blackboard.get(bb.CHOSEN_ROLE)
            # Convention: pass with red pipe on the same side as at gate
            # Simple approach: strafe right to put red pipe on right
            sway_dir = 0.4 if role == 'survey_repair' else -0.4

            cmd.sway = sway_dir
            cmd.surge = 0.1  # Slight forward motion
            node.cmd_pub.publish(cmd)

            # Check if pipe is now on the correct side of frame
            if detections:
                for det in detections.detections:
                    if det.class_name == 'pipe_red':
                        img_w = detections.image_width or 1280
                        normalized_x = det.center_x / img_w
                        # Red pipe should be on right (>0.6) for survey_repair
                        if role == 'survey_repair' and normalized_x > 0.65:
                            self.phase = 'pass'
                        elif role == 'search_rescue' and normalized_x < 0.35:
                            self.phase = 'pass'

            # Timeout strafe after 5 seconds
            if (time.time() - self.start_time) > 15.0:
                self.phase = 'pass'

            return py_trees.common.Status.RUNNING

        elif self.phase == 'pass':
            # Surge past the pipe
            cmd.surge = 0.4
            node.cmd_pub.publish(cmd)
            if (time.time() - self.start_time) > 25.0:
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.RUNNING

        return py_trees.common.Status.RUNNING


def create_slalom_subtree() -> py_trees.behaviour.Behaviour:
    """Build the Task 2 (Slalom) behavior subtree."""
    return py_trees.composites.Sequence(
        name='Task2_Slalom',
        memory=True,
        children=[
            LogBehavior('Slalom_Start', 'Starting Task 2: Slalom'),
            WaitForDetection('FindPathToSlalom', 'path_marker', timeout=30.0),
            WaitForDuration('FollowPath', duration_sec=3.0),
            SlalomPipe('SlalomPipe1', pipe_number=1),
            SlalomPipe('SlalomPipe2', pipe_number=2),
            SlalomPipe('SlalomPipe3', pipe_number=3),
            StopMotion('StopAfterSlalom'),
            LogBehavior('Slalom_Done', 'Task 2 Slalom COMPLETE'),
        ],
    )
