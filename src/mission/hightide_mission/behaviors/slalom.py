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
        self.blackboard.register_key(key=bb.GATE_DIVIDER_SIDE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def _divider_side(self):
        """Which gate half we passed through ('right'/'left'), default 'right'."""
        try:
            side = self.blackboard.get(bb.GATE_DIVIDER_SIDE)
        except KeyError:
            side = None
        return side if side in ('right', 'left') else 'right'

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
            # Surge forward until we see a red pole close enough. The ffc model's
            # 'slalom' class IS the red poles (white pipes aren't a trained
            # class), which is exactly the divider we align relative to.
            cmd.surge = 0.3
            pipe_det = None
            if detections:
                for det in detections.detections:
                    if det.class_name == 'slalom':
                        pipe_det = det
                        break

            if pipe_det and pipe_det.depth_m > 0 and pipe_det.depth_m < 1.5:
                self.phase = 'strafe'
                node.get_logger().info(
                    f'Pipe {self.pipe_number} at {pipe_det.depth_m:.1f}m — strafing')
            node.cmd_pub.publish(cmd)
            return py_trees.common.Status.RUNNING

        elif self.phase == 'strafe':
            # Keep the red divider on the SAME side we passed at the gate.
            # If we passed on the RIGHT half of the gate, the red divider was
            # on our left, so here we stay right of the red pipe (strafe right,
            # driving the red pipe toward the left of frame). Mirror for left.
            # NOTE: sign convention assumes +sway = strafe right; flip if the
            # slalom bonus side comes out wrong in pool testing.
            side = self._divider_side()
            if side == 'right':
                sway_dir = 0.4          # strafe right, red pipe -> left of frame
                red_passed = lambda nx: nx < 0.4
            else:
                sway_dir = -0.4         # strafe left, red pipe -> right of frame
                red_passed = lambda nx: nx > 0.6

            cmd.sway = sway_dir
            cmd.surge = 0.1  # Slight forward motion
            node.cmd_pub.publish(cmd)

            # Check if the red pole ('slalom') has moved to the correct side of frame
            if detections:
                for det in detections.detections:
                    if det.class_name == 'slalom':
                        img_w = detections.image_width or 1280
                        normalized_x = det.center_x / img_w
                        if red_passed(normalized_x):
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
            # No path_marker class in the ffc model — head straight for the first
            # red pole instead of following a path lead-in.
            WaitForDetection('FindSlalom', 'slalom', timeout=30.0),
            WaitForDuration('ApproachSettle', duration_sec=2.0),
            SlalomPipe('SlalomPipe1', pipe_number=1),
            SlalomPipe('SlalomPipe2', pipe_number=2),
            SlalomPipe('SlalomPipe3', pipe_number=3),
            StopMotion('StopAfterSlalom'),
            LogBehavior('Slalom_Done', 'Task 2 Slalom COMPLETE'),
        ],
    )
