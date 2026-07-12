"""
Task 2: Avoid Debris (Slalom) — Navigate around RED and WHITE pipes.

Strategy: Lock heading via FOG. Do NOT snake. Strafe laterally past each
pipe set while maintaining forward heading, then strafe back to re-center
on the course line so the next set is found straight ahead. Stay on the
correct side based on which side the red divider was at the gate.
"""

import py_trees
from .common import (WaitForDetection, WaitForDuration,
                     LogBehavior, StopMotion, SearchForDetection)
from . import blackboard_keys as bb


class SlalomPipe(py_trees.behaviour.Behaviour):
    """
    Navigate around a single pipe set with per-phase timing:
      approach  — surge until the red pole is close (< 1.5 m)
      strafe    — move laterally until the red pole clears the correct side
      pass      — surge past the pipe set for a fixed duration
      recenter  — strafe back for as long as we strafed out, re-centering on
                  the course line so the next set appears ahead
    Each phase has its own timer; previously every deadline was measured from
    behavior start, so the pass duration depended on how long strafing took
    (up to ~19 s of open-loop surge — enough to overshoot the next set).
    """

    APPROACH_TIMEOUT = 15.0
    STRAFE_TIMEOUT = 6.0
    PASS_DURATION = 5.0

    def __init__(self, name='SlalomPipe', pipe_number=1):
        super().__init__(name)
        self.pipe_number = pipe_number
        self.phase = 'approach'
        self.phase_start = None
        self.strafe_duration = 0.0
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
        self.phase = 'approach'
        self.phase_start = time.time()
        self.strafe_duration = 0.0

    def _enter_phase(self, phase):
        import time
        self.phase = phase
        self.phase_start = time.time()

    def update(self):
        import time
        from hightide_interfaces.msg import ThrusterCommand
        node = self.blackboard.get(bb.ROS_NODE)
        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()
        phase_elapsed = time.time() - self.phase_start

        try:
            detections = self.blackboard.get(bb.DETECTIONS)
        except KeyError:
            detections = None

        red_pole = None
        if detections:
            for det in detections.detections:
                if det.class_name == 'slalom':
                    red_pole = det
                    break

        if self.phase == 'approach':
            # Surge forward until we see a red pole close enough. The ffc model's
            # 'slalom' class IS the red poles (white pipes aren't a trained
            # class), which is exactly the divider we align relative to.
            cmd.surge = 0.3
            if red_pole and red_pole.depth_m > 0 and red_pole.depth_m < 1.5:
                node.get_logger().info(
                    f'Pipe {self.pipe_number} at {red_pole.depth_m:.1f}m — strafing')
                self._enter_phase('strafe')
            elif phase_elapsed > self.APPROACH_TIMEOUT:
                # Never got close to a red pole — just drive through the set.
                node.get_logger().warn(
                    f'Pipe {self.pipe_number} not acquired — passing straight')
                self._enter_phase('pass')
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

            done = phase_elapsed > self.STRAFE_TIMEOUT
            if red_pole and detections:
                img_w = detections.image_width or 1280
                if red_passed(red_pole.center_x / img_w):
                    done = True

            if done:
                self.strafe_duration = phase_elapsed
                self._enter_phase('pass')
            return py_trees.common.Status.RUNNING

        elif self.phase == 'pass':
            # Surge past the pipe for a fixed time.
            cmd.surge = 0.4
            node.cmd_pub.publish(cmd)
            if phase_elapsed > self.PASS_DURATION:
                self._enter_phase('recenter')
            return py_trees.common.Status.RUNNING

        elif self.phase == 'recenter':
            # Strafe back toward the course line for as long as we strafed
            # out (slightly less, to avoid crossing to the wrong side), so the
            # next pipe set is found roughly straight ahead.
            recenter_time = 0.8 * self.strafe_duration
            if phase_elapsed >= recenter_time:
                node.cmd_pub.publish(ThrusterCommand())
                return py_trees.common.Status.SUCCESS
            side = self._divider_side()
            cmd.sway = -0.4 if side == 'right' else 0.4
            cmd.surge = 0.2
            node.cmd_pub.publish(cmd)
            return py_trees.common.Status.RUNNING

        return py_trees.common.Status.RUNNING


def create_slalom_subtree() -> py_trees.behaviour.Behaviour:
    """Build the Task 2 (Slalom) behavior subtree."""
    return py_trees.composites.Sequence(
        name='Task2_Slalom',
        memory=True,
        children=[
            LogBehavior('Slalom_Start', 'Starting Task 2: Slalom'),
            # No path_marker class in the ffc model — creep forward with a
            # lateral sweep until the first red pole is seen (the slalom is
            # NOT on a straight line from the gate, per the course rules).
            SearchForDetection('FindSlalom', {'slalom'}, timeout=45.0,
                               surge=0.2, sway_amplitude=0.2),
            WaitForDuration('ApproachSettle', duration_sec=2.0),
            SlalomPipe('SlalomPipe1', pipe_number=1),
            SlalomPipe('SlalomPipe2', pipe_number=2),
            SlalomPipe('SlalomPipe3', pipe_number=3),
            StopMotion('StopAfterSlalom'),
            LogBehavior('Slalom_Done', 'Task 2 Slalom COMPLETE'),
        ],
    )
