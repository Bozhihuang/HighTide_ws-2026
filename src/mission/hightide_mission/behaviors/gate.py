"""
Task 1: Begin Assessment (Gate) — pass through the half matching our role.

The role (Survey & Repair vs Search & Rescue) is a PRE-RACE decision set via
the `chosen_role` parameter on the mission node — it is NOT read off the gate
(both symbols are always present, so detecting one tells us nothing about our
choice). Strategy: find the gate, strafe (crab walk) to center our role symbol,
record which half it was on so the slalom can stay on the matching side of the
red divider, surge through, then a heading-safe 360° yaw spin for style points.
"""

import py_trees
from .common import (CallTriggerService, LogBehavior, StopMotion, RecordPose,
                     WaitForStableDetection, YawSweepSearch, lock_heading, yaw_hold)
from . import blackboard_keys as bb


# Role -> the gate symbols that mark that role's half of the gate (ffc model
# class names). The Search & Rescue half only exposes an SOS symbol to the
# model (the life-ring is not a trained class), which is enough to align on.
ROLE_SYMBOLS = {
    'survey_repair': {'compass', 'hammer_and_wrench'},
    'search_rescue': {'sos'},
}

# The ffc model has no dedicated 'gate' box — we detect the gate's presence by
# seeing ANY of the role symbols mounted on it (both halves are always present).
GATE_SYMBOLS = {'compass', 'hammer_and_wrench', 'sos'}


class ConfirmGateRole(py_trees.behaviour.Behaviour):
    """
    Log/confirm the pre-chosen role by checking its symbol is visible on the
    gate. This never CHANGES the role (that is our own coin-flip decision) — it
    just waits briefly for the expected symbol so downstream alignment has a
    target, and succeeds best-effort on timeout.
    """

    def __init__(self, name='ConfirmGateRole', timeout=15.0):
        super().__init__(name)
        self.timeout = timeout
        self.start_time = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.CHOSEN_ROLE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        import time
        self.start_time = time.time()

    def update(self):
        import time
        node = self.blackboard.get(bb.ROS_NODE)
        role = self.blackboard.get(bb.CHOSEN_ROLE)
        target_classes = ROLE_SYMBOLS.get(role, ROLE_SYMBOLS['survey_repair'])

        if (time.time() - self.start_time) > self.timeout:
            node.get_logger().warn(
                f'Role symbol for {role} not seen — proceeding anyway')
            return py_trees.common.Status.SUCCESS

        try:
            detections = self.blackboard.get(bb.DETECTIONS)
        except KeyError:
            detections = None

        if detections:
            for det in detections.detections:
                if det.class_name in target_classes and det.confidence > 0.5:
                    node.get_logger().info(f'Confirmed role half visible: {role}')
                    return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class AlignWithGateHalf(py_trees.behaviour.Behaviour):
    """
    Crab-walk to center our role symbol in the frame, keeping heading locked
    (FOG holds yaw). Records which half of the gate the role symbol sits on
    into GATE_DIVIDER_SIDE so the slalom can keep the red divider on the same
    side. Succeeds once centered, or best-effort on timeout.
    """

    def __init__(self, name='AlignWithGateHalf', timeout=20.0, center_tol=0.12):
        super().__init__(name)
        self.timeout = timeout
        self.center_tol = center_tol
        self.start_time = None
        self.side_recorded = False
        self._locked_heading = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.CHOSEN_ROLE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.GATE_DIVIDER_SIDE, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        import time
        self.start_time = time.time()
        self.side_recorded = False
        role = self.blackboard.get(bb.CHOSEN_ROLE)
        node = self.blackboard.get(bb.ROS_NODE)
        self._locked_heading = lock_heading(node)
        node.get_logger().info(f'Aligning with gate half for role: {role}')

    def update(self):
        import time
        from hightide_interfaces.msg import ThrusterCommand
        node = self.blackboard.get(bb.ROS_NODE)
        role = self.blackboard.get(bb.CHOSEN_ROLE)
        target_classes = ROLE_SYMBOLS.get(role, ROLE_SYMBOLS['survey_repair'])

        if (time.time() - self.start_time) > self.timeout:
            node.get_logger().warn('Gate align timed out — proceeding best-effort')
            node.cmd_pub.publish(ThrusterCommand())
            return py_trees.common.Status.SUCCESS

        try:
            detections = self.blackboard.get(bb.DETECTIONS)
        except KeyError:
            detections = None

        target = None
        if detections:
            for det in detections.detections:
                if det.class_name in target_classes and det.confidence > 0.4:
                    target = det
                    break

        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.yaw = yaw_hold(node, self._locked_heading)  # hold heading while aligning

        if target is None:
            # Symbol not visible yet — creep forward slowly to bring it into view.
            cmd.surge = 0.1
            node.cmd_pub.publish(cmd)
            return py_trees.common.Status.RUNNING

        img_w = detections.image_width or 1280
        normalized_x = target.center_x / img_w

        # Record which half the role symbol is on the first time we see it.
        # This is the side of the gate we pass through; the slalom keeps the
        # red divider/pipe on the matching side for the bonus.
        if not self.side_recorded:
            side = 'right' if normalized_x > 0.5 else 'left'
            self.blackboard.set(bb.GATE_DIVIDER_SIDE, side)
            self.side_recorded = True
            node.get_logger().info(f'Role symbol on {side} half of gate')

        lateral_error = normalized_x - 0.5
        cmd.sway = max(-0.4, min(0.4, lateral_error * 2.0))  # + = strafe right
        cmd.surge = 0.05  # gentle forward creep while centering
        node.cmd_pub.publish(cmd)

        if abs(lateral_error) < self.center_tol:
            node.get_logger().info('Centered on role half of gate')
            node.cmd_pub.publish(ThrusterCommand())
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class SurgeThrough(py_trees.behaviour.Behaviour):
    """Surge forward through the gate for a fixed duration."""

    def __init__(self, name='SurgeThrough', duration=5.0, speed=0.5):
        super().__init__(name)
        self.duration = duration
        self.speed = speed
        self.start_time = None
        self._locked_heading = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        import time
        self.start_time = time.time()
        self._locked_heading = lock_heading(self.blackboard.get(bb.ROS_NODE))

    def update(self):
        import time
        from hightide_interfaces.msg import ThrusterCommand
        if (time.time() - self.start_time) > self.duration:
            return py_trees.common.Status.SUCCESS

        node = self.blackboard.get(bb.ROS_NODE)
        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.surge = self.speed
        cmd.yaw = yaw_hold(node, self._locked_heading)  # drive straight, hold heading
        node.cmd_pub.publish(cmd)
        return py_trees.common.Status.RUNNING


class HeadingTurn(py_trees.behaviour.Behaviour):
    """Turns the AUV a specific number of degrees using closed-loop PID control."""

    def __init__(self, name='HeadingTurn', degrees=180.0, tolerance=2.0, timeout=10.0):
        super().__init__(name)
        self.target_degrees_offset = degrees
        self.tolerance_deg = tolerance
        self.timeout = timeout
        self.start_time = None
        self.target_heading = None
        
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.CURRENT_HEADING, access=py_trees.common.Access.READ)

    def initialise(self):
        import time
        import math
        from hightide_navigation import normalize_angle, PIDController
        self.start_time = time.time()
        
        # We need a local PID controller for this maneuver
        self.pid = PIDController(kp=1.5, ki=0.05, kd=0.1, output_max=0.6)
        
        current_heading = self.blackboard.get(bb.CURRENT_HEADING)
        if current_heading is None:
            current_heading = 0.0
        
        offset_rad = math.radians(self.target_degrees_offset)
        self.target_heading = normalize_angle(current_heading + offset_rad)
        
        node = self.blackboard.get(bb.ROS_NODE)
        node.get_logger().info(f'Turning by {self.target_degrees_offset}° to heading {math.degrees(self.target_heading):.1f}°')
        self.last_t = time.time()

    def update(self):
        import time
        import math
        from hightide_navigation import normalize_angle
        from hightide_interfaces.msg import ThrusterCommand
        
        now = time.time()
        if (now - self.start_time) > self.timeout:
            return py_trees.common.Status.SUCCESS
            
        current_heading = self.blackboard.get(bb.CURRENT_HEADING)
        if current_heading is None:
            return py_trees.common.Status.RUNNING

        error_rad = normalize_angle(self.target_heading - current_heading)
        
        if abs(math.degrees(error_rad)) <= self.tolerance_deg:
            # We have reached the heading
            return py_trees.common.Status.SUCCESS

        dt = now - self.last_t
        self.last_t = now
        
        yaw_cmd = self.pid.compute(error_rad, dt)

        node = self.blackboard.get(bb.ROS_NODE)
        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.yaw = yaw_cmd
        node.cmd_pub.publish(cmd)
        
        return py_trees.common.Status.RUNNING


def create_gate_subtree() -> py_trees.behaviour.Behaviour:
    """Build the Task 1 (Gate) behavior subtree."""

    # Find the gate WITHOUT assuming which way we're pointing. The old logic
    # only handled "gate dead ahead" or "gate exactly 180° behind"; if the sub
    # started 90° off (parallel to the wall) it could never find the gate and
    # just crept into it. Instead:
    #   1. If we're already facing the gate, confirm it robustly (4 of the last
    #      5 frames) so a single false positive can't commit us forward.
    #   2. Otherwise rotate in place and sweep for the gate at ANY bearing,
    #      confirming with the same M-of-N filter. Yaw-only — never surge — so a
    #      wrong orientation can't drive us into a wall while searching.
    # memory=True: once QuickFindGate fails, stay on the sweep instead of
    # restarting the quick check every tick.
    find_gate_logic = py_trees.composites.Selector(
        name='FindGate',
        memory=True,
        children=[
            WaitForStableDetection('QuickFindGate', GATE_SYMBOLS,
                                   window=5, min_hits=4, timeout=5.0),
            YawSweepSearch('SweepForGate', GATE_SYMBOLS,
                           window=5, min_hits=4, timeout=45.0),
        ]
    )

    return py_trees.composites.Sequence(
        name='Task1_Gate',
        memory=True,
        children=[
            LogBehavior('Gate_Start', 'Starting Task 1: Gate'),
            find_gate_logic,
            SurgeThrough('ApproachGate', duration=3.0, speed=0.3),
            ConfirmGateRole('ConfirmRole'),
            AlignWithGateHalf('AlignGate'),
            SurgeThrough('PassThrough', duration=5.0, speed=0.5),
            StopMotion('StopAfterGate'),
            # Remember the pose just PAST the gate (odometry) so Return Home
            # can dead-reckon back to the far side of the gate and cross it
            # once, camera-first — we have no pinger to home on. Recording
            # before the gate (the old placement) made return-home target the
            # start box, forcing a blind reverse crossing.
            RecordPose('RecordGatePose', bb.GATE_POSITION),
            # Real 360° yaw spin for style — heading-safe (FOG returns to start).
            LogBehavior('Gate_StyleSpin', 'Executing style yaw spin'),
            CallTriggerService('YawSpinStyle', '/hightide/yaw_spin'),
            StopMotion('StopAfterSpin'),
            LogBehavior('Gate_Done', 'Task 1 Gate COMPLETE'),
        ],
    )