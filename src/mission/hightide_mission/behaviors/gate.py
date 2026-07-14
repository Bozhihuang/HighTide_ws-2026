"""
Task 1: Begin Assessment (Gate) — pass through the half matching our role.

The role (Survey & Repair vs Search & Rescue) is a PRE-RACE decision set via
the `chosen_role` parameter on the mission node — it is NOT read off the gate
(both symbols are always present, so detecting one tells us nothing about our
choice). Strategy: drive forward blind (no "is the gate in frame" check first),
confirm the role symbol, strafe (crab walk) to center it, record which half it
was on, surge through, then a heading-safe 2x 360° yaw spin for style points.
"""

import py_trees
from .common import (CallTriggerService, LogBehavior, StopMotion, RecordPose,
                     DeadReckonTransit, lock_heading, yaw_hold)
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
    gate. This never CHANGES the role (that is our own coin-flip decision) —
    it just needs the expected symbol in frame so the strafe-align has a
    target, and succeeds best-effort on timeout.

    No motion: FindGate already yawed to bring the gate into view (both halves
    are one structure, so the role symbol is in/near frame), and the follow-up
    AlignWithGateHalf does the searching/centering by STRAFING with heading
    locked. Keeping this behavior stationary avoids adding any yaw drift right
    before that strafe-align.
    """

    def __init__(self, name='ConfirmGateRole', timeout=10.0):
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
                f'Role symbol for {role} not seen — proceeding to strafe-align anyway')
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
    STRAFE (no forward motion) until our role symbol is centered in the frame,
    keeping heading locked (FOG holds yaw). This is the "strafe until the
    symbol centers, THEN go forward" step: the forward approach happens before
    this behavior and the pass-through happens after — here we only translate
    laterally so the sub crosses the gate through the correct half.

    If the symbol isn't visible yet it strafes in ONE fixed direction
    (search_side param — 'left' or 'right', whichever way the symbol is
    expected to be) instead of creeping forward, until either the symbol
    comes into frame (then it centers normally) or the timeout hits.
    Records which half the role symbol sits on into GATE_DIVIDER_SIDE (kept
    for downstream logic). Succeeds once centered; on timeout it FAILS —
    Task1_Gate is a Sequence, so a failed align abandons the rest of the gate
    task (no PassThrough, no style spin) and the mission's outer
    FailureIsSuccess wrapper moves straight on to the next task instead.
    """

    def __init__(self, name='AlignWithGateHalf', timeout=20.0, center_tol=0.12,
                 strafe_gain=2.0, strafe_max=0.15,
                 search_side='right', search_speed=0.15):
        super().__init__(name)
        self.timeout = timeout
        self.center_tol = center_tol
        self.strafe_gain = strafe_gain
        # Same slow cap for both phases — strafe slowly the whole way in,
        # search or center, no fast snap once the symbol comes into frame.
        self.strafe_max = strafe_max
        # + = strafe right, - = strafe left (same sign convention as the
        # centering command below).
        self.search_sway = search_speed if search_side == 'right' else -search_speed
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
        node.get_logger().info(f'Strafe-aligning with gate half for role: {role}')

    def update(self):
        import time
        from hightide_interfaces.msg import ThrusterCommand
        node = self.blackboard.get(bb.ROS_NODE)
        role = self.blackboard.get(bb.CHOSEN_ROLE)
        target_classes = ROLE_SYMBOLS.get(role, ROLE_SYMBOLS['survey_repair'])
        elapsed = time.time() - self.start_time

        if elapsed > self.timeout:
            node.get_logger().warn(
                'Gate align timed out — abandoning gate task, moving on')
            node.cmd_pub.publish(ThrusterCommand())
            return py_trees.common.Status.FAILURE

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
        cmd.yaw = yaw_hold(node, self._locked_heading)  # hold heading throughout

        if target is None:
            # Symbol not visible — strafe in one fixed direction (heading
            # locked, NO forward motion) toward where it's expected.
            cmd.sway = self.search_sway
            node.cmd_pub.publish(cmd)
            return py_trees.common.Status.RUNNING

        img_w = detections.image_width or 1280
        normalized_x = target.center_x / img_w

        # Record which half the role symbol is on the first time we see it.
        if not self.side_recorded:
            side = 'right' if normalized_x > 0.5 else 'left'
            self.blackboard.set(bb.GATE_DIVIDER_SIDE, side)
            self.side_recorded = True
            node.get_logger().info(f'Role symbol on {side} half of gate')

        lateral_error = normalized_x - 0.5
        cmd.sway = max(-self.strafe_max,
                       min(self.strafe_max, lateral_error * self.strafe_gain))  # + = strafe right
        node.cmd_pub.publish(cmd)  # pure strafe — no surge

        if abs(lateral_error) < self.center_tol:
            node.get_logger().info('Centered on role half of gate — will pass through')
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
        node = self.blackboard.get(bb.ROS_NODE)
        if (time.time() - self.start_time) > self.duration:
            node.cmd_pub.publish(ThrusterCommand())   # stop-on-exit: don't coast
            return py_trees.common.Status.SUCCESS

        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.surge = self.speed
        cmd.yaw = yaw_hold(node, self._locked_heading)  # drive straight, hold heading
        node.cmd_pub.publish(cmd)
        return py_trees.common.Status.RUNNING


class HeadingTurn(py_trees.behaviour.Behaviour):
    """Turns the AUV a specific number of degrees using closed-loop PID control.

    Gains default to the same values as pre_dive.YawToRecordedHeading and
    yaw_controller_node's rotate_to_heading() — all three do the identical
    physical action (PID-turn the same vehicle to a target heading), so they
    should be tuned together rather than drifting apart. mission_node passes
    its heading_turn_kp/ki/kd/output_limit params here so there's one source
    of truth instead of each hardcoding its own copy.
    """

    def __init__(self, name='HeadingTurn', degrees=180.0, tolerance=2.0, timeout=10.0,
                 kp=0.225, ki=0.0, kd=0.2, output_limit=0.6):
        super().__init__(name)
        self.target_degrees_offset = degrees
        self.tolerance_deg = tolerance
        self.timeout = timeout
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_limit = output_limit
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

        # Clamp BOTH sides — output_min defaults to -1.0, which made clockwise
        # turns (negative error, e.g. a 270° request that normalizes to -90°)
        # spin at full rate while counterclockwise ones were capped.
        self.pid = PIDController(kp=self.kp, ki=self.ki, kd=self.kd,
                                 output_min=-self.output_limit,
                                 output_max=self.output_limit)
        
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
        
        node = self.blackboard.get(bb.ROS_NODE)
        now = time.time()
        if (now - self.start_time) > self.timeout:
            # stop-on-exit: don't leave the last yaw command spinning the sub
            node.cmd_pub.publish(ThrusterCommand())
            node.get_logger().warn(
                f'{self.name}: turn timed out — proceeding at current heading')
            return py_trees.common.Status.SUCCESS

        current_heading = self.blackboard.get(bb.CURRENT_HEADING)
        if current_heading is None:
            return py_trees.common.Status.RUNNING

        error_rad = normalize_angle(self.target_heading - current_heading)

        if abs(math.degrees(error_rad)) <= self.tolerance_deg:
            # We have reached the heading — stop-on-exit so we don't coast past
            node.cmd_pub.publish(ThrusterCommand())
            node.get_logger().info(
                f'{self.name}: reached target heading '
                f'({math.degrees(self.target_heading):.1f}°)')
            return py_trees.common.Status.SUCCESS

        dt = now - self.last_t
        self.last_t = now

        # Negate: IMU yaw (ENU) is CCW-positive but ArduSub's yaw channel is
        # CW-positive — see yaw_hold() in common.py for the same fix.
        yaw_cmd = -self.pid.compute(error_rad, dt)

        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.yaw = yaw_cmd
        node.cmd_pub.publish(cmd)
        
        return py_trees.common.Status.RUNNING


def create_gate_subtree(confirm_timeout=15.0, align_timeout=30.0,
                        approach_forward_m=1.5, passthrough_forward_m=2.5,
                        transit_speed=0.6, transit_kp=0.2, transit_ki=0.0,
                        transit_kd=0.05, align_search_side='right',
                        align_search_speed=0.15,
                        align_strafe_max=0.15) -> py_trees.behaviour.Behaviour:
    """Build the Task 1 (Gate) behavior subtree.

    Flow: drive forward `approach_forward_m` (closed-loop ZED/PID, no vision
    check first) → confirm the role symbol is visible → strafe until it's
    centered → drive forward `passthrough_forward_m` through the gate →
    record pose → 2x style yaw spin.

    confirm_timeout/align_timeout are fixed deadlines (NOT scaled against a
    total mission budget) — if AlignGate hasn't centered on the symbol within
    align_timeout (30s default), it gives up best-effort and proceeds straight
    to PassThrough anyway. The approach/pass-through forward distances are
    closed-loop ZED-odometry legs (same PID transit as the inter-task legs).
    """
    return py_trees.composites.Sequence(
        name='Task1_Gate',
        memory=True,
        children=[
            LogBehavior('Gate_Start', 'Starting Task 1: Gate'),
            # Drive forward blind (closed-loop ZED/PID, heading held) — no
            # "is the gate already in frame" check first.
            DeadReckonTransit('ApproachGate', forward_m=approach_forward_m,
                              speed=transit_speed, kp=transit_kp, ki=transit_ki,
                              kd=transit_kd),
            ConfirmGateRole('ConfirmRole', timeout=confirm_timeout),
            AlignWithGateHalf('AlignGate', timeout=align_timeout,
                              search_side=align_search_side,
                              search_speed=align_search_speed,
                              strafe_max=align_strafe_max),
            # Then drive forward THROUGH the gate (closed-loop ZED/PID).
            DeadReckonTransit('PassThrough', forward_m=passthrough_forward_m,
                              speed=transit_speed, kp=transit_kp, ki=transit_ki,
                              kd=transit_kd),
            StopMotion('StopAfterGate'),
            # Remember the pose just PAST the gate (odometry) so Return Home
            # can dead-reckon back to the far side of the gate and cross it
            # once, camera-first — we have no pinger to home on. Recording
            # before the gate (the old placement) made return-home target the
            # start box, forcing a blind reverse crossing.
            RecordPose('RecordGatePose', bb.GATE_POSITION),
            # Double 360° yaw spin for style — heading-safe (FOG returns to
            # start). Count is set by yaw_controller_node's spin_count param.
            LogBehavior('Gate_StyleSpin', 'Executing 2x style yaw spin'),
            CallTriggerService('YawSpinStyle', '/hightide/yaw_spin'),
            StopMotion('StopAfterSpin'),
            LogBehavior('Gate_Done', 'Task 1 Gate COMPLETE'),
        ],
    )