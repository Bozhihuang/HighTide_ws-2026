"""
Task 5: Restore (Octagon) — surface inside the octagon, then face an image.

Hardware reality for hightide: we have NO manipulator/claw, so the resupply
portion (collecting nut/bolt/plug/pill/bandage and placing them in baskets) is
out of scope. We also have NO hydrophone, so we cannot home on the acoustic
pinger. This task does the achievable core: get to the octagon, surface, then
slowly yaw to face one of the images hanging around it for the facing bonus.

No vision for getting there: the Transit_Octagon leg (mission_node's
octagon_{course}_leg_*) already deadreckons the sub to the octagon's vicinity,
same as every other inter-task leg this mission. EnterOctagon is one more
closed-loop ZED/PID forward advance (DeadReckonTransit, advance_distance_m)
into the interior, then settle and surface.

Facing bonus: the ffc model has no dedicated "octagon image" class, so this
reuses the same role-icon classes already used at the gate/torpedo board —
compass/hammer_and_wrench/fire for survey_repair, sos/blood for search_rescue.
Facing ANY of them is worth something; facing the one matching our chosen role
is worth the max. FaceOctagonImage slowly yaws in place, tracking whichever
icon is currently the best candidate (role match preferred), and once one is
centered in frame it stops and holds facing it for hold_sec to bank the points.
"""

import py_trees
from .common import (WaitForDuration, LogBehavior, StopMotion,
                     DeadReckonTransit, distribute_timeout, detection_size)
from . import blackboard_keys as bb

# Any of these hanging images is worth SOME facing points.
ANY_ICON_CLASSES = {'compass', 'hammer_and_wrench', 'fire', 'sos', 'blood'}
# The subset matching our chosen role is worth the MAX facing points.
ROLE_ICON_CLASSES = {
    'survey_repair': {'compass', 'hammer_and_wrench', 'fire'},
    'search_rescue': {'sos', 'blood'},
}


class SurfaceInOctagon(py_trees.behaviour.Behaviour):
    """Surface inside the octagon by commanding depth setpoint to 0."""

    def __init__(self, name='SurfaceInOctagon', timeout=30.0, surface_depth_m=0.3):
        super().__init__(name)
        self.timeout = timeout
        self.surface_depth_m = surface_depth_m
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
            if depth is not None and depth < self.surface_depth_m:  # Close to surface
                return py_trees.common.Status.SUCCESS
        except KeyError:
            pass

        return py_trees.common.Status.RUNNING


class FaceOctagonImage(py_trees.behaviour.Behaviour):
    """
    After surfacing, slowly yaw in place scanning for the role-icon images
    hanging around the octagon. Tracks the BEST candidate each tick — a
    role-matching icon always beats a non-matching one — and once it's
    centered in frame, stops and holds facing it for hold_sec (3s) to bank
    the points, fine-tuning with small yaw corrections to stay centered.

    Best-effort: if nothing is ever centered within timeout, gives up and
    proceeds anyway — a missed facing bonus never stalls the mission.
    """

    def __init__(self, name='FaceOctagonImage', yaw_rate=0.15, turn_gain=1.5,
                 turn_max=0.2, center_tol=0.1, confidence=0.4,
                 hold_sec=3.0, timeout=60.0):
        super().__init__(name)
        self.yaw_rate = yaw_rate
        self.turn_gain = turn_gain
        self.turn_max = turn_max
        self.center_tol = center_tol
        self.confidence = confidence
        self.hold_sec = hold_sec
        self.timeout = timeout
        self.start_time = None
        self.held_since = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.CHOSEN_ROLE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        import time
        self.start_time = time.time()
        self.held_since = None

    def _best_candidate(self, detections, correct_classes):
        """(det, is_correct) for the best icon in view, or (None, False).
        A role-matching icon always outranks a non-matching one; ties within
        a rank go to the larger (closer/more centered) detection."""
        best, best_correct = None, False
        if not detections:
            return None, False
        for det in detections.detections:
            if det.confidence <= self.confidence:
                continue
            is_any = det.class_name in ANY_ICON_CLASSES
            is_correct = det.class_name in correct_classes
            if not is_any:
                continue
            if best is None or (is_correct and not best_correct) or (
                    is_correct == best_correct and detection_size(det) > detection_size(best)):
                best, best_correct = det, is_correct
        return best, best_correct

    def update(self):
        import time
        from hightide_interfaces.msg import ThrusterCommand
        node = self.blackboard.get(bb.ROS_NODE)
        role = self.blackboard.get(bb.CHOSEN_ROLE)
        correct_classes = ROLE_ICON_CLASSES.get(role, set())

        if (time.time() - self.start_time) > self.timeout:
            node.cmd_pub.publish(ThrusterCommand())
            node.get_logger().warn('FaceOctagonImage timed out — proceeding without facing bonus')
            return py_trees.common.Status.SUCCESS

        try:
            detections = self.blackboard.get(bb.DETECTIONS)
        except KeyError:
            detections = None

        target, is_correct = self._best_candidate(detections, correct_classes)

        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()

        if target is None:
            # Nothing in view — keep slowly rotating to scan for one.
            self.held_since = None
            cmd.yaw = self.yaw_rate
            node.cmd_pub.publish(cmd)
            return py_trees.common.Status.RUNNING

        img_w = detections.image_width or 1280
        lateral_error = (target.center_x / img_w) - 0.5

        if abs(lateral_error) < self.center_tol:
            # Centered — hold facing it.
            cmd.yaw = 0.0
            node.cmd_pub.publish(cmd)
            if self.held_since is None:
                self.held_since = time.time()
                node.get_logger().info(
                    f'Facing {"CORRECT" if is_correct else "an"} icon '
                    f'({target.class_name}) — holding {self.hold_sec:.0f}s')
            elif (time.time() - self.held_since) >= self.hold_sec:
                node.cmd_pub.publish(ThrusterCommand())
                node.get_logger().info(
                    f'Held facing {target.class_name} for {self.hold_sec:.0f}s — scored')
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.RUNNING

        # Not yet centered — fine-tune yaw toward it (small correction, not
        # the slow scan rate). Losing/regaining center resets the hold timer.
        self.held_since = None
        cmd.yaw = max(-self.turn_max, min(self.turn_max, lateral_error * self.turn_gain))
        node.cmd_pub.publish(cmd)
        return py_trees.common.Status.RUNNING


def create_octagon_subtree(total_timeout=240.0, advance_distance_m=3.0,
                           surge=0.3, settle_sec=2.0, surface_depth_m=0.3,
                           transit_kp=0.2, transit_ki=0.0, transit_kd=0.05,
                           face_yaw_rate=0.15, face_turn_gain=1.5,
                           face_turn_max=0.2, face_center_tol=0.1,
                           face_hold_sec=3.0
                           ) -> py_trees.behaviour.Behaviour:
    """Build the Task 5 (Octagon) behavior subtree — surface, then face an image.

    No vision for getting there: EnterOctagon is a closed-loop ZED/PID forward
    advance (same DeadReckonTransit primitive as every other leg). After
    surfacing, FaceOctagonImage slowly yaws to find and hold facing a role
    icon for the facing bonus (best-effort — see its own docstring).
    total_timeout is split across the Surface and FaceImage deadlines.
    """
    t = distribute_timeout(total_timeout, {'surface': 30.0, 'face': 60.0})

    return py_trees.composites.Sequence(
        name='Task5_Octagon',
        memory=True,
        children=[
            LogBehavior('Oct_Start', 'Starting Task 5: Octagon'),
            DeadReckonTransit('EnterOctagon', forward_m=advance_distance_m,
                              speed=surge, kp=transit_kp, ki=transit_ki,
                              kd=transit_kd),
            StopMotion('StopInOctagon'),
            WaitForDuration('SettleInOctagon', duration_sec=settle_sec),
            SurfaceInOctagon('Surface', timeout=t['surface'],
                             surface_depth_m=surface_depth_m),
            FaceOctagonImage('FaceImage', yaw_rate=face_yaw_rate,
                             turn_gain=face_turn_gain, turn_max=face_turn_max,
                             center_tol=face_center_tol, hold_sec=face_hold_sec,
                             timeout=t['face']),
            LogBehavior('Oct_Done', 'Task 5 Octagon COMPLETE'),
        ],
    )
