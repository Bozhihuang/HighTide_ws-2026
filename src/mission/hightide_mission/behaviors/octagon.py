"""
Task 5: Restore (Octagon) — surface inside the octagon.

Hardware reality for hightide: we have NO manipulator/claw, so the resupply
portion (collecting nut/bolt/plug/pill/bandage and placing them in baskets, and
the inventory-based facing/rotation bonuses) is out of scope. We also have NO
hydrophone, so we cannot home on the acoustic pinger. This task therefore does
the achievable core: get to the octagon (drive toward it visually if the camera
picks it up, otherwise advance a fixed distance using ZED odometry) and surface
inside it.

Visual localization has TWO cues, checked in priority order:
  1. The under-octagon "capability-matrix" table — a large patterned board on
     the floor inside the octagon. When the forward ZED picks it up we center on
     it and treat it filling the frame as "we're inside". This is the better
     cue because the table is physically inside the octagon (the buoy border can
     be seen from outside it). The table is NOT a trained ffc-YOLO class; it is
     a sentinel class (see hightide_perception SENTINEL_CLASS_NAMES) published by
     an auxiliary detector, so this branch simply stays dormant until such a
     detection appears in the tracked stream.
  2. The octagon buoy border ('buoy', a trained class) — the fallback visual cue.

If neither is visible we advance a fixed real-world distance on odometry.
"""

import math
import py_trees
from .common import (WaitForDetection, WaitForDuration,
                     LogBehavior, StopMotion, PublishDepthSetpoint,
                     lock_heading, yaw_hold, distribute_timeout, detection_size)
from . import blackboard_keys as bb


class NavigateIntoOctagon(py_trees.behaviour.Behaviour):
    """
    Drive forward into the octagon. If a visual cue (under-octagon table, else
    the octagon buoy) is detected we center on it and stop once it fills the
    frame (we're underneath/inside it). Otherwise we advance a fixed real-world
    distance using ZED odometry (CURRENT_POSE) as the measuring stick — not a
    timed guess — then stop, on the assumption the octagon is roughly
    `advance_distance_m` ahead. A timeout is kept only as a safety fallback in
    case odometry never arrives.
    """

    def __init__(self, name='NavigateIntoOctagon', advance_distance_m=3.0,
                 surge=0.3, timeout=20.0, buoy_fill_frac=0.7,
                 table_fill_frac=0.6, table_class='octagon_table',
                 confidence=0.4, sway_gain=1.5, sway_limit=0.3):
        super().__init__(name)
        self.advance_distance_m = advance_distance_m
        self.surge = surge
        self.timeout = timeout
        self.buoy_fill_frac = buoy_fill_frac
        self.table_fill_frac = table_fill_frac
        self.table_class = table_class
        self.confidence = confidence
        self.sway_gain = sway_gain
        self.sway_limit = sway_limit
        self.start_time = None
        self.start_pos = None
        self._locked_heading = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.CURRENT_POSE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        import time
        self.start_time = time.time()
        self.start_pos = None
        self._locked_heading = lock_heading(self.blackboard.get(bb.ROS_NODE))
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

    def _pick_visual_cue(self, detections):
        """Return (det, fill_frac) for the best visual cue, or (None, None).

        The under-octagon table is preferred over the buoy border because it
        sits physically inside the octagon. Both must clear the confidence gate.
        """
        if not detections:
            return None, None
        table = None
        buoy = None
        for det in detections.detections:
            if det.confidence <= self.confidence:
                continue
            if det.class_name == self.table_class:
                # Keep the largest table detection (closest / most centered).
                # Size by segmented mask area when available, else bbox area.
                if table is None or detection_size(det) > detection_size(table):
                    table = det
            elif det.class_name == 'buoy':
                if buoy is None or detection_size(det) > detection_size(buoy):
                    buoy = det
        if table is not None:
            return table, self.table_fill_frac
        if buoy is not None:
            return buoy, self.buoy_fill_frac
        return None, None

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
        cmd.yaw = yaw_hold(node, self._locked_heading)  # hold heading while advancing

        # Prefer a real observation of the octagon (table, else buoy). Center on
        # it and treat "we're inside" as its box filling most of the frame.
        # Takes priority over the blind odometry advance.
        cue, fill_frac = self._pick_visual_cue(detections)
        if cue is not None:
            img_w = detections.image_width or 1280
            cmd.sway = max(-self.sway_limit, min(self.sway_limit,
                           ((cue.center_x / img_w) - 0.5) * self.sway_gain))
            if cue.width > img_w * fill_frac:
                node.cmd_pub.publish(ThrusterCommand())
                node.get_logger().info(
                    f'Reached octagon interior (visual: {cue.class_name})')
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


def create_octagon_subtree(total_timeout=240.0, advance_distance_m=3.0,
                           surge=0.3, buoy_fill_frac=0.7, table_fill_frac=0.6,
                           table_class='octagon_table', confidence=0.4,
                           settle_sec=2.0, surface_depth_m=0.3
                           ) -> py_trees.behaviour.Behaviour:
    """Build the Task 5 (Octagon) behavior subtree — surface only (no claw).

    total_timeout is this mission's time budget, split across navigate-in and
    surface deadlines (ratios preserve the old 20:30 tuning). The settle wait is
    a fixed motion and is NOT scaled. The remaining knobs (approach distance,
    surge, frame-fill thresholds, table class name, confidence gate, surface
    depth) are surfaced so they can be tuned as ROS params without code edits.
    """
    t = distribute_timeout(total_timeout, {
        'navigate': 20.0, 'surface': 30.0})

    return py_trees.composites.Sequence(
        name='Task5_Octagon',
        memory=True,
        children=[
            LogBehavior('Oct_Start', 'Starting Task 5: Octagon (surface only)'),
            NavigateIntoOctagon('EnterOctagon', advance_distance_m=advance_distance_m,
                                surge=surge, timeout=t['navigate'],
                                buoy_fill_frac=buoy_fill_frac,
                                table_fill_frac=table_fill_frac,
                                table_class=table_class, confidence=confidence),
            StopMotion('StopInOctagon'),
            WaitForDuration('SettleInOctagon', duration_sec=settle_sec),
            SurfaceInOctagon('Surface', timeout=t['surface'],
                             surface_depth_m=surface_depth_m),
            LogBehavior('Oct_Done', 'Task 5 Octagon COMPLETE'),
        ],
    )
