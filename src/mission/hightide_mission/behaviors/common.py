"""
Common behavior tree leaf nodes shared across all tasks.

Each behavior is a py_trees.behaviour.Behaviour subclass that accesses
ROS2 through the blackboard-stored node reference.
"""

import math
import time as pytime
from collections import deque
import py_trees
from std_msgs.msg import Float64
from hightide_interfaces.msg import ThrusterCommand, DetectionArray
from . import blackboard_keys as bb


def lock_heading(node):
    """Snapshot the current FOG/IMU heading (radians) to hold, or None if not
    yet available. Call in a behavior's initialise() and stash the result."""
    return getattr(node, 'current_heading', None)


def yaw_hold(node, locked_heading, kp=1.5, limit=0.3):
    """Proportional yaw command that drives the vehicle back to `locked_heading`.

    Nothing else holds heading while a behavior strafes/surges (in MANUAL the
    FCU won't, and even in ALT_HOLD small disturbances accumulate), so every
    translating behavior feeds this into cmd.yaw to stay on its locked FOG/IMU
    heading. Returns 0.0 when heading is unknown so it degrades to "no yaw
    command" rather than spinning on bad data.

    IMU yaw (ENU) is CCW-positive, but ThrusterCommand.yaw / ArduSub ch4 are
    CW-positive (see waypoint_navigator_node's yaw_command_sign) — the raw PID
    output must be negated or this becomes positive feedback: any drift off
    locked_heading gets reinforced instead of corrected, so the vehicle spins
    instead of holding straight while it surges.
    """
    if locked_heading is None:
        return 0.0
    current = getattr(node, 'current_heading', None)
    if current is None:
        return 0.0
    from hightide_navigation import normalize_angle
    err = normalize_angle(locked_heading - current)
    return max(-limit, min(limit, -kp * err))


def read_use_odometry(blackboard, default=True):
    """Global movement mode from the blackboard: True = ZED odometry (closed-loop
    on CURRENT_POSE), False = open-loop timed dead reckoning. Defaults to
    odometry if the key was never set (mission_node sets it at startup)."""
    try:
        v = blackboard.get(bb.USE_ODOMETRY)
    except KeyError:
        return default
    return default if v is None else bool(v)


def read_dead_reckon_mps(blackboard, default=0.4):
    """Calibrated forward speed (m/s) used to convert distance→time in dead-reckon
    mode. From the blackboard, else `default`."""
    try:
        v = blackboard.get(bb.DEAD_RECKON_MPS)
    except KeyError:
        return default
    return float(v) if v else default


def estimate_travel(use_odom, start_pos, current_pose, start_time, now, mps):
    """Distance (m) traveled since a leg started.

    ZED-odometry mode uses the straight-line CURRENT_POSE delta (accurate,
    drift-aside). Dead-reckon mode — or odometry mode with no pose available —
    falls back to open-loop elapsed_time * mps, which is only as good as the mps
    calibration but needs no position feedback at all.
    """
    if use_odom and start_pos is not None and current_pose is not None:
        p = current_pose.pose.pose.position
        return math.hypot(p.x - start_pos[0], p.y - start_pos[1])
    return max(0.0, now - start_time) * mps


def detection_size(det):
    """Pixel size of a detection, preferring the TRUE segmented mask area.

    Seg (YOLO*-seg) models populate `mask_area` — the count of segmented pixels,
    which is a cleaner size cue than the bounding box for thin / rotated /
    occluded shapes. Detect-only models (and the classical RGB detectors) leave
    it 0.0, so we fall back to bbox area. Use this anywhere the old code did
    `det.width * det.height` for a size comparison.
    """
    area = getattr(det, 'mask_area', 0.0)
    if area and area > 0.0:
        return float(area)
    return float(det.width * det.height)


def distribute_timeout(total, weights):
    """Split a mission's total time budget across its timeout-bearing behaviors.

    `weights` is a {name: weight} dict — we pass the hand-tuned default timeouts
    as the weights, so the RELATIVE ratios between behaviors are preserved while
    the absolute values are scaled to sum to `total`. Returns {name: seconds}.

    IMPORTANT: only feed *failure / search deadlines* (how long to wait or search
    before giving up) through here. Fixed motion durations — SurgeThrough,
    SlalomPipe.PASS_DURATION, WaitForDuration settle times — must stay constant:
    those are open-loop "drive for N seconds" motions, and scaling one would
    change how far the sub actually travels (e.g. straight into a wall).
    """
    names = list(weights)
    if not names:
        return {}
    s = float(sum(weights.values()))
    if s <= 0:
        return {n: total / len(names) for n in names}
    return {n: total * (float(weights[n]) / s) for n in names}


class DeadReckonTransit(py_trees.behaviour.Behaviour):
    """Blind inter-task transit across open water to a body-frame offset.

    Course props are far apart, so "creep forward until the next prop appears"
    is too slow and misses anything off the current axis. This drives a fixed
    BODY-FRAME (forward, lateral) offset from where the leg started — heading
    held by the FOG, so it strafes rather than turns — then hands off to the
    task's own visual search to acquire the prop.

    Two movement modes, chosen by the global blackboard flag USE_ODOMETRY:
      * ZED (odometry): closed-loop — build a world target from the offset and
        drive to it, stopping within pos_tol. Accurate.
      * dead reckon: open-loop — drive the offset's direction at `speed` for
        time = distance / DEAD_RECKON_MPS. No position feedback; only as good as
        the mps calibration, but works with ZED odometry down.

    forward_m: +ahead in the held heading. lateral_m: +right (course A/D set the
    value upstream). Best-effort: succeeds on arrival, on timeout, or immediately
    for a zero offset — it never stalls the mission (the supervisor is the only
    hard stop).
    """

    def __init__(self, name='DeadReckonTransit', forward_m=0.0, lateral_m=0.0,
                 speed=0.35, pos_tol=0.4, timeout=45.0,
                 target_classes=None, confidence=0.4):
        super().__init__(name)
        self.forward_m = forward_m
        self.lateral_m = lateral_m
        self.speed = speed
        self.pos_tol = pos_tol
        self.timeout = timeout
        # If the next prop's class(es) appear while transiting, bail early and let
        # the task's vision take over — the preset distance/time is only a cap so
        # we don't overshoot when the prop is closer than measured. None = drive
        # the full blind leg (no vision hand-off).
        self.target_classes = set(target_classes) if target_classes else None
        self.confidence = confidence
        self.start_time = None
        self.target = None          # world (x, y) goal (odometry mode only)
        self.dr_duration = None     # timed drive length (dead-reckon mode only)
        self.dr_surge = 0.0
        self.dr_sway = 0.0
        self._locked_heading = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.CURRENT_POSE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.CURRENT_HEADING, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.USE_ODOMETRY, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.DEAD_RECKON_MPS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        self.start_time = pytime.time()
        self.target = None
        self.dr_duration = None
        node = self.blackboard.get(bb.ROS_NODE)
        self._locked_heading = lock_heading(node)

        mag = math.hypot(self.forward_m, self.lateral_m)
        if mag < 1e-6:
            return  # zero offset — update() succeeds immediately

        use_odom = read_use_odometry(self.blackboard)
        if use_odom:
            try:
                pose = self.blackboard.get(bb.CURRENT_POSE)
                yaw = self.blackboard.get(bb.CURRENT_HEADING)
            except KeyError:
                pose = yaw = None
            if pose is not None and yaw is not None:
                # Body-frame (forward=surge, lateral=sway) → world displacement,
                # the exact inverse of update()'s world→body decomposition so a
                # lateral offset commands the matching +sway sign.
                cos_y, sin_y = math.cos(yaw), math.sin(yaw)
                wx = self.forward_m * cos_y - self.lateral_m * sin_y
                wy = self.forward_m * sin_y + self.lateral_m * cos_y
                p = pose.pose.pose.position
                self.target = (p.x + wx, p.y + wy)
                node.get_logger().info(
                    f'{self.name}: ZED transit fwd={self.forward_m:.1f} lat={self.lateral_m:.1f}m')
                return

        # Dead-reckon mode (or odometry requested but unavailable): drive the
        # offset direction open-loop for distance/mps seconds.
        mps = read_dead_reckon_mps(self.blackboard)
        self.dr_duration = mag / max(mps, 1e-3)
        self.dr_surge = self.speed * self.forward_m / mag
        self.dr_sway = self.speed * self.lateral_m / mag
        node.get_logger().info(
            f'{self.name}: dead-reckon transit fwd={self.forward_m:.1f} '
            f'lat={self.lateral_m:.1f}m over {self.dr_duration:.1f}s @ {mps:.2f}m/s')

    def update(self):
        from hightide_interfaces.msg import ThrusterCommand
        node = self.blackboard.get(bb.ROS_NODE)

        # Zero offset → nothing to do.
        if self.target is None and self.dr_duration is None:
            return py_trees.common.Status.SUCCESS
        if (pytime.time() - self.start_time) > self.timeout:
            node.get_logger().warn(f'{self.name}: transit timed out — proceeding')
            node.cmd_pub.publish(ThrusterCommand())
            return py_trees.common.Status.SUCCESS

        # Vision hand-off: the instant the next prop is in view, stop transiting
        # and let the task's own search/align take over (preset distance is just a
        # ceiling). Applies in both ZED and dead-reckon modes.
        if self.target_classes:
            try:
                detections = self.blackboard.get(bb.DETECTIONS)
            except KeyError:
                detections = None
            if detections:
                for det in detections.detections:
                    if (det.class_name in self.target_classes
                            and det.confidence > self.confidence):
                        node.cmd_pub.publish(ThrusterCommand())
                        node.get_logger().info(
                            f'{self.name}: {det.class_name} in view — handing off to vision')
                        return py_trees.common.Status.SUCCESS

        # ---- Dead-reckon (open-loop timed) ----
        if self.target is None:
            if (pytime.time() - self.start_time) >= self.dr_duration:
                node.cmd_pub.publish(ThrusterCommand())
                node.get_logger().info(f'{self.name}: dead-reckon leg complete')
                return py_trees.common.Status.SUCCESS
            cmd = ThrusterCommand()
            cmd.header.stamp = node.get_clock().now().to_msg()
            cmd.surge = self.dr_surge
            cmd.sway = self.dr_sway
            cmd.yaw = yaw_hold(node, self._locked_heading)
            node.cmd_pub.publish(cmd)
            return py_trees.common.Status.RUNNING

        # ---- ZED odometry (closed-loop to world target) ----
        try:
            current = self.blackboard.get(bb.CURRENT_POSE)
            yaw = self.blackboard.get(bb.CURRENT_HEADING)
        except KeyError:
            current = yaw = None
        if current is None or yaw is None:
            return py_trees.common.Status.SUCCESS  # lost odom mid-leg — best effort

        pos = current.pose.pose.position
        dx = self.target[0] - pos.x
        dy = self.target[1] - pos.y
        dist = math.hypot(dx, dy)
        if dist < self.pos_tol:
            node.cmd_pub.publish(ThrusterCommand())
            node.get_logger().info(f'{self.name}: reached transit target')
            return py_trees.common.Status.SUCCESS

        # World error → body frame (surge ahead, +sway right); heading FOG-held.
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        surge = dx * cos_y + dy * sin_y
        sway = -dx * sin_y + dy * cos_y
        norm = max(dist, 1e-3)
        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.surge = max(-self.speed, min(self.speed, self.speed * surge / norm))
        cmd.sway = max(-self.speed, min(self.speed, self.speed * sway / norm))
        cmd.yaw = yaw_hold(node, self._locked_heading)
        node.cmd_pub.publish(cmd)
        return py_trees.common.Status.RUNNING


class PublishThrusterCommand(py_trees.behaviour.Behaviour):
    """Publish a single ThrusterCommand and succeed."""

    def __init__(self, name, surge=0.0, sway=0.0, heave=0.0,
                 yaw=0.0, pitch=0.0, roll=0.0):
        super().__init__(name)
        self.cmd = ThrusterCommand()
        self.cmd.surge = surge
        self.cmd.sway = sway
        self.cmd.heave = heave
        self.cmd.yaw = yaw
        self.cmd.pitch = pitch
        self.cmd.roll = roll
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def update(self):
        node = self.blackboard.get(bb.ROS_NODE)
        pub = node.cmd_pub
        self.cmd.header.stamp = node.get_clock().now().to_msg()
        pub.publish(self.cmd)
        return py_trees.common.Status.SUCCESS


class PublishDepthSetpoint(py_trees.behaviour.Behaviour):
    """Publish a target depth and succeed."""

    def __init__(self, name, depth_m=1.0):
        super().__init__(name)
        self.depth_m = depth_m
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def update(self):
        node = self.blackboard.get(bb.ROS_NODE)
        msg = Float64()
        msg.data = self.depth_m
        node.depth_pub.publish(msg)
        return py_trees.common.Status.SUCCESS


class WaitForDuration(py_trees.behaviour.Behaviour):
    """Tick RUNNING for duration_sec, then SUCCESS."""

    def __init__(self, name, duration_sec=3.0):
        super().__init__(name)
        self.duration = duration_sec
        self.start_time = None

    def initialise(self):
        self.start_time = pytime.time()

    def update(self):
        elapsed = pytime.time() - self.start_time
        if elapsed >= self.duration:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class WaitForDetection(py_trees.behaviour.Behaviour):
    """Tick RUNNING until a specific class is detected. Writes detection to blackboard."""

    def __init__(self, name, target_class, confidence_threshold=0.5, timeout=30.0):
        super().__init__(name)
        self.target_class = target_class
        self.conf_thresh = confidence_threshold
        self.timeout = timeout
        self.start_time = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.TARGET_DETECTION, access=py_trees.common.Access.WRITE)

    def initialise(self):
        self.start_time = pytime.time()

    def update(self):
        if (pytime.time() - self.start_time) > self.timeout:
            return py_trees.common.Status.FAILURE

        try:
            detections = self.blackboard.get(bb.DETECTIONS)
        except KeyError:
            return py_trees.common.Status.RUNNING

        if detections is None:
            return py_trees.common.Status.RUNNING

        for det in detections.detections:
            if (det.class_name == self.target_class and
                    det.confidence >= self.conf_thresh):
                self.blackboard.set(bb.TARGET_DETECTION, det)
                return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class WaitForAnyDetection(py_trees.behaviour.Behaviour):
    """
    Like WaitForDetection but succeeds on ANY of several class names. Used now
    that structural objects (the gate, a bin, the torpedo board) have no
    dedicated class in the ffc model and are instead recognized by whichever
    role symbol sits on them (e.g. the gate is 'seen' via compass/hammer/sos).
    Writes the first matching detection to TARGET_DETECTION.
    """

    def __init__(self, name, target_classes, confidence_threshold=0.5, timeout=30.0):
        super().__init__(name)
        self.target_classes = set(target_classes)
        self.conf_thresh = confidence_threshold
        self.timeout = timeout
        self.start_time = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.TARGET_DETECTION, access=py_trees.common.Access.WRITE)

    def initialise(self):
        self.start_time = pytime.time()

    def update(self):
        if (pytime.time() - self.start_time) > self.timeout:
            return py_trees.common.Status.FAILURE

        try:
            detections = self.blackboard.get(bb.DETECTIONS)
        except KeyError:
            return py_trees.common.Status.RUNNING

        if detections is None:
            return py_trees.common.Status.RUNNING

        for det in detections.detections:
            if (det.class_name in self.target_classes and
                    det.confidence >= self.conf_thresh):
                self.blackboard.set(bb.TARGET_DETECTION, det)
                return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class SearchForDetection(py_trees.behaviour.Behaviour):
    """
    Actively search for any of several classes instead of waiting stationary:
    creep forward while sweeping laterally (sinusoidal sway). Heading stays
    FOG-locked — the sweep widens the camera's effective corridor without
    losing the course heading. The course guarantees no three elements are in
    a straight line, so a pure stationary wait (the old behavior) usually
    stares at empty water.

    Succeeds (and stops motion) when a matching detection appears; fails on
    timeout. min_y_frac/max_y_frac optionally constrain the detection's
    normalized vertical center — used to tell floor props (bins) apart from
    mid-water props (torpedo board) since both carry fire/blood symbols.
    """

    def __init__(self, name, target_classes, confidence_threshold=0.5,
                 timeout=45.0, surge=0.15, sway_amplitude=0.2, sway_period=8.0,
                 min_y_frac=None, max_y_frac=None):
        super().__init__(name)
        self.target_classes = set(target_classes)
        self.conf_thresh = confidence_threshold
        self.timeout = timeout
        self.surge = surge
        self.sway_amplitude = sway_amplitude
        self.sway_period = sway_period
        self.min_y_frac = min_y_frac
        self.max_y_frac = max_y_frac
        self.start_time = None
        self._locked_heading = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.TARGET_DETECTION, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        self.start_time = pytime.time()
        self._locked_heading = lock_heading(self.blackboard.get(bb.ROS_NODE))

    def _matches(self, det, img_h):
        if det.class_name not in self.target_classes:
            return False
        if det.confidence < self.conf_thresh:
            return False
        y_frac = det.center_y / img_h if img_h else 0.5
        if self.min_y_frac is not None and y_frac < self.min_y_frac:
            return False
        if self.max_y_frac is not None and y_frac > self.max_y_frac:
            return False
        return True

    def update(self):
        import math
        node = self.blackboard.get(bb.ROS_NODE)
        elapsed = pytime.time() - self.start_time

        if elapsed > self.timeout:
            node.cmd_pub.publish(ThrusterCommand())
            return py_trees.common.Status.FAILURE

        try:
            detections = self.blackboard.get(bb.DETECTIONS)
        except KeyError:
            detections = None

        if detections is not None:
            img_h = detections.image_height or 720
            for det in detections.detections:
                if self._matches(det, img_h):
                    self.blackboard.set(bb.TARGET_DETECTION, det)
                    node.cmd_pub.publish(ThrusterCommand())
                    node.get_logger().info(
                        f'{self.name}: found {det.class_name} '
                        f'(conf {det.confidence:.2f})')
                    return py_trees.common.Status.SUCCESS

        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.surge = self.surge
        cmd.sway = self.sway_amplitude * math.sin(
            2.0 * math.pi * elapsed / self.sway_period)
        cmd.yaw = yaw_hold(node, self._locked_heading)  # hold heading while sweeping
        node.cmd_pub.publish(cmd)
        return py_trees.common.Status.RUNNING


class WaitForStableDetection(py_trees.behaviour.Behaviour):
    """
    Succeed only when a target class shows up in at least `min_hits` of the last
    `window` ticks — a sliding-window M-of-N filter (default 4-of-5). This is the
    robust replacement for trusting a single frame: one spurious detection can't
    commit the mission, and one dropped frame can't reset it either (unlike a
    strict-consecutive requirement).

    Writes the most recent matching detection to TARGET_DETECTION so a follow-up
    align/approach behavior has something to work with. Fails on timeout.
    """

    def __init__(self, name, target_classes, window=5, min_hits=4,
                 confidence_threshold=0.5, timeout=30.0):
        super().__init__(name)
        self.target_classes = set(target_classes)
        self.window = window
        self.min_hits = min_hits
        self.conf_thresh = confidence_threshold
        self.timeout = timeout
        self.start_time = None
        self.history = deque(maxlen=window)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.TARGET_DETECTION, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        self.start_time = pytime.time()
        self.history.clear()

    def _best_match(self, detections):
        """Highest-confidence detection of a target class this tick, or None."""
        best = None
        if detections is not None:
            for det in detections.detections:
                if (det.class_name in self.target_classes and
                        det.confidence >= self.conf_thresh):
                    if best is None or det.confidence > best.confidence:
                        best = det
        return best

    def update(self):
        if (pytime.time() - self.start_time) > self.timeout:
            return py_trees.common.Status.FAILURE

        try:
            detections = self.blackboard.get(bb.DETECTIONS)
        except KeyError:
            detections = None

        match = self._best_match(detections)
        self.history.append(match is not None)
        if match is not None:
            # Keep the freshest good detection for the downstream aligner.
            self.blackboard.set(bb.TARGET_DETECTION, match)

        # Only decide once the window is full, so we always judge over N frames.
        if len(self.history) >= self.window and sum(self.history) >= self.min_hits:
            node = self.blackboard.get(bb.ROS_NODE)
            node.get_logger().info(
                f'{self.name}: stable detection '
                f'({sum(self.history)}/{self.window} of last frames)')
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class YawSweepSearch(py_trees.behaviour.Behaviour):
    """
    Rotate in place looking for any of `target_classes`, confirming via the same
    sliding-window M-of-N filter. This replaces the old "gate is either dead
    ahead or exactly 180° behind" assumption: by sweeping yaw it finds the gate
    at ANY bearing (e.g. 90° off to the side). Heading is driven open-loop at a
    fixed yaw rate — one direction, so a long-enough timeout covers a full turn.

    Stops (zero motion) and SUCCEEDS on a stable detection; FAILS on timeout.
    Only yaw is commanded — never surge — so a bad orientation can't drive the
    sub into a wall while it looks.
    """

    def __init__(self, name, target_classes, yaw_rate=0.3, window=5, min_hits=4,
                 confidence_threshold=0.5, timeout=40.0):
        super().__init__(name)
        self.target_classes = set(target_classes)
        self.yaw_rate = yaw_rate
        self.window = window
        self.min_hits = min_hits
        self.conf_thresh = confidence_threshold
        self.timeout = timeout
        self.start_time = None
        self.history = deque(maxlen=window)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.TARGET_DETECTION, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        self.start_time = pytime.time()
        self.history.clear()

    def _best_match(self, detections):
        best = None
        if detections is not None:
            for det in detections.detections:
                if (det.class_name in self.target_classes and
                        det.confidence >= self.conf_thresh):
                    if best is None or det.confidence > best.confidence:
                        best = det
        return best

    def update(self):
        node = self.blackboard.get(bb.ROS_NODE)

        if (pytime.time() - self.start_time) > self.timeout:
            node.cmd_pub.publish(ThrusterCommand())
            node.get_logger().warn(f'{self.name}: swept without a stable detection')
            return py_trees.common.Status.FAILURE

        try:
            detections = self.blackboard.get(bb.DETECTIONS)
        except KeyError:
            detections = None

        match = self._best_match(detections)
        self.history.append(match is not None)
        if match is not None:
            self.blackboard.set(bb.TARGET_DETECTION, match)

        if len(self.history) >= self.window and sum(self.history) >= self.min_hits:
            node.cmd_pub.publish(ThrusterCommand())  # stop rotating
            node.get_logger().info(
                f'{self.name}: gate found while sweeping '
                f'({sum(self.history)}/{self.window} of last frames)')
            return py_trees.common.Status.SUCCESS

        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.yaw = self.yaw_rate
        node.cmd_pub.publish(cmd)
        return py_trees.common.Status.RUNNING


class CheckDetectionVisible(py_trees.behaviour.Behaviour):
    """Condition: checks if a class is currently visible in detections."""

    def __init__(self, name, target_class, min_confidence=0.4):
        super().__init__(name)
        self.target_class = target_class
        self.min_conf = min_confidence
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)

    def update(self):
        try:
            detections = self.blackboard.get(bb.DETECTIONS)
        except KeyError:
            return py_trees.common.Status.FAILURE

        if detections:
            for det in detections.detections:
                if (det.class_name == self.target_class and
                        det.confidence >= self.min_conf):
                    return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class SetBlackboardValue(py_trees.behaviour.Behaviour):
    """Write a value to the blackboard and succeed."""

    def __init__(self, name, key, value):
        super().__init__(name)
        self.key = key
        self.value = value
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)

    def update(self):
        self.blackboard.set(self.key, self.value)
        return py_trees.common.Status.SUCCESS


class LogBehavior(py_trees.behaviour.Behaviour):
    """Log a message and succeed."""

    def __init__(self, name, message=''):
        super().__init__(name)
        self.message = message
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def update(self):
        node = self.blackboard.get(bb.ROS_NODE)
        node.get_logger().info(f'[BT] {self.message}')
        return py_trees.common.Status.SUCCESS


class StopMotion(py_trees.behaviour.Behaviour):
    """Publish zero command to stop all motion."""

    def __init__(self, name='StopMotion'):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def update(self):
        node = self.blackboard.get(bb.ROS_NODE)
        node.cmd_pub.publish(ThrusterCommand())
        return py_trees.common.Status.SUCCESS


class CallTriggerService(py_trees.behaviour.Behaviour):
    """
    Call a std_srvs/Trigger service once and report the result.

    Used for the style maneuvers (/hightide/yaw_spin, /hightide/barrel_roll),
    which are long-running blocking services living in their own nodes — so the
    call is async and this behavior ticks RUNNING until the future resolves.
    In best_effort mode a missing service or a failed response still returns
    SUCCESS so a bonus maneuver never aborts the mission.
    """

    def __init__(self, name, service_name, wait_timeout=1.0, best_effort=True):
        super().__init__(name)
        self.service_name = service_name
        self.wait_timeout = wait_timeout
        self.best_effort = best_effort
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)
        self.client = None
        self.future = None

    def initialise(self):
        from std_srvs.srv import Trigger
        node = self.blackboard.get(bb.ROS_NODE)
        self.client = node.create_client(Trigger, self.service_name)
        self.future = None

    def _finish(self, ok):
        if ok or self.best_effort:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE

    def update(self):
        from std_srvs.srv import Trigger
        node = self.blackboard.get(bb.ROS_NODE)

        if not self.client.wait_for_service(timeout_sec=self.wait_timeout):
            node.get_logger().warn(f'{self.service_name} not available')
            return self._finish(False)

        if self.future is None:
            self.future = self.client.call_async(Trigger.Request())
            return py_trees.common.Status.RUNNING

        if not self.future.done():
            return py_trees.common.Status.RUNNING

        result = self.future.result()
        ok = bool(result and result.success)
        node.get_logger().info(f'{self.service_name} -> {"ok" if ok else "failed"}')
        return self._finish(ok)


class RecordPose(py_trees.behaviour.Behaviour):
    """
    Snapshot the current odometry pose into a blackboard key so it can be
    navigated back to later (used to remember the gate position for Return
    Home, since we have no acoustic pinger to home on).
    """

    def __init__(self, name, dest_key):
        super().__init__(name)
        self.dest_key = dest_key
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.CURRENT_POSE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.dest_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def update(self):
        node = self.blackboard.get(bb.ROS_NODE)
        try:
            pose = self.blackboard.get(bb.CURRENT_POSE)
        except KeyError:
            pose = None

        self.blackboard.set(self.dest_key, pose)
        if pose is not None:
            p = pose.pose.pose.position
            node.get_logger().info(
                f'Recorded {self.dest_key}: x={p.x:.2f} y={p.y:.2f}')
        else:
            node.get_logger().warn(
                f'No odometry available to record {self.dest_key}')
        return py_trees.common.Status.SUCCESS