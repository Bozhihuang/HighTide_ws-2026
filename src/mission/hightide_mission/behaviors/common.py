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


def yaw_hold(node, locked_heading, kp=None, kd=None, limit=None):
    """PD yaw command that drives the vehicle back to `locked_heading`.

    This mirrors the pool-proven pidtest companion heading hold: PD (not
    P-only) with the tuned stable gains kp=0.45 / kd=0.2. The derivative term
    matters — an undamped P heading-hold limit-cycles on a vehicle with yaw
    inertia, constantly banging yaw left/right, which on real (asymmetric)
    thrusters bleeds into lateral drift: the sub crabs instead of driving
    straight. Derivative state is kept on the node; a >0.5 s gap between
    calls (behavior handoff) resets it so a stale error can't spike the D.

    Gains default to the mission node's `yaw_hold_kp` / `yaw_hold_kd` /
    `yaw_hold_limit` / `yaw_hold_sign` ROS params — ALL live-tunable in-water
    (`ros2 param set /mission_node yaw_hold_kp 0.3`) unless the caller
    overrides them. Returns 0.0 when heading is unknown so it degrades to
    "no yaw command" rather than spinning on bad data.

    IMU yaw (ENU) is CCW-positive, but ThrusterCommand.yaw / ArduSub ch4 are
    CW-positive — hence yaw_hold_sign = -1.0, the SAME convention pidtest's
    yaw_command_sign uses and that HeadingTurn converged with in the water.
    Set yaw_hold_sign to +1.0 live if you ever need to test the opposite.
    Note the rc_override deadzone (0.05): outputs below it go out as neutral
    PWM, and in ALT_HOLD the FCU then holds heading itself — small errors are
    deliberately left to ArduSub instead of fighting it.
    """
    if locked_heading is None:
        return 0.0
    current = getattr(node, 'current_heading', None)
    if current is None:
        return 0.0
    if kp is None:
        kp = float(getattr(node, 'yaw_hold_kp', 0.45))
    if kd is None:
        kd = float(getattr(node, 'yaw_hold_kd', 0.2))
    if limit is None:
        limit = float(getattr(node, 'yaw_hold_limit', 0.25))
    sign = float(getattr(node, 'yaw_hold_sign', -1.0))
    from hightide_navigation import normalize_angle
    err = normalize_angle(locked_heading - current)

    now = pytime.time()
    d = 0.0
    state = getattr(node, '_yaw_hold_state', None)
    if state is not None:
        prev_err, prev_t = state
        dt = now - prev_t
        if 0.0 < dt < 0.5:
            d = normalize_angle(err - prev_err) / dt
    node._yaw_hold_state = (err, now)

    out = sign * (kp * err + kd * d)
    return max(-limit, min(limit, out))


def pose_yaw(pose):
    """Yaw (radians) of an Odometry message, in the SAME frame as its position.

    All world<->body geometry done on CURRENT_POSE must use this — NOT the
    IMU/FOG heading. The IMU yaw reference is arbitrary relative to the ZED
    odometry frame (ZED zeroes wherever the camera pointed at boot), and
    decomposing ZED world errors with an offset heading rotates every commanded
    velocity by that offset: instead of driving at its target the vehicle
    pursues it in a spiral, and at a ~90° offset it orbits the target in a
    circle without ever arriving (observed in-pool on the opening advance).
    The IMU heading remains correct for yaw_hold — holding a heading needs no
    shared frame with the position source.
    """
    if pose is None:
        return None
    from hightide_navigation import quaternion_to_yaw
    try:
        return quaternion_to_yaw(pose.pose.pose.orientation)
    except Exception:
        return None


def read_use_odometry(blackboard, default=True):
    """Global movement mode from the blackboard: True = ZED odometry (closed-loop
    on CURRENT_POSE), False = open-loop timed dead reckoning. Defaults to
    odometry if the key was never set (mission_node sets it at startup)."""
    try:
        v = blackboard.get(bb.USE_ODOMETRY)
    except KeyError:
        return default
    return default if v is None else bool(v)


def lock_track(node, blackboard):
    """Snapshot the world TRACK LINE to hold laterally: the ZED position and
    heading right now. Returns (x0, y0, yaw0) or None if it can't be held.

    This is the sway-axis twin of lock_heading(): call it in initialise() and
    stash the result, then feed it to sway_hold() every tick. Where lock_heading
    remembers "which way we were pointing", this remembers "which LINE we were
    driving down" — the line through (x0, y0) along yaw0.

    Returns None (→ sway_hold is a no-op) when there is no trustworthy position:
    in dead_reckon movement mode we have deliberately chosen not to believe ZED
    odometry, and holding a line against a drifting position estimate would walk
    the sub sideways across the pool rather than keep it straight.

    yaw0 comes from the ZED pose, NOT the IMU — the line and the position it is
    compared against must live in one frame (see pose_yaw()).
    """
    if not read_use_odometry(blackboard):
        return None
    try:
        pose = blackboard.get(bb.CURRENT_POSE)
    except KeyError:
        pose = None
    yaw = pose_yaw(pose)
    if pose is None or yaw is None:
        return None
    p = pose.pose.pose.position
    return (p.x, p.y, yaw)


def sway_hold(node, blackboard, locked_track, kp=None, kd=None, limit=None):
    """PD sway command that drives the vehicle back onto `locked_track`.

    The sway-axis twin of yaw_hold(), and it exists for the same reason: pool
    waves push the sub sideways off its line over a long open-water leg, and
    nothing else corrects it — a surge leg commands zero sway, so the drift is
    integrated straight into every downstream dead-reckoned distance. Same PD
    shape, same live-tunable params, same degrade-to-zero-on-bad-data rule.

    Only the CROSS-TRACK component of the error is used: the error is projected
    onto the axis perpendicular to the LOCKED heading yaw0, so distance traveled
    ALONG the line produces no sway command and the leg's own surge controller
    keeps sole ownership of the forward axis. Projecting with the current
    heading instead would bleed along-track distance into the sway command as
    soon as yaw drifted a few degrees (3 m down a leg at 10° off = half a metre
    of phantom cross-track), and the sub would chase a drift that isn't there.

    Sign convention matches DeadReckonTransit's lateral legs: the error is
    measured TOWARD the line on the body-lateral axis (e_sway = -dx*sin + dy*cos),
    and a positive command drives that way — so sway_hold_sign defaults to +1.0.
    Flip it live if the sub runs AWAY from the line in the pool.

    Gains/clamp/deadband default to the node's `sway_hold_kp` / `sway_hold_kd` /
    `sway_hold_limit` / `sway_hold_sign` / `sway_hold_deadband_m` params, all
    live-tunable in-water (`ros2 param set /mission_node sway_hold_kp 0.15`).
    Set sway_hold_enabled false to disable the whole thing mid-run.
    """
    if locked_track is None:
        return 0.0
    if not bool(getattr(node, 'sway_hold_enabled', True)):
        return 0.0
    try:
        pose = blackboard.get(bb.CURRENT_POSE)
    except KeyError:
        pose = None
    if pose is None:
        return 0.0

    if kp is None:
        kp = float(getattr(node, 'sway_hold_kp', 0.5))
    if kd is None:
        kd = float(getattr(node, 'sway_hold_kd', 0.15))
    if limit is None:
        limit = float(getattr(node, 'sway_hold_limit', 0.2))
    sign = float(getattr(node, 'sway_hold_sign', 1.0))
    deadband = float(getattr(node, 'sway_hold_deadband_m', 0.1))

    x0, y0, yaw0 = locked_track
    p = pose.pose.pose.position
    # Signed error toward the line, on the axis perpendicular to yaw0.
    err = -(x0 - p.x) * math.sin(yaw0) + (y0 - p.y) * math.cos(yaw0)

    now = pytime.time()
    d = 0.0
    state = getattr(node, '_sway_hold_state', None)
    if state is not None:
        prev_err, prev_t = state
        dt = now - prev_t
        if 0.0 < dt < 0.5:   # >0.5s gap = behavior handoff; stale D would spike
            d = (err - prev_err) / dt
    node._sway_hold_state = (err, now)

    # Deadband AFTER the D state update: ZED odometry noise and the sub's own
    # bobbing are centimetre-scale, and correcting them just burns lateral
    # thrusters and adds wake. Waves move us a lot further than that.
    if abs(err) < deadband:
        return 0.0

    out = sign * (kp * err + kd * d)
    return max(-limit, min(limit, out))


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
                 target_classes=None, confidence=0.4,
                 kp=0.2, ki=0.0, kd=0.05):
        super().__init__(name)
        self.forward_m = forward_m
        self.lateral_m = lateral_m
        self.speed = speed          # PID output clamp (max surge/sway), like the tester's max_output
        self.pos_tol = pos_tol
        self.timeout = timeout
        # Position-PID gains for the ZED (closed-loop) drive — SAME control law as
        # the pool pidtest so travel decelerates onto the target instead of
        # constant-thrust-then-cut. Set these to the surge gains you tuned there.
        self.kp, self.ki, self.kd = kp, ki, kd
        self.surge_pid = None
        self.sway_pid = None
        self.last_time = None
        # If the next prop's class(es) appear while transiting, bail early and let
        # the task's vision take over — the preset distance/time is only a cap so
        # we don't overshoot when the prop is closer than measured. None = drive
        # the full blind leg (no vision hand-off).
        self.target_classes = set(target_classes) if target_classes else None
        self.confidence = confidence
        self.start_time = None
        self.target = None          # world (x, y) goal ('world' mode only)
        self.dr_duration = None     # timed drive length ('timed' mode only)
        self.dr_surge = 0.0
        self.dr_sway = 0.0
        # 'zed'   = closed-loop to a world target — a literal port of the
        #           pool-proven pidtest surge loop (see initialise)
        # 'timed' = open-loop dead reckon; None = zero offset, nothing to do
        self.mode = None
        self.pure_forward = False
        self.companion_yaw_pid = None
        self.start_pos = None
        self.leg_mag = 0.0
        self.effective_timeout = timeout
        self._locked_heading = None
        self._locked_track = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.CURRENT_POSE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.CURRENT_HEADING, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.USE_ODOMETRY, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.DEAD_RECKON_MPS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        from hightide_navigation import PIDController
        self.start_time = pytime.time()
        self.last_time = self.start_time
        self.target = None
        self.dr_duration = None
        self.mode = None
        self.pure_forward = False
        self.start_pos = None
        self.effective_timeout = self.timeout
        node = self.blackboard.get(bb.ROS_NODE)
        self._locked_heading = lock_heading(node)
        # The line this leg drives down — held on the sway axis exactly like the
        # heading is held on the yaw axis (see sway_hold). Only pure-forward legs
        # use it; a lateral leg's own sway PID already owns that axis.
        self._locked_track = lock_track(node, self.blackboard)
        # Fresh position PIDs each leg (output clamped to the speed cap) — the
        # same PIDController + clamp arrangement as pidtest's main axis.
        self.surge_pid = PIDController(self.kp, self.ki, self.kd,
                                       output_min=-self.speed, output_max=self.speed)
        self.sway_pid = PIDController(self.kp, self.ki, self.kd,
                                      output_min=-self.speed, output_max=self.speed)
        # Companion heading hold — pidtest's companion PID, verbatim (the
        # PIDController class with its filtered derivative), gains from the
        # live yaw_hold params. Fresh each leg, like pidtest resets on step.
        self.companion_yaw_pid = PIDController(
            float(getattr(node, 'yaw_hold_kp', 0.45)), 0.0,
            float(getattr(node, 'yaw_hold_kd', 0.2)))

        mag = math.hypot(self.forward_m, self.lateral_m)
        self.leg_mag = mag
        if mag < 1e-6:
            return  # zero offset — update() succeeds immediately

        # Bound the leg by its DISTANCE: 3x the nominal travel time (min 8 s).
        # The flat 45 s default let a misbehaving 2 m leg grind against a wall
        # for most of a minute before giving up.
        mps = read_dead_reckon_mps(self.blackboard)
        self.effective_timeout = min(
            self.timeout, max(8.0, 3.0 * mag / max(mps, 1e-3)))

        use_odom = read_use_odometry(self.blackboard)
        if use_odom:
            try:
                pose = self.blackboard.get(bb.CURRENT_POSE)
            except KeyError:
                pose = None
            # Yaw from the ZED pose itself, NOT the IMU — see pose_yaw().
            yaw = pose_yaw(pose)
            if pose is not None and yaw is not None:
                # Literal pidtest step: world target = pos + offset rotated by
                # the ZED pose's own yaw (pidtest builds the same thing from
                # its auto-calibrated imu+frame_offset heading). Pure-forward
                # legs behave exactly like a pidtest surge step: signed
                # body-projected error, NO sway ever commanded.
                p = pose.pose.pose.position
                self.start_pos = (p.x, p.y)
                cos_y, sin_y = math.cos(yaw), math.sin(yaw)
                wx = self.forward_m * cos_y - self.lateral_m * sin_y
                wy = self.forward_m * sin_y + self.lateral_m * cos_y
                self.target = (p.x + wx, p.y + wy)
                self.pure_forward = abs(self.lateral_m) < 1e-6
                self.mode = 'zed'
                node.get_logger().info(
                    f'{self.name}: ZED transit fwd={self.forward_m:.1f} '
                    f'lat={self.lateral_m:.1f}m'
                    + (' (pidtest surge law)' if self.pure_forward else '')
                    + f' (timeout {self.effective_timeout:.0f}s)')
                return

        # Dead-reckon mode (or odometry requested but unavailable): drive the
        # offset direction open-loop for distance/mps seconds.
        self.mode = 'timed'
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
        if self.mode is None:
            return py_trees.common.Status.SUCCESS
        if (pytime.time() - self.start_time) > self.effective_timeout:
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
        if self.mode == 'timed':
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

        # ---- ZED mode: the pidtest surge loop, ported verbatim ----
        from hightide_navigation import normalize_angle
        try:
            current = self.blackboard.get(bb.CURRENT_POSE)
        except KeyError:
            current = None
        yaw = pose_yaw(current)
        if current is None or yaw is None:
            node.cmd_pub.publish(ThrusterCommand())   # lost odom mid-leg — stop, best effort
            return py_trees.common.Status.SUCCESS
        pos = current.pose.pose.position

        # Overshoot guard: if we've physically moved well past the leg length
        # the target estimate is bad (odometry drift/jump) — stop chasing it.
        traveled = math.hypot(pos.x - self.start_pos[0], pos.y - self.start_pos[1])
        if traveled > 1.5 * self.leg_mag + 0.5:
            node.cmd_pub.publish(ThrusterCommand())
            node.get_logger().warn(
                f'{self.name}: traveled {traveled:.1f}m on a {self.leg_mag:.1f}m '
                'leg — bad odometry? stopping here')
            return py_trees.common.Status.SUCCESS

        # pidtest's _measure(): world error projected onto the body axes with
        # the current ZED-frame heading. e_surge is SIGNED, so an overshoot
        # drives back instead of latching done.
        dx = self.target[0] - pos.x
        dy = self.target[1] - pos.y
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        e_surge = dx * cos_y + dy * sin_y
        e_sway = -dx * sin_y + dy * cos_y

        done = (abs(e_surge) < self.pos_tol if self.pure_forward
                else math.hypot(dx, dy) < self.pos_tol)
        if done:
            node.cmd_pub.publish(ThrusterCommand())
            node.get_logger().info(f'{self.name}: reached transit target')
            return py_trees.common.Status.SUCCESS

        now = pytime.time()
        dt = now - self.last_time
        self.last_time = now

        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.surge = self.surge_pid.compute(e_surge, dt)
        # pidtest never commands sway on a surge run, and neither did we — but
        # pidtest ran a short step in still water, while a competition leg is
        # long enough for pool waves to push the sub metres off its line, and a
        # pure-forward leg has nothing else watching the sway axis. So hold the
        # line with the gentle, deadbanded sway_hold PD (NOT the surge-strength
        # position PID) — same treatment the yaw axis already gets. Lateral legs
        # are unchanged: their own sway PID already owns that axis.
        cmd.sway = (sway_hold(node, self.blackboard, self._locked_track)
                    if self.pure_forward else self.sway_pid.compute(e_sway, dt))
        # pidtest's companion stabilizer, verbatim:
        #   yaw_error = ref_yaw - imu_heading; cmd.yaw = sign * pid(yaw_error)
        imu_now = getattr(node, 'current_heading', None)
        if self._locked_heading is not None and imu_now is not None:
            yaw_err = normalize_angle(self._locked_heading - imu_now)
            y = (float(getattr(node, 'yaw_hold_sign', -1.0))
                 * self.companion_yaw_pid.compute(yaw_err, dt))
            lim = float(getattr(node, 'yaw_hold_limit', 1.0))
            cmd.yaw = max(-lim, min(lim, y))
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

    wait_timeout is a budget spread ACROSS ticks (checked via the non-blocking
    service_is_ready(), not a single blocking wait_for_service() call) — DDS
    discovery for a freshly created client easily exceeds one second right
    after full_system.launch.py brings up ~15 nodes at once, and a single
    blocking 1s check used to give up (best-effort SUCCESS, service never
    actually called) before discovery finished. Polling across ticks over a
    longer budget gives discovery time to complete without stalling the
    node's own executor on a blocking call each tick.
    """

    def __init__(self, name, service_name, wait_timeout=10.0, best_effort=True):
        super().__init__(name)
        self.service_name = service_name
        self.wait_timeout = wait_timeout
        self.best_effort = best_effort
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)
        self.client = None
        self.future = None
        self.wait_start = None

    def initialise(self):
        from std_srvs.srv import Trigger
        node = self.blackboard.get(bb.ROS_NODE)
        self.client = node.create_client(Trigger, self.service_name)
        self.future = None
        self.wait_start = pytime.time()

    def _finish(self, ok):
        if ok or self.best_effort:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE

    def update(self):
        from std_srvs.srv import Trigger
        node = self.blackboard.get(bb.ROS_NODE)

        if self.future is None:
            if not self.client.service_is_ready():
                if (pytime.time() - self.wait_start) > self.wait_timeout:
                    node.get_logger().warn(
                        f'{self.service_name} not available after '
                        f'{self.wait_timeout:.1f}s — giving up')
                    return self._finish(False)
                return py_trees.common.Status.RUNNING
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