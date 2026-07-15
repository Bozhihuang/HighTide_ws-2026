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


def odom_is_fresh(node, max_age=None):
    """True when the ZED pose can be trusted RIGHT NOW.

    Three independent kill switches, any of which vetoes trust:
      1. mission_node's odom callback stamps `node.odom_rx_time` only when the
         pose VALUE changes — so a ZED that keeps republishing a bit-identical
         pose (tracking frozen, the classic murky-water failure) goes stale here
         exactly like one that stopped publishing. Real VO output always jitters
         at the millimetre level, even holding station, so value-change is a
         safe liveness signal.
      2. age > odom_stale_sec (node param, default 0.7 s).
      3. the nav tier manager demoted to DEAD_RECKONING (covariance blowup /
         its own staleness check) — `node.nav_tier_is_dr`.

    This gate exists because the old code only ever checked `pose is None`,
    which is false forever after the FIRST message: a mid-leg ZED freeze left
    the transit PIDing against a frozen position — error never shrinks, so it
    surged at full clamp for the whole 3x-nominal timeout (up to 3x the leg
    distance) and then logged SUCCESS.
    """
    rx = getattr(node, 'odom_rx_time', None)
    if rx is None:
        return False
    if max_age is None:
        max_age = float(getattr(node, 'odom_stale_sec', 0.7))
    if (pytime.time() - rx) > max_age:
        return False
    if bool(getattr(node, 'nav_tier_is_dr', False)):
        return False
    return True


def resolve_course_heading(node, blackboard):
    """The heading a straight leg should HOLD: the INTENDED course heading
    (threaded through bb.INTENDED_HEADING by the preceding turn) when we are
    plausibly pointed at it, else the live FOG heading. Returns (heading, src)
    where src is 'intended' / 'live' / 'none' (heading None).

    Why not just lock the live heading (the old behavior)? HeadingTurn declares
    success anywhere inside tolerance_deg of its target, and the sub still has
    yaw rate when it does — so each leg locked "wherever the nose happened to
    be at that instant". 5° off on a 6 m leg is half a metre of lateral miss,
    and the error COMPOUNDS turn over turn through the slalom. Locking the
    intended heading lets yaw_hold pull the leg onto the true course line
    during the drive, so turn tolerance stops accumulating; the FOG (the one
    sensor we fully trust) becomes the spine of the whole course.

    The sanity window (intended_heading_max_dev_deg node param, default 30°)
    is what makes a stale intent harmless with NO invalidation bookkeeping:
    after a yaw sweep / octagon facing scan the nose is far from the last
    written intent, and an intent that far away is no longer a plausible
    "finish the turn" correction — fall back to the live heading and log it.
    """
    cur = lock_heading(node)
    try:
        intended = blackboard.get(bb.INTENDED_HEADING)
    except (KeyError, AttributeError):
        intended = None
    if intended is None or cur is None:
        return (cur, 'live' if cur is not None else 'none')
    from hightide_navigation import normalize_angle
    dev = abs(normalize_angle(intended - cur))
    max_dev = math.radians(float(getattr(node, 'intended_heading_max_dev_deg', 30.0)))
    if dev <= max_dev:
        return (intended, 'intended')
    node.get_logger().warn(
        f'Intended heading is {math.degrees(dev):.0f}° from the nose '
        f'(> {math.degrees(max_dev):.0f}°) — stale after a sweep? '
        'Holding live heading instead')
    return (cur, 'live')


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
    if not odom_is_fresh(node):
        return None   # don't lock a line to a frozen/stale pose
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
    if pose is None or not odom_is_fresh(node):
        return 0.0   # never trim against a stale/frozen pose

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

    Two movement modes; unlike before, a leg can DEMOTE mid-flight:
      * 'zed': closed-loop on ZED odometry. Pure-forward legs (the whole live
        course) drive FOG-primary: yaw_hold on the intended course heading,
        progress = ZED displacement projected onto the leg's START axis (a
        scalar), sway_hold for cross-track. A 1-D along-track error is immune
        to mid-leg ZED yaw wander, which used to bend the old world-target
        trajectory. Diagonal legs keep the original 2-D world-target PID.
      * 'dr': open-loop — drive the offset direction at `speed` for
        time = remaining / DEAD_RECKON_MPS.
    A leg starts in 'zed' whenever ZED is healthy (odom_is_fresh + nav tier),
    and the moment ZED goes stale/frozen mid-leg it falls back to 'dr' FOR THE
    REMAINING distance (anchored at the last good along-track progress) instead
    of silently truncating the leg (the old `pose is None` check could never
    fire and the old fallback stopped dead where it was). Once demoted it stays
    demoted for the leg — predictability over cleverness.

    While cruising in 'zed' mode at the saturated speed cap, the leg measures
    its own actual m/s and EMA-updates bb.DEAD_RECKON_MPS (mps autocal), so a
    later ZED dropout dead-reckons with TODAY's measured speed — battery, trim,
    depth included — not a config constant from some other day.

    Heading: locks the INTENDED course heading via resolve_course_heading, so
    turn tolerance doesn't leak into leg direction (see that docstring).

    Every exit path emits one `LEG <name>: ...` telemetry line (commanded vs
    estimated distance, elapsed, mode history, mean sway trim, exit reason) —
    one pool run of these is a full calibration dataset.

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
        self.target = None          # world (x, y) goal (diagonal 'zed' legs only)
        self.mode = None            # None (zero offset) | 'zed' | 'dr'
        self.mode_history = 'none'  # e.g. 'zed', 'zed>dr@1.4m', 'dr' — for the LEG line
        self.anchor = None          # ZED position at leg start
        self.axis = None            # ZED-world unit vector of the leg direction
        self.body_dir = (1.0, 0.0)  # unit drive direction in (surge, sway)
        self.last_s = 0.0           # last good along-track progress (m)
        self.mps = 0.4
        self.dr_start_time = None   # when open-loop driving began
        self.dr_duration = None     # open-loop drive length (s)
        self.dr_base_s = 0.0        # progress already banked when 'dr' began
        self.dr_surge = 0.0
        self.dr_sway = 0.0
        self.pure_forward = False
        self.companion_yaw_pid = None
        self.start_pos = None
        self.leg_mag = 0.0
        self.effective_timeout = timeout
        self._locked_heading = None
        self._locked_track = None
        self._sway_sum = 0.0        # telemetry: mean sway trim over the leg
        self._tick_n = 0
        self._cruise_v = []         # mps autocal: per-tick speed samples at saturation
        self._prev_s = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.CURRENT_POSE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.CURRENT_HEADING, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.INTENDED_HEADING, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.USE_ODOMETRY, access=py_trees.common.Access.READ)
        # WRITE (which includes read) — mps autocal updates this key.
        self.blackboard.register_key(key=bb.DEAD_RECKON_MPS, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        from hightide_navigation import PIDController
        self.start_time = pytime.time()
        self.last_time = self.start_time
        self.target = None
        self.mode = None
        self.mode_history = 'none'
        self.pure_forward = False
        self.start_pos = None
        self.anchor = None
        self.axis = None
        self.last_s = 0.0
        self.dr_start_time = None
        self.dr_duration = None
        self.dr_base_s = 0.0
        self.effective_timeout = self.timeout
        self._locked_track = None
        self._sway_sum = 0.0
        self._tick_n = 0
        self._cruise_v = []
        self._prev_s = None
        node = self.blackboard.get(bb.ROS_NODE)
        # Hold the INTENDED course heading (threaded through the blackboard by
        # the preceding turn) rather than wherever the nose happens to point —
        # see resolve_course_heading for why.
        self._locked_heading, hd_src = resolve_course_heading(node, self.blackboard)
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
        self.body_dir = (self.forward_m / mag, self.lateral_m / mag)

        # Bound the leg by its DISTANCE: 3x the nominal travel time (min 8 s).
        # The flat 45 s default let a misbehaving 2 m leg grind against a wall
        # for most of a minute before giving up.
        self.mps = read_dead_reckon_mps(self.blackboard)
        self.effective_timeout = min(
            self.timeout, max(8.0, 3.0 * mag / max(self.mps, 1e-3)))

        use_odom = read_use_odometry(self.blackboard)
        pose = None
        if use_odom:
            try:
                pose = self.blackboard.get(bb.CURRENT_POSE)
            except KeyError:
                pose = None
        # Yaw from the ZED pose itself, NOT the IMU — see pose_yaw().
        yaw = pose_yaw(pose)
        if use_odom and pose is not None and yaw is not None and odom_is_fresh(node):
            p = pose.pose.pose.position
            self.start_pos = (p.x, p.y)
            self.anchor = (p.x, p.y)
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            wx = self.forward_m * cos_y - self.lateral_m * sin_y
            wy = self.forward_m * sin_y + self.lateral_m * cos_y
            # The leg's fixed ZED-world axis: locked at start so progress is a
            # scalar along it — mid-leg ZED yaw wander can't bend the leg.
            self.axis = (wx / mag, wy / mag)
            self.pure_forward = abs(self.lateral_m) < 1e-6
            if self.pure_forward:
                # Cross-track held by sway_hold against this same anchor line.
                self._locked_track = (p.x, p.y, yaw)
            else:
                # Diagonal legs keep the original 2-D world-target PID.
                self.target = (p.x + wx, p.y + wy)
            self.mode = 'zed'
            self.mode_history = 'zed'
            node.get_logger().info(
                f'{self.name}: ZED transit fwd={self.forward_m:.1f} '
                f'lat={self.lateral_m:.1f}m'
                + (' (along-track law)' if self.pure_forward else '')
                + f', heading={hd_src}'
                + f' (timeout {self.effective_timeout:.0f}s)')
            return

        # No trustworthy ZED at leg start (dead_reckon mode, or odometry
        # requested but missing/stale) — the whole leg is open-loop.
        self._enter_dr(node, base_s=0.0,
                       reason=('movement_mode dead_reckon' if not use_odom
                               else 'no fresh ZED odometry at leg start'),
                       expected=not use_odom)

    def _enter_dr(self, node, base_s, reason, expected=False):
        """Switch to open-loop timed driving for the REMAINING distance.

        base_s = along-track progress already banked (from ZED, before it went
        stale). The old code had no mid-leg fallback at all: ZED dying mid-leg
        silently truncated the leg where it stood, and every later leg started
        from the wrong place. Once demoted, a leg never re-promotes to 'zed' —
        a recovering ZED that re-anchors mid-leg with a jumped pose would be
        worse than finishing the leg on the clock.
        """
        remaining = max(0.0, self.leg_mag - base_s)
        self.dr_base_s = base_s
        self.dr_duration = remaining / max(self.mps, 1e-3)
        self.dr_start_time = pytime.time()
        self.dr_surge = self.speed * self.body_dir[0]
        self.dr_sway = self.speed * self.body_dir[1]
        if self.mode == 'zed':
            self.mode_history += f'>dr@{base_s:.1f}m'
            self._locked_track = None   # position no longer trusted
        else:
            self.mode_history = 'dr'
        self.mode = 'dr'
        log = node.get_logger().info if expected else node.get_logger().warn
        log(f'{self.name}: open-loop dead reckon — remaining {remaining:.1f}m '
            f'over {self.dr_duration:.1f}s @ {self.mps:.2f}m/s ({reason})')

    def update(self):
        from hightide_interfaces.msg import ThrusterCommand
        node = self.blackboard.get(bb.ROS_NODE)

        # Zero offset → nothing to do.
        if self.mode is None:
            return py_trees.common.Status.SUCCESS
        if (pytime.time() - self.start_time) > self.effective_timeout:
            node.cmd_pub.publish(ThrusterCommand())
            return self._finish(node, 'TIMED OUT — proceeding')

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
                        return self._finish(
                            node, f'{det.class_name} in view — vision hand-off')

        now = pytime.time()
        dt = now - self.last_time
        self.last_time = now

        if self.mode == 'zed':
            status = self._tick_zed(node, dt)
            if status is not None:
                return status
            # status None = ZED went stale THIS tick and _enter_dr already ran;
            # fall through and drive the first open-loop tick immediately.

        # ---- Open-loop dead reckon (from the start, or after a demotion) ----
        if (pytime.time() - self.dr_start_time) >= self.dr_duration:
            node.cmd_pub.publish(ThrusterCommand())
            return self._finish(node, 'dead-reckon leg complete')
        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.surge = self.dr_surge
        cmd.sway = self.dr_sway
        cmd.yaw = self._yaw_cmd(node, dt)
        self._sway_sum += cmd.sway
        self._tick_n += 1
        node.cmd_pub.publish(cmd)
        return py_trees.common.Status.RUNNING

    def _tick_zed(self, node, dt):
        """One closed-loop tick. Returns a Status, or None if ZED just went
        stale and the leg demoted to 'dr' (caller drives the same tick)."""
        from hightide_interfaces.msg import ThrusterCommand
        try:
            current = self.blackboard.get(bb.CURRENT_POSE)
        except KeyError:
            current = None
        yaw = pose_yaw(current)
        if current is None or yaw is None or not odom_is_fresh(node):
            # ZED stale/frozen mid-leg. Finish the remaining distance on the
            # clock instead of truncating the leg where it stood.
            self._enter_dr(node, base_s=self.last_s,
                           reason='ZED odometry went stale/frozen mid-leg')
            return None
        pos = current.pose.pose.position

        rel_x = pos.x - self.anchor[0]
        rel_y = pos.y - self.anchor[1]
        # Progress = displacement projected on the leg's FIXED start axis.
        s = rel_x * self.axis[0] + rel_y * self.axis[1]
        self.last_s = s

        # Overshoot guard: if we've physically moved well past the leg length
        # the target estimate is bad (odometry drift/jump) — stop chasing it.
        traveled = math.hypot(rel_x, rel_y)
        if traveled > 1.5 * self.leg_mag + 0.5:
            node.cmd_pub.publish(ThrusterCommand())
            node.get_logger().warn(
                f'{self.name}: traveled {traveled:.1f}m on a {self.leg_mag:.1f}m '
                'leg — bad odometry? stopping here')
            return self._finish(node, 'overshoot guard')

        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.yaw = self._yaw_cmd(node, dt)

        if self.pure_forward:
            # FOG-primary along-track law: 1-D signed error on the fixed axis
            # (an overshoot drives back, like pidtest's signed e_surge), immune
            # to mid-leg ZED yaw wander bending the trajectory.
            e = self.leg_mag - s
            if abs(e) < self.pos_tol:
                node.cmd_pub.publish(ThrusterCommand())
                return self._finish(node, 'arrived')
            out = self.surge_pid.compute(e, dt)
            cmd.surge = out
            # Cross-track held by the gentle, deadbanded sway_hold PD (NOT the
            # surge-strength position PID) — waves get trimmed out, the surge
            # axis stays owned by the along-track law.
            cmd.sway = sway_hold(node, self.blackboard, self._locked_track)
            # mps autocal sampling: per-tick speed while the PID is pinned at
            # the cap (the only moments the sub is actually driven AT the
            # thrust that dead_reckon_mps is defined for), with a sane-delta
            # gate against ZED jumps. Spin-up samples are collected too and
            # rejected statistically in _maybe_update_mps (75th percentile).
            if (self._prev_s is not None and 0.0 < dt < 0.5
                    and abs(out) >= 0.95 * self.speed):
                ds = s - self._prev_s
                if 0.0 <= ds <= 1.2 * dt:
                    self._cruise_v.append(ds / dt)
            self._prev_s = s
        else:
            # Diagonal legs: the original pidtest world-target PID, verbatim —
            # world error projected onto the body axes with the current yaw.
            dx = self.target[0] - pos.x
            dy = self.target[1] - pos.y
            if math.hypot(dx, dy) < self.pos_tol:
                node.cmd_pub.publish(ThrusterCommand())
                return self._finish(node, 'arrived')
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            e_surge = dx * cos_y + dy * sin_y
            e_sway = -dx * sin_y + dy * cos_y
            cmd.surge = self.surge_pid.compute(e_surge, dt)
            cmd.sway = self.sway_pid.compute(e_sway, dt)

        self._sway_sum += cmd.sway
        self._tick_n += 1
        node.cmd_pub.publish(cmd)
        return py_trees.common.Status.RUNNING

    def _yaw_cmd(self, node, dt):
        """pidtest's companion heading stabilizer, verbatim:
        yaw_error = ref_yaw - imu_heading; cmd.yaw = sign * pid(yaw_error)."""
        from hightide_navigation import normalize_angle
        imu_now = getattr(node, 'current_heading', None)
        if self._locked_heading is None or imu_now is None:
            return 0.0
        yaw_err = normalize_angle(self._locked_heading - imu_now)
        y = (float(getattr(node, 'yaw_hold_sign', -1.0))
             * self.companion_yaw_pid.compute(yaw_err, dt))
        lim = float(getattr(node, 'yaw_hold_limit', 1.0))
        return max(-lim, min(lim, y))

    def _finish(self, node, reason):
        """Every exit path funnels through here: one LEG telemetry line, plus
        the mps autocal update on a clean arrival. One pool run of LEG lines is
        a complete calibration dataset (commanded vs measured distance, times,
        mode demotions, mean lateral trim)."""
        elapsed = pytime.time() - self.start_time
        if self.mode == 'dr' and self.dr_start_time is not None:
            run = min(pytime.time() - self.dr_start_time, self.dr_duration or 0.0)
            est = self.dr_base_s + run * self.mps
        else:
            est = self.last_s
        zed_disp = 'n/a'
        if self.anchor is not None and odom_is_fresh(node):
            try:
                pose = self.blackboard.get(bb.CURRENT_POSE)
            except KeyError:
                pose = None
            if pose is not None:
                p = pose.pose.pose.position
                zed_disp = (f'{math.hypot(p.x - self.anchor[0], p.y - self.anchor[1]):.2f}m')
        sway_avg = self._sway_sum / self._tick_n if self._tick_n else 0.0
        node.get_logger().info(
            f'LEG {self.name}: cmd fwd={self.forward_m:.2f} lat={self.lateral_m:.2f}m | '
            f'progress~{est:.2f}m | zed_disp={zed_disp} | {elapsed:.1f}s | '
            f'mode={self.mode_history} | sway_avg={sway_avg:+.2f} | {reason}')
        if reason == 'arrived':
            self._maybe_update_mps(node)
        return py_trees.common.Status.SUCCESS

    def _maybe_update_mps(self, node):
        """mps autocal: EMA today's measured cruise speed into DEAD_RECKON_MPS.

        Only from clean data: pure-forward legs whose speed cap IS the transit
        thrust that dead_reckon_mps is defined at, with ≥1 s of samples taken
        while the PID was pinned at that cap. The cruise speed is the 75th
        PERCENTILE of the per-tick samples: saturation starts during spin-up
        (those samples read low) and ZED noise scatters both ways, so the
        upper quartile lands on steady cruise without needing a hand-tuned
        spin-up exclusion window. Note the PID only saturates while more than
        speed/kp metres of error remain (0.6/0.2 = 3 m at the defaults), so
        only legs longer than that ever contribute — short legs just don't
        calibrate, which is the safe direction.

        The payoff: when ZED drops out later in the run, the timed fallback
        uses today's battery/trim/depth speed, not a yaml constant measured on
        some other day.
        """
        if not bool(getattr(node, 'mps_autocal_enabled', True)):
            return
        if not self.pure_forward or len(self._cruise_v) < 20:
            return
        ref = getattr(node, 'transit_thrust', None)
        if ref is None or abs(self.speed - float(ref)) > 0.02:
            return
        v = sorted(self._cruise_v)[int(0.75 * len(self._cruise_v))]
        if not (0.05 <= v <= 1.5):
            return
        alpha = float(getattr(node, 'mps_autocal_alpha', 0.3))
        old = read_dead_reckon_mps(self.blackboard)
        new = old + alpha * (v - old)
        self.blackboard.set(bb.DEAD_RECKON_MPS, new)
        node.get_logger().info(
            f'{self.name}: mps autocal — {len(self._cruise_v)} cruise samples, '
            f'p75 {v:.2f}m/s; dead_reckon_mps {old:.2f} -> {new:.2f}')


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