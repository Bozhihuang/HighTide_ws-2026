#!/usr/bin/env python3
"""
Mission Node — Root behavior tree that orchestrates all competition tasks.

This is the master brain of the hightide AUV. It builds a py_trees behavior
tree with the following structure:

  Root (Sequence "MainMission"):
  ├── PreDive (Arm → AltHold → Submerge → Stabilize)
  ├── Task1_Gate
  ├── Task2_Slalom
  ├── Task3_Bins
  ├── Task4_Torpedoes
  ├── Task5_Octagon
  ├── Task6_ReturnHome
  └── StylePoints (barrel roll)

Safety is enforced at the NODE level, not inside the tree: a behavior-tree
parallel cannot cleanly halt the mission branch while also driving an
emergency maneuver (both branches keep publishing conflicting commands). So
every tick the node first checks the mission timeout; on timeout — or when
the tree finishes — it stops ticking the tree entirely and runs a
surface-then-disarm shutdown sequence itself. Once disarmed it calls
rclpy.shutdown(), which makes the node exit; full_system.launch.py's
OnProcessExit handler then tears down the rest of the stack.

The node also manages ROS2 subscriptions to keep the blackboard updated
with fresh sensor data every tick.
"""

import time as pytime
import py_trees
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float64
from std_srvs.srv import SetBool
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from mavros_msgs.msg import State
from hightide_interfaces.msg import (
    ThrusterCommand, DetectionArray, MissionState, NavigationTier)

from hightide_mission.behaviors import blackboard_keys as bb
from hightide_mission.behaviors.pre_dive import (
    ArmVehicle, SetAltHoldMode, SubmergeToDepth, WaitForStable)
from hightide_mission.behaviors.gate import create_gate_subtree, HeadingTurn
from hightide_mission.behaviors.slalom import create_slalom_subtree
from hightide_mission.behaviors.bins import create_bins_subtree
from hightide_mission.behaviors.torpedoes import create_torpedoes_subtree
from hightide_mission.behaviors.octagon import create_octagon_subtree
from hightide_mission.behaviors.return_home import create_return_home_subtree
from hightide_mission.behaviors.common import (
    LogBehavior, CallTriggerService, DeadReckonTransit)


class MissionNode(Node):
    """Root ROS2 node that runs the py_trees behavior tree."""

    def __init__(self):
        super().__init__('mission_node')

        # Parameters
        self.declare_parameter('mission_depth_m', 1.0)
        self.declare_parameter('mission_timeout_sec', 900.0)  # 15 minutes
        self.declare_parameter('tick_rate', 10.0)
        self.declare_parameter('skip_tasks', '')  # Comma-separated task names to skip
        # Pre-race role choice (our coin-flip decision), NOT read off the gate:
        # 'survey_repair' or 'search_rescue'.
        self.declare_parameter('chosen_role', 'survey_repair')

        # Per-mission time budgets (seconds). Each subtree factory splits its
        # budget across its own timeout-bearing behaviors, preserving their tuned
        # ratios (see behaviors.common.distribute_timeout). NOTE: the sum of the
        # six budgets (default 6*240 = 1440 s) intentionally exceeds
        # mission_timeout_sec (900 s) — the global cap wins, so later tasks are
        # squeezed if earlier ones run long. Fixed open-loop motions (surge/pass/
        # settle) are NOT part of these budgets and never scale.
        self.declare_parameter('gate_timeout_sec', 240.0)
        self.declare_parameter('slalom_timeout_sec', 240.0)
        self.declare_parameter('bins_timeout_sec', 240.0)
        self.declare_parameter('torpedoes_timeout_sec', 240.0)
        self.declare_parameter('octagon_timeout_sec', 240.0)
        self.declare_parameter('return_home_timeout_sec', 240.0)

        # Slalom (Task 2): if the poles are never visually acquired, blind-drive
        # this far straight through the slalom instead of skipping it (skipping
        # strands the sub on the near side and wrecks later transit distances).
        self.declare_parameter('slalom_fallback_forward_m', 5.0)

        # Octagon (Task 5) tuning knobs — surfaced so they can be tuned in-water
        # via `ros2 param set` without editing code.
        self.declare_parameter('octagon_advance_distance_m', 3.0)  # blind odom advance
        self.declare_parameter('octagon_surge', 0.3)
        self.declare_parameter('octagon_buoy_fill_frac', 0.7)      # buoy box fill => inside
        self.declare_parameter('octagon_table_fill_frac', 0.6)     # table box fill => inside
        self.declare_parameter('octagon_table_class', 'octagon_table')
        self.declare_parameter('octagon_confidence', 0.4)          # min det confidence
        self.declare_parameter('octagon_settle_sec', 2.0)          # fixed settle (not scaled)
        self.declare_parameter('octagon_surface_depth_m', 0.3)     # "at surface" threshold

        # ---- Course selection + inter-mission dead-reckon transits ----
        # Courses A and D are DIFFERENT layouts (not mirror images), so each has
        # its OWN transit distances. `course` selects which set is used at run
        # time; the other set is ignored.
        # Movement mode for the odometry/dead-reckon behaviors (transits, the
        # bin/octagon "advance N metres" legs). 'zed' = measure travel with ZED
        # odometry (accurate, closed-loop); 'dead_reckon' = open-loop timed drive
        # (distance / dead_reckon_mps) for when ZED odometry is down/unreliable.
        # Opening maneuver, right after submerge: turn a set amount, then advance.
        # initial_turn_deg = 0 skips the turn; initial_forward_m = 0 skips the
        # advance. The forward move uses the movement_mode (ZED / dead-reckon).
        self.declare_parameter('initial_turn_deg', 0.0)            # compass-style: + = CLOCKWISE. 90/180/270 (0 = none)
        self.declare_parameter('initial_forward_m', 2.0)          # advance after the turn
        self.declare_parameter('initial_turn_timeout_sec', 15.0)  # give 270° enough time

        # Heading-hold PD gains + clamp + sign used by EVERY translating
        # behavior (common.yaw_hold reads these off the node). Defaults are the
        # pool-proven pidtest companion values (kp 0.45 / kd 0.2). Live-tunable
        # in-water:  ros2 param set /mission_node yaw_hold_kp 0.3
        # P-only or too-high kp makes the sub WEAVE/crab instead of driving
        # straight (observed in pool with the old hardcoded P-only 1.5).
        # yaw_hold_sign flips the correction direction (pidtest's
        # yaw_command_sign equivalent); -1.0 is the validated convention.
        self.declare_parameter('yaw_hold_kp', 0.45)
        self.declare_parameter('yaw_hold_kd', 0.2)
        self.declare_parameter('yaw_hold_limit', 0.25)
        self.declare_parameter('yaw_hold_sign', -1.0)

        self.declare_parameter('movement_mode', 'zed')             # 'zed' or 'dead_reckon'
        # transit_thrust = the POWER (normalized -1..1 cmd) the transit legs drive
        # at, in BOTH modes. dead_reckon_mps = the actual speed the vehicle moves
        # AT that thrust (measure it in-pool) — used only in dead_reckon mode to
        # turn each leg's distance into a drive TIME (time = distance / mps). The
        # two are coupled: re-measure mps whenever you change transit_thrust.
        self.declare_parameter('transit_thrust', 0.35)             # drive power CAP for transits (PID clamp)
        self.declare_parameter('dead_reckon_mps', 0.4)             # measured speed (m/s) at transit_thrust
        # Position-PID gains for ZED-mode transits/advance — SET THESE TO THE
        # SURGE GAINS YOU TUNED IN THE POOL pidtest so mission moves match it.
        self.declare_parameter('transit_kp', 0.2)
        self.declare_parameter('transit_ki', 0.0)
        self.declare_parameter('transit_kd', 0.05)

        self.declare_parameter('course', 'A')                      # 'A' or 'D'
        # Props are far apart, so between tasks we blind-drive a fixed body-frame
        # (forward, lateral) offset on ZED odometry, THEN let the next task's
        # visual search acquire. Metres; lateral is +right. 0.0 = no transit
        # (pure creep-and-search, the old behavior). MEASURE PER COURSE IN POOL.
        for course in ('A', 'D'):
            for leg in ('to_slalom', 'to_bins', 'to_torpedoes', 'to_octagon'):
                self.declare_parameter(f'transit_{course}_{leg}_forward_m', 0.0)
                self.declare_parameter(f'transit_{course}_{leg}_lateral_m', 0.0)

        self.mission_depth = self.get_parameter('mission_depth_m').value
        self.mission_timeout = self.get_parameter('mission_timeout_sec').value
        self.tick_rate = self.get_parameter('tick_rate').value
        self.chosen_role = self.get_parameter('chosen_role').value

        # Per-mission budgets
        self.gate_timeout = self.get_parameter('gate_timeout_sec').value
        self.slalom_timeout = self.get_parameter('slalom_timeout_sec').value
        self.slalom_fallback_forward_m = float(
            self.get_parameter('slalom_fallback_forward_m').value)
        self.bins_timeout = self.get_parameter('bins_timeout_sec').value
        self.torpedoes_timeout = self.get_parameter('torpedoes_timeout_sec').value
        self.octagon_timeout = self.get_parameter('octagon_timeout_sec').value
        self.return_home_timeout = self.get_parameter('return_home_timeout_sec').value

        # Opening maneuver
        self.initial_turn_deg = float(self.get_parameter('initial_turn_deg').value)
        self.initial_forward_m = float(self.get_parameter('initial_forward_m').value)
        self.initial_turn_timeout = float(self.get_parameter('initial_turn_timeout_sec').value)

        # Heading-hold gains as plain attributes — common.yaw_hold reads them
        # via getattr every tick, so the set-parameters callback below makes
        # `ros2 param set` take effect immediately (no restart).
        self.yaw_hold_kp = float(self.get_parameter('yaw_hold_kp').value)
        self.yaw_hold_kd = float(self.get_parameter('yaw_hold_kd').value)
        self.yaw_hold_limit = float(self.get_parameter('yaw_hold_limit').value)
        self.yaw_hold_sign = float(self.get_parameter('yaw_hold_sign').value)
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # Movement mode → blackboard flag the nav behaviors read.
        mode = str(self.get_parameter('movement_mode').value).lower()
        self.use_odometry = (mode != 'dead_reckon')   # anything but dead_reckon = zed
        self.dead_reckon_mps = float(self.get_parameter('dead_reckon_mps').value)
        self.transit_thrust = float(self.get_parameter('transit_thrust').value)
        self.transit_kp = float(self.get_parameter('transit_kp').value)
        self.transit_ki = float(self.get_parameter('transit_ki').value)
        self.transit_kd = float(self.get_parameter('transit_kd').value)

        # Selected course (A/D) — picks which transit distance set is used.
        self.course = str(self.get_parameter('course').value).upper()
        if self.course not in ('A', 'D'):
            self.get_logger().warn(f"Unknown course '{self.course}' — defaulting to A")
            self.course = 'A'

        # Octagon knobs
        self.octagon_params = dict(
            advance_distance_m=self.get_parameter('octagon_advance_distance_m').value,
            surge=self.get_parameter('octagon_surge').value,
            buoy_fill_frac=self.get_parameter('octagon_buoy_fill_frac').value,
            table_fill_frac=self.get_parameter('octagon_table_fill_frac').value,
            table_class=self.get_parameter('octagon_table_class').value,
            confidence=self.get_parameter('octagon_confidence').value,
            settle_sec=self.get_parameter('octagon_settle_sec').value,
            surface_depth_m=self.get_parameter('octagon_surface_depth_m').value,
        )

        # Publishers
        self.cmd_pub = self.create_publisher(ThrusterCommand, '/hightide/cmd_vel', 10)
        self.depth_pub = self.create_publisher(Float64, '/hightide/target_depth', 10)
        self.state_pub = self.create_publisher(MissionState, '/hightide/mission_state', 10)

        # MAVROS sensor topics (depth, IMU, odometry) are published BEST_EFFORT.
        # A default (RELIABLE) subscriber is QoS-incompatible with them and
        # receives NOTHING — which silently pinned CURRENT_DEPTH at 0.0 and made
        # SubmergeToDepth time out forever. Match the publisher with sensor QoS.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscribers — update blackboard
        self.create_subscription(
            DetectionArray, '/hightide/tracked_targets',
            self._detections_cb, 10)
        self.create_subscription(
            Odometry, '/mavros/zed/odom',
            self._odom_cb, sensor_qos)
        self.create_subscription(
            Float64, '/mavros/global_position/rel_alt',
            self._depth_cb, sensor_qos)
        self.create_subscription(
            Imu, '/mavros/imu/data',
            self._imu_cb, sensor_qos)
        self.create_subscription(
            State, '/mavros/state',
            self._mavros_state_cb, 10)
        self.create_subscription(
            NavigationTier, '/hightide/navigation_tier',
            self._nav_tier_cb, 10)

        # Blackboard setup
        self.blackboard = py_trees.blackboard.Client(name='MissionNode')
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.CURRENT_DEPTH, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.CURRENT_HEADING, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.CURRENT_POSE, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.VEHICLE_ARMED, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.VEHICLE_MODE, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.NAV_TIER, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.MISSION_START_TIME, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.MISSION_TIMEOUT, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.CHOSEN_ROLE, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.CURRENT_TASK, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.MARKERS_DROPPED, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.TORPEDOES_FIRED, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.OBJECTS_COLLECTED, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.GATE_POSITION, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.GATE_DIVIDER_SIDE, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.USE_ODOMETRY, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.DEAD_RECKON_MPS, access=py_trees.common.Access.WRITE)

        # Initialize blackboard
        self.mission_start_time = pytime.time()
        self.blackboard.set(bb.ROS_NODE, self)
        self.blackboard.set(bb.MISSION_START_TIME, self.mission_start_time)
        self.blackboard.set(bb.MISSION_TIMEOUT, self.mission_timeout)
        self.blackboard.set(bb.CHOSEN_ROLE, self.chosen_role)
        self.blackboard.set(bb.CURRENT_TASK, MissionState.IDLE)
        self.blackboard.set(bb.MARKERS_DROPPED, 0)
        self.blackboard.set(bb.TORPEDOES_FIRED, 0)
        self.blackboard.set(bb.OBJECTS_COLLECTED, 0)
        self.blackboard.set(bb.GATE_POSITION, None)
        self.blackboard.set(bb.GATE_DIVIDER_SIDE, 'right')
        self.blackboard.set(bb.USE_ODOMETRY, self.use_odometry)
        self.blackboard.set(bb.DEAD_RECKON_MPS, self.dead_reckon_mps)
        self.blackboard.set(bb.CURRENT_DEPTH, 0.0)
        self.blackboard.set(bb.CURRENT_HEADING, 0.0)
        self.blackboard.set(bb.VEHICLE_ARMED, False)
        self.blackboard.set(bb.VEHICLE_MODE, '')

        # Live FOG/IMU heading (radians), updated by _imu_cb. Behaviors read
        # this for active yaw-hold. None until the first IMU message arrives.
        self.current_heading = None

        # Safe-shutdown state machine: None → 'surfacing' → 'disarming' → 'done'
        self._shutdown_state = None
        self._shutdown_start = None
        self._disarm_client = None
        self._disarm_future = None

        # Build behavior tree
        self.tree = self._build_tree()

        # Tick timer
        self.tick_timer = self.create_timer(
            1.0 / self.tick_rate, self._tick)

        self.get_logger().info('=== MISSION NODE STARTED ===')
        self.get_logger().info(f'Chosen role: {self.chosen_role}')
        self.get_logger().info(f'Mission depth: {self.mission_depth}m')
        self.get_logger().info(f'Timeout: {self.mission_timeout}s')
        self.get_logger().info(
            f'Movement mode: {"ZED odometry" if self.use_odometry else "DEAD RECKON"}'
            + ('' if self.use_odometry else f' ({self.dead_reckon_mps} m/s)')
            + f' | Course: {self.course}')
        self.get_logger().info(
            'Opening maneuver: '
            + (f'turn {abs(self.initial_turn_deg):.0f}deg '
               + ('CW' if self.initial_turn_deg > 0 else 'CCW')
               if abs(self.initial_turn_deg) > 0.01 else 'no turn')
            + ' + '
            + (f'advance {self.initial_forward_m:.1f}m'
               if self.initial_forward_m > 0.01 else 'no advance'))

    def _on_set_parameters(self, params):
        """Apply live `ros2 param set` updates for the in-water tuning knobs."""
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name in ('yaw_hold_kp', 'yaw_hold_kd',
                          'yaw_hold_limit', 'yaw_hold_sign'):
                setattr(self, p.name, float(p.value))
                self.get_logger().info(f'{p.name} -> {float(p.value)}')
        return SetParametersResult(successful=True)

    def _build_tree(self) -> py_trees.trees.BehaviourTree:
        """Build the full competition behavior tree."""

        # Pre-dive sequence: arm, dive, stabilize, then the opening maneuver
        # (turn a set amount + advance) before the tasks begin.
        predive_children = [
            LogBehavior('Mission_Start', '=== MISSION BEGINNING ==='),
            ArmVehicle('Arm'),
            SetAltHoldMode('SetAltHold'),
            SubmergeToDepth('Submerge', depth_m=self.mission_depth),
            WaitForStable('Stabilize'),
        ]
        if abs(self.initial_turn_deg) > 0.01:
            # The param is compass-style (positive = clockwise, what a driver
            # expects); HeadingTurn is ENU (positive = counterclockwise) — negate.
            predive_children.append(
                HeadingTurn('InitialTurn', degrees=-self.initial_turn_deg,
                            tolerance=3.0, timeout=self.initial_turn_timeout))
        if self.initial_forward_m > 0.01:
            predive_children.append(
                DeadReckonTransit('InitialAdvance', forward_m=self.initial_forward_m,
                                  speed=self.transit_thrust,
                                  kp=self.transit_kp, ki=self.transit_ki,
                                  kd=self.transit_kd))
        predive_children.append(
            LogBehavior('PreDive_Complete', 'Pre-dive complete — starting tasks'))
        pre_dive = py_trees.composites.Sequence(
            name='PreDive', memory=True, children=predive_children)

        # Competition tasks in order. Each is wrapped so that a FAILURE (e.g. a
        # prop never detected) is converted to SUCCESS — the sequence then moves
        # on to the next task instead of aborting the whole run. RUNNING still
        # passes through, so a task keeps ticking until it finishes or times out.
        def resilient(subtree):
            return py_trees.decorators.FailureIsSuccess(
                name=f'Try_{subtree.name}', child=subtree)

        # A blind dead-reckon leg to cross the open water into the NEXT prop's
        # vicinity before that task's visual search runs. Distances come from
        # ROS params (0.0 = disabled); course A/D mirrors the lateral sign.
        # The prop class(es) that end each transit early — the moment the next
        # task's target is in view, the transit bails and hands off to vision
        # (the preset distance is only a ceiling). Class names per the ffc model.
        transit_targets = {
            'to_slalom': {'slalom'},
            'to_bins': {'fire', 'blood'},
            'to_torpedoes': {'fire', 'blood'},
            'to_octagon': {'buoy', 'octagon_table'},
        }

        def transit(name, leg):
            fwd = self.get_parameter(f'transit_{self.course}_{leg}_forward_m').value
            lat = self.get_parameter(f'transit_{self.course}_{leg}_lateral_m').value
            return DeadReckonTransit(name, forward_m=fwd, lateral_m=lat,
                                     speed=self.transit_thrust,
                                     target_classes=transit_targets.get(leg),
                                     kp=self.transit_kp, ki=self.transit_ki,
                                     kd=self.transit_kd)

        tasks = py_trees.composites.Sequence(
            name='CompetitionTasks',
            memory=True,
            children=[
                resilient(create_gate_subtree(total_timeout=self.gate_timeout)),
                transit('Transit_Slalom', 'to_slalom'),
                resilient(create_slalom_subtree(
                    total_timeout=self.slalom_timeout,
                    fallback_forward_m=self.slalom_fallback_forward_m)),
                transit('Transit_Bins', 'to_bins'),
                resilient(create_bins_subtree(total_timeout=self.bins_timeout)),
                transit('Transit_Torpedoes', 'to_torpedoes'),
                resilient(create_torpedoes_subtree(total_timeout=self.torpedoes_timeout)),
                transit('Transit_Octagon', 'to_octagon'),
                resilient(create_octagon_subtree(total_timeout=self.octagon_timeout,
                                                 **self.octagon_params)),
                # Octagon → gate is handled by Return Home's own dead-reckon.
                resilient(create_return_home_subtree(total_timeout=self.return_home_timeout)),
            ],
        )

        # Barrel roll as the final style maneuver. This is LAST on purpose — it
        # switches to MANUAL and destroys the FOG heading reference, so it must
        # run only after every heading-dependent task is complete.
        style_finale = CallTriggerService('BarrelRollFinale', '/hightide/barrel_roll')

        # Main mission sequence — this IS the root. Timeout/emergency handling
        # lives in _tick(), not in the tree (see module docstring).
        main_mission = py_trees.composites.Sequence(
            name='MainMission',
            memory=True,
            children=[
                pre_dive,
                tasks,
                LogBehavior('Style_Finale', 'Executing barrel roll finale'),
                style_finale,
                LogBehavior('Mission_Complete', '=== ALL TASKS COMPLETED ==='),
            ],
        )

        tree = py_trees.trees.BehaviourTree(root=main_mission)
        return tree

    def _tick(self):
        """Tick the behavior tree once, with node-level safety supervision."""
        # Once a shutdown has begun the tree never ticks again — a finished or
        # timed-out memory-sequence would otherwise reset and re-run the whole
        # mission (re-arm, re-submerge) on the next tick.
        if self._shutdown_state is not None:
            self._run_safe_shutdown()
            self._publish_mission_state()
            return

        if (pytime.time() - self.mission_start_time) > self.mission_timeout:
            self._begin_shutdown('MISSION TIMEOUT — emergency surface + disarm',
                                 error=True)
            self._run_safe_shutdown()
            self._publish_mission_state()
            return

        try:
            self.tree.tick()
        except Exception as e:
            self.get_logger().error(f'Behavior tree tick error: {e}')

        if self.tree.root.status == py_trees.common.Status.SUCCESS:
            self._begin_shutdown('Mission complete — surfacing and disarming')
        elif self.tree.root.status == py_trees.common.Status.FAILURE:
            # Only pre-dive can fail the root (tasks are FailureIsSuccess).
            # Let the tree restart and retry arming/diving; the mission
            # timeout above bounds how long this can go on.
            self.get_logger().warn('Mission tree FAILED (pre-dive) — retrying')

        self._log_health()

        # Publish mission state
        self._publish_mission_state()

    def _log_health(self):
        """Throttled warnings when a motion-critical input is missing, so a
        'the sub just sits there' is diagnosable straight from the mission log.
        None of these block motion by themselves — they explain WHY a behavior
        might not be commanding surge/sway/yaw."""
        now = pytime.time()
        if now - getattr(self, '_last_health_log', 0.0) < 3.0:
            return
        self._last_health_log = now
        issues = []
        if self.current_heading is None:
            issues.append('no IMU heading (yaw-hold + turns disabled)')
        try:
            if self.blackboard.get(bb.CURRENT_POSE) is None:
                issues.append('no ZED odometry (ZED-mode moves fall back to timed)')
        except KeyError:
            issues.append('no ZED odometry (ZED-mode moves fall back to timed)')
        try:
            det = self.blackboard.get(bb.DETECTIONS)
            if det is None or len(det.detections) == 0:
                issues.append('no detections (vision behaviors blind)')
        except KeyError:
            issues.append('no detections (vision behaviors blind)')
        if issues:
            self.get_logger().warn('HEALTH: ' + '; '.join(issues))

    # ==================== Safe shutdown (complete / timeout) ====================

    def _begin_shutdown(self, reason, error=False):
        """Latch the mission over; _tick only runs the shutdown sequence now."""
        self._shutdown_state = 'surfacing'
        self._shutdown_start = pytime.time()
        log = self.get_logger().error if error else self.get_logger().info
        log(f'=== {reason} ===')

    def _run_safe_shutdown(self):
        """One tick of the surface-then-disarm sequence."""
        # Kill lateral/forward motion and command the surface every tick.
        self.cmd_pub.publish(ThrusterCommand())
        surface = Float64()
        surface.data = 0.0
        self.depth_pub.publish(surface)

        elapsed = pytime.time() - self._shutdown_start

        if self._shutdown_state == 'surfacing':
            try:
                depth = self.blackboard.get(bb.CURRENT_DEPTH)
            except KeyError:
                depth = None
            # 30 s is a generous ascent budget from mission depth; if depth
            # feedback is dead we disarm anyway rather than spin forever.
            if (depth is not None and depth < 0.3) or elapsed > 30.0:
                self.get_logger().info('At surface — disarming')
                self._disarm_client = self.create_client(SetBool, '/hightide/arm')
                req = SetBool.Request()
                req.data = False
                self._disarm_future = self._disarm_client.call_async(req)
                self._shutdown_state = 'disarming'
                self._shutdown_start = pytime.time()

        elif self._shutdown_state == 'disarming':
            if self._disarm_future.done() or elapsed > 10.0:
                if self._disarm_future.done():
                    self.get_logger().info('Disarmed — mission node exiting')
                else:
                    self.get_logger().warn(
                        'Disarm service did not respond — exiting anyway')
                self._shutdown_state = 'done'
                # Exiting the node triggers full_system.launch.py's
                # OnProcessExit → the whole stack shuts down.
                self.tick_timer.cancel()
                rclpy.shutdown()

    def _publish_mission_state(self):
        """Publish current mission state for monitoring."""
        msg = MissionState()
        try:
            msg.current_task = self.blackboard.get(bb.CURRENT_TASK)
        except (KeyError, TypeError):
            msg.current_task = MissionState.IDLE

        # Get current behavior name from tree tip
        tip = self.tree.root.tip()
        msg.current_behavior = tip.name if tip else 'idle'
        msg.status_message = str(tip.status) if tip else 'IDLE'
        if self._shutdown_state is not None:
            msg.current_behavior = f'shutdown:{self._shutdown_state}'
        msg.task_start_time = self.get_clock().now().to_msg()

        try:
            msg.chosen_role = self.blackboard.get(bb.CHOSEN_ROLE)
        except KeyError:
            msg.chosen_role = 'unknown'

        self.state_pub.publish(msg)

    # ==================== Blackboard update callbacks ====================

    def _detections_cb(self, msg):
        self.blackboard.set(bb.DETECTIONS, msg)

    def _odom_cb(self, msg):
        self.blackboard.set(bb.CURRENT_POSE, msg)

    def _depth_cb(self, msg):
        self.blackboard.set(bb.CURRENT_DEPTH, -msg.data)  # Positive = deeper

    def _imu_cb(self, msg):
        from hightide_navigation import quaternion_to_yaw
        try:
            yaw = quaternion_to_yaw(msg.orientation)
            self.blackboard.set(bb.CURRENT_HEADING, yaw)
            # Mirror onto a plain attribute so translating behaviors can read the
            # live FOG/IMU heading for active yaw-hold without registering a
            # blackboard key. This is the single heading source of truth (IMU,
            # NOT ZED) used everywhere for heading.
            self.current_heading = yaw
        except Exception:
            pass

    def _mavros_state_cb(self, msg):
        self.blackboard.set(bb.VEHICLE_ARMED, msg.armed)
        self.blackboard.set(bb.VEHICLE_MODE, msg.mode)

    def _nav_tier_cb(self, msg):
        self.blackboard.set(bb.NAV_TIER, msg.current_tier)


def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Emergency: stop all motors (context may already be down on clean exit)
        try:
            node.cmd_pub.publish(ThrusterCommand())
            node.get_logger().info('Mission node shutdown — motors stopped')
        except Exception:
            pass
        node.destroy_node()

if __name__ == '__main__':
    main()