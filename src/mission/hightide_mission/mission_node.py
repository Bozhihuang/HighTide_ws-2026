#!/usr/bin/env python3
"""
Mission Node — Root behavior tree that orchestrates all competition tasks.

This is the master brain of the hightide AUV. It builds a py_trees behavior
tree with the following structure:

  Root (Parallel/Selector):
  ├── Safety Guard (Sequence):
  │   ├── CheckMissionTimeout
  │   └── EmergencySurface (triggered on timeout)
  └── Main Mission (Sequence):
      ├── PreDive (Arm → AltHold → Submerge → Stabilize)
      ├── Task1_Gate
      ├── Task2_Slalom
      ├── Task3_Bins
      ├── Task4_Torpedoes
      ├── Task5_Octagon
      ├── Task6_ReturnHome
      └── StylePoints (barrel roll)

The node also manages ROS2 subscriptions to keep the blackboard updated
with fresh sensor data every tick.
"""

import time as pytime
import py_trees
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from mavros_msgs.msg import State
from hightide_interfaces.msg import (
    ThrusterCommand, DetectionArray, MissionState, NavigationTier)

from hightide_mission.behaviors import blackboard_keys as bb
from hightide_mission.behaviors.pre_dive import (
    ArmVehicle, SetAltHoldMode, SubmergeToDepth, WaitForStable)
from hightide_mission.behaviors.gate import create_gate_subtree
from hightide_mission.behaviors.slalom import create_slalom_subtree
from hightide_mission.behaviors.bins import create_bins_subtree
from hightide_mission.behaviors.torpedoes import create_torpedoes_subtree
from hightide_mission.behaviors.octagon import create_octagon_subtree
from hightide_mission.behaviors.return_home import create_return_home_subtree
from hightide_mission.behaviors.emergency import CheckMissionTimeout, EmergencySurface
from hightide_mission.behaviors.common import LogBehavior, WaitForDuration


class MissionNode(Node):
    """Root ROS2 node that runs the py_trees behavior tree."""

    def __init__(self):
        super().__init__('mission_node')

        # Parameters
        self.declare_parameter('mission_depth_m', 1.0)
        self.declare_parameter('mission_timeout_sec', 900.0)  # 15 minutes
        self.declare_parameter('tick_rate', 10.0)
        self.declare_parameter('skip_tasks', '')  # Comma-separated task names to skip

        self.mission_depth = self.get_parameter('mission_depth_m').value
        self.mission_timeout = self.get_parameter('mission_timeout_sec').value
        self.tick_rate = self.get_parameter('tick_rate').value

        # Publishers
        self.cmd_pub = self.create_publisher(ThrusterCommand, '/hightide/cmd_vel', 10)
        self.depth_pub = self.create_publisher(Float64, '/hightide/target_depth', 10)
        self.state_pub = self.create_publisher(MissionState, '/hightide/mission_state', 10)

        # Subscribers — update blackboard
        self.create_subscription(
            DetectionArray, '/hightide/tracked_targets',
            self._detections_cb, 10)
        self.create_subscription(
            Odometry, '/hightide/odometry/filtered',
            self._odom_cb, 10)
        self.create_subscription(
            Float64, '/mavros/global_position/rel_alt',
            self._depth_cb, 10)
        self.create_subscription(
            Imu, '/mavros/imu/data',
            self._imu_cb, 10)
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

        # Initialize blackboard
        self.blackboard.set(bb.ROS_NODE, self)
        self.blackboard.set(bb.MISSION_START_TIME, pytime.time())
        self.blackboard.set(bb.MISSION_TIMEOUT, self.mission_timeout)
        self.blackboard.set(bb.CHOSEN_ROLE, 'survey_repair')
        self.blackboard.set(bb.CURRENT_TASK, MissionState.IDLE)
        self.blackboard.set(bb.MARKERS_DROPPED, 0)
        self.blackboard.set(bb.TORPEDOES_FIRED, 0)
        self.blackboard.set(bb.OBJECTS_COLLECTED, 0)
        self.blackboard.set(bb.CURRENT_DEPTH, 0.0)
        self.blackboard.set(bb.CURRENT_HEADING, 0.0)
        self.blackboard.set(bb.VEHICLE_ARMED, False)
        self.blackboard.set(bb.VEHICLE_MODE, '')

        # Additional attributes for behaviors to use
        self.vision_servo_class = ''

        # Build behavior tree
        self.tree = self._build_tree()

        # Tick timer
        self.tick_timer = self.create_timer(
            1.0 / self.tick_rate, self._tick)

        self.get_logger().info('=== MISSION NODE STARTED ===')
        self.get_logger().info(f'Mission depth: {self.mission_depth}m')
        self.get_logger().info(f'Timeout: {self.mission_timeout}s')

    def _build_tree(self) -> py_trees.trees.BehaviourTree:
        """Build the full competition behavior tree."""

        # Pre-dive sequence
        pre_dive = py_trees.composites.Sequence(
            name='PreDive',
            memory=True,
            children=[
                LogBehavior('Mission_Start', '=== MISSION BEGINNING ==='),
                ArmVehicle('Arm'),
                SetAltHoldMode('SetAltHold'),
                SubmergeToDepth('Submerge', depth_m=self.mission_depth),
                WaitForStable('Stabilize'),
                LogBehavior('PreDive_Complete', 'Pre-dive complete — starting tasks'),
            ],
        )

        # Competition tasks in order
        tasks = py_trees.composites.Sequence(
            name='CompetitionTasks',
            memory=True,
            children=[
                create_gate_subtree(),
                create_slalom_subtree(),
                create_bins_subtree(),
                create_torpedoes_subtree(),
                create_octagon_subtree(),
                create_return_home_subtree(),
            ],
        )

        # Main mission sequence
        main_mission = py_trees.composites.Sequence(
            name='MainMission',
            memory=True,
            children=[
                pre_dive,
                tasks,
                LogBehavior('Mission_Complete', '=== ALL TASKS COMPLETED ==='),
            ],
        )

        # Safety guard — runs in parallel, emergency surfaces on timeout
        safety = py_trees.composites.Sequence(
            name='SafetyGuard',
            memory=False,
            children=[
                CheckMissionTimeout('TimeoutCheck'),
            ],
        )

        # Root: Safety runs alongside main mission
        root = py_trees.composites.Parallel(
            name='MissionRoot',
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=[
                safety,
                main_mission,
            ],
        )

        tree = py_trees.trees.BehaviourTree(root=root)
        return tree

    def _tick(self):
        """Tick the behavior tree once."""
        try:
            self.tree.tick()
        except Exception as e:
            self.get_logger().error(f'Behavior tree tick error: {e}')

        # Publish mission state
        self._publish_mission_state()

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
            self.blackboard.set(bb.CURRENT_HEADING, quaternion_to_yaw(msg.orientation))
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
    except KeyboardInterrupt:
        pass
    finally:
        # Emergency: stop all motors
        cmd = ThrusterCommand()
        node.cmd_pub.publish(cmd)
        node.get_logger().info('Mission node shutdown — motors stopped')
        node.destroy_node()

if __name__ == '__main__':
    main()
