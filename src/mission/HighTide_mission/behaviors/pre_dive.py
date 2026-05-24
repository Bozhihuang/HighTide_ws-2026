"""
Pre-dive behaviors: Arm vehicle, set Alt Hold mode, submerge to depth.
"""

import time as pytime
import py_trees
from std_msgs.msg import Float64
from std_srvs.srv import SetBool, Trigger
from . import blackboard_keys as bb


class ArmVehicle(py_trees.behaviour.Behaviour):
    """Arm the vehicle via /HighTide/arm service."""

    def __init__(self, name='ArmVehicle'):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)
        self.client = None
        self.future = None

    def initialise(self):
        node = self.blackboard.get(bb.ROS_NODE)
        self.client = node.create_client(SetBool, '/HighTide/arm')

    def update(self):
        node = self.blackboard.get(bb.ROS_NODE)

        if not self.client.wait_for_service(timeout_sec=1.0):
            node.get_logger().warn('Arm service not available')
            return py_trees.common.Status.RUNNING

        if self.future is None:
            req = SetBool.Request()
            req.data = True
            self.future = self.client.call_async(req)
            return py_trees.common.Status.RUNNING

        if not self.future.done():
            return py_trees.common.Status.RUNNING

        result = self.future.result()
        if result and result.success:
            node.get_logger().info('Vehicle ARMED')
            return py_trees.common.Status.SUCCESS
        else:
            node.get_logger().error('Failed to arm')
            return py_trees.common.Status.FAILURE


class SetAltHoldMode(py_trees.behaviour.Behaviour):
    """Set Alt Hold flight mode via /HighTide/set_alt_hold service."""

    def __init__(self, name='SetAltHoldMode'):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)
        self.client = None
        self.future = None

    def initialise(self):
        node = self.blackboard.get(bb.ROS_NODE)
        self.client = node.create_client(Trigger, '/HighTide/set_alt_hold')

    def update(self):
        node = self.blackboard.get(bb.ROS_NODE)

        if not self.client.wait_for_service(timeout_sec=1.0):
            return py_trees.common.Status.RUNNING

        if self.future is None:
            self.future = self.client.call_async(Trigger.Request())
            return py_trees.common.Status.RUNNING

        if not self.future.done():
            return py_trees.common.Status.RUNNING

        result = self.future.result()
        if result and result.success:
            node.get_logger().info('Alt Hold mode SET')
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class SubmergeToDepth(py_trees.behaviour.Behaviour):
    """
    Publish target depth and wait until vehicle reaches it.
    In Alt Hold: publishes to /HighTide/target_depth which the depth
    controller uses to adjust throttle.
    """

    def __init__(self, name='SubmergeToDepth', depth_m=1.0, tolerance=0.2):
        super().__init__(name)
        self.target_depth = depth_m
        self.tolerance = tolerance
        self.start_time = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.CURRENT_DEPTH, access=py_trees.common.Access.READ)

    def initialise(self):
        self.start_time = pytime.time()
        node = self.blackboard.get(bb.ROS_NODE)
        msg = Float64()
        msg.data = self.target_depth
        node.depth_pub.publish(msg)
        node.get_logger().info(f'Submerging to {self.target_depth}m')

    def update(self):
        # Timeout after 30 seconds
        if (pytime.time() - self.start_time) > 30.0:
            return py_trees.common.Status.FAILURE

        try:
            current = self.blackboard.get(bb.CURRENT_DEPTH)
        except KeyError:
            return py_trees.common.Status.RUNNING

        if abs(current - self.target_depth) < self.tolerance:
            return py_trees.common.Status.SUCCESS

        # Keep publishing target
        node = self.blackboard.get(bb.ROS_NODE)
        msg = Float64()
        msg.data = self.target_depth
        node.depth_pub.publish(msg)

        return py_trees.common.Status.RUNNING


class WaitForStable(py_trees.behaviour.Behaviour):
    """Wait until vehicle velocities are low for 2 seconds."""

    def __init__(self, name='WaitForStable', velocity_threshold=0.1):
        super().__init__(name)
        self.vel_thresh = velocity_threshold
        self.stable_since = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.CURRENT_POSE, access=py_trees.common.Access.READ)

    def initialise(self):
        self.stable_since = None

    def update(self):
        try:
            odom = self.blackboard.get(bb.CURRENT_POSE)
        except KeyError:
            return py_trees.common.Status.RUNNING

        if odom is None:
            return py_trees.common.Status.RUNNING

        vx = abs(odom.twist.twist.linear.x)
        vy = abs(odom.twist.twist.linear.y)
        vz = abs(odom.twist.twist.linear.z)
        total_vel = (vx**2 + vy**2 + vz**2) ** 0.5

        if total_vel < self.vel_thresh:
            if self.stable_since is None:
                self.stable_since = pytime.time()
            elif (pytime.time() - self.stable_since) > 2.0:
                return py_trees.common.Status.SUCCESS
        else:
            self.stable_since = None

        return py_trees.common.Status.RUNNING
