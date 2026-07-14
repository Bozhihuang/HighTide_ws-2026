"""
Task 5: Restore (Octagon) — surface inside the octagon.

Hardware reality for hightide: we have NO manipulator/claw, so the resupply
portion (collecting nut/bolt/plug/pill/bandage and placing them in baskets, and
the inventory-based facing/rotation bonuses) is out of scope. We also have NO
hydrophone, so we cannot home on the acoustic pinger. This task therefore does
the achievable core: get to the octagon and surface inside it.

No vision: the Transit_Octagon leg (mission_node's octagon_{course}_leg_*)
already deadreckons the sub to the octagon's vicinity, same as every other
inter-task leg this mission. EnterOctagon is one more closed-loop ZED/PID
forward advance (DeadReckonTransit, advance_distance_m) into the interior,
then settle and surface.
"""

import py_trees
from .common import (WaitForDuration, LogBehavior, StopMotion,
                     DeadReckonTransit, distribute_timeout)
from . import blackboard_keys as bb


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
                           surge=0.3, settle_sec=2.0, surface_depth_m=0.3,
                           transit_kp=0.2, transit_ki=0.0, transit_kd=0.05
                           ) -> py_trees.behaviour.Behaviour:
    """Build the Task 5 (Octagon) behavior subtree — surface only (no claw).

    No vision: EnterOctagon is a closed-loop ZED/PID forward advance (same
    DeadReckonTransit primitive as every other leg), not a timeout-bearing
    search, so total_timeout only covers the Surface deadline.
    """
    t = distribute_timeout(total_timeout, {'surface': 30.0})

    return py_trees.composites.Sequence(
        name='Task5_Octagon',
        memory=True,
        children=[
            LogBehavior('Oct_Start', 'Starting Task 5: Octagon (surface only)'),
            DeadReckonTransit('EnterOctagon', forward_m=advance_distance_m,
                              speed=surge, kp=transit_kp, ki=transit_ki,
                              kd=transit_kd),
            StopMotion('StopInOctagon'),
            WaitForDuration('SettleInOctagon', duration_sec=settle_sec),
            SurfaceInOctagon('Surface', timeout=t['surface'],
                             surface_depth_m=surface_depth_m),
            LogBehavior('Oct_Done', 'Task 5 Octagon COMPLETE'),
        ],
    )
