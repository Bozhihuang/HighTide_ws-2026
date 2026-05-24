"""
Task 3: Recon (Bins) — Drop markers into correct bins.

Strategy: Approach bins, identify correct symbol (🔥 for Survey, 🩸 for Rescue).
Calculate bin X/Y coordinate from ZED depth while visible. Switch to dead
reckoning when bin goes below camera FOV. Hover and drop markers.
"""

import py_trees
from .common import (WaitForDetection, WaitForDuration,
                     LogBehavior, StopMotion, PublishDepthSetpoint)
from . import blackboard_keys as bb


class IdentifyCorrectBin(py_trees.behaviour.Behaviour):
    """Find the correct bin based on chosen role."""

    def __init__(self, name='IdentifyCorrectBin', timeout=30.0):
        super().__init__(name)
        self.timeout = timeout
        self.start_time = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.CHOSEN_ROLE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.TARGET_DETECTION, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        import time
        self.start_time = time.time()

    def update(self):
        import time
        if (time.time() - self.start_time) > self.timeout:
            return py_trees.common.Status.FAILURE

        role = self.blackboard.get(bb.CHOSEN_ROLE)
        target_symbol = 'symbol_fire' if role == 'survey_repair' else 'symbol_blood'

        try:
            detections = self.blackboard.get(bb.DETECTIONS)
        except KeyError:
            return py_trees.common.Status.RUNNING

        if detections:
            for det in detections.detections:
                if det.class_name == target_symbol and det.confidence > 0.4:
                    self.blackboard.set(bb.TARGET_DETECTION, det)
                    node = self.blackboard.get(bb.ROS_NODE)
                    node.get_logger().info(
                        f'Found correct bin: {target_symbol} at depth {det.depth_m:.1f}m')
                    return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class NavigateOverBin(py_trees.behaviour.Behaviour):
    """
    Dead reckon over the bin position. Since camera is forward-facing,
    we calculate distance from last known position and surge forward.
    """

    def __init__(self, name='NavigateOverBin', surge_distance=2.0):
        super().__init__(name)
        self.surge_dist = surge_distance
        self.start_time = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.TARGET_DETECTION, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        import time
        self.start_time = time.time()
        # Calculate surge distance from last known detection depth
        try:
            det = self.blackboard.get(bb.TARGET_DETECTION)
            if det and det.depth_m > 0:
                self.surge_dist = det.depth_m
        except KeyError:
            pass

    def update(self):
        import time
        from HighTide_interfaces.msg import ThrusterCommand
        node = self.blackboard.get(bb.ROS_NODE)

        # Time-based dead reckoning
        # Assuming ~0.2 m/s at 0.4 thrust
        travel_time = self.surge_dist / 0.2
        elapsed = time.time() - self.start_time

        if elapsed > travel_time:
            node.cmd_pub.publish(ThrusterCommand())  # Stop
            return py_trees.common.Status.SUCCESS

        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.surge = 0.4
        node.cmd_pub.publish(cmd)
        return py_trees.common.Status.RUNNING


class DropMarker(py_trees.behaviour.Behaviour):
    """Drop a marker via the actuator driver service."""

    def __init__(self, name='DropMarker', dropper_id=1):
        super().__init__(name)
        self.dropper_id = dropper_id
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.MARKERS_DROPPED, access=py_trees.common.Access.WRITE)
        self.client = None
        self.future = None

    def initialise(self):
        from HighTide_interfaces.srv import DropMarker as DropMarkerSrv
        node = self.blackboard.get(bb.ROS_NODE)
        self.client = node.create_client(DropMarkerSrv, '/HighTide/drop_marker')
        self.future = None

    def update(self):
        from HighTide_interfaces.srv import DropMarker as DropMarkerSrv
        node = self.blackboard.get(bb.ROS_NODE)

        if not self.client.wait_for_service(timeout_sec=1.0):
            return py_trees.common.Status.RUNNING

        if self.future is None:
            req = DropMarkerSrv.Request()
            req.dropper_id = self.dropper_id
            self.future = self.client.call_async(req)
            return py_trees.common.Status.RUNNING

        if not self.future.done():
            return py_trees.common.Status.RUNNING

        result = self.future.result()
        if result and result.success:
            try:
                dropped = self.blackboard.get(bb.MARKERS_DROPPED)
            except KeyError:
                dropped = 0
            self.blackboard.set(bb.MARKERS_DROPPED, dropped + 1)
            node.get_logger().info(f'Marker {self.dropper_id} dropped!')
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


def create_bins_subtree() -> py_trees.behaviour.Behaviour:
    """Build the Task 3 (Bins) behavior subtree."""
    return py_trees.composites.Sequence(
        name='Task3_Bins',
        memory=True,
        children=[
            LogBehavior('Bins_Start', 'Starting Task 3: Bins'),
            WaitForDetection('FindPathToBins', 'path_marker', timeout=30.0),
            WaitForDuration('FollowPathToBins', duration_sec=3.0),
            WaitForDetection('FindBins', 'bin', timeout=45.0),
            IdentifyCorrectBin('IdentifyBin'),
            NavigateOverBin('NavigateOverBin1'),
            StopMotion('HoverOverBin1'),
            WaitForDuration('StabilizeOverBin1', duration_sec=3.0),
            DropMarker('DropMarker1', dropper_id=1),
            WaitForDuration('WaitAfterDrop1', duration_sec=2.0),
            DropMarker('DropMarker2', dropper_id=2),
            StopMotion('StopAfterBins'),
            LogBehavior('Bins_Done', 'Task 3 Bins COMPLETE'),
        ],
    )
