"""
Task 3: Recon (Bins) — Drop markers into correct bins.

Strategy: Approach bins, identify correct symbol (🔥 for Survey, 🩸 for Rescue).
Calculate bin X/Y coordinate from ZED depth while visible. Switch to dead
reckoning when bin goes below camera FOV. Hover and drop markers.
"""

import math
import py_trees
from .common import (WaitForDetection, WaitForAnyDetection, WaitForDuration,
                     LogBehavior, StopMotion, PublishDepthSetpoint,
                     SearchForDetection, lock_heading, yaw_hold,
                     distribute_timeout)
from . import blackboard_keys as bb

# Bins sit on the pool floor, so their symbols appear in the LOWER part of the
# forward camera frame. The torpedo board carries the SAME fire/blood symbols
# but hangs mid-water (appears near frame center). This normalized-y cutoff is
# how the two tasks avoid latching onto each other's prop — tune in the pool.
BIN_MIN_Y_FRAC = 0.40


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
        target_symbol = 'fire' if role == 'survey_repair' else 'blood'

        try:
            detections = self.blackboard.get(bb.DETECTIONS)
        except KeyError:
            return py_trees.common.Status.RUNNING

        if detections:
            img_h = detections.image_height or 720
            for det in detections.detections:
                if (det.class_name == target_symbol and det.confidence > 0.4
                        and (det.center_y / img_h) >= BIN_MIN_Y_FRAC):
                    self.blackboard.set(bb.TARGET_DETECTION, det)
                    node = self.blackboard.get(bb.ROS_NODE)
                    node.get_logger().info(
                        f'Found correct bin: {target_symbol} at depth {det.depth_m:.1f}m')
                    return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class NavigateOverBin(py_trees.behaviour.Behaviour):
    """
    Drive forward over the bin position. The camera is forward-facing, so we
    grab the ZED range to the bin (det.depth_m) while it is still visible, then
    advance that measured distance using ZED odometry (CURRENT_POSE) as the
    ruler — not a timed guess. A timeout is kept only as a safety fallback if
    odometry never arrives.

    NOTE: with no downward camera we cannot confirm we are actually over the
    bin — odometry makes the travel distance accurate, but the drop position
    stays an inference. This is a hardware limitation, not a code one.
    """

    def __init__(self, name='NavigateOverBin', surge_distance=2.0,
                 surge=0.3, timeout=20.0, floor_depth_m=2.5):
        super().__init__(name)
        self.surge_dist = surge_distance
        self.surge = surge
        self.timeout = timeout
        self._locked_heading = None
        # Venue pool-floor depth under the bins (m). Used to convert the ZED
        # slant range into horizontal travel — set per venue (TRANSDEC ~3.8).
        self.floor_depth_m = floor_depth_m
        self.start_time = None
        self.start_pos = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.TARGET_DETECTION, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.CURRENT_POSE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.CURRENT_DEPTH, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        import time
        self.start_time = time.time()
        self.start_pos = None
        self._locked_heading = lock_heading(self.blackboard.get(bb.ROS_NODE))

        # Distance to cover = HORIZONTAL component of the last-seen ZED range.
        # The camera looks forward and the bin is on the floor below, so the
        # slant range overshoots the horizontal distance; correct for the
        # height difference between our depth and the floor.
        try:
            det = self.blackboard.get(bb.TARGET_DETECTION)
            if det and det.depth_m > 0:
                try:
                    current_depth = self.blackboard.get(bb.CURRENT_DEPTH) or 0.0
                except KeyError:
                    current_depth = 0.0
                dz = max(0.0, self.floor_depth_m - current_depth)
                slant = det.depth_m
                horizontal = math.sqrt(max(slant * slant - dz * dz, 0.0))
                # Never fully zero — if slant < dz the range was noisy/short;
                # creep a little rather than declaring ourselves on top of it.
                self.surge_dist = max(horizontal, 0.3)
        except KeyError:
            pass

        # Snapshot starting odometry position to measure travel against.
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

    def update(self):
        import time
        from hightide_interfaces.msg import ThrusterCommand
        node = self.blackboard.get(bb.ROS_NODE)

        if (time.time() - self.start_time) > self.timeout:
            node.get_logger().warn('NavigateOverBin timed out (no odom / never advanced)')
            node.cmd_pub.publish(ThrusterCommand())
            return py_trees.common.Status.SUCCESS

        traveled = self._distance_traveled()
        if traveled is not None and traveled >= self.surge_dist:
            node.cmd_pub.publish(ThrusterCommand())  # Stop
            node.get_logger().info(
                f'Advanced {traveled:.2f}m over bin (odometry) — stopping')
            return py_trees.common.Status.SUCCESS

        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.surge = self.surge
        cmd.yaw = yaw_hold(node, self._locked_heading)  # advance straight over the bin
        node.cmd_pub.publish(cmd)
        return py_trees.common.Status.RUNNING


class DropMarker(py_trees.behaviour.Behaviour):
    """Drop a marker via the actuator driver service. Fails after `timeout`
    so a dead actuator node can't stall the whole mission forever."""

    def __init__(self, name='DropMarker', dropper_id=1, timeout=15.0):
        super().__init__(name)
        self.dropper_id = dropper_id
        self.timeout = timeout
        self.start_time = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.MARKERS_DROPPED, access=py_trees.common.Access.WRITE)
        self.client = None
        self.future = None

    def initialise(self):
        import time
        from hightide_interfaces.srv import DropMarker as DropMarkerSrv
        node = self.blackboard.get(bb.ROS_NODE)
        self.client = node.create_client(DropMarkerSrv, '/hightide/drop_marker')
        self.future = None
        self.start_time = time.time()

    def update(self):
        import time
        from hightide_interfaces.srv import DropMarker as DropMarkerSrv
        node = self.blackboard.get(bb.ROS_NODE)

        if (time.time() - self.start_time) > self.timeout:
            node.get_logger().error('DropMarker service timed out')
            return py_trees.common.Status.FAILURE

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


def create_bins_subtree(total_timeout=240.0) -> py_trees.behaviour.Behaviour:
    """Build the Task 3 (Bins) behavior subtree.

    total_timeout is this mission's time budget, split across the search,
    identify, navigate-over and marker-drop deadlines (ratios preserve the old
    45:30:20:15:15 tuning). NavigateOverBin's travel distance is measured off
    ZED odometry, so scaling its timeout only extends the safety fallback, not
    the distance driven. WaitForDuration hovers are fixed and NOT scaled.
    """
    t = distribute_timeout(total_timeout, {
        'find': 45.0, 'identify': 30.0, 'navigate': 20.0,
        'drop1': 15.0, 'drop2': 15.0})

    return py_trees.composites.Sequence(
        name='Task3_Bins',
        memory=True,
        children=[
            LogBehavior('Bins_Start', 'Starting Task 3: Bins'),
            # No path_marker / generic 'bin' class in the ffc model — the bins are
            # recognized directly by their fire/blood symbols. Creep-and-sweep
            # search (course elements are never on a straight line), and only
            # accept symbols low in frame so we don't latch the torpedo board
            # (which carries the same symbols mid-water).
            SearchForDetection('FindBins', {'fire', 'blood'}, timeout=t['find'],
                               surge=0.2, min_y_frac=BIN_MIN_Y_FRAC),
            IdentifyCorrectBin('IdentifyBin', timeout=t['identify']),
            NavigateOverBin('NavigateOverBin1', timeout=t['navigate']),
            StopMotion('HoverOverBin1'),
            WaitForDuration('StabilizeOverBin1', duration_sec=3.0),
            DropMarker('DropMarker1', dropper_id=1, timeout=t['drop1']),
            WaitForDuration('WaitAfterDrop1', duration_sec=2.0),
            DropMarker('DropMarker2', dropper_id=2, timeout=t['drop2']),
            StopMotion('StopAfterBins'),
            LogBehavior('Bins_Done', 'Task 3 Bins COMPLETE'),
        ],
    )