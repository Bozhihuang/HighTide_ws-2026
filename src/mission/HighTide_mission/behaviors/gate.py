"""
Task 1: Begin Assessment (Gate) — Identify role and pass through correct side.

Strategy: Use YOLO to find gate and role symbols (🧭⚒️ vs 🛟🆘).
Strafe (crab walk) to align with correct half. Surge through.
Execute 360° yaw spin for style points — FOG snaps back to heading.
"""

import py_trees
from .common import (WaitForDetection, PublishThrusterCommand,
                     WaitForDuration, SetBlackboardValue, LogBehavior, StopMotion)
from . import blackboard_keys as bb


class IdentifyGateRole(py_trees.behaviour.Behaviour):
    """
    Look at gate detections to determine role assignment.
    Searches for symbol_compass/symbol_pickaxe (Survey & Repair)
    or symbol_lifering/symbol_sos (Search & Rescue).
    """

    def __init__(self, name='IdentifyGateRole', timeout=15.0):
        super().__init__(name)
        self.timeout = timeout
        self.start_time = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.CHOSEN_ROLE, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        import time
        self.start_time = time.time()

    def update(self):
        import time
        if (time.time() - self.start_time) > self.timeout:
            # Default to survey_repair if can't identify
            self.blackboard.set(bb.CHOSEN_ROLE, 'survey_repair')
            return py_trees.common.Status.SUCCESS

        try:
            detections = self.blackboard.get(bb.DETECTIONS)
        except KeyError:
            return py_trees.common.Status.RUNNING

        if detections is None:
            return py_trees.common.Status.RUNNING

        survey_symbols = {'symbol_compass', 'symbol_pickaxe'}
        rescue_symbols = {'symbol_lifering', 'symbol_sos'}

        for det in detections.detections:
            if det.class_name in survey_symbols and det.confidence > 0.5:
                self.blackboard.set(bb.CHOSEN_ROLE, 'survey_repair')
                node = self.blackboard.get(bb.ROS_NODE)
                node.get_logger().info('Role identified: SURVEY & REPAIR')
                return py_trees.common.Status.SUCCESS
            elif det.class_name in rescue_symbols and det.confidence > 0.5:
                self.blackboard.set(bb.CHOSEN_ROLE, 'search_rescue')
                node = self.blackboard.get(bb.ROS_NODE)
                node.get_logger().info('Role identified: SEARCH & RESCUE')
                return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class AlignWithGateHalf(py_trees.behaviour.Behaviour):
    """
    Strafe to align with the correct gate half based on chosen role.
    Uses vision servo to center on the role symbol.
    """

    def __init__(self, name='AlignWithGateHalf', timeout=20.0):
        super().__init__(name)
        self.timeout = timeout
        self.start_time = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.DETECTIONS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.CHOSEN_ROLE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        import time
        self.start_time = time.time()
        node = self.blackboard.get(bb.ROS_NODE)
        # Enable vision servo for the correct symbols
        role = self.blackboard.get(bb.CHOSEN_ROLE)
        if role == 'survey_repair':
            target = 'symbol_compass'
        else:
            target = 'symbol_lifering'

        # Set vision servo parameters dynamically
        from rclpy.parameter import Parameter
        node.vision_servo_class = target
        node.get_logger().info(f'Aligning with gate half for {role} — target: {target}')

    def update(self):
        import time
        if (time.time() - self.start_time) > self.timeout:
            return py_trees.common.Status.SUCCESS  # Best effort

        try:
            detections = self.blackboard.get(bb.DETECTIONS)
            role = self.blackboard.get(bb.CHOSEN_ROLE)
        except KeyError:
            return py_trees.common.Status.RUNNING

        target_classes = ({'symbol_compass', 'symbol_pickaxe'}
                         if role == 'survey_repair'
                         else {'symbol_lifering', 'symbol_sos'})

        if detections:
            for det in detections.detections:
                if det.class_name in target_classes:
                    img_w = detections.image_width or 1280
                    # Check if approximately centered (within 15% of center)
                    normalized_x = det.center_x / img_w
                    if abs(normalized_x - 0.5) < 0.15:
                        return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class SurgeThrough(py_trees.behaviour.Behaviour):
    """Surge forward through the gate for a fixed duration."""

    def __init__(self, name='SurgeThrough', duration=5.0, speed=0.5):
        super().__init__(name)
        self.duration = duration
        self.speed = speed
        self.start_time = None
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)

    def initialise(self):
        import time
        self.start_time = time.time()

    def update(self):
        import time
        from HighTide_interfaces.msg import ThrusterCommand
        if (time.time() - self.start_time) > self.duration:
            return py_trees.common.Status.SUCCESS

        node = self.blackboard.get(bb.ROS_NODE)
        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.surge = self.speed
        node.cmd_pub.publish(cmd)
        return py_trees.common.Status.RUNNING


class HeadingTurn(py_trees.behaviour.Behaviour):
    """Turns the AUV a specific number of degrees using closed-loop PID control."""

    def __init__(self, name='HeadingTurn', degrees=180.0, tolerance=2.0, timeout=10.0):
        super().__init__(name)
        self.target_degrees_offset = degrees
        self.tolerance_deg = tolerance
        self.timeout = timeout
        self.start_time = None
        self.target_heading = None
        
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=bb.ROS_NODE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=bb.CURRENT_HEADING, access=py_trees.common.Access.READ)

    def initialise(self):
        import time
        import math
        from HighTide_navigation import normalize_angle, PIDController
        self.start_time = time.time()
        
        # We need a local PID controller for this maneuver
        self.pid = PIDController(kp=1.5, ki=0.05, kd=0.1, output_max=0.6)
        
        current_heading = self.blackboard.get(bb.CURRENT_HEADING)
        if current_heading is None:
            current_heading = 0.0
        
        offset_rad = math.radians(self.target_degrees_offset)
        self.target_heading = normalize_angle(current_heading + offset_rad)
        
        node = self.blackboard.get(bb.ROS_NODE)
        node.get_logger().info(f'Turning by {self.target_degrees_offset}° to heading {math.degrees(self.target_heading):.1f}°')
        self.last_t = time.time()

    def update(self):
        import time
        import math
        from HighTide_navigation import normalize_angle
        from HighTide_interfaces.msg import ThrusterCommand
        
        now = time.time()
        if (now - self.start_time) > self.timeout:
            return py_trees.common.Status.SUCCESS
            
        current_heading = self.blackboard.get(bb.CURRENT_HEADING)
        if current_heading is None:
            return py_trees.common.Status.RUNNING

        error_rad = normalize_angle(self.target_heading - current_heading)
        
        if abs(math.degrees(error_rad)) <= self.tolerance_deg:
            # We have reached the heading
            return py_trees.common.Status.SUCCESS

        dt = now - self.last_t
        self.last_t = now
        
        yaw_cmd = self.pid.compute(error_rad, dt)

        node = self.blackboard.get(bb.ROS_NODE)
        cmd = ThrusterCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.yaw = yaw_cmd
        node.cmd_pub.publish(cmd)
        
        return py_trees.common.Status.RUNNING


def create_gate_subtree() -> py_trees.behaviour.Behaviour:
    """Build the Task 1 (Gate) behavior subtree."""

    # Logic to handle Coin Flip (Heads = Front, Tails = Back)
    find_gate_logic = py_trees.composites.Selector(
        name='CoinFlipLogic',
        memory=False,
        children=[
            # 1. Quick check for Heads (gate in front)
            WaitForDetection('QuickFindGate', 'gate', timeout=3.0),
            # 2. If it fails, assume Tails (gate in back). Turn exactly 180° using FOG and search again.
            py_trees.composites.Sequence(
                name='AssumeTails',
                memory=True,
                children=[
                    LogBehavior('LogTails', 'Gate not found immediately. Assuming TAILS. Turning 180°.'),
                    HeadingTurn('Turn180', degrees=180.0, tolerance=2.0, timeout=10.0),
                    StopMotion('StopTurn'),
                    WaitForDuration('SettleDown', duration_sec=1.0),
                    WaitForDetection('FindGateAfterTurn', 'gate', timeout=60.0),
                ]
            )
        ]
    )

    return py_trees.composites.Sequence(
        name='Task1_Gate',
        memory=True,
        children=[
            LogBehavior('Gate_Start', 'Starting Task 1: Gate'),
            find_gate_logic,
            SurgeThrough('ApproachGate', duration=3.0, speed=0.3),
            IdentifyGateRole('IdentifyRole'),
            AlignWithGateHalf('AlignGate'),
            SurgeThrough('PassThrough', duration=5.0, speed=0.5),
            StopMotion('StopAfterGate'),
            LogBehavior('Gate_StyleSpin', 'Executing style yaw spin'),
            # Yaw spin handled by the mission node calling yaw_controller
            WaitForDuration('StyleSpinDelay', duration_sec=5.0),
            LogBehavior('Gate_Done', 'Task 1 Gate COMPLETE'),
        ],
    )
