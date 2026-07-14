"""
Task 6: Return Home — dive, face back toward the gate, drive straight home.

Simple, fully dead-reckoned finish (no vision, no recorded-pose homing):
  1. Dive back down from wherever we ended up (after the octagon we surfaced).
  2. PID-yaw to the gate-return heading: the coin-flip's recorded start heading
     (INITIAL_HEADING, which pointed AT the gate from the start box) + 180°, so
     the camera now faces back toward the gate/home.
  3. Drive straight `forward_m` (4 m default) holding that heading with the same
     surge + heading-hold PID every other transit leg uses.

If the coin flip was disabled (INITIAL_HEADING never recorded), the yaw step
best-effort skips and the drive just goes straight ahead on the current heading.
"""

import py_trees
from .common import LogBehavior, StopMotion, WaitForDuration, DeadReckonTransit
from .pre_dive import SubmergeToDepth, YawToRecordedHeading
from . import blackboard_keys as bb


def create_return_home_subtree(depth_m=0.5, forward_m=4.0,
                               heading_offset_deg=180.0, settle_sec=3.0,
                               turn_tolerance_deg=3.0, turn_timeout=20.0,
                               drive_timeout=60.0,
                               transit_speed=0.6, transit_kp=0.2, transit_ki=0.0,
                               transit_kd=0.05, turn_kp=0.225, turn_ki=0.0,
                               turn_kd=0.2, turn_output_limit=0.6
                               ) -> py_trees.behaviour.Behaviour:
    """Build the Task 6 (Return Home) behavior subtree.

    Dive to depth_m, PID-yaw to (INITIAL_HEADING + heading_offset_deg), then
    drive forward forward_m holding that heading. All params come from
    mission_node so the turn uses the shared heading-turn PID and the drive
    uses the shared transit PID.
    """
    return py_trees.composites.Sequence(
        name='Task6_ReturnHome',
        memory=True,
        children=[
            LogBehavior('ReturnHome_Start', 'Starting Task 6: Return Home'),
            # 1. Dive back down (we surfaced in the octagon) — actually WAIT
            # until depth_m is reached (not just publish-and-move-on), so the
            # turn/drive below start from the correct depth, not mid-transition.
            SubmergeToDepth('SubmergeForReturn', depth_m=depth_m),
            WaitForDuration('WaitSubmerge', duration_sec=settle_sec),
            # 2. Face back toward the gate: coin-flip start heading + 180°.
            YawToRecordedHeading('TurnToGateHeading',
                                 heading_key=bb.INITIAL_HEADING,
                                 offset_deg=heading_offset_deg,
                                 tolerance_deg=turn_tolerance_deg,
                                 timeout=turn_timeout,
                                 kp=turn_kp, ki=turn_ki, kd=turn_kd,
                                 output_limit=turn_output_limit),
            StopMotion('StopAfterTurnHome'),
            WaitForDuration('SettleAfterTurn', duration_sec=1.0),
            # 3. Drive straight home holding that heading (transit PID).
            DeadReckonTransit('DriveHome', forward_m=forward_m,
                              speed=transit_speed, timeout=drive_timeout,
                              kp=transit_kp, ki=transit_ki, kd=transit_kd),
            StopMotion('StopAfterReturn'),
            LogBehavior('ReturnHome_Done', 'Task 6 Return Home COMPLETE'),
        ],
    )
