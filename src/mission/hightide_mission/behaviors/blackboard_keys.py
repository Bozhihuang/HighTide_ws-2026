"""Blackboard key constants for the mission behavior tree."""

# Vehicle state
VEHICLE_ARMED = 'vehicle_armed'
VEHICLE_MODE = 'vehicle_mode'
CURRENT_DEPTH = 'current_depth'
CURRENT_HEADING = 'current_heading'
CURRENT_POSE = 'current_pose'

# Mission state
CHOSEN_ROLE = 'chosen_role'       # 'survey_repair' or 'search_rescue'
CURRENT_TASK = 'current_task'
MISSION_START_TIME = 'mission_start_time'
MISSION_TIMEOUT = 'mission_timeout'

# Perception
DETECTIONS = 'detections'
TARGET_DETECTION = 'target_detection'
GATE_SIDE = 'gate_side'

# Navigation
NAV_TIER = 'nav_tier'
TARGET_WAYPOINT = 'target_waypoint'
GATE_POSITION = 'gate_position'   # For return home
# Movement mode: True = measure travel with ZED odometry (closed-loop on
# CURRENT_POSE); False = open-loop dead reckoning (elapsed time * DEAD_RECKON_MPS).
USE_ODOMETRY = 'use_odometry'
DEAD_RECKON_MPS = 'dead_reckon_mps'  # calibrated forward speed (m/s) for timed mode

# Task progress
MARKERS_DROPPED = 'markers_dropped'
TORPEDOES_FIRED = 'torpedoes_fired'
OBJECTS_COLLECTED = 'objects_collected'
GATE_DIVIDER_SIDE = 'gate_divider_side'  # Which side to pass slalom

# ROS node reference
ROS_NODE = 'ros_node'
