#!/usr/bin/env python3
"""
HighTide AUV - Python Node Parameter Type Auto-Fixer
Recursively scans the workspace 'src/' directory to find and fix parameter
declarations, changing '0' to '0.0' for float/double configurations to prevent
InvalidParameterTypeExceptions in ROS 2.
"""

import os
import re

# Comprehensive list of float/double parameters across all HighTide nodes
TARGET_PARAMS = [
    'mission_depth_m', 'mission_timeout_sec', 'tick_rate',
    'publish_rate', 'deadzone', 'timeout_sec',
    'kp', 'ki', 'kd', 'depth_tolerance', 'integral_max',
    'barrel_roll_duration_sec',
    'confidence_threshold', 'nms_threshold',
    'max_tracking_age',
    'vslam_confidence_threshold', 'vio_confidence_threshold', 'stale_timeout_sec',
    'surge_kp', 'surge_ki', 'surge_kd',
    'sway_kp', 'sway_ki', 'sway_kd',
    'yaw_kp', 'yaw_ki', 'yaw_kd',
    'position_tolerance', 'yaw_tolerance', 'max_speed',
    'lateral_kp', 'lateral_ki', 'lateral_kd',
    'vertical_kp', 'vertical_ki', 'vertical_kd',
    'range_kp', 'range_ki', 'range_kd', 'approach_distance_m',
    'yaw_tolerance', 'spin_speed', 'spin_timeout',
    'surge_speed', 'sway_speed', 'speed_to_mps',
    'leg_length_m', 'leg_increment_m', 'search_speed',
    'search_kp', 'search_ki', 'search_kd'
]

def fix_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    modified = False
    for param in TARGET_PARAMS:
        # Regex matches self.declare_parameter('param_name', 0) or self.declare_parameter("param_name", 0)
        pattern = rf"(self\.declare_parameter\(\s*['\"]{param}['\"]\s*,\s*)0(\s*\))"
        if re.search(pattern, content):
            content = re.sub(pattern, r"\g<1>0.0\g<2>", content)
            print(f"  [FIXED] Parameter '{param}' updated to float (0.0) default in: {os.path.basename(file_path)}")
            modified = True

    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

def main():
    workspace_dir = os.path.expanduser('~/HighTide_ws-2026/src')
    if not os.path.exists(workspace_dir):
        print(f"Error: Directory {workspace_dir} does not exist.")
        return

    print("====================================================")
    print("      HighTide Workspace Parameter Auto-Fixer       ")
    print("====================================================")
    
    for root, _, files in os.walk(workspace_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                fix_file(file_path)

    print("\nScan complete. All float configurations have been updated to '0.0'.")

if __name__ == '__main__':
    main()