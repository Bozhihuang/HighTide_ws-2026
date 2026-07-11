#!/usr/bin/env python3
"""
System-ID launch — minimal stack for running pidexcitation.py (PID tuning data harvest).

Brings up ONLY what the excitation harness needs:
  - MAVROS            : FCU link, /mavros/rc/override, IMU, rel_alt
  - mode_manager_node : /hightide/arm, /hightide/set_alt_hold, /hightide/set_manual
  - ZED camera        : /mavros/zed/odom (position feedback for surge/sway/yaw ID)

Deliberately EXCLUDED: rc_override_node and depth_controller_node. Both publish to
/mavros/rc/override at 20 Hz; if either is running it races the excitation harness
(which commands raw RC override directly) and corrupts the identification data.
Nav / mission nodes are excluded for the same isolation reason.

The exciter itself is NOT auto-started — bring this stack up, confirm the pool is
clear, then run it manually so you control when thrusters go live:

    ros2 run hightide_tests sys_id_exciter
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import UnlessCondition
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource, AnyLaunchDescriptionSource)
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ============== ARGUMENTS ==============
    sim_arg = DeclareLaunchArgument('sim', default_value='false',
                                    description='Enable simulation mode (skips ZED)')
    fcu_url_arg = DeclareLaunchArgument(
        'fcu_url', default_value='udp://192.168.2.1:14550@',
        description='FCU connection URL')
    system_id_arg = DeclareLaunchArgument(
        'system_id', default_value='1', description='MAVROS system ID')

    sim = LaunchConfiguration('sim')
    fcu_url = LaunchConfiguration('fcu_url')
    system_id = LaunchConfiguration('system_id')

    mavros_config = os.path.join(
        get_package_share_directory('hightide_launch'), 'config', 'mavros.yaml')

    # ============== MAVROS ==============
    mavros_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource([
            os.path.join(get_package_share_directory('mavros'),
                         'launch', 'apm.launch')
        ]),
        launch_arguments={
            'fcu_url': fcu_url,
            'system_id': system_id,
            'config_yaml': mavros_config,
        }.items(),
    )

    # ============== MODE MANAGER (arm / alt_hold / manual services) ==============
    mode_manager = Node(
        package='hightide_control',
        executable='mode_manager_node',
        name='mode_manager_node',
        output='screen',
    )

    # ============== ZED CAMERA (odom source for horizontal ID) ==============
    zed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('zed_wrapper'),
                         'launch', 'zed_camera.launch.py')
        ]),
        launch_arguments={
            'camera_model': 'zedxm',
            'publish_tf': 'true',
            'depth_mode': 'NEURAL_LIGHT',
            'pos_tracking_enabled': 'true',
            'spatial_memory_enabled': 'true',
        }.items(),
        condition=UnlessCondition(sim),
    )

    return LaunchDescription([
        sim_arg, fcu_url_arg, system_id_arg,
        mavros_launch,
        mode_manager,
        zed_launch,
    ])