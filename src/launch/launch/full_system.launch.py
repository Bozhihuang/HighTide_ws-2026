#!/usr/bin/env python3
"""
Full system launch — Brings up all hightide nodes for competition.

Usage:
  ros2 launch hightide_launch full_system.launch.py
  ros2 launch hightide_launch full_system.launch.py run_mission:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource, AnyLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.actions import TimerAction
import os


def generate_launch_description():
    # ============== ARGUMENTS ==============
    sim_arg = DeclareLaunchArgument('sim', default_value='false',
                                    description='Enable simulation mode')
    run_mission_arg = DeclareLaunchArgument('run_mission', default_value='true', 
                                    description='Run the autonomous mission tree')                                
    depth_arg = DeclareLaunchArgument('mission_depth', default_value='1.0',
                                      description='Mission depth in meters')
    timeout_arg = DeclareLaunchArgument('mission_timeout', default_value='900.0',
                                        description='Mission timeout in seconds')
    engine_arg = DeclareLaunchArgument('yolo_engine', default_value='',
                                       description='Path to TensorRT engine file')
    
    # Defaults the argument to your verified working tether connection parameter
    fcu_url_arg = DeclareLaunchArgument('fcu_url',
                                         default_value='udp://192.168.2.1:14550@',
                                         description='FCU connection URL')

    # Resolve Launch Configurations
    sim = LaunchConfiguration('sim')
    run_mission = LaunchConfiguration('run_mission')
    mission_depth = LaunchConfiguration('mission_depth')
    mission_timeout = LaunchConfiguration('mission_timeout')
    yolo_engine = LaunchConfiguration('yolo_engine')
    fcu_url = LaunchConfiguration('fcu_url')

    # Locate Configuration File Paths
    localization_config = os.path.join(
        get_package_share_directory('hightide_localization'),
        'config', 'ekf_params.yaml')

    global_config = os.path.join(
        get_package_share_directory('hightide_launch'),
        'config', 'params.yaml')

    # ============== MAVROS ==============
    # Exactly mirrors your working alias: 
    # ros2 launch mavros apm.launch fcu_url:=udp://192.168.2.1:14550@ system_id:=1
    mavros_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource([
            os.path.join(get_package_share_directory('mavros'),
                         'launch', 'apm.launch')
        ]),
        launch_arguments={
            'fcu_url': fcu_url,
            'system_id': '1',
        }.items(),
    )

    # ============== ZED CAMERA ==============
    zed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('zed_wrapper'),
                         'launch', 'zed_camera.launch.py')
        ]),
        launch_arguments={
            'camera_model': 'zedxm',
            'publish_tf': 'true',
            'depth_mode': 'NEURAL',
            'pos_tracking_enabled': 'true',
            'spatial_memory_enabled': 'true',
        }.items(),
        condition=UnlessCondition(sim),
    )

    # ============== EKF (robot_localization) ==============
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[localization_config],
    )

    # ============== CONTROL NODES ==============
    rc_override = Node(
        package='hightide_control',
        executable='rc_override_node',
        name='rc_override_node',
        output='screen',
        parameters=[global_config],
    )

    depth_controller = Node(
        package='hightide_control',
        executable='depth_controller_node',
        name='depth_controller_node',
        output='screen',
        parameters=[global_config],
    )

    mode_manager = Node(
        package='hightide_control',
        executable='mode_manager_node',
        name='mode_manager_node',
        output='screen',
    )

    # ============== PERCEPTION NODES ==============
    # Commented out pending YOLO Engine deployment
    # yolo_detector = Node(
    #     package='hightide_perception',
    #     executable='yolo_detector_node',
    #     name='yolo_detector_node',
    #     output='screen',
    #     parameters=[
    #         global_config,
    #         {'engine_path': yolo_engine}
    #     ],
    # )

    # target_tracker = Node(
    #     package='hightide_perception',
    #     executable='target_tracker_node',
    #     name='target_tracker_node',
    #     output='screen',
    #     parameters=[global_config],
    # )

    detection_viz = Node(
        package='hightide_perception',
        executable='detection_viz_node',
        name='detection_viz_node',
        output='screen',
    )

    # ============== LOCALIZATION NODES ==============
    nav_tier_manager = Node(
        package='hightide_localization',
        executable='nav_tier_manager_node',
        name='nav_tier_manager_node',
        output='screen',
    )

    # ============== NAVIGATION NODES ==============
    waypoint_navigator = Node(
        package='hightide_navigation',
        executable='waypoint_navigator_node',
        name='waypoint_navigator_node',
        output='screen',
        parameters=[global_config],
    )

    vision_servo = Node(
        package='hightide_navigation',
        executable='vision_servo_node',
        name='vision_servo_node',
        output='screen',
        parameters=[global_config],
    )

    yaw_controller = Node(
        package='hightide_navigation',
        executable='yaw_controller_node',
        name='yaw_controller_node',
        output='screen',
        parameters=[global_config],
    )

    search_pattern = Node(
        package='hightide_navigation',
        executable='search_pattern_node',
        name='search_pattern_node',
        output='screen',
        parameters=[global_config],
    )

    # ============== DRIVERS ==============
    actuator_driver = Node(
        package='hightide_drivers',
        executable='actuator_driver_node',
        name='actuator_driver_node',
        output='screen',
        parameters=[global_config],
    )

    # ============== MISSION ==============
    mission_node = Node(
        package='hightide_mission',
        executable='mission_node',
        name='mission_node',
        output='screen',
        condition=IfCondition(run_mission),
        parameters=[
            global_config,
            {
                'mission_depth_m': mission_depth,
                'mission_timeout_sec': mission_timeout,
            }
        ],
    )

    delayed_mission = TimerAction(
        period=5.0,
        actions=[mission_node]
    )
    shutdown_handler = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=mission_node,
            on_exit=[
                EmitEvent(event=Shutdown(reason='mission node exited'))
            ]
        )
    )
    return LaunchDescription([
        sim_arg, depth_arg, timeout_arg, engine_arg, fcu_url_arg, run_mission_arg,
        # mavros_launch,
        zed_launch,
        ekf_node,
        rc_override,
        depth_controller,
        mode_manager,
        # yolo_detector,
        # target_tracker,
        detection_viz,
        nav_tier_manager,
        waypoint_navigator,
        vision_servo,
        yaw_controller,
        search_pattern,
        actuator_driver,
        delayed_mission,
        shutdown_handler,
    ])