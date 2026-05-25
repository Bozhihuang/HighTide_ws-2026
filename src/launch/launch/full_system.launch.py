"""
Full system launch — Brings up all hightide nodes for competition.

Usage:
  ros2 launch hightide_launch full_system.launch.py
  ros2 launch hightide_launch full_system.launch.py sim:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # ============== ARGUMENTS ==============
    sim_arg = DeclareLaunchArgument('sim', default_value='false',
                                    description='Enable simulation mode')
    depth_arg = DeclareLaunchArgument('mission_depth', default_value='1.0',
                                      description='Mission depth in meters')
    timeout_arg = DeclareLaunchArgument('mission_timeout', default_value='900.0',
                                        description='Mission timeout in seconds')
    engine_arg = DeclareLaunchArgument('yolo_engine', default_value='',
                                       description='Path to TensorRT engine file')
    fcu_url_arg = DeclareLaunchArgument('fcu_url',
                                         default_value='/dev/ttyACM0:115200',
                                         description='FCU connection URL')

    sim = LaunchConfiguration('sim')
    mission_depth = LaunchConfiguration('mission_depth')
    mission_timeout = LaunchConfiguration('mission_timeout')
    yolo_engine = LaunchConfiguration('yolo_engine')
    fcu_url = LaunchConfiguration('fcu_url')

    localization_config = os.path.join(
        get_package_share_directory('hightide_localization'),
        'config', 'ekf_params.yaml')

    # ============== MAVROS ==============
    mavros_node = Node(
        package='mavros',
        executable='mavros_node',
        name='mavros',
        output='screen',
        parameters=[{
            'fcu_url': fcu_url,
            'gcs_url': '',
            'target_system_id': 1,
            'target_component_id': 1,
            'system_id': 255,  # Required for ArduSub to accept GCS commands
        }],
    )

    # ============== ZED CAMERA ==============
    zed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('zed_wrapper'),
                         'launch', 'zed_camera.launch.py')
        ]),
        launch_arguments={
            'camera_model': 'zedxm',  # ZED X Mini
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
        remappings=[
            ('odometry/filtered', '/hightide/odometry/filtered'),
        ],
    )

    # ============== CONTROL NODES ==============
    rc_override = Node(
        package='hightide_control',
        executable='rc_override_node',
        name='rc_override_node',
        output='screen',
        parameters=[{
            'publish_rate': 20.0,
            'deadzone': 0.05,
        }],
    )

    depth_controller = Node(
        package='hightide_control',
        executable='depth_controller_node',
        name='depth_controller_node',
        output='screen',
        parameters=[{
            'kp': 100.0, 'ki': 5.0, 'kd': 20.0,
            'publish_rate': 20.0,
        }],
    )

    mode_manager = Node(
        package='hightide_control',
        executable='mode_manager_node',
        name='mode_manager_node',
        output='screen',
    )

    # ============== PERCEPTION NODES ==============
    yolo_detector = Node(
        package='hightide_perception',
        executable='yolo_detector_node',
        name='yolo_detector_node',
        output='screen',
        parameters=[{
            'engine_path': yolo_engine,
            'confidence_threshold': 0.5,
            'nms_threshold': 0.45,
            'publish_viz': True,
        }],
    )

    target_tracker = Node(
        package='hightide_perception',
        executable='target_tracker_node',
        name='target_tracker_node',
        output='screen',
        parameters=[{
            'max_tracking_age': 1.0,
            'iou_threshold': 0.3,
        }],
    )

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
    )

    vision_servo = Node(
        package='hightide_navigation',
        executable='vision_servo_node',
        name='vision_servo_node',
        output='screen',
        parameters=[{
            'enabled': False,  # Enabled by mission when needed
        }],
    )

    yaw_controller = Node(
        package='hightide_navigation',
        executable='yaw_controller_node',
        name='yaw_controller_node',
        output='screen',
    )

    search_pattern = Node(
        package='hightide_navigation',
        executable='search_pattern_node',
        name='search_pattern_node',
        output='screen',
    )

    # ============== DRIVERS ==============
    actuator_driver = Node(
        package='hightide_drivers',
        executable='actuator_driver_node',
        name='actuator_driver_node',
        output='screen',
        parameters=[{
            'torpedo_1_pin': 27,
            'torpedo_2_pin': 22,
            'dropper_1_pin': 23,
            'dropper_2_pin': 24,
            'pulse_duration_ms': 500,
        }],
    )

    # ============== MISSION ==============
    mission = Node(
        package='hightide_mission',
        executable='mission_node',
        name='mission_node',
        output='screen',
        parameters=[{
            'mission_depth_m': mission_depth,
            'mission_timeout_sec': mission_timeout,
            'tick_rate': 10.0,
        }],
    )

    return LaunchDescription([
        sim_arg, depth_arg, timeout_arg, engine_arg, fcu_url_arg,
        mavros_node,
        zed_launch,
        ekf_node,
        rc_override,
        depth_controller,
        mode_manager,
        yolo_detector,
        target_tracker,
        detection_viz,
        nav_tier_manager,
        waypoint_navigator,
        vision_servo,
        yaw_controller,
        search_pattern,
        actuator_driver,
        mission,
    ])
