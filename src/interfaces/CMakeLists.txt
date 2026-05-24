cmake_minimum_required(VERSION 3.8)
project(HighTide_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(action_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection.msg"
  "msg/DetectionArray.msg"
  "msg/NavigationTier.msg"
  "msg/ThrusterCommand.msg"
  "msg/MissionState.msg"
  "srv/FireTorpedo.srv"
  "srv/DropMarker.srv"
  "srv/SetDepth.srv"
  "srv/SetNavigationTier.srv"
  "action/NavigateToWaypoint.action"
  "action/ExecuteTask.action"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs builtin_interfaces action_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
