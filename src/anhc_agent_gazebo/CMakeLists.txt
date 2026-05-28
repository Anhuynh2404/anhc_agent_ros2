cmake_minimum_required(VERSION 3.8)
project(anhc_agent_gazebo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(ros_gz_sim REQUIRED)
find_package(anhc_agent_description REQUIRED)
find_package(anhc_agent_interfaces REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY
    config
    launch
    models
    worlds
  DESTINATION
    share/${PROJECT_NAME}/
)

# Install Python scripts so they are runnable via ros2 run
install(
  PROGRAMS
    scripts/pedsim_gz_bridge.py
  DESTINATION
    lib/${PROJECT_NAME}
)

ament_package()
