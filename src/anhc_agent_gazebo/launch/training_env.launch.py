#!/usr/bin/python3
"""
training_env.launch.py

Launches the DRL agent simulation using the custom Blender training environment.
Usage:
    ros2 launch anhc_agent_gazebo training_env.launch.py
    ros2 launch anhc_agent_gazebo training_env.launch.py use_gazebo_gui:=false rviz:=false
"""

import os

from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch.substitutions.launch_configuration import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python import get_package_share_directory


ARGUMENTS = [
    DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        choices=["true", "false"],
        description="Use simulation (Gazebo) clock",
    ),
    DeclareLaunchArgument(
        "namespace",
        default_value="",
        description="Robot namespace",
    ),
    DeclareLaunchArgument(
        "rviz",
        default_value="true",
        choices=["true", "false"],
        description="Start RViz2 visualization",
    ),
    DeclareLaunchArgument(
        "use_gazebo_gui",
        default_value="true",
        choices=["true", "false"],
        description="Start Gazebo client GUI (gzclient)",
    ),
    DeclareLaunchArgument(
        "slam",
        default_value="false",
        choices=["true", "false"],
        description="Whether or not to launch SLAMToolBox",
    ),
]


def generate_launch_description():
    # Package directories
    anhc_agent_gazebo_pkg = "anhc_agent_gazebo"
    anhc_agent_description_pkg = "anhc_agent_description"
    anhc_agent_gazebo_share = get_package_share_directory(anhc_agent_gazebo_pkg)
    anhc_agent_description_share = get_package_share_directory(anhc_agent_description_pkg)

    # World file: use the Blender training environment world
    training_env_world = os.path.join(
        anhc_agent_gazebo_share, "worlds", "training_env.world"
    )

    # Launch file paths
    agent_description_launch = PathJoinSubstitution(
        [anhc_agent_description_share, "launch", "agent_description.launch.py"]
    )
    agent_gazebo_world_launch = PathJoinSubstitution(
        [anhc_agent_gazebo_share, "launch", "gazebo_world.launch.py"]
    )
    agent_spawn_launch = PathJoinSubstitution(
        [anhc_agent_gazebo_share, "launch", "spawn_agent.launch.py"]
    )
    rviz_launch = PathJoinSubstitution(
        [anhc_agent_gazebo_share, "launch", "rviz.launch.py"]
    )
    slam_launch = PathJoinSubstitution(
        [anhc_agent_gazebo_share, "launch", "slam.launch.py"]
    )

    # Launch configurations
    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_gazebo_gui = LaunchConfiguration("use_gazebo_gui")

    # 1. Robot description (URDF → robot_state_publisher)
    agent_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([agent_description_launch]),
        launch_arguments=[
            ("namespace", namespace),
            ("use_sim_time", use_sim_time),
        ],
    )

    # 2. Gazebo world — pass the Blender training_env world file
    agent_gazebo_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([agent_gazebo_world_launch]),
        launch_arguments=[
            ("use_gazebo_gui", use_gazebo_gui),
            ("world_path", training_env_world),
        ],
    )

    # 3. Spawn agent robot into Gazebo
    agent_spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([agent_spawn_launch])
    )

    # 4. RViz2 visualization
    rviz2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([rviz_launch]),
        launch_arguments=[
            ("namespace", namespace),
            ("use_sim_time", use_sim_time),
        ],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    # 5. SLAM (optional)
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([slam_launch]),
        launch_arguments=[
            ("namespace", namespace),
            ("use_sim_time", use_sim_time),
        ],
        condition=IfCondition(LaunchConfiguration("slam")),
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(agent_description)
    ld.add_action(agent_gazebo_world)
    ld.add_action(agent_spawn)
    ld.add_action(rviz2)
    ld.add_action(slam)

    return ld
