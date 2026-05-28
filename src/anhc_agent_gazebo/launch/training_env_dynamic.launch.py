#!/usr/bin/python3
"""
training_env_dynamic.launch.py

Launches the DRL agent simulation with DYNAMIC PEDESTRIANS powered by pedsim_ros2.

Architecture:
  1. Gazebo + training_env world  (same as training_env.launch.py)
  2. Robot description + spawn
  3. pedsim_simulator             — computes pedestrian social-force dynamics
  4. spawn_pedsim_agents          — mirrors pedsim positions into Gazebo entities
  5. RViz2 (optional)

Usage:
    ros2 launch anhc_agent_gazebo training_env_dynamic.launch.py
    ros2 launch anhc_agent_gazebo training_env_dynamic.launch.py num_pedestrians:=5
    ros2 launch anhc_agent_gazebo training_env_dynamic.launch.py use_gazebo_gui:=false rviz:=false
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory


# ---------------------------------------------------------------------------
# Launch arguments
# ---------------------------------------------------------------------------
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
    DeclareLaunchArgument(
        "pedsim_scene_file",
        default_value="/root/anhc_agent_ws/install/anhc_agent_gazebo/share/anhc_agent_gazebo/config/pedsim_training_env.xml",
        description="Path to the pedsim scenario XML file",
    ),
    DeclareLaunchArgument(
        "pedsim_update_rate",
        default_value="25.0",
        description="Pedsim simulator update rate (Hz)",
    ),
    DeclareLaunchArgument(
        "pedsim_start_delay",
        default_value="3.0",
        description="Delay before starting pedsim simulator (s)",
    ),
    DeclareLaunchArgument(
        "bridge_start_delay",
        default_value="5.0",
        description="Delay before starting the pedsim-to-Gazebo bridge (s)",
    ),
    DeclareLaunchArgument(
        "ped_radius",
        default_value="0.25",
        description="Pedestrian collision radius in metres",
    ),
    DeclareLaunchArgument(
        "ped_height",
        default_value="1.70",
        description="Pedestrian collision height in metres",
    ),
    DeclareLaunchArgument(
        "world_name",
        default_value="training_env_world",
        description="Gazebo world name used by the bridge",
    ),
]


# ---------------------------------------------------------------------------
def generate_launch_description():
    # ── Package directories ──────────────────────────────────────────────────
    gazebo_pkg    = "anhc_agent_gazebo"
    desc_pkg      = "anhc_agent_description"
    pedsim_sim_pkg = "pedsim_simulator"

    gazebo_share = get_package_share_directory(gazebo_pkg)
    pedsim_share = get_package_share_directory(pedsim_sim_pkg)

    # ── Key file paths ───────────────────────────────────────────────────────
    training_env_world = os.path.join(gazebo_share, "worlds", "training_env.world")
    pedsim_scenario    = os.path.join(gazebo_share, "config", "pedsim_training_env.xml")
    pedsim_config      = os.path.join(pedsim_share, "config", "params.yaml")

    # ── Sub-launch files (reuse existing anhc_agent_gazebo launchers) ─────────
    desc_share = get_package_share_directory(desc_pkg)

    agent_description_launch = PathJoinSubstitution(
        [desc_share, "launch", "agent_description.launch.py"]
    )
    gazebo_world_launch = PathJoinSubstitution(
        [gazebo_share, "launch", "gazebo_world.launch.py"]
    )
    spawn_agent_launch = PathJoinSubstitution(
        [gazebo_share, "launch", "spawn_agent.launch.py"]
    )
    rviz_launch = PathJoinSubstitution(
        [gazebo_share, "launch", "rviz.launch.py"]
    )
    slam_launch = PathJoinSubstitution(
        [gazebo_share, "launch", "slam.launch.py"]
    )

    # ── Launch configurations ────────────────────────────────────────────────
    namespace      = LaunchConfiguration("namespace")
    use_sim_time   = LaunchConfiguration("use_sim_time")
    use_gazebo_gui = LaunchConfiguration("use_gazebo_gui")
    pedsim_scene = LaunchConfiguration("pedsim_scene_file")
    pedsim_rate = LaunchConfiguration("pedsim_update_rate")
    pedsim_start_delay = LaunchConfiguration("pedsim_start_delay")
    bridge_start_delay = LaunchConfiguration("bridge_start_delay")
    ped_radius = LaunchConfiguration("ped_radius")
    ped_height = LaunchConfiguration("ped_height")
    world_name = LaunchConfiguration("world_name")

    # ── 1. Robot description ─────────────────────────────────────────────────
    agent_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([agent_description_launch]),
        launch_arguments=[
            ("namespace", namespace),
            ("use_sim_time", use_sim_time),
        ],
    )

    # ── 2. Gazebo world ──────────────────────────────────────────────────────
    agent_gazebo_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([gazebo_world_launch]),
        launch_arguments=[
            ("use_gazebo_gui", use_gazebo_gui),
            ("world_path", training_env_world),
        ],
    )

    # ── 3. Spawn robot ───────────────────────────────────────────────────────
    agent_spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([spawn_agent_launch])
    )

    # ── 4. RViz2 ─────────────────────────────────────────────────────────────
    rviz2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([rviz_launch]),
        launch_arguments=[
            ("namespace", namespace),
            ("use_sim_time", use_sim_time),
        ],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    # ── 5. SLAM (optional) ───────────────────────────────────────────────────
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([slam_launch]),
        launch_arguments=[
            ("namespace", namespace),
            ("use_sim_time", use_sim_time),
        ],
        condition=IfCondition(LaunchConfiguration("slam")),
    )

    # ── 6. Pedsim simulator node ─────────────────────────────────────────────
    #   Starts after a configurable delay to give Gazebo time to initialise.
    pedsim_simulator = TimerAction(
        period=pedsim_start_delay,
        actions=[
            Node(
                package="pedsim_simulator",
                executable="pedsim_simulator",
                name="pedsim_simulator",
                output="screen",
                parameters=[
                    pedsim_config,
                    {
                        "scene_file": pedsim_scene,
                        "simulation_factor": 1.0,   # real-time factor
                        "update_rate": pedsim_rate,
                        "robot_mode": 1,             # teleoperation mode (no nav2)
                        "enable_groups": True,
                        "max_robot_speed": 1.5,
                    },
                ],
            )
        ],
    )

    # ── 7. PEDSIM Gazebo plugin spawner (human model visual) ─────────────────
    #   Starts after a configurable delay (pedsim must publish states first).
    spawn_pedestrians = TimerAction(
        period=bridge_start_delay,
        actions=[
            Node(
                package="pedsim_gazebo_plugin",
                executable="spawn_pedsim_agents.py",
                name="spawn_pedsim_agents",
                output="screen",
                parameters=[{"use_sim_time": use_sim_time}],
            )
        ],
    )

    # ── Assemble LaunchDescription ───────────────────────────────────────────
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(agent_description)
    ld.add_action(agent_gazebo_world)
    ld.add_action(agent_spawn)
    ld.add_action(rviz2)
    ld.add_action(slam)
    ld.add_action(pedsim_simulator)
    ld.add_action(spawn_pedestrians)

    return ld
