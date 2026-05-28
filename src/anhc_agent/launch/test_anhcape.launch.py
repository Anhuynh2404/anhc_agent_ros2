from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    gym_env_node = Node(
        package="anhc_agent",
        executable="environment.py",
        name="environment_node",
        output="screen",
        emulate_tty=True,
        parameters=[{"environment_mode": "test"}],
    )

    test_anhcape_node = Node(
        package="anhc_agent",
        executable="test_anhcape_agent.py",
        name="test_anhcape_node",
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            gym_env_node,
            test_anhcape_node,
        ]
    )
