"""
Launch file để train TD3 Baseline.

KHÔNG bao gồm environment.py vì node đó đã được chạy riêng ở Terminal 2,
đúng theo quy trình 3-terminal của dự án (giống AnhcApe).

Dùng trực tiếp lệnh:
    ros2 run anhc_agent train_td3_agent.py
thay vì launch file này, để nhất quán hoàn toàn với workflow AnhcApe.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Chỉ chạy TD3 training node.
    # environment.py đã chạy sẵn ở Terminal 2.
    train_td3_node = Node(
        package="anhc_agent",
        executable="train_td3_agent.py",
        name="train_td3_node",
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([train_td3_node])

