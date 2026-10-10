from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='traversable_node',
            executable='traversable_node',
            name='traversable_node',
            output='screen',
        ),
    ])
