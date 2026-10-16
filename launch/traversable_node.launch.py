from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    footprint_to_base_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='footprint_to_base_tf',
        arguments=[
            '--x', '0.0',
            '--y', '0.0',
            '--z', '1.0',
            '--roll', '0.0',
            '--pitch', '0.0',
            '--yaw', '0.0',
            '--frame-id', 'base_footprint',
            '--child-frame-id', 'base_link'
        ],
        output='screen'
    )

    base_to_camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_tf',
        arguments=[
            '--x', '0.2',
            '--y', '0.0',
            '--z', '0.5',
            '--roll', '-1.5707963267948966',
            '--pitch', '0.0',
            '--yaw', '-1.5707963267948966',
            '--frame-id', 'base_link',
            '--child-frame-id', 'camera_optical_frame'
        ],
        output='screen'
    )

    corridor_detect = Node(
        package='traversable_corridor',
        executable='traversable_node',
        name='traversable_node',
        output='screen'
    )

    ld = LaunchDescription()
    ld.add_action(footprint_to_base_tf)
    ld.add_action(base_to_camera_tf)
    ld.add_action(corridor_detect)
    return ld