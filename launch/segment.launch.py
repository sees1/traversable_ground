import os

from ament_index_python.packages import get_package_prefix, get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, RegisterEventHandler, OpaqueFunction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ARGUMENTS = [
    DeclareLaunchArgument('venv_path', 
                          description='Absolute inference venv Python path'),
    DeclareLaunchArgument('device', default_value='cuda'),
    DeclareLaunchArgument('threshold', default_value='0.5'),
    DeclareLaunchArgument('weights_path', default_value='runs/unet/best.pt'),
    DeclareLaunchArgument('socket_host', default_value='127.0.0.1'),
    DeclareLaunchArgument('socket_port', default_value='5055'),
    DeclareLaunchArgument('startup_timeout', default_value='120.0'),
    DeclareLaunchArgument('inference_timeout', default_value='60.0'),
    DeclareLaunchArgument('input_reliability', default_value='auto'),
    DeclareLaunchArgument('use_video', default_value='true'),
    DeclareLaunchArgument('video_path', default_value='video/vineyard/video1.mp4'),
    DeclareLaunchArgument('image_topic_name', default_value='/fake_camera/image_raw'),
    DeclareLaunchArgument('inference_topic_name', default_value='/inference'),
]

def launch(context, *args, **kwargs):
    share = get_package_share_path('traversable_corridor')
    package_prefix = get_package_prefix('traversable_corridor')

    segment_path = os.path.join(package_prefix, 'lib', 'traversable_corridor', 'segment.py')
    weights_path = os.path.join(share, LaunchConfiguration('weights_path').perform(context))
    video_path = os.path.join(share, LaunchConfiguration('video_path').perform(context))

    input_reliability = LaunchConfiguration('input_reliability').perform(context)
    if input_reliability == 'auto':
        use_video = LaunchConfiguration('use_video').perform(context).lower() in ('true', '1')
        input_reliability = 'reliable' if use_video else 'best_effort'

    segment = ExecuteProcess(
        cmd=[LaunchConfiguration('venv_path'), '-u', segment_path,
            '--host', LaunchConfiguration('socket_host'),
            '--port', LaunchConfiguration('socket_port'),
            '--weights',weights_path,
            '--device', LaunchConfiguration('device'),
            '--threshold', LaunchConfiguration('threshold')],
        output='screen'
    )

    bridge = Node(
        package='traversable_corridor', 
        executable='image_socket_bridge.py',
        name='image_socket_bridge', 
        output='screen',
        parameters=[{
            'input_reliability': input_reliability,
            'socket_host': LaunchConfiguration('socket_host'),
            'socket_port': ParameterValue(LaunchConfiguration('socket_port'), value_type=int),
            'startup_timeout': ParameterValue(LaunchConfiguration('startup_timeout'), value_type=float),
            'inference_timeout': ParameterValue(LaunchConfiguration('inference_timeout'), value_type=float),
            'image_topic_name': LaunchConfiguration('image_topic_name'),
            'inference_topic_name': LaunchConfiguration('inference_topic_name'),
        }]
    )

    video = Node(
        package='traversable_corridor',
        executable='mp4_publisher.py',
        name='mp4_publisher', 
        output='screen',
        parameters=[{'make_loop': True,
                    'video_path': video_path,
                    'image_topic_name': LaunchConfiguration('image_topic_name')
        }],
        condition=IfCondition(LaunchConfiguration('use_video'))
    )

    # Neither process owns the other; launch closes the pipeline when either exits.
    handlers = [RegisterEventHandler(OnProcessExit(
        target_action=process,
        on_exit=[EmitEvent(event=Shutdown(reason='Image bridge or inference exited'))]))
        for process in (segment, bridge)]

    return [*handlers, segment, bridge, video]

def generate_launch_description():
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(OpaqueFunction(function=launch))
    return ld
