"""ASR モニター GUI（マイク入力用）."""
import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import launch_ros.actions  # noqa: I201


def generate_launch_description():  # noqa: D103
    return launch.LaunchDescription([
        DeclareLaunchArgument(
            'mic_device_index', default_value='-1',
            description='使用するマイクのデバイスインデックス（-1でデフォルト）',
        ),
        launch_ros.actions.Node(
            package='susumu_asr',
            executable='asr_monitor_node',
            name='asr_monitor_node',
            output='screen',
            parameters=[{
                'mic_device_index': LaunchConfiguration('mic_device_index'),
            }],
        ),
    ])
