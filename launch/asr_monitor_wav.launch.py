"""ASR モニター GUI（WAVファイル再生デバッグ向け）."""
import launch
import launch_ros.actions  # noqa: I201


def generate_launch_description():  # noqa: D103
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='susumu_asr',
            executable='asr_monitor_node',
            name='asr_monitor_node',
            output='screen',
            parameters=[{
                'mic_device_index': -2,  # マイク無効
            }],
        ),
    ])
