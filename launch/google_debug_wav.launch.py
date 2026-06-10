"""Silero VAD + Google Cloud ASR (WAVファイル デバッグ)."""

import launch
from launch import LaunchService
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import launch_ros.actions  # noqa: I201


_DEBUG_DIR = '/home/taro/ros2_ws/src/susumu_asr/debug'
_ENV_FILE = '/home/taro/ros2_ws/src/susumu_asr/.env'
_WAV_FILE = '/home/taro/ros2_ws/src/susumu_asr/test/audio/multi_utterance_42s.wav'


def generate_launch_description():
    return launch.LaunchDescription([
        DeclareLaunchArgument(
            'language_code', default_value='ja-JP',
            description='Google Cloud ASR 言語コード',
        ),
        DeclareLaunchArgument(
            'input_file', default_value=_WAV_FILE,
            description='入力WAVファイルパス',
        ),
        DeclareLaunchArgument(
            'debug_dir', default_value=_DEBUG_DIR,
            description='デバッグ出力フォルダ',
        ),
        DeclareLaunchArgument(
            'env_file', default_value=_ENV_FILE,
            description='.env ファイルのパス',
        ),
        launch_ros.actions.Node(
            package='susumu_asr',
            executable='susumu_asr_node',
            name='susumu_asr_node',
            output='screen',
            additional_env={
                'SUSUMU_ASR_ENV_FILE': LaunchConfiguration('env_file'),
            },
            parameters=[{
                'env_file': _ENV_FILE,
                'vad_plugin': 'silero_vad',
                'wakeword_plugin': 'passthrough',
                'asr_plugin': 'google_cloud',
                'input_file': LaunchConfiguration('input_file'),
                'debug': True,
                'debug_dir': LaunchConfiguration('debug_dir'),
                'google_cloud.language_code': LaunchConfiguration('language_code'),
            }],
        ),
        launch_ros.actions.Node(
            package='susumu_asr',
            executable='asr_monitor_node',
            name='asr_monitor_node',
            output='screen',
            parameters=[{'mic_device_index': -2}],
        ),
    ])


if __name__ == '__main__':
    launch_service = LaunchService()
    launch_service.include_launch_description(generate_launch_description())
    launch_service.run()
