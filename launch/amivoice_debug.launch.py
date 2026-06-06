"""Silero VAD + AmiVoice ACP ASR (デバッグモード)."""
import launch
from launch import LaunchService
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import launch_ros.actions  # noqa: I201

_DEBUG_DIR = '/home/taro/ros2_ws/src/susumu_asr_ros/debug'
_ENV_FILE = '/home/taro/ros2_ws/src/susumu_asr_ros/.env'


def generate_launch_description():
    return launch.LaunchDescription([
        DeclareLaunchArgument(
            'amivoice_engine', default_value='-a-general',
            description='AmiVoice ACP 認識エンジン名',
        ),
        DeclareLaunchArgument(
            'amivoice_profile_words', default_value='今日 きょう',
            description='ユーザー辞書 (表記 読み 形式、複数は | 区切り)',
        ),
        DeclareLaunchArgument(
            'input_device_index', default_value='-1',
            description='マイク入力デバイスインデックス（-1 でシステムデフォルト）',
        ),
        DeclareLaunchArgument(
            'input_file', default_value='',
            description='WAVファイルパス（空文字でマイク入力）',
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
            package='susumu_asr_ros',
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
                'asr_plugin': 'amivoice',
                'input_device_index': LaunchConfiguration('input_device_index'),
                'input_file': LaunchConfiguration('input_file'),
                'debug': True,
                'debug_dir': LaunchConfiguration('debug_dir'),
                'amivoice.engine': LaunchConfiguration('amivoice_engine'),
                'amivoice.profile_words': LaunchConfiguration('amivoice_profile_words'),
            }],
        ),
    ])


if __name__ == '__main__':
    launch_service = LaunchService()
    launch_service.include_launch_description(generate_launch_description())
    launch_service.run()
