"""livekit-wakeword + faster-whisper ASR (デバッグモード)."""
import os
import launch
from launch import LaunchService
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import launch_ros.actions  # noqa: I201

_DEBUG_DIR = '/home/taro/ros2_ws/src/susumu_asr_ros/launch/debug'
_ENV_FILE = '/home/taro/ros2_ws/src/susumu_asr_ros/.env'


def generate_launch_description():
    return launch.LaunchDescription([
        DeclareLaunchArgument(
            'model_name', default_value='hey_mycroft_v0.1.onnx',
            description='ウェイクワードモデルファイル名',
        ),
        DeclareLaunchArgument(
            'model_folder', default_value='models',
            description='ウェイクワードモデルフォルダ',
        ),
        DeclareLaunchArgument(
            'whisper_model_name', default_value='turbo',
            description='Whisper モデル名',
        ),
        DeclareLaunchArgument(
            'whisper_language_code', default_value='ja',
            description='Whisper 言語コード（auto で自動判別）',
        ),
        DeclareLaunchArgument(
            'whisper_device', default_value='auto',
            description='推論デバイス（auto / cpu / cuda）',
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
                'env_file': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'),
                'vad_plugin': 'silero_vad',
                'wakeword_plugin': 'livekit_wakeword',
                'asr_plugin': 'whisper',
                'input_device_index': LaunchConfiguration('input_device_index'),
                'input_file': LaunchConfiguration('input_file'),
                'debug': True,
                'debug_dir': LaunchConfiguration('debug_dir'),
                'livekit_wakeword.model_name': LaunchConfiguration('model_name'),
                'livekit_wakeword.model_folder': LaunchConfiguration('model_folder'),
                'whisper.model_name': LaunchConfiguration('whisper_model_name'),
                'whisper.language_code': LaunchConfiguration('whisper_language_code'),
                'whisper.device': LaunchConfiguration('whisper_device'),
            }],
        ),
    ])


if __name__ == '__main__':
    launch_service = LaunchService()
    launch_service.include_launch_description(generate_launch_description())
    launch_service.run()
