"""livekit-wakeword + Google Cloud ASR (デバッグモード)."""
import launch
from launch import LaunchService
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import launch_ros.actions  # noqa: I201

_DEBUG_DIR = '/home/taro/ros2_ws/src/susumu_asr_ros/launch/debug'


def generate_launch_description():
    return launch.LaunchDescription([
        DeclareLaunchArgument(
            'language_code', default_value='ja-JP',
            description='Google Cloud ASR 言語コード',
        ),
        DeclareLaunchArgument(
            'model_name', default_value='hey_mycroft_v0.1.onnx',
            description='ウェイクワードモデルファイル名',
        ),
        DeclareLaunchArgument(
            'model_folder', default_value='models',
            description='ウェイクワードモデルフォルダ',
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
        launch_ros.actions.Node(
            package='susumu_asr_ros',
            executable='susumu_asr_node',
            name='susumu_asr_node',
            output='screen',
            parameters=[{
                'vad_plugin': 'silero_vad',
                'wakeword_plugin': 'livekit_wakeword',
                'asr_plugin': 'google_cloud',
                'input_device_index': LaunchConfiguration('input_device_index'),
                'input_file': LaunchConfiguration('input_file'),
                'debug': True,
                'debug_dir': LaunchConfiguration('debug_dir'),
                'google_cloud.language_code': LaunchConfiguration('language_code'),
                'livekit_wakeword.model_name': LaunchConfiguration('model_name'),
                'livekit_wakeword.model_folder': LaunchConfiguration('model_folder'),
            }],
        ),
    ])


if __name__ == '__main__':
    launch_service = LaunchService()
    launch_service.include_launch_description(generate_launch_description())
    launch_service.run()
