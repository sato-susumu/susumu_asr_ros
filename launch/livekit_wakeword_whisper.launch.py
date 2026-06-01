"""livekit-wakeword + faster-whisper ASR."""
import launch
from launch import LaunchService
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import launch_ros.actions  # noqa: I201


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
            'debug', default_value='false',
            description='デバッグモード（音声ファイル出力）',
        ),
        launch_ros.actions.Node(
            package='susumu_asr_ros',
            executable='susumu_asr_node',
            name='susumu_asr_node',
            output='screen',
            parameters=[{
                'vad_plugin': 'silero_vad',
                'wakeword_plugin': 'livekit_wakeword',
                'asr_plugin': 'whisper',
                'input_device_index': LaunchConfiguration('input_device_index'),
                'debug': LaunchConfiguration('debug'),
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
