"""livekit-wakeword + AmiVoice ACP ASR."""
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
            'amivoice_engine', default_value='-a-general-dn',
            description='AmiVoice ACP 認識エンジン名',
        ),
        DeclareLaunchArgument(
            'amivoice_profile_words', default_value='今日 きょう|ヘイ、マイクロフト へいまいくろふと',
            description='ユーザー辞書 (表記 読み 形式、複数は | 区切り)',
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
                'asr_plugin': 'amivoice',
                'input_device_index': LaunchConfiguration('input_device_index'),
                'debug': LaunchConfiguration('debug'),
                'livekit_wakeword.model_name': LaunchConfiguration('model_name'),
                'livekit_wakeword.model_folder': LaunchConfiguration('model_folder'),
                'amivoice.engine': LaunchConfiguration('amivoice_engine'),
                'amivoice.profile_words': LaunchConfiguration('amivoice_profile_words'),
            }],
        ),
    ])


if __name__ == '__main__':
    launch_service = LaunchService()
    launch_service.include_launch_description(generate_launch_description())
    launch_service.run()
