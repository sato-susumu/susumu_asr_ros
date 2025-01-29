import launch
import launch_ros.actions

from launch import LaunchService
from susumu_asr_ros.susumu_asr import ASR_WHISPER, VAD_OPENWAKEWORD


def generate_launch_description():
    return launch.LaunchDescription(
        [
            launch_ros.actions.Node(
                package="susumu_asr_ros",
                executable="susumu_asr_node",
                name="susumu_asr_node",
                output="screen",
                parameters=[
                    {
                        "vad_type": VAD_OPENWAKEWORD,
                        "asr_type": ASR_WHISPER,
                        "debug": True,
                        "whisper_language_code": "ja",
                    }
                ],
            )
        ]
    )


if __name__ == '__main__':
    launch_service = LaunchService()
    launch_service.include_launch_description(generate_launch_description())
    launch_service.run()
