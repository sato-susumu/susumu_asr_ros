import launch
import launch_ros.actions

from launch import LaunchService


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
                        "vad_type": "openwakeword",
                        "asr_type": "whisper",
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
