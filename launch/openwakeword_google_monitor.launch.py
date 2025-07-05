from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    language_code_arg = DeclareLaunchArgument(
        'language_code',
        default_value='ja-JP',
        description='Language code for ASR (default: ja-JP)'
    )

    input_device_index_arg = DeclareLaunchArgument(
        'input_device_index',
        default_value='',
        description='Input device index for microphone (empty for default)'
    )

    oww_model_folder_arg = DeclareLaunchArgument(
        'oww_model_folder',
        default_value='models',
        description='OpenWakeWord model folder path'
    )

    oww_model_name_arg = DeclareLaunchArgument(
        'oww_model_name',
        default_value='hey_mycroft_v0.1.tflite',
        description='OpenWakeWord model name'
    )

    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Enable debug mode for audio output'
    )

    monitor_update_interval_arg = DeclareLaunchArgument(
        'monitor_update_interval',
        default_value='2.0',
        description='Monitor update interval in seconds'
    )

    monitor_show_details_arg = DeclareLaunchArgument(
        'monitor_show_details',
        default_value='false',
        description='Show detailed event history in monitor'
    )

    # ASR Node
    asr_node = Node(
        package='susumu_asr_ros',
        executable='susumu_asr_node',
        name='susumu_asr_node',
        output='screen',
        parameters=[
            {'list_mic_devices': False},
            {'vad_type': 'openwakeword'},
            {'asr_type': 'google_cloud'},
            {'language_code': LaunchConfiguration('language_code')},
            {'oww_model_folder': LaunchConfiguration('oww_model_folder')},
            {'oww_model_name': LaunchConfiguration('oww_model_name')},
            {'input_device_index': LaunchConfiguration('input_device_index')},
            {'debug': LaunchConfiguration('debug')},
        ]
    )

    # Monitor process
    monitor_process = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'susumu_asr_ros', 'susumu_asr_monitor',
            '--update-interval', LaunchConfiguration('monitor_update_interval'),
            ('--show-details' if LaunchConfiguration('monitor_show_details') == 'true'
             else '')
        ],
        output='screen',
        shell=False
    )

    return LaunchDescription([
        language_code_arg,
        input_device_index_arg,
        oww_model_folder_arg,
        oww_model_name_arg,
        debug_arg,
        monitor_update_interval_arg,
        monitor_show_details_arg,
        asr_node,
        monitor_process,
    ])
