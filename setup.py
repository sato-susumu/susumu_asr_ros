from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'susumu_asr'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*')),
        ),
    ],
    install_requires=[
        'setuptools',
        'numpy>=1.26,<2',
        'pyaudio',
        'torch',
        'torchaudio',
        'google-cloud-speech',
        'faster-whisper',
        'click',
        'loguru',
        'python-dotenv',
    ],
    zip_safe=True,
    maintainer='Sato Susumu',
    maintainer_email='75652942+sato-susumu@users.noreply.github.com',
    description='TODO: Package description',
    license='MIT',
    tests_require=[
        'pytest<8.0.0'
    ],
    entry_points={
        'console_scripts': [
            'susumu_asr_node = susumu_asr.susumu_asr_node:main',
            'asr_monitor_node = susumu_asr.asr_monitor_node:main',
        ],
        'susumu_asr.asr_plugins': [
            'google_cloud = susumu_asr.asr_google:GoogleCloudASRPlugin',
            'whisper = susumu_asr.asr_whisper:WhisperASRPlugin',
            'amivoice = susumu_asr.asr_amivoice:AmiVoiceASRPlugin',
        ],
        'susumu_asr.vad_plugins': [
            'silero_vad = susumu_asr.vad_silero:SileroVADPlugin',
        ],
        'susumu_asr.wakeword_plugins': [
            'passthrough = susumu_asr.wakeword_passthrough:PassthroughWakewordPlugin',
            'livekit_wakeword = susumu_asr.wakeword_livekit:LivekitWakewordPlugin',
            'openwakeword = susumu_asr.wakeword_openwakeword:OpenWakewordPlugin',
        ],
    },
)
