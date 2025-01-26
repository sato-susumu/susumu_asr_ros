import os
from glob import glob
from setuptools import find_packages, setup

package_name = "susumu_asr_ros"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
    ],
    install_requires=[
        "setuptools",
        "numpy>=1.26,<2",
        "pyaudio",
        "torch",
        "torchaudio",
        "vosk",
        "google-cloud-speech",
        "openwakeword",
        "tflite_runtime==2.14.0",
        "faster-whisper",
    ],
    zip_safe=True,
    maintainer="Sato Susumu",
    maintainer_email="75652942+sato-susumu@users.noreply.github.com",
    description="TODO: Package description",
    license="MIT",
    tests_require=[
        "pytest<8.0.0"
    ],
    entry_points={
        "console_scripts": ["susumu_asr_node = susumu_asr_ros.susumu_asr_node:main"],
    },
)
