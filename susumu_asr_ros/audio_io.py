"""音声録音・書き込みクラス群."""
from abc import ABC, abstractmethod
from datetime import datetime
import os
import time
import wave

import pyaudio
from rclpy.logging import get_logger

from susumu_asr_ros.constants import (
    CHANNELS,
    FRAME_LENGTH_MS,
    MS_PER_SEC,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
)


class AudioRecorderBase(ABC):
    """音声入力のための抽象基底クラス."""

    @abstractmethod
    def open(self):  # noqa: A003
        pass

    @abstractmethod
    def read_frame(self) -> bytes:
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def get_device_info(self) -> dict:
        pass


class MicAudioRecorder(AudioRecorderBase):
    """マイク入力専用のAudioRecorderクラス."""

    @staticmethod
    def list_devices() -> None:
        """利用可能なマイクデバイスを一覧表示する."""
        pa = pyaudio.PyAudio()
        print('利用可能なマイクデバイス一覧:')
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"デバイス {i}: {info['name']}")
        pa.terminate()

    def __init__(self, read_frame_size, input_device_index=None):
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.read_frame_size = read_frame_size
        self.input_device_index = input_device_index
        self.logger = get_logger('mic_audio_recorder')

    def get_device_info(self) -> dict:
        """デバイス情報を取得して返す."""
        device_count = self.pa.get_device_count()
        device_info = {}
        for i in range(device_count):
            info = self.pa.get_device_info_by_index(i)
            device_info[i] = info
        return device_info

    def open(self):  # noqa: A003
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.read_frame_size,
            input_device_index=self.input_device_index,
            stream_callback=None,
        )
        self.stream.start_stream()
        self.logger.info(f'input_device_index: {self.input_device_index}')
        self.logger.info('マイク入力開始')

    def read_frame(self) -> bytes:
        if self.stream is not None:
            frame = self.stream.read(self.read_frame_size, exception_on_overflow=False)
            return frame
        return b''

    def close(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()
        self.logger.info('マイク入力終了')


class WavAudioRecorder(AudioRecorderBase):
    """WAVファイル入力専用のAudioRecorderクラス."""

    def __init__(
        self, read_frame_size, input_file: str, simulate_realtime: bool = False
    ):
        self.read_frame_size = read_frame_size
        self.input_file = input_file
        self.simulate_realtime = simulate_realtime
        self.wav_handle = None
        self.logger = get_logger('wav_audio_recorder')

    def get_device_info(self) -> dict:
        return {}

    def open(self):  # noqa: A003
        if not self.input_file:
            raise ValueError('input_file が指定されていません。')
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f'WAVファイルが見つかりません: {self.input_file}')
        self.wav_handle = wave.open(self.input_file, 'rb')
        if self.wav_handle.getnchannels() != CHANNELS:
            raise ValueError(f'WAVファイルのチャンネル数が {CHANNELS} ではありません。')
        if self.wav_handle.getsampwidth() != SAMPLE_WIDTH:
            raise ValueError(
                f'WAVファイルのサンプル幅が {SAMPLE_WIDTH} バイトではありません。'
            )
        if self.wav_handle.getframerate() != SAMPLE_RATE:
            raise ValueError(
                f'WAVファイルのサンプリングレートが {SAMPLE_RATE} Hz ではありません。'
            )
        self.logger.info(f'WAVファイル入力開始: {self.input_file}')

    def read_frame(self) -> bytes:
        if self.wav_handle is None:
            return b''

        data = self.wav_handle.readframes(self.read_frame_size)
        if not data:
            return b''

        expected = self.read_frame_size * SAMPLE_WIDTH
        if len(data) < expected:
            data = data + b'\x00' * (expected - len(data))

        if self.simulate_realtime:
            time.sleep(FRAME_LENGTH_MS / MS_PER_SEC)

        return data

    def close(self):
        if self.wav_handle is not None:
            self.wav_handle.close()
        self.logger.info('WAVファイル入力終了')


class AudioWriterBase(ABC):

    @abstractmethod
    def open(self) -> None:  # noqa: A003
        pass

    @abstractmethod
    def write(self, data: bytes) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class LabelWriterBase(ABC):

    @abstractmethod
    def write_segment(self, start: float, end: float, label: str) -> None:
        pass


class DummyAudioWriter(AudioWriterBase):

    def open(self) -> None:  # noqa: A003
        pass

    def write(self, data: bytes) -> None:
        pass

    def close(self) -> None:
        pass


class DummyLabelWriter(LabelWriterBase):

    def write_segment(self, start: float, end: float, label: str) -> None:
        pass


class FullAudioWriter(AudioWriterBase):

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.full_wav_handle = None
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def open(self) -> None:  # noqa: A003
        self.full_wav_handle = wave.open(self.file_path, 'wb')
        self.full_wav_handle.setnchannels(CHANNELS)
        self.full_wav_handle.setsampwidth(SAMPLE_WIDTH)
        self.full_wav_handle.setframerate(SAMPLE_RATE)

    def write(self, data: bytes) -> None:
        if self.full_wav_handle:
            self.full_wav_handle.writeframes(data)

    def close(self) -> None:
        if self.full_wav_handle:
            self.full_wav_handle.close()
            self.full_wav_handle = None


class LabelWriter(LabelWriterBase):

    def __init__(self, vad_text_file_path: str):
        self.file_path = vad_text_file_path
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def write_segment(self, start: float, end: float, label: str) -> None:
        with open(self.file_path, mode='a', encoding='utf-8') as f:
            f.write(f'{start:.2f}\t{end:.2f}\t{label}\n')


class DummySpeechAudioWriter(AudioWriterBase):
    """音声認識1回分の音声を出力しないダミークラス."""

    def open(self) -> None:  # noqa: A003
        pass

    def write(self, data: bytes) -> None:
        pass

    def close(self) -> None:
        pass


class SpeechAudioWriter(AudioWriterBase):
    """音声認識1回分の音声を出力するクラス."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.current_wav_handle = None
        self.current_file_path = None
        self.logger = get_logger('speech_audio_writer')
        os.makedirs(self.output_dir, exist_ok=True)

    def open(self) -> None:  # noqa: A003
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.current_file_path = os.path.join(self.output_dir, f'speech_{timestamp}.wav')
        self.current_wav_handle = wave.open(self.current_file_path, 'wb')
        self.current_wav_handle.setnchannels(CHANNELS)
        self.current_wav_handle.setsampwidth(SAMPLE_WIDTH)
        self.current_wav_handle.setframerate(SAMPLE_RATE)
        self.logger.info(f'音声認識セッションの音声出力開始: {self.current_file_path}')

    def write(self, data: bytes) -> None:
        if self.current_wav_handle:
            self.current_wav_handle.writeframes(data)

    def close(self) -> None:
        if self.current_wav_handle:
            self.current_wav_handle.close()
            self.current_wav_handle = None
            self.logger.info(f'音声認識セッションの音声出力終了: {self.current_file_path}')
