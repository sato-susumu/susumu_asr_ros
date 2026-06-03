"""音声録音・書き込みクラス群."""
import os
from abc import ABC, abstractmethod
from datetime import datetime
import time
import wave

import matplotlib
matplotlib.use('Agg')  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyaudio  # noqa: E402
from susumu_asr_ros.ros_logger import get_logger  # noqa: E402

from susumu_asr_ros.constants import (  # noqa: E402
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
        self.current_file_path = os.path.join(
            self.output_dir, f'speech_{timestamp}.wav'
        )
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


_LABEL_COLORS = {
    'vad_speech': '#4a90d9',
    'ww_detected': '#e67e22',
}
_DEFAULT_LABEL_COLOR = '#27ae60'
_SYSTEM_LABELS = {'vad_speech', 'ww_detected'}


class WaveformImageWriter:
    """WAV波形とラベルを重ねた画像を生成するクラス."""

    def __init__(self, wav_path: str, label_path: str, image_path: str):
        self.wav_path = wav_path
        self.label_path = label_path
        self.image_path = image_path
        self.logger = get_logger('waveform_image_writer')

    def generate(self) -> None:
        """WAVとラベルファイルを読み込み、画像を出力する."""
        if not os.path.exists(self.wav_path):
            self.logger.warning(f'WAVファイルが見つかりません: {self.wav_path}')
            return

        samples, duration = self._load_wav()
        labels = self._load_labels()
        self._plot(samples, duration, labels)
        self.logger.info(f'波形画像を出力: {self.image_path}')

    def _load_wav(self):
        with wave.open(self.wav_path, 'rb') as wf:
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
        samples = (
            np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        )
        duration = len(samples) / SAMPLE_RATE
        return samples, duration

    def _load_labels(self) -> list[tuple[float, float, str]]:
        labels = []
        if not os.path.exists(self.label_path):
            return labels
        with open(self.label_path, encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 3:
                    continue
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    label = parts[2]
                    labels.append((start, end, label))
                except ValueError:
                    continue
        return labels

    def _plot(self, samples, duration: float, labels: list) -> None:
        plt.rcParams['font.family'] = 'Noto Sans CJK JP'
        times = np.linspace(0, duration, len(samples))

        fig, (ax_wave, ax_label) = plt.subplots(
            2, 1, figsize=(18, 5),
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True,
        )
        fig.subplots_adjust(hspace=0.05)

        # --- 上段: 波形 ---
        ax_wave.plot(times, samples, linewidth=0.3, color='#2c3e50')
        ax_wave.set_ylim(-1.05, 1.05)
        ax_wave.set_ylabel('Amplitude')
        ax_wave.grid(True, axis='x', linestyle='--', linewidth=0.4, alpha=0.5)

        # 区間ラベルを波形に薄く塗り、点ラベル(ww_detected等)は垂直線で表示
        for start, end, label in labels:
            if label not in _SYSTEM_LABELS:
                continue
            color = _LABEL_COLORS.get(label, _DEFAULT_LABEL_COLOR)
            if start == end:
                ax_wave.axvline(start, color=color, linewidth=1.2, alpha=0.9)
            else:
                ax_wave.axvspan(start, end, alpha=0.15, color=color)

        # ASR結果を波形上段に表示
        for start, end, label in labels:
            if label in _SYSTEM_LABELS:
                continue
            cx = (start + end) / 2
            ax_wave.text(
                cx, 0.92, label,
                ha='center', va='top', fontsize=8, color='#2c3e50',
                transform=ax_wave.get_xaxis_transform(),
                clip_on=True,
                bbox=dict(
                    facecolor='#f9e79f', edgecolor='#d4ac0d',
                    alpha=0.9, pad=2, boxstyle='round,pad=0.3',
                ),
            )

        # --- 下段: ラベルバー ---
        ax_label.set_ylim(0, 1)
        ax_label.set_yticks([])
        ax_label.set_xlabel('Time (s)')

        seen_labels = {}
        for start, end, label in labels:
            if label not in _SYSTEM_LABELS:
                continue
            color = _LABEL_COLORS.get(label, _DEFAULT_LABEL_COLOR)
            if start == end:
                # 点イベント: 垂直線＋上部テキスト
                ax_label.axvline(start, color=color, linewidth=1.5, alpha=0.9)
                ax_label.text(
                    start, 0.92, label,
                    ha='center', va='top', fontsize=7, color=color,
                    clip_on=True,
                    bbox=dict(
                        facecolor='white', edgecolor='none',
                        alpha=0.7, pad=1,
                    ),
                )
            else:
                width = end - start
                rect = mpatches.FancyBboxPatch(
                    (start, 0.1), width, 0.8,
                    boxstyle='round,pad=0.01',
                    facecolor=color, edgecolor='white',
                    linewidth=0.5, alpha=0.85,
                )
                ax_label.add_patch(rect)
                short = label if len(label) <= 20 else label[:18] + '…'
                ax_label.text(
                    start + width / 2, 0.5, short,
                    ha='center', va='center', fontsize=7, color='white',
                    clip_on=True,
                )
            if label not in seen_labels:
                seen_labels[label] = mpatches.Patch(color=color, label=label)

        if seen_labels:
            ax_label.legend(
                handles=list(seen_labels.values()),
                loc='upper right', fontsize=7, framealpha=0.7,
            )

        ax_wave.set_title(os.path.basename(self.wav_path))
        plt.savefig(self.image_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
