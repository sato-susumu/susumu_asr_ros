"""Silero VAD を用いた発話区間検出プラグイン."""
import collections
from enum import Enum
import math

import numpy as np
from rclpy.logging import get_logger
from susumu_asr_ros.constants import AUDIO_FRAME_SAMPLES, FRAME_LENGTH_MS, INT16_MAX
from susumu_asr_ros.plugin_base import PluginParam, VADEvent, VADPluginBase, VADResult
import torch


class _SileroEvent(str, Enum):
    """SilenceAwareVADIterator の内部イベント."""
    START = 'start'
    END = 'end'
    NONE = 'none'


class SilenceAwareVADIterator:
    """
    Silero VADIterator のラッパー.

    Silero が返す 'start'/'end' イベントをもとに、silence_threshold_frames 分の
    無音継続を確認してから VAD_END を確定する。
    """

    MODEL_NAME = 'silero_vad'
    REPO = 'snakers4/silero-vad:v4.0'

    def __init__(self, silence_threshold_ms: int, threshold: float):
        self.logger = get_logger('silence_aware_vad')
        self.logger.info('Torch Hub からモデルをロードします...')
        self.model, self.utils = torch.hub.load(
            repo_or_dir=self.REPO,
            model=self.MODEL_NAME,
            force_reload=False,
        )
        _, _, _, self.VADIterator, _ = self.utils

        self.vad_iterator = self.VADIterator(self.model, threshold=threshold)
        self.silence_threshold_frames = int(math.ceil(silence_threshold_ms / FRAME_LENGTH_MS))
        self._silence_frame_count = 0
        self._waiting_for_end = False

    def __call__(self, audio_float32) -> _SileroEvent:
        """フレームを処理して _SileroEvent を返す."""
        if len(audio_float32) < AUDIO_FRAME_SAMPLES:
            raise ValueError(
                f'Silero VAD には {AUDIO_FRAME_SAMPLES} サンプル以上必要です。'
                f'受け取ったサイズ: {len(audio_float32)}'
            )

        raw = self.vad_iterator(audio_float32, return_seconds=False)

        if raw and 'start' in raw:
            self._waiting_for_end = False
            self._silence_frame_count = 0
            return _SileroEvent.START

        if raw and 'end' in raw:
            self._waiting_for_end = True
            self._silence_frame_count = 1

        if self._waiting_for_end:
            if raw is None:
                self._silence_frame_count += 1
            if self._silence_frame_count >= self.silence_threshold_frames:
                self._waiting_for_end = False
                self._silence_frame_count = 0
                return _SileroEvent.END

        return _SileroEvent.NONE


class SileroVADPlugin(VADPluginBase):
    """Silero VAD を用いた発話検知プラグイン."""

    plugin_name = 'silero_vad'

    DEFAULT_THRESHOLD = 0.5
    DEFAULT_SILENCE_THRESHOLD_MS = 1000
    DEFAULT_PRE_SPEECH_MS = 400

    def get_param_declarations(self) -> list[PluginParam]:
        return [
            PluginParam('threshold', self.DEFAULT_THRESHOLD, 'VAD 検出しきい値 (0.0–1.0)'),
            PluginParam('silence_threshold_ms', self.DEFAULT_SILENCE_THRESHOLD_MS, '発話終了とみなす無音時間 (ms)'),
            PluginParam('pre_speech_ms', self.DEFAULT_PRE_SPEECH_MS, '発話開始時に遡って送るバッファ時間 (ms)'),
        ]

    def load_params(self, params: dict) -> None:
        self._threshold = float(params.get('threshold', self.DEFAULT_THRESHOLD))
        self._silence_ms = int(params.get('silence_threshold_ms', self.DEFAULT_SILENCE_THRESHOLD_MS))
        self._pre_speech_ms = int(params.get('pre_speech_ms', self.DEFAULT_PRE_SPEECH_MS))

    def setup(self) -> None:
        self.logger = get_logger('silero_vad')
        self.logger.info('SilenceAwareVADIterator を初期化します...')
        self._vad_it = SilenceAwareVADIterator(
            silence_threshold_ms=self._silence_ms,
            threshold=self._threshold,
        )
        pre_speech_frames = self._pre_speech_ms // FRAME_LENGTH_MS
        self._pre_speech_buffer = collections.deque(maxlen=pre_speech_frames)
        self.in_speech = False

    def extend_silence_threshold(self, silence_threshold_ms: int) -> None:
        """発話終了とみなす無音時間を変更する."""
        self._vad_it.silence_threshold_frames = int(math.ceil(silence_threshold_ms / FRAME_LENGTH_MS))

    def process_frame(self, frame: bytes) -> VADResult:
        self._pre_speech_buffer.append(frame)
        audio_float32 = torch.from_numpy(
            np.frombuffer(frame, dtype=np.int16)
        ).float() / INT16_MAX

        event = self._vad_it(audio_float32)

        if event == _SileroEvent.START and not self.in_speech:
            self.in_speech = True
            return VADResult(VADEvent.VAD_START, list(self._pre_speech_buffer))
        if event == _SileroEvent.END and self.in_speech:
            self.in_speech = False
            return VADResult(VADEvent.VAD_END, [frame])
        if self.in_speech:
            return VADResult(VADEvent.VAD_CONT, [frame])
        return VADResult(VADEvent.SILENCE, [])
