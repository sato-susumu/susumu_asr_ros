"""Silero VAD を用いた発話区間検出プラグイン."""
import collections
import math

import numpy as np
from rclpy.logging import get_logger
from susumu_asr_ros.constants import AUDIO_FRAME_SAMPLES, FRAME_LENGTH_MS, INT16_MAX
from susumu_asr_ros.plugin_base import PluginParam, VADEvent, VADPluginBase, VADResult
import torch


class SilenceAwareVADIterator:
    """
    Silero VADIterator のラッパー.

    発話終了判定を指定された無音継続時間に基づいて行う.
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
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = self.utils

        self.vad_iterator = self.VADIterator(self.model, threshold=threshold)
        self.silence_frame_count = 0
        self.silence_threshold_frames = int(
            math.ceil(silence_threshold_ms / FRAME_LENGTH_MS)
        )
        self.in_speech = False
        self.last_result = None

    def __call__(self, audio_float32, return_seconds=False):
        if len(audio_float32) < AUDIO_FRAME_SAMPLES:
            raise ValueError(
                f'Silero VAD には {AUDIO_FRAME_SAMPLES} サンプル以上必要です。'
                f'受け取ったサイズ: {len(audio_float32)}'
            )
        result = self.vad_iterator(audio_float32, return_seconds=return_seconds)
        self.last_result = result

        if result and 'start' in result:
            self.in_speech = True
            self.silence_frame_count = 0
            return result
        elif result and 'end' in result:
            if self.in_speech:
                self.silence_frame_count = 1
                self.in_speech = False
                return None
            else:
                self.silence_frame_count += 1
                if self.silence_frame_count >= self.silence_threshold_frames:
                    self.silence_frame_count = 0
                    return result
                return None
        else:
            if self.in_speech:
                self.silence_frame_count = 0
                return result
            else:
                if self.silence_frame_count > 0:
                    self.silence_frame_count += 1
                    if self.silence_frame_count >= self.silence_threshold_frames:
                        self.silence_frame_count = 0
                        return {'end': True}
                return None


class SileroVADPlugin(VADPluginBase):
    """Silero VAD を用いた発話検知プラグイン."""

    plugin_name = 'silero_vad'

    DEFAULT_THRESHOLD = 0.5
    DEFAULT_SILENCE_THRESHOLD_MS = 1000
    DEFAULT_PRE_SPEECH_MS = 300

    def get_param_declarations(self) -> list[PluginParam]:
        return [
            PluginParam(
                'threshold', self.DEFAULT_THRESHOLD,
                'VAD 検出しきい値 (0.0–1.0)',
            ),
            PluginParam(
                'silence_threshold_ms', self.DEFAULT_SILENCE_THRESHOLD_MS,
                '発話終了とみなす無音時間 (ms)',
            ),
            PluginParam(
                'pre_speech_ms', self.DEFAULT_PRE_SPEECH_MS,
                '発話開始時に遡って送るバッファ時間 (ms)',
            ),
        ]

    def load_params(self, params: dict) -> None:
        self._threshold = float(params.get('threshold', self.DEFAULT_THRESHOLD))
        self._silence_ms = int(
            params.get('silence_threshold_ms', self.DEFAULT_SILENCE_THRESHOLD_MS)
        )
        self._pre_speech_ms = int(
            params.get('pre_speech_ms', self.DEFAULT_PRE_SPEECH_MS)
        )

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
        self._vad_it.silence_threshold_frames = int(
            math.ceil(silence_threshold_ms / FRAME_LENGTH_MS)
        )

    def process_frame(self, frame: bytes) -> VADResult:
        self._pre_speech_buffer.append(frame)
        data_np = np.frombuffer(frame, dtype=np.int16)
        audio_float32 = torch.from_numpy(data_np).float() / INT16_MAX
        result = self._vad_it(audio_float32, return_seconds=False)

        if result and 'start' in result:
            if not self.in_speech:
                self.in_speech = True
                return VADResult(VADEvent.VAD_START, list(self._pre_speech_buffer))
            return VADResult(VADEvent.VAD_CONT, [frame])
        elif result and 'end' in result:
            if self.in_speech:
                self.in_speech = False
                return VADResult(VADEvent.VAD_END, [frame])
            return VADResult(VADEvent.SILENCE, [])
        else:
            if self.in_speech:
                return VADResult(VADEvent.VAD_CONT, [frame])
            return VADResult(VADEvent.SILENCE, [])
