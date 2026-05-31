"""OpenWakeWord ウェイクワード検出 + Silero VAD 発話終了検出プラグイン."""
import os

import numpy as np
import openwakeword
from openwakeword.model import Model
from rclpy.logging import get_logger
from susumu_asr_ros.constants import FRAME_LENGTH_MS, INT16_MAX, MS_PER_SEC
from susumu_asr_ros.plugin_base import PluginParam, VADEvent, VADPluginBase, VADResult
from susumu_asr_ros.vad_silero import SilenceAwareVADIterator
import torch


class OpenWakeWordPlugin(VADPluginBase):
    """
    OpenWakeWord ウェイクワード検出 + Silero VAD 発話終了検出プラグイン.

    ウェイクワード検出後、Silero VAD で発話終了またはタイムアウトで停止。
    """

    plugin_name = 'openwakeword'

    DEFAULT_MODEL_FOLDER = 'models'
    DEFAULT_MODEL_NAME = 'hey_mycroft_v0.1.tflite'
    DEFAULT_THRESHOLD = 0.5
    DEFAULT_SPEECH_TIMEOUT_SEC = 8.0
    DEFAULT_SPEECH_END_MIN_SEC = 2.0
    DEFAULT_SILENCE_THRESHOLD_MS = 2000
    DEFAULT_VAD_THRESHOLD = 0.5

    def get_param_declarations(self) -> list[PluginParam]:
        return [
            PluginParam('model_folder', self.DEFAULT_MODEL_FOLDER,
                        'モデルファイルが置かれたディレクトリ'),
            PluginParam('model_name', self.DEFAULT_MODEL_NAME,
                        '使用するウェイクワードモデルのファイル名'),
            PluginParam('threshold', self.DEFAULT_THRESHOLD,
                        'ウェイクワード検出しきい値 (0.0–1.0)'),
            PluginParam('speech_timeout_sec', self.DEFAULT_SPEECH_TIMEOUT_SEC,
                        'ウェイクワード後の最大録音時間 (秒)'),
            PluginParam('speech_end_min_sec', self.DEFAULT_SPEECH_END_MIN_SEC,
                        '発話終了を認める最短経過時間 (秒)'),
            PluginParam('silence_threshold_ms', self.DEFAULT_SILENCE_THRESHOLD_MS,
                        '発話終了とみなす無音時間 (ms)'),
            PluginParam('vad_threshold', self.DEFAULT_VAD_THRESHOLD,
                        'Silero VAD のしきい値 (0.0–1.0)'),
        ]

    def load_params(self, params: dict) -> None:
        self._model_folder = params.get('model_folder', self.DEFAULT_MODEL_FOLDER)
        self._model_name = params.get('model_name', self.DEFAULT_MODEL_NAME)
        self._threshold = float(params.get('threshold', self.DEFAULT_THRESHOLD))
        self._speech_timeout_sec = float(
            params.get('speech_timeout_sec', self.DEFAULT_SPEECH_TIMEOUT_SEC)
        )
        self._speech_end_min_sec = float(
            params.get('speech_end_min_sec', self.DEFAULT_SPEECH_END_MIN_SEC)
        )
        self._silence_ms = int(
            params.get('silence_threshold_ms', self.DEFAULT_SILENCE_THRESHOLD_MS)
        )
        self._vad_threshold = float(
            params.get('vad_threshold', self.DEFAULT_VAD_THRESHOLD)
        )

    def setup(self) -> None:
        self.logger = get_logger('open_wake_word')

        self._vad_it = SilenceAwareVADIterator(
            silence_threshold_ms=self._silence_ms,
            threshold=self._vad_threshold,
        )

        self.logger.info('OpenWakeWord のモデルをロードします...')
        openwakeword.utils.download_models()
        os.makedirs(self._model_folder, exist_ok=True)
        openwakeword.utils.download_models(target_directory=self._model_folder)

        model_path = os.path.join(self._model_folder, self._model_name)
        self._oww_model = Model(wakeword_models=[model_path])

        self.in_speech = False
        self._frame_count_since_start = 0
        self._total_frame_count = 0

    def process_frame(self, frame: bytes) -> VADResult:
        self._total_frame_count += 1

        data_np = np.frombuffer(frame, dtype=np.int16)
        self._oww_model.predict(data_np)

        oww_score = 0.0
        for scores in self._oww_model.prediction_buffer.values():
            oww_score = scores[-1] if scores else 0.0

        audio_float32 = torch.from_numpy(data_np).float() / INT16_MAX
        silero_result = self._vad_it(audio_float32, return_seconds=False)
        silero_end = (silero_result is not None) and ('end' in silero_result)

        if not self.in_speech:
            if oww_score > self._threshold:
                self.in_speech = True
                self._frame_count_since_start = 0
                return VADResult(VADEvent.SPEECH_START, [frame])
            return VADResult(VADEvent.SILENCE, [])
        else:
            self._frame_count_since_start += 1
            elapsed = (self._frame_count_since_start * FRAME_LENGTH_MS) / MS_PER_SEC

            if (silero_end and elapsed > self._speech_end_min_sec) or (
                elapsed >= self._speech_timeout_sec
            ):
                self.in_speech = False
                return VADResult(VADEvent.SPEECH_STOP, [frame])
            return VADResult(VADEvent.SPEECH_CONT, [frame])
