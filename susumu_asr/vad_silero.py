"""Silero VAD を用いた発話区間検出プラグイン."""
import collections

import numpy as np
from susumu_asr.constants import FRAME_LENGTH_MS, INT16_MAX, SAMPLE_RATE
from susumu_asr.plugin_base import (
    PluginParam,
    VADEvent,
    VADPluginBase,
    VADResult,
)
from susumu_asr.ros_logger import get_logger
import torch


class SileroVADPlugin(VADPluginBase):
    """Silero VAD を用いた発話検知プラグイン."""

    plugin_name = 'silero_vad'

    DEFAULT_THRESHOLD = 0.5
    DEFAULT_SILENCE_THRESHOLD_MS = 2000
    DEFAULT_PRE_SPEECH_MS = 300
    DEFAULT_SPEECH_PAD_MS = 30

    def get_param_declarations(self) -> list[PluginParam]:
        """パラメータ宣言."""
        return [
            PluginParam(
                'threshold',
                self.DEFAULT_THRESHOLD,
                'VAD 検出しきい値 (0.0–1.0)',
            ),
            PluginParam(
                'silence_threshold_ms',
                self.DEFAULT_SILENCE_THRESHOLD_MS,
                '発話終了とみなす無音時間 (ms)',
            ),
            PluginParam(
                'pre_speech_ms',
                self.DEFAULT_PRE_SPEECH_MS,
                '発話開始時に遡って送るバッファ時間 (ms)',
            ),
            PluginParam(
                'speech_pad_ms',
                self.DEFAULT_SPEECH_PAD_MS,
                'start/end タイムスタンプに付加するパディング (ms)',
            ),
        ]

    def load_params(self, params: dict) -> None:
        """パラメータをロードする."""
        self._threshold = float(
            params.get('threshold', self.DEFAULT_THRESHOLD)
        )
        self._silence_ms = int(
            params.get(
                'silence_threshold_ms', self.DEFAULT_SILENCE_THRESHOLD_MS
            )
        )
        self._pre_speech_ms = int(
            params.get('pre_speech_ms', self.DEFAULT_PRE_SPEECH_MS)
        )
        self._speech_pad_ms = int(
            params.get('speech_pad_ms', self.DEFAULT_SPEECH_PAD_MS)
        )

    def setup(self) -> None:
        """モデルをロードして内部状態を初期化する."""
        self.logger = get_logger('silero_vad')
        torch.set_num_threads(1)
        self._torch_threads_set = False
        self.logger.info('Torch Hub からモデルをロードします...')
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad:v4.0',
            model='silero_vad',
            force_reload=False,
        )
        _, _, _, VADIterator, _ = utils

        self._vad_iterator = VADIterator(
            model,
            threshold=self._threshold,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=self._silence_ms,
            speech_pad_ms=self._speech_pad_ms,
        )
        pre_speech_frames = self._pre_speech_ms // FRAME_LENGTH_MS
        self._pre_speech_buffer = collections.deque(maxlen=pre_speech_frames)
        self.in_speech = False

    def extend_silence_threshold(self, silence_threshold_ms: int) -> None:
        """発話終了とみなす無音時間を変更する."""
        self._vad_iterator.min_silence_samples = (
            SAMPLE_RATE * silence_threshold_ms / 1000
        )

    def process_frame(self, frame: bytes) -> VADResult:
        """1フレームを処理して VADResult を返す."""
        if not self._torch_threads_set:
            # Silero は 512 サンプルの極小推論を 32ms ごとに繰り返すため、
            # デフォルトの OpenMP プールではワーカーが推論の合間に
            # スピンウェイトし続け CPU を浪費する（実測 956%→14%）。
            # OpenMP のスレッド数設定は呼び出したスレッドにしか効かない
            # ので、setup()（メインスレッド）ではなく推論スレッド自身で
            # 設定する必要がある
            torch.set_num_threads(1)
            self._torch_threads_set = True
        self._pre_speech_buffer.append(frame)
        audio_float32 = torch.from_numpy(
            np.frombuffer(frame, dtype=np.int16).copy()
        ).float() / INT16_MAX

        raw = self._vad_iterator(audio_float32, return_seconds=False)

        if raw and 'start' in raw:
            if not self.in_speech:
                self.in_speech = True
                return VADResult(
                    VADEvent.VAD_START,
                    list(self._pre_speech_buffer),
                    speech_start_sec=raw['start'] / SAMPLE_RATE,
                )

        if raw and 'end' in raw:
            if self.in_speech:
                self.in_speech = False
                return VADResult(
                    VADEvent.VAD_END,
                    [frame],
                    speech_end_sec=raw['end'] / SAMPLE_RATE,
                )

        if self.in_speech:
            return VADResult(VADEvent.VAD_CONT, [frame])
        return VADResult(VADEvent.SILENCE, [])
