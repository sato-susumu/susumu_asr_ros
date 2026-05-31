"""livekit-wakeword によるウェイクワード検出 + Silero VAD 発話終了検出プラグイン."""
import os

import numpy as np
from rclpy.logging import get_logger
from susumu_asr_ros.constants import (
    FRAME_LENGTH_MS,
    INT16_MAX,
    MS_PER_SEC,
    SAMPLE_RATE,
)
from susumu_asr_ros.plugin_base import PluginParam, VADEvent, VADPluginBase, VADResult
from susumu_asr_ros.vad_silero import SilenceAwareVADIterator
import torch

_WINDOW_SEC = 2.0
_STRIDE_SEC = 0.2
_WINDOW_SAMPLES = int(SAMPLE_RATE * _WINDOW_SEC)
_STRIDE_SAMPLES = int(SAMPLE_RATE * _STRIDE_SEC)


def _download_model(model_path: str, logger) -> None:
    """モデルファイルが存在しない場合 openWakeWord GitHub から自動ダウンロードする."""
    import urllib.request
    dest = os.path.abspath(model_path)
    if os.path.exists(dest):
        return
    base = 'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1'
    url = f'{base}/{os.path.basename(dest)}'
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    logger.info(f'モデルが見つかりません。ダウンロードします: {url}')
    tmp = dest + '.tmp'
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
        logger.info(f'ダウンロード完了: {dest}')
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f'モデルのダウンロードに失敗しました: {url}\n{e}') from e


class LivekitWakeWordPlugin(VADPluginBase):
    """
    livekit-wakeword によるウェイクワード検出 + Silero VAD 発話終了検出プラグイン.

    - ONNX 形式のモデル（hey_mycroft_v0.1.onnx 等）を使用する。
    - 2 秒のリングバッファに音声を蓄積し、0.2 秒ごとに WakeWordModel.predict() を呼ぶ。
    - スコアが threshold を超えたら SPEECH_START を返す。
    - 発話開始後は Silero VAD で終了を検出するか、タイムアウトで SPEECH_STOP を返す。
    """

    plugin_name = 'livekit_wakeword'

    DEFAULT_MODEL_FOLDER = 'models'
    DEFAULT_MODEL_NAME = 'hey_mycroft_v0.1.onnx'
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
                        '使用するウェイクワード ONNX モデルのファイル名'),
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
        from livekit.wakeword import WakeWordModel

        self.logger = get_logger('livekit_wakeword')

        self._vad_it = SilenceAwareVADIterator(
            silence_threshold_ms=self._silence_ms,
            threshold=self._vad_threshold,
        )

        model_path = os.path.join(self._model_folder, self._model_name)
        _download_model(model_path, self.logger)

        self.logger.info(f'livekit-wakeword モデルをロードします: {model_path}')
        self._wakeword_model = WakeWordModel(models=[model_path])
        self._model_key = os.path.splitext(self._model_name)[0]

        # リングバッファ (2 秒ウィンドウ)
        self._audio_buf = np.zeros(_WINDOW_SAMPLES, dtype=np.float32)
        self._write_pos = 0
        self._filled = 0
        self._samples_since_infer = 0

        self.in_speech = False
        self._frame_count_since_start = 0
        self.last_score = 0.0

    def process_frame(self, frame: bytes) -> VADResult:
        data_np = np.frombuffer(frame, dtype=np.int16)
        audio_f32 = data_np.astype(np.float32) / INT16_MAX

        # Silero VAD（発話終了判定用）
        audio_tensor = torch.from_numpy(data_np).float() / INT16_MAX
        silero_result = self._vad_it(audio_tensor, return_seconds=False)
        silero_end = (silero_result is not None) and ('end' in silero_result)

        # 発話中は終了判定のみ行う
        if self.in_speech:
            self._frame_count_since_start += 1
            elapsed = (self._frame_count_since_start * FRAME_LENGTH_MS) / MS_PER_SEC
            if (silero_end and elapsed > self._speech_end_min_sec) or (
                elapsed >= self._speech_timeout_sec
            ):
                self.in_speech = False
                return VADResult(VADEvent.SPEECH_STOP, [frame])
            return VADResult(VADEvent.SPEECH_CONT, [frame])

        # リングバッファ更新
        n = len(audio_f32)
        end = self._write_pos + n
        if end <= _WINDOW_SAMPLES:
            self._audio_buf[self._write_pos:end] = audio_f32
        else:
            first = _WINDOW_SAMPLES - self._write_pos
            self._audio_buf[self._write_pos:] = audio_f32[:first]
            self._audio_buf[:end - _WINDOW_SAMPLES] = audio_f32[first:]
        self._write_pos = end % _WINDOW_SAMPLES
        self._filled = min(self._filled + n, _WINDOW_SAMPLES)
        self._samples_since_infer += n

        # ウィンドウが埋まるまで、またはストライド未満はスキップ
        if self._filled < _WINDOW_SAMPLES:
            return VADResult(VADEvent.SILENCE, [])
        if self._samples_since_infer < _STRIDE_SAMPLES:
            return VADResult(VADEvent.SILENCE, [])
        self._samples_since_infer = 0

        # リングバッファを時系列順に並べて推論
        window = np.concatenate([
            self._audio_buf[self._write_pos:],
            self._audio_buf[:self._write_pos],
        ])
        scores = self._wakeword_model.predict(window)
        self.last_score = max(scores.values()) if scores else 0.0

        if self.last_score >= self._threshold:
            self.logger.info(
                f'ウェイクワード検出: {self._model_key} score={self.last_score:.3f}'
            )
            self.in_speech = True
            self._frame_count_since_start = 0
            return VADResult(VADEvent.SPEECH_START, [frame])

        return VADResult(VADEvent.SILENCE, [])
