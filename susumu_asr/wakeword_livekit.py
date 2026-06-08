"""livekit-wakeword によるウェイクワード検出プラグイン."""
import os

import numpy as np
from susumu_asr.ros_logger import get_logger
from susumu_asr.constants import INT16_MAX, SAMPLE_RATE
from susumu_asr.plugin_base import (
    PluginParam, WakewordEvent, WakewordPluginBase, WakewordResult,
)

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


class LivekitWakewordPlugin(WakewordPluginBase):
    """
    livekit-wakeword による ONNX ウェイクワード検出プラグイン.

    2 秒のリングバッファに音声を蓄積し、0.2 秒ごとに推論する。
    スコアが threshold を超えたら DETECTED を返す。
    """

    plugin_name = 'livekit_wakeword'

    DEFAULT_MODEL_FOLDER = 'models'
    DEFAULT_MODEL_NAME = 'hey_mycroft_v0.1.onnx'
    DEFAULT_THRESHOLD = 0.5

    def get_param_declarations(self) -> list[PluginParam]:
        return [
            PluginParam('model_folder', self.DEFAULT_MODEL_FOLDER,
                        'モデルファイルが置かれたディレクトリ'),
            PluginParam('model_name', self.DEFAULT_MODEL_NAME,
                        '使用するウェイクワード ONNX モデルのファイル名'),
            PluginParam('threshold', self.DEFAULT_THRESHOLD,
                        'ウェイクワード検出しきい値 (0.0–1.0)'),
        ]

    def load_params(self, params: dict) -> None:
        self._model_folder = params.get('model_folder', self.DEFAULT_MODEL_FOLDER)
        self._model_name = params.get('model_name', self.DEFAULT_MODEL_NAME)
        self._threshold = float(params.get('threshold', self.DEFAULT_THRESHOLD))

    def setup(self) -> None:
        from livekit.wakeword import WakeWordModel

        self.logger = get_logger('livekit_wakeword')
        model_path = os.path.join(self._model_folder, self._model_name)
        _download_model(model_path, self.logger)

        self.logger.info(f'livekit-wakeword モデルをロードします: {model_path}')
        self._wakeword_model = WakeWordModel(models=[model_path])
        self._model_key = os.path.splitext(self._model_name)[0]
        self._audio_buf = np.zeros(_WINDOW_SAMPLES, dtype=np.float32)
        self._write_pos = 0
        self._filled = 0
        self._samples_since_infer = 0

    def reset(self) -> None:
        self._samples_since_infer = 0

    def process_frame(self, frame: bytes) -> WakewordResult:
        data_np = np.frombuffer(frame, dtype=np.int16)
        audio_f32 = data_np.astype(np.float32) / INT16_MAX

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

        if self._filled < _WINDOW_SAMPLES:
            return WakewordResult(event=WakewordEvent.SEARCHING)
        if self._samples_since_infer < _STRIDE_SAMPLES:
            return WakewordResult(event=WakewordEvent.SEARCHING)
        self._samples_since_infer = 0

        window = np.concatenate([
            self._audio_buf[self._write_pos:],
            self._audio_buf[:self._write_pos],
        ])
        scores = self._wakeword_model.predict(window)
        score = max(scores.values()) if scores else 0.0

        if score >= self._threshold:
            self.logger.info(
                f'ウェイクワード検出: {self._model_key} score={score:.3f}'
            )
            return WakewordResult(event=WakewordEvent.DETECTED, score=score)
        return WakewordResult(event=WakewordEvent.SEARCHING)
