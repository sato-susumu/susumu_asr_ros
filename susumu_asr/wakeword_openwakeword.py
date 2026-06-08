"""OpenWakeWord によるウェイクワード検出プラグイン."""
import os

import numpy as np
import openwakeword
from openwakeword.model import Model
from susumu_asr.ros_logger import get_logger
from susumu_asr.plugin_base import (
    PluginParam, WakewordEvent, WakewordPluginBase, WakewordResult,
)


class OpenWakewordPlugin(WakewordPluginBase):
    """
    OpenWakeWord による tflite ウェイクワード検出プラグイン.

    フレームごとにスコアを算出し、threshold を超えたら DETECTED を返す。
    """

    plugin_name = 'openwakeword'

    DEFAULT_MODEL_FOLDER = 'models'
    DEFAULT_MODEL_NAME = 'hey_mycroft_v0.1.tflite'
    DEFAULT_THRESHOLD = 0.5

    def get_param_declarations(self) -> list[PluginParam]:
        return [
            PluginParam('model_folder', self.DEFAULT_MODEL_FOLDER,
                        'モデルファイルが置かれたディレクトリ'),
            PluginParam('model_name', self.DEFAULT_MODEL_NAME,
                        '使用するウェイクワードモデルのファイル名'),
            PluginParam('threshold', self.DEFAULT_THRESHOLD,
                        'ウェイクワード検出しきい値 (0.0–1.0)'),
        ]

    def load_params(self, params: dict) -> None:
        self._model_folder = params.get('model_folder', self.DEFAULT_MODEL_FOLDER)
        self._model_name = params.get('model_name', self.DEFAULT_MODEL_NAME)
        self._threshold = float(params.get('threshold', self.DEFAULT_THRESHOLD))

    def setup(self) -> None:
        self.logger = get_logger('openwakeword')
        self.logger.info('OpenWakeWord のモデルをロードします...')
        openwakeword.utils.download_models()
        os.makedirs(self._model_folder, exist_ok=True)
        openwakeword.utils.download_models(target_directory=self._model_folder)
        model_path = os.path.join(self._model_folder, self._model_name)
        self._oww_model = Model(wakeword_models=[model_path])

    def reset(self) -> None:
        for buf in self._oww_model.prediction_buffer.values():
            buf.clear()

    def process_frame(self, frame: bytes) -> WakewordResult:
        data_np = np.frombuffer(frame, dtype=np.int16)
        self._oww_model.predict(data_np)

        score = 0.0
        for scores in self._oww_model.prediction_buffer.values():
            score = scores[-1] if scores else 0.0

        if score > self._threshold:
            self.logger.info(f'ウェイクワード検出: score={score:.3f}')
            return WakewordResult(event=WakewordEvent.DETECTED, score=score)
        return WakewordResult(event=WakewordEvent.SEARCHING, score=score)
