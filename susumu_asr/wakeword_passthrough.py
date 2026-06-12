"""ウェイクワード検出をスキップするパススループラグイン."""
from susumu_asr.constants import FRAME_LENGTH_MS, MS_PER_SEC
from susumu_asr.plugin_base import (
    PluginParam,
    WakewordEvent,
    WakewordPluginBase,
    WakewordResult,
)
from susumu_asr.ros_logger import get_logger


class PassthroughWakewordPlugin(WakewordPluginBase):
    """
    ウェイクワード検出をスキップするパススループラグイン.

    VAD_START から delay_sec 秒後に DETECTED を返す。
    SileroVAD 単体モードで使用し、イベントフローを他のプラグインと統一する。
    """

    plugin_name = 'passthrough'
    extend_silence_on_detected = False

    DEFAULT_DELAY_SEC = 0.5
    DEFAULT_THRESHOLD = 0.5

    def get_param_declarations(self) -> list[PluginParam]:
        return [
            PluginParam('delay_sec', self.DEFAULT_DELAY_SEC,
                        'VAD_START から DETECTED を返すまでの遅延秒数'),
            PluginParam('threshold', self.DEFAULT_THRESHOLD,
                        'ウェイクワード検出しきい値'),
        ]

    def load_params(self, params: dict) -> None:
        self._delay_sec = float(params.get('delay_sec', self.DEFAULT_DELAY_SEC))
        self._threshold = float(params.get('threshold', self.DEFAULT_THRESHOLD))

    def setup(self) -> None:
        self.logger = get_logger('passthrough_wakeword')
        self._frame_count = 0
        self._delay_frames = int(self._delay_sec * MS_PER_SEC / FRAME_LENGTH_MS)

    def reset(self) -> None:
        self._frame_count = 0

    def process_frame(self, frame: bytes) -> WakewordResult:
        self._frame_count += 1
        score = 1.0 if self._frame_count >= self._delay_frames else 0.0
        if score >= self._threshold:
            self.logger.info(f'パススルー検出: score={score:.3f}')
            return WakewordResult(event=WakewordEvent.DETECTED, score=score)
        return WakewordResult(event=WakewordEvent.SEARCHING, score=score)
