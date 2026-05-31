"""ウェイクワード検出をスキップするパススループラグイン."""
from susumu_asr_ros.constants import FRAME_LENGTH_MS, MS_PER_SEC
from susumu_asr_ros.plugin_base import (
    PluginParam, WakewordEvent, WakewordPluginBase, WakewordResult,
)


class PassthroughWakewordPlugin(WakewordPluginBase):
    """
    ウェイクワード検出をスキップするパススループラグイン.

    VAD_START から delay_sec 秒後に DETECTED を返す。
    SileroVAD 単体モードで使用し、イベントフローを他のプラグインと統一する。
    """

    plugin_name = 'passthrough'

    DEFAULT_DELAY_SEC = 0.5

    def get_param_declarations(self) -> list[PluginParam]:
        return [
            PluginParam('delay_sec', self.DEFAULT_DELAY_SEC,
                        'VAD_START から DETECTED を返すまでの遅延秒数'),
        ]

    def load_params(self, params: dict) -> None:
        self._delay_sec = float(params.get('delay_sec', self.DEFAULT_DELAY_SEC))

    def setup(self) -> None:
        self._frame_count = 0
        self._delay_frames = int(self._delay_sec * MS_PER_SEC / FRAME_LENGTH_MS)

    def reset(self) -> None:
        self._frame_count = 0

    def process_frame(self, frame: bytes) -> WakewordResult:
        self._frame_count += 1
        if self._frame_count >= self._delay_frames:
            return WakewordResult(event=WakewordEvent.DETECTED, score=1.0)
        return WakewordResult(event=WakewordEvent.SEARCHING)
