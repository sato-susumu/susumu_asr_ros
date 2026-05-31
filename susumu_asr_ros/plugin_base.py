"""ASR/VAD プラグインの抽象基底クラス、列挙型、データクラス、パラメータ宣言型."""
import queue
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class VADEvent(str, Enum):
    """VADプラグインの process_frame() が返すイベント名."""

    SILENCE = "silence"          # 無音・待機中（処理不要）
    SPEECH_START = "speech_start"
    SPEECH_CONT = "speech_cont"
    SPEECH_STOP = "speech_stop"
    SPEECH_TIMEOUT = "speech_timeout"


class ASRCommand(str, Enum):
    """audio_queue に送るコマンド名."""

    START = "start"
    AUDIO = "audio"
    STOP = "stop"
    STOP_ALL = "stop_all"


@dataclass
class VADResult:
    """process_frame() の戻り値.

    event が SILENCE のとき frames は空リスト。
    event が SPEECH_START のとき frames には発話開始前のバッファ＋現フレームが含まれる。
    それ以外のとき frames には現フレームのみが含まれる。
    """

    event: VADEvent
    frames: list[bytes]


@dataclass
class ASRResult:
    """result_queue から返る認識結果."""

    is_final: bool
    text: str
    start: float
    end: float | None  # partial結果では未確定のため None


@dataclass
class PluginParam:
    """プラグインが宣言するパラメータ1件."""

    name: str
    default: object
    description: str = ""


class ASRPluginBase(ABC):
    """
    ASRプラグインの基底クラス.

    ライフサイクル:
        __init__()  →  load_params()  →  setup()  →  run()
    """

    plugin_name: str = ""

    # setup() 後にセットされるキュー
    audio_queue: queue.Queue
    result_queue: queue.Queue

    def get_param_declarations(self) -> list[PluginParam]:
        """このプラグインが使うパラメータ一覧を返す。デフォルトは空。"""
        return []

    def load_params(self, params: dict) -> None:
        """ノードが解決したパラメータ値を受け取る。params のキーはプレフィックスなしの name。"""

    @abstractmethod
    def setup(
        self,
        audio_queue: queue.Queue,
        result_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> "ASRPluginBase":
        """モデルロードとキューの受け取りを行い self を返す。"""

    @abstractmethod
    def run(self) -> None:
        """ASRワーカースレッドのエントリポイント。"""


class VADPluginBase(ABC):
    """
    VADプラグインの基底クラス.

    ライフサイクル:
        __init__()  →  load_params()  →  setup()  →  process_frame()
    """

    plugin_name: str = ""

    # 発話中かどうかを示すフラグ。SpeechRecognitionSystem が WAV終端処理などで参照する。
    in_speech: bool = False

    def get_param_declarations(self) -> list[PluginParam]:
        """このプラグインが使うパラメータ一覧を返す。デフォルトは空。"""
        return []

    def load_params(self, params: dict) -> None:
        """ノードが解決したパラメータ値を受け取る。params のキーはプレフィックスなしの name。"""

    @abstractmethod
    def setup(self) -> None:
        """モデルロードなど重い初期化。load_params() の後に呼ばれる。"""

    @abstractmethod
    def process_frame(self, frame: bytes) -> VADResult:
        """1フレームを処理して VADResult を返す。詳細は VADResult の docstring を参照。"""
