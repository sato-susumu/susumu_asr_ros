"""ASR/VAD プラグインの抽象基底クラス、列挙型、データクラス、パラメータ宣言型."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import queue
import threading


class VADEvent(str, Enum):
    """VADプラグインの process_frame() が返すイベント名."""

    SILENCE = 'silence'      # 無音・待機中（処理不要）
    VAD_START = 'vad_start'
    VAD_CONT = 'vad_cont'
    VAD_END = 'vad_end'


class WakewordEvent(str, Enum):
    """WakewordPlugin の process_frame() が返すイベント名."""

    SEARCHING = 'searching'      # 検索中（未検出）
    DETECTED = 'detected'        # ウェイクワード検出


class ASRCommand(str, Enum):
    """audio_queue に送るコマンド名."""

    START = 'start'
    AUDIO = 'audio'
    STOP = 'stop'
    STOP_ALL = 'stop_all'


class ASREventType(str, Enum):
    """on_asr_event / on_status コールバックに渡すイベント種別."""

    VAD_START = 'vad_speech_start'
    VAD_STOP = 'vad_speech_stop'
    WAKEWORD_LISTENING_STARTED = 'ww_listening_started'
    WAKEWORD_DETECTED = 'ww_detected'
    ASR_PARTIAL_RESULT = 'asr_partial_result'
    ASR_FINAL_RESULT = 'asr_final_result'


@dataclass
class VadStartEvent:
    """VAD 発話開始イベント。on_asr_event 経由で通知される."""

    start: float
    pre_start: float | None = None  # pre_speech を含む開始時刻（None なら start と同じ）
    event_type: ASREventType = field(
        default=ASREventType.VAD_START, init=False
    )


@dataclass
class VadStopEvent:
    """VAD 発話終了イベント。on_asr_event 経由で通知される."""

    start: float
    end: float
    event_type: ASREventType = field(
        default=ASREventType.VAD_STOP, init=False
    )


@dataclass
class WakewordListeningStartedEvent:
    """ウェイクワード検出処理開始イベント。on_asr_event 経由で通知される."""

    start: float
    event_type: ASREventType = field(
        default=ASREventType.WAKEWORD_LISTENING_STARTED, init=False
    )


@dataclass
class WakewordDetectedEvent:
    """ウェイクワード検出イベント。on_asr_event 経由で通知される."""

    start: float
    score: float
    event_type: ASREventType = field(
        default=ASREventType.WAKEWORD_DETECTED, init=False
    )


@dataclass
class AsrPartialResultEvent:
    """ASR 途中認識結果イベント。on_asr_event 経由で通知される."""

    start: float
    text: str
    event_type: ASREventType = field(
        default=ASREventType.ASR_PARTIAL_RESULT, init=False
    )


@dataclass
class AsrFinalResultEvent:
    """ASR 確定認識結果イベント。on_asr_event 経由で通知される."""

    start: float
    end: float
    text: str
    event_type: ASREventType = field(
        default=ASREventType.ASR_FINAL_RESULT, init=False
    )


# on_asr_event / on_status コールバックに渡す型の Union
ASREventUnion = (
    VadStartEvent
    | VadStopEvent
    | WakewordListeningStartedEvent
    | WakewordDetectedEvent
    | AsrPartialResultEvent
    | AsrFinalResultEvent
)


@dataclass
class VADResult:
    """
    process_frame() の戻り値.

    event が SILENCE のとき frames は空リスト.
    event が VAD_START のとき frames には発話開始前のバッファ＋現フレームが含まれる.
    それ以外のとき frames には現フレームのみが含まれる.
    speech_start_sec: VAD が検出した発話開始タイムスタンプ（秒）。VAD_START のみ有効。
    speech_end_sec:   VAD が検出した発話終了タイムスタンプ（秒）。VAD_END のみ有効。
    """

    event: VADEvent
    frames: list[bytes]
    speech_start_sec: float | None = None
    speech_end_sec: float | None = None


@dataclass
class WakewordResult:
    """WakewordPlugin の process_frame() の戻り値."""

    event: WakewordEvent
    score: float = 0.0


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
    description: str = ''


class ASRPluginBase(ABC):
    """
    ASRプラグインの基底クラス.

    ライフサイクル:
        __init__()  ->  load_params()  ->  setup()  ->  run()
    """

    plugin_name: str = ''
    extend_silence_on_wakeword: bool = True

    # setup() 後にセットされるキュー
    audio_queue: queue.Queue
    result_queue: queue.Queue

    def get_param_declarations(self) -> list[PluginParam]:
        """このプラグインが使うパラメータ一覧を返す。デフォルトは空."""
        return []

    def load_params(self, params: dict) -> None:
        """ノードが解決したパラメータ値を受け取る。params のキーはプレフィックスなしの name."""

    @abstractmethod
    def setup(
        self,
        audio_queue: queue.Queue,
        result_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> 'ASRPluginBase':
        """モデルロードとキューの受け取りを行い self を返す."""

    @abstractmethod
    def run(self) -> None:
        """ASRワーカースレッドのエントリポイント."""


class VADPluginBase(ABC):
    """
    VADプラグインの基底クラス.

    ライフサイクル:
        __init__()  ->  load_params()  ->  setup()  ->  process_frame()
    """

    plugin_name: str = ''

    # 発話中かどうかを示すフラグ。SpeechRecognitionSystem が WAV終端処理などで参照する。
    in_speech: bool = False

    def get_param_declarations(self) -> list[PluginParam]:
        """このプラグインが使うパラメータ一覧を返す。デフォルトは空."""
        return []

    def load_params(self, params: dict) -> None:
        """ノードが解決したパラメータ値を受け取る。params のキーはプレフィックスなしの name."""

    @abstractmethod
    def setup(self) -> None:
        """モデルロードなど重い初期化。load_params() の後に呼ばれる."""

    @abstractmethod
    def process_frame(self, frame: bytes) -> VADResult:
        """1フレームを処理して VADResult を返す。詳細は VADResult の docstring を参照."""

    def extend_silence_threshold(self, silence_threshold_ms: int) -> None:
        """Change the silence duration threshold for end-of-speech detection."""


class WakewordPluginBase(ABC):
    """
    ウェイクワード検出プラグインの基底クラス.

    ライフサイクル:
        __init__()  ->  load_params()  ->  setup()  ->  reset() + process_frame()
    """

    plugin_name: str = ''
    extend_silence_on_detected: bool = True

    def get_param_declarations(self) -> list[PluginParam]:
        """このプラグインが使うパラメータ一覧を返す。デフォルトは空."""
        return []

    def load_params(self, params: dict) -> None:
        """ノードが解決したパラメータ値を受け取る。params のキーはプレフィックスなしの name."""

    @abstractmethod
    def setup(self) -> None:
        """モデルロードなど重い初期化。load_params() の後に呼ばれる."""

    @abstractmethod
    def reset(self) -> None:
        """内部状態をリセットする。VAD_START のたびに SRS が呼ぶ."""

    @abstractmethod
    def process_frame(self, frame: bytes) -> WakewordResult:
        """1フレームを処理して WakewordResult を返す."""
