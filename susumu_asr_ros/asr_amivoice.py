"""AmiVoice ACP WebSocket ストリーミング ASR プラグイン（公式 Wrp ライブラリ使用）."""
import json
import os
import queue
import threading

from susumu_asr_ros.ros_logger import get_logger

from susumu_asr_ros.com.amivoice.wrp.Wrp import Wrp
from susumu_asr_ros.com.amivoice.wrp.WrpListener import WrpListener
from susumu_asr_ros.constants import SAMPLE_RATE
from susumu_asr_ros.plugin_base import (
    ASRCommand, ASRPluginBase, ASRResult, PluginParam,
)

_ENV_APP_KEY = 'AMIVOICE_APP_KEY'
_WS_URL = 'wss://acp-api.amivoice.com/v1/nolog/'
_STREAM_END = object()


class _WrpListener(WrpListener):
    """Wrp イベントを ASRResult に変換して result_queue に流す."""

    def __init__(self, result_queue: queue.Queue, logger):
        self._result_queue = result_queue
        self._logger = logger
        self._start_time: float | None = None
        self._stop_time: float | None = None

    def set_times(self, start: float, stop: float | None) -> None:
        self._start_time = start
        self._stop_time = stop

    def utteranceStarted(self, startTime):  # noqa: N802
        self._logger.info(f'AmiVoice: 発話開始 startTime={startTime}')

    def utteranceEnded(self, endTime):  # noqa: N802
        self._logger.info(f'AmiVoice: 発話終了 endTime={endTime}')

    def resultCreated(self):  # noqa: N802
        pass

    def resultUpdated(self, result):  # noqa: N802
        text = self._extract_text(result)
        if text:
            self._logger.info(f'[Partial] {text}')
            self._result_queue.put(
                ASRResult(False, text, self._start_time, end=None)
            )

    def resultFinalized(self, result):  # noqa: N802
        text = self._extract_text(result)
        if text:
            self._logger.info(f'[Final] {text}')
            self._result_queue.put(
                ASRResult(True, text, self._start_time, self._stop_time)
            )

    def eventNotified(self, eventId, eventMessage):  # noqa: N802
        self._logger.info(f'AmiVoice: イベント {eventId} {eventMessage}')

    def TRACE(self, message):  # noqa: N802
        self._logger.info(f'[Wrp] {message}')

    @staticmethod
    def _extract_text(result: str) -> str:
        try:
            return json.loads(result).get('text', '').strip()
        except (json.JSONDecodeError, AttributeError):
            return ''


class AmiVoiceASRPlugin(ASRPluginBase):
    """
    AmiVoice ACP (WebSocket) ストリーミング ASR プラグイン.

    公式 Wrp ライブラリを使用。発話ごとに feedDataResume/feedDataPause を呼ぶ。
    """

    plugin_name = 'amivoice'
    extend_silence_on_wakeword = False

    def get_param_declarations(self) -> list[PluginParam]:
        return [
            PluginParam('engine', '-a-general', '認識エンジン名'),
            PluginParam('profile_words', '', 'ユーザー辞書 (表記 読み 形式、複数は | 区切り)'),
        ]

    def load_params(self, params: dict) -> None:
        self._engine = params.get('engine', '-a-general')
        self._profile_words = params.get('profile_words', '')

    def setup(
        self,
        audio_queue: queue.Queue,
        result_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> 'AmiVoiceASRPlugin':  # noqa: F821
        self.audio_queue = audio_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.logger = get_logger('amivoice_asr')

        self._app_key = os.environ.get(_ENV_APP_KEY, '')
        if not self._app_key:
            self.logger.warning(
                f'環境変数 {_ENV_APP_KEY} が設定されていません'
            )

        self._listener = _WrpListener(result_queue, self.logger)
        codec = f'LSB{SAMPLE_RATE // 1000}K'

        self._wrp = Wrp.construct()
        self._wrp.setListener(self._listener)
        self._wrp.setServerURL(_WS_URL)
        self._wrp.setCodec(codec)
        self._wrp.setGrammarFileNames(self._engine)
        self._wrp.setAuthorization(self._app_key)
        if self._profile_words:
            self._wrp.setProfileWords(self._profile_words)
            self.logger.info(f'ユーザー辞書設定: {self._profile_words}')

        self.call_active = False
        self._audio_buffer_queue: queue.Queue = queue.Queue()
        self._feed_thread: threading.Thread | None = None
        self._start_time: float | None = None
        self._stop_time: float | None = None
        return self

    def run(self) -> None:
        self.logger.info('AmiVoice ASR スレッド起動')

        if not self._wrp.connect():
            self.logger.error(
                f'AmiVoice WebSocket 接続失敗: {self._wrp.getLastMessage()}'
            )
            return
        self.logger.info('AmiVoice WebSocket 接続成功')

        try:
            while not self.stop_event.is_set():
                try:
                    command, data = self.audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if command == ASRCommand.START:
                    self._handle_start(data)
                elif command == ASRCommand.AUDIO:
                    self._handle_audio(data)
                elif command == ASRCommand.STOP:
                    self._handle_stop(data)
                elif command == ASRCommand.STOP_ALL:
                    self._handle_stop_all(data)
                    return
        finally:
            self._wrp.disconnect()

    def _handle_start(self, data: bytes) -> None:
        self._start_time = float(data.decode())
        self._stop_time = None
        if self.call_active:
            self.logger.info("既にストリーミング中 → 'start' を無視")
            return
        self.logger.info('ストリーミング認識 開始')
        self._listener.set_times(self._start_time, None)
        while not self._audio_buffer_queue.empty():
            self._audio_buffer_queue.get_nowait()
        if not self._wrp.feedDataResume():
            self.logger.error(
                f'feedDataResume 失敗: {self._wrp.getLastMessage()}'
            )
            return
        self.call_active = True
        self._feed_thread = threading.Thread(
            target=self._feed_loop, daemon=True
        )
        self._feed_thread.start()

    def _handle_audio(self, data: bytes) -> None:
        if self.call_active:
            self._audio_buffer_queue.put(data)

    def _handle_stop(self, data: bytes) -> None:
        self._stop_time = float(data.decode())
        if not self.call_active:
            self.logger.info("ストリーミング中ではない → 'stop' を無視")
            return
        self.logger.info('ストリーミング終了を要求')
        self._listener.set_times(self._start_time, self._stop_time)
        self._audio_buffer_queue.put(_STREAM_END)
        if self._feed_thread:
            self._feed_thread.join(timeout=10.0)
            self._feed_thread = None
        self.call_active = False

    def _handle_stop_all(self, data: bytes) -> None:
        self.logger.info('stop_all 受信 → 終了処理')
        if self.call_active:
            self._handle_stop(data)

    def _feed_loop(self) -> None:
        """audio_buffer_queue から音声を取り出して Wrp に送り続ける."""
        while True:
            try:
                chunk = self._audio_buffer_queue.get(timeout=0.1)
            except queue.Empty:
                if not self.call_active:
                    break
                continue
            if chunk is _STREAM_END:
                break
            self._wrp.feedData(chunk, 0, len(chunk))

        if not self._wrp.feedDataPause():
            self.logger.error(
                f'feedDataPause 失敗: {self._wrp.getLastMessage()}'
            )
