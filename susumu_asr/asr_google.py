"""Google Cloud Speech-to-Text ストリーミング ASR プラグイン."""
import queue
import threading

from google.cloud import speech
from susumu_asr.constants import SAMPLE_RATE
from susumu_asr.plugin_base import ASRCommand, ASRPluginBase, ASRResult, PluginParam
from susumu_asr.ros_logger import get_logger

# _audio_buffer_queue のストリーム終了シグナル
_STREAM_END = object()


class GoogleCloudASRPlugin(ASRPluginBase):
    """
    Google Cloud Speech-to-Text (ストリーミング認識) ASR プラグイン.

    single_utterance=True で1回の発話が終わると自動的にストリーム終了.
    interim_results=True で途中経過(Partial)を取得.
    """

    plugin_name = 'google_cloud'

    def get_param_declarations(self) -> list[PluginParam]:
        return [
            PluginParam('language_code', 'ja-JP', '認識言語コード (例: ja-JP, en-US)'),
        ]

    def load_params(self, params: dict) -> None:
        self._language_code = params.get('language_code', 'ja-JP')

    def setup(
        self,
        audio_queue: queue.Queue,
        result_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> 'GoogleCloudASRPlugin':  # noqa: F821
        self.audio_queue = audio_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.logger = get_logger('google_cloud_asr')

        self.client = speech.SpeechClient()
        self.call_active = False
        self._audio_buffer_queue = queue.Queue()
        self._response_thread = None
        self._start_time = None
        self._stop_time = None
        return self

    def run(self) -> None:
        self.logger.info('スレッド起動: Streaming Speech API (single_utterance=True)')
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

    def _handle_start(self, data: bytes) -> None:
        self._start_time = float(data.decode())
        self._stop_time = None

        if not self.call_active:
            self.logger.info('ストリーミング認識 開始')
            self.call_active = True
            while not self._audio_buffer_queue.empty():
                self._audio_buffer_queue.get_nowait()
            self._response_thread = threading.Thread(
                target=self._streaming_recognize_loop, daemon=True
            )
            self._response_thread.start()
        else:
            self.logger.info("既にストリーミング中 → 'start' を無視")

    def _handle_audio(self, data: bytes) -> None:
        if self.call_active:
            self._audio_buffer_queue.put(data)

    def _handle_stop(self, data: bytes) -> None:
        self._stop_time = float(data.decode())
        if self.call_active:
            self.logger.info('明示的にストリーミング終了を要求')
            self.call_active = False
            self._audio_buffer_queue.put(_STREAM_END)
            if self._response_thread:
                self._response_thread.join()
                self._response_thread = None
        else:
            self.logger.info("ストリーミング中ではない → 'stop' を無視")

    def _handle_stop_all(self, data: bytes) -> None:
        self.logger.info('stop_all受信 → スレッド終了処理')
        if self.call_active:
            self._handle_stop(data)

    def _streaming_recognize_loop(self) -> None:
        streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=SAMPLE_RATE,
                language_code=self._language_code,
                enable_automatic_punctuation=True,
            ),
            interim_results=True,
            single_utterance=True,
        )
        try:
            for response in self.client.streaming_recognize(
                config=streaming_config,
                requests=self._request_stream(),
            ):
                for result in response.results:
                    if not result.alternatives:
                        continue
                    transcript = result.alternatives[0].transcript
                    if result.is_final:
                        self.result_queue.put(
                            ASRResult(True, transcript, self._start_time, self._stop_time)
                        )
                        self.call_active = False
                        self._audio_buffer_queue.put(_STREAM_END)
                        return
                    else:
                        self.result_queue.put(
                            ASRResult(False, transcript, self._start_time, end=None)
                        )
        except Exception as e:
            self.logger.error(f'ストリーミング認識中に例外: {e}')
        finally:
            self.call_active = False
            self.logger.info('ストリーミング終了')

    def _request_stream(self):
        """Queue から音声チャンクを受け取って StreamingRecognizeRequest に包んで yield する."""
        while True:
            try:
                chunk = self._audio_buffer_queue.get(timeout=0.1)
            except queue.Empty:
                if not self.call_active:
                    return
                continue
            if chunk is _STREAM_END:
                return
            yield speech.StreamingRecognizeRequest(audio_content=chunk)
