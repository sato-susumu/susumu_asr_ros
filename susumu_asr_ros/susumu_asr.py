"""音声認識パイプラインのメインループ."""
import collections
from enum import Enum
import queue
import signal
import sys
import threading
import time

from rclpy.logging import get_logger

from susumu_asr_ros.audio_io import (
    AudioRecorderBase,
    DummyAudioWriter,
    DummyLabelWriter,
    DummySpeechAudioWriter,
    FullAudioWriter,
    LabelWriter,
    SpeechAudioWriter,
    WavAudioRecorder,
)
from susumu_asr_ros.constants import SAMPLE_RATE, SAMPLE_WIDTH
from susumu_asr_ros.plugin_base import (
    ASRCommand,
    AsrFinalResultEvent,
    AsrPartialResultEvent,
    ASRPluginBase,
    ASRResult,
    VADEvent,
    VADPluginBase,
    VadStartEvent,
    VadStopEvent,
    WakewordDetectedEvent,
    WakewordEvent,
    WakewordListeningStartedEvent,
    WakewordPluginBase,
)

# ウェイクワード検出後に適用する無音閾値（ ms）。8秒タイムアウト相当。
_WAKEWORD_SILENCE_THRESHOLD_MS = 8000


class SRSState(str, Enum):
    """SpeechRecognitionSystem の内部状態."""

    IDLE = 'idle'
    BUFFERING = 'buffering'
    IN_SPEECH = 'in_speech'


class SpeechRecognitionSystem:
    """
    システム全体の制御を行うクラス.

    状態機械:
        IDLE
          → [VAD_START] → BUFFERING
              VAD音声をバッファに蓄積しながら WakewordPlugin に渡す
          → [WAKEWORD_DETECTED] → IN_SPEECH
              silence_threshold を延長、バッファ先頭から ASR に送信開始
          → [VAD_END in BUFFERING] → IDLE
              ウェイクワード未検出のまま終了、音声を捨てる
          → [VAD_END in IN_SPEECH] → IDLE
              ASR に STOP 送信
    """

    def __init__(
        self,
        vad_plugin: VADPluginBase,
        wakeword_plugin: WakewordPluginBase,
        asr_plugin: ASRPluginBase,
        recorder: AudioRecorderBase,
        full_audio_writer: FullAudioWriter | None = None,
        label_writer: LabelWriter | None = None,
        speech_audio_writer: SpeechAudioWriter | None = None,
        on_asr_event=None,
    ):
        self.stop_event = threading.Event()
        signal.signal(signal.SIGINT, self._signal_handler)

        self.audio_queue = asr_plugin.audio_queue
        self.result_queue = asr_plugin.result_queue

        self.asr_plugin = asr_plugin
        self.asr_thread = threading.Thread(target=self.asr_plugin.run, daemon=True)

        self.vad_plugin = vad_plugin
        self.wakeword_plugin = wakeword_plugin
        self.recorder = recorder

        self.full_audio_writer = full_audio_writer or DummyAudioWriter()
        self.label_writer = label_writer or DummyLabelWriter()
        self.speech_audio_writer = speech_audio_writer or DummySpeechAudioWriter()

        self.on_asr_event = on_asr_event or (lambda d: None)

        self.processed_size: int = 0
        self.current_time: float = 0.0
        self.vad_start: float = 0.0

        # BUFFERING 中に蓄積する音声バッファ（VAD_STARTのpre_speechも含む）
        self._speech_buffer: list[bytes] = []
        self._state: SRSState = SRSState.IDLE
        # IDLE 中の音声を 2 秒分保持し、VAD_START 時に wakeword プラグインへ供給する
        _prebuf_frames = int(2.0 * SAMPLE_RATE * SAMPLE_WIDTH // (512 * SAMPLE_WIDTH))
        self._wakeword_prebuffer: collections.deque = collections.deque(maxlen=_prebuf_frames)

        self.logger = get_logger('speech_recognition_system')

    def _signal_handler(self, sig, frame):
        self.logger.info('捕捉: Ctrl+C で停止要求')
        self.stop_event.set()
        sys.exit(0)

    def start(self):
        self.logger.info('システム起動。Ctrl+C で終了')
        for idx, info in self.recorder.get_device_info().items():
            if info['maxInputChannels'] > 0:
                self.logger.info(f"マイクデバイス {idx}: {info['name']}")

        self.asr_thread.start()
        self.recorder.open()
        try:
            while not self.stop_event.is_set():
                frame = self.recorder.read_frame()
                self.full_audio_writer.write(frame)

                if not frame:
                    if isinstance(self.recorder, WavAudioRecorder):
                        if self._state == SRSState.IN_SPEECH:
                            self.logger.info('WAV 終端: 発話セッションを強制終了')
                            self.audio_queue.put(
                                (ASRCommand.STOP, str(self.current_time).encode())
                            )
                            self.speech_audio_writer.close()
                        elif self._state == SRSState.BUFFERING:
                            self.logger.info('WAV 終端: ウェイクワード未検出のまま終了')
                        self.vad_plugin.in_speech = False
                        self._state = SRSState.IDLE
                        self.stop_event.set()
                        break
                    time.sleep(0.01)
                    continue

                vad_result = self.vad_plugin.process_frame(frame)
                self._handle_vad(vad_result, frame)

                self._fetch_results()
                self._update_current_time(frame)
                time.sleep(0.03)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            self.logger.error(f'エラー: {e}')
            raise
        finally:
            self.stop_event.set()
            self.recorder.close()
            self.logger.info('終了処理。stop_all をワーカーへ送信')
            self.audio_queue.put((ASRCommand.STOP_ALL, None))
            self.asr_thread.join()
            self._fetch_results()
            self.full_audio_writer.close()
            self.speech_audio_writer.close()
            if isinstance(self.recorder, WavAudioRecorder):
                time.sleep(1.0)
            self.logger.info('プログラム終了')

    def _handle_vad(self, vad_result, frame: bytes) -> None:
        # IDLE 中のフレームを 2 秒分のリングバッファに蓄積する
        if self._state == SRSState.IDLE:
            self._wakeword_prebuffer.append(frame)

        if vad_result.event == VADEvent.VAD_START:
            self.logger.info('VAD 発話開始 → BUFFERING')
            self._state = SRSState.BUFFERING
            self._speech_buffer = list(vad_result.frames)
            self.vad_start = self.current_time
            self.wakeword_plugin.reset()
            self.on_asr_event(VadStartEvent(start=self.current_time))
            self.on_asr_event(WakewordListeningStartedEvent(start=self.current_time))
            # IDLE 中に蓄積した音声 → pre_speech_buffer の順で wakeword プラグインに供給
            for f in self._wakeword_prebuffer:
                self._feed_wakeword(f)
            self._wakeword_prebuffer.clear()
            for f in vad_result.frames:
                self._feed_wakeword(f)

        elif vad_result.event == VADEvent.VAD_CONT:
            if self._state == SRSState.BUFFERING:
                self._speech_buffer.extend(vad_result.frames)
                for f in vad_result.frames:
                    self._feed_wakeword(f)
            elif self._state == SRSState.IN_SPEECH:
                for f in vad_result.frames:
                    self.audio_queue.put((ASRCommand.AUDIO, f))
                    self.speech_audio_writer.write(f)

        elif vad_result.event == VADEvent.VAD_END:
            if self._state == SRSState.BUFFERING:
                self.logger.info('VAD 発話終了（ウェイクワード未検出）→ 音声を捨てる')
                self._speech_buffer.clear()
                self._state = SRSState.IDLE
                self.on_asr_event(VadStopEvent(
                    start=self.vad_start, end=self.current_time
                ))
            elif self._state == SRSState.IN_SPEECH:
                for f in vad_result.frames:
                    self.audio_queue.put((ASRCommand.AUDIO, f))
                    self.speech_audio_writer.write(f)
                self.logger.info('VAD 発話終了 → ASR STOP')
                self.audio_queue.put((ASRCommand.STOP, str(self.current_time).encode()))
                self.speech_audio_writer.close()
                self.label_writer.write_segment(self.vad_start, self.current_time, 'speech')
                self._state = SRSState.IDLE
                self.on_asr_event(VadStopEvent(
                    start=self.vad_start, end=self.current_time
                ))

        elif vad_result.event != VADEvent.SILENCE:
            raise ValueError(f'未知のイベント: {vad_result.event}')

    def _feed_wakeword(self, frame: bytes) -> None:
        """BUFFERINGフェーズでウェイクワードプラグインにフレームを渡す."""
        ww_result = self.wakeword_plugin.process_frame(frame)
        if ww_result.event == WakewordEvent.DETECTED:
            self._on_wakeword_detected(ww_result.score)

    def _on_wakeword_detected(self, score: float) -> None:
        """ウェイクワード検出時の処理。バッファを ASR に流し IN_SPEECH へ遷移する."""
        self.logger.info(f'ウェイクワード検出 score={score:.3f} → IN_SPEECH')
        self.on_asr_event(WakewordDetectedEvent(start=self.current_time, score=score))
        self._state = SRSState.IN_SPEECH

        # VADの無音閾値を延長してタイムアウトまで発話を継続させる
        self.vad_plugin.extend_silence_threshold(_WAKEWORD_SILENCE_THRESHOLD_MS)

        # バッファ先頭（VAD_START時点）からASRに送信
        self.audio_queue.put((ASRCommand.START, str(self.vad_start).encode()))
        self.speech_audio_writer.open()
        for f in self._speech_buffer:
            self.audio_queue.put((ASRCommand.AUDIO, f))
            self.speech_audio_writer.write(f)
        self._speech_buffer.clear()

    def _update_current_time(self, frame: bytes) -> None:
        self.processed_size += len(frame)
        self.current_time = self.processed_size / SAMPLE_RATE / SAMPLE_WIDTH

    def _fetch_results(self):
        """ワーカーからの認識結果を受け取って出力."""
        while True:
            try:
                result: ASRResult = self.result_queue.get_nowait()
            except queue.Empty:
                break
            if not result.text:
                continue
            if result.is_final:
                self.logger.info(f'[Final] {result.text}')
                self.on_asr_event(AsrFinalResultEvent(
                    start=result.start,
                    end=result.end if result.end is not None else self.current_time,
                    text=result.text,
                ))
                if result.end is not None:
                    self.label_writer.write_segment(result.start, result.end, result.text)
            else:
                self.logger.info(f'[Partial] {result.text}')
                self.on_asr_event(AsrPartialResultEvent(
                    start=result.start,
                    text=result.text,
                ))
