"""音声認識パイプラインのメインループ."""
import collections
from enum import Enum
import queue
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

# ウェイクワード検出後に適用する無音閾値（ms）。8秒タイムアウト相当。
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
        on_stop=None,
    ):
        self.stop_event = threading.Event()

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
        self.on_stop = on_stop or (lambda: None)

        self.processed_size: int = 0
        self.current_time: float = 0.0
        self.vad_start: float = 0.0

        self._speech_buffer: list[bytes] = []
        self._state: SRSState = SRSState.IDLE
        _prebuf_frames = int(2.0 * SAMPLE_RATE * SAMPLE_WIDTH // (512 * SAMPLE_WIDTH))
        self._wakeword_prebuffer: collections.deque = collections.deque(maxlen=_prebuf_frames)

        self.logger = get_logger('speech_recognition_system')

    # ------------------------------------------------------------------
    # メインループ
    # ------------------------------------------------------------------

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
                        self._handle_wav_eof()
                        break
                    time.sleep(0.01)
                    continue

                vad_result = self.vad_plugin.process_frame(frame)
                self._handle_vad(vad_result, frame)
                self._fetch_results()
                self._update_current_time(frame)
                if not isinstance(self.recorder, WavAudioRecorder):
                    time.sleep(0.03)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            self.logger.error(f'エラー: {e}')
            raise
        finally:
            self._shutdown()

    def _handle_wav_eof(self):
        """WAVファイル終端の処理。"""
        self._finalize_state('WAV終端')
        self.vad_plugin.in_speech = False
        self.stop_event.set()

    def _shutdown(self):
        """終了処理。"""
        self.stop_event.set()
        self.recorder.close()
        self._finalize_state('シャットダウン')
        self.logger.info('終了処理。stop_all をワーカーへ送信')
        self.audio_queue.put((ASRCommand.STOP_ALL, str(self.current_time).encode()))
        self.asr_thread.join()
        self._fetch_results()
        self.full_audio_writer.close()
        self.speech_audio_writer.close()
        if isinstance(self.recorder, WavAudioRecorder):
            time.sleep(1.0)
        self.logger.info('プログラム終了')
        self.on_stop()

    # ------------------------------------------------------------------
    # 状態遷移
    # ------------------------------------------------------------------

    def _transition_to_idle(self, end_time: float) -> None:
        """BUFFERING/IN_SPEECH → IDLE への唯一の出口。"""
        self._state = SRSState.IDLE
        self.label_writer.write_segment(self.vad_start, end_time, 'vad_speech')
        self.on_asr_event(VadStopEvent(start=self.vad_start, end=end_time))

    def _finalize_state(self, reason: str) -> None:
        """IDLE以外の状態で終了する場合に後始末して _transition_to_idle を呼ぶ."""
        if self._state == SRSState.IN_SPEECH:
            self.logger.info(f'{reason}: IN_SPEECH を強制終了')
            self.audio_queue.put((ASRCommand.STOP, str(self.current_time).encode()))
            self.speech_audio_writer.close()
            self._transition_to_idle(self.current_time)
        elif self._state == SRSState.BUFFERING:
            self.logger.info(f'{reason}: BUFFERING を強制終了（ウェイクワード未検出）')
            self._speech_buffer.clear()
            self._transition_to_idle(self.current_time)

    def _should_extend_silence(self) -> bool:
        """ウェイクワード検出後に VAD の無音閾値を延長すべきか判定する."""
        return self.wakeword_plugin.extend_silence_on_detected and self.asr_plugin.extend_silence_on_wakeword

    # ------------------------------------------------------------------
    # VAD イベント処理
    # ------------------------------------------------------------------

    def _handle_vad(self, vad_result, frame: bytes) -> None:
        if vad_result.event not in (VADEvent.SILENCE, VADEvent.VAD_CONT):
            self.logger.info(f'[VAD] state={self._state} event={vad_result.event}')

        if self._state == SRSState.IDLE:
            self._wakeword_prebuffer.append(frame)

        if vad_result.event == VADEvent.VAD_START:
            self._on_vad_start(vad_result)
        elif vad_result.event == VADEvent.VAD_CONT:
            self._on_vad_cont(vad_result)
        elif vad_result.event == VADEvent.VAD_END:
            self._on_vad_end(vad_result)
        elif vad_result.event != VADEvent.SILENCE:
            raise ValueError(f'未知のイベント: {vad_result.event}')

    def _on_vad_start(self, vad_result) -> None:
        if self._state != SRSState.IDLE:
            self.logger.info(f'VAD_START を無視 (state={self._state})')
            return
        self.logger.info('VAD 発話開始 → BUFFERING')
        self._state = SRSState.BUFFERING
        self._speech_buffer = list(vad_result.frames)
        self.vad_start = self.current_time
        self.wakeword_plugin.reset()
        self.on_asr_event(VadStartEvent(start=self.current_time))
        self.on_asr_event(WakewordListeningStartedEvent(start=self.current_time))
        self._feed_wakeword_frames(list(self._wakeword_prebuffer) + list(vad_result.frames))
        self._wakeword_prebuffer.clear()

    def _on_vad_cont(self, vad_result) -> None:
        if self._state == SRSState.BUFFERING:
            self._speech_buffer.extend(vad_result.frames)
            self._feed_wakeword_frames(vad_result.frames)
        elif self._state == SRSState.IN_SPEECH:
            for f in vad_result.frames:
                self.audio_queue.put((ASRCommand.AUDIO, f))
                self.speech_audio_writer.write(f)

    def _on_vad_end(self, vad_result) -> None:
        if self._state == SRSState.BUFFERING:
            self.logger.info('VAD 発話終了（ウェイクワード未検出）→ 音声を捨てる')
            self._speech_buffer.clear()
            self._transition_to_idle(self.current_time)
        elif self._state == SRSState.IN_SPEECH:
            for f in vad_result.frames:
                self.audio_queue.put((ASRCommand.AUDIO, f))
                self.speech_audio_writer.write(f)
            self.logger.info('VAD 発話終了 → ASR STOP')
            if self._should_extend_silence():
                self.vad_plugin.extend_silence_threshold(self.vad_plugin._silence_ms)
            self.audio_queue.put((ASRCommand.STOP, str(self.current_time).encode()))
            self.speech_audio_writer.close()
            self._transition_to_idle(self.current_time)

    # ------------------------------------------------------------------
    # ウェイクワード処理
    # ------------------------------------------------------------------

    def _feed_wakeword_frames(self, frames: list[bytes]) -> None:
        """フレームリストをウェイクワードプラグインに順に渡す。BUFFERING でなくなったら中断する."""
        for f in frames:
            if self._state != SRSState.BUFFERING:
                break
            self._feed_wakeword(f)

    def _feed_wakeword(self, frame: bytes) -> None:
        ww_result = self.wakeword_plugin.process_frame(frame)
        if ww_result.event == WakewordEvent.DETECTED:
            self._on_wakeword_detected(ww_result.score)

    def _on_wakeword_detected(self, score: float) -> None:
        self.logger.info(f'ウェイクワード検出 score={score:.3f} → IN_SPEECH')
        self.on_asr_event(WakewordDetectedEvent(start=self.current_time, score=score))
        self.label_writer.write_segment(self.current_time, self.current_time, 'ww_detected')
        self._state = SRSState.IN_SPEECH

        if self._should_extend_silence():
            self.vad_plugin.extend_silence_threshold(_WAKEWORD_SILENCE_THRESHOLD_MS)

        self.audio_queue.put((ASRCommand.START, str(self.vad_start).encode()))
        self.speech_audio_writer.open()
        for f in self._speech_buffer:
            self.audio_queue.put((ASRCommand.AUDIO, f))
            self.speech_audio_writer.write(f)
        self._speech_buffer.clear()

    # ------------------------------------------------------------------
    # ASR 結果処理・時刻管理
    # ------------------------------------------------------------------

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
                self._on_asr_final(result)
            else:
                self.logger.info(f'[Partial] {result.text}')
                self.on_asr_event(AsrPartialResultEvent(start=result.start, text=result.text))

    def _on_asr_final(self, result: ASRResult) -> None:
        self.logger.info(f'[Final] {result.text}')
        end = result.end if result.end is not None else self.current_time
        self.on_asr_event(AsrFinalResultEvent(start=result.start, end=end, text=result.text))
        self.label_writer.write_segment(result.start, end, result.text)
        if self._should_extend_silence():
            self.vad_plugin.extend_silence_threshold(self.vad_plugin._silence_ms)
        if self._state == SRSState.IN_SPEECH:
            self._transition_to_idle(end)
