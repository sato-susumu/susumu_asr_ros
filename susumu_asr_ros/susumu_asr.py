"""音声認識パイプラインのメインループ."""
import queue
import signal
import sys
import threading
import time

import numpy as np
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
from susumu_asr_ros.constants import INT16_MAX, SAMPLE_RATE, SAMPLE_WIDTH
from susumu_asr_ros.plugin_base import (
    ASRCommand,
    ASRPluginBase,
    ASRResult,
    VADEvent,
    VADPluginBase,
)
from susumu_asr_ros.vad_openwakeword import OpenWakeWordPlugin


class SpeechRecognitionSystem:
    """
    システム全体の制御を行うクラス.

    - スレッド生成（音声認識ワーカー）
    - メインループで AudioRecorder → VAD → ASR へコマンド送信 → 結果出力
    - シグナルハンドラ登録
    """

    def __init__(
        self,
        vad_plugin: VADPluginBase,
        asr_plugin: ASRPluginBase,
        recorder: AudioRecorderBase,
        full_audio_writer: FullAudioWriter | None = None,
        label_writer: LabelWriter | None = None,
        speech_audio_writer: SpeechAudioWriter | None = None,
        on_asr_event=None,
        on_audio_level=None,
        on_wakeword_score=None,
        on_status=None,
    ):
        self.stop_event = threading.Event()
        signal.signal(signal.SIGINT, self._signal_handler)

        self.audio_queue = asr_plugin.audio_queue
        self.result_queue = asr_plugin.result_queue

        self.asr_plugin = asr_plugin
        self.asr_thread = threading.Thread(target=self.asr_plugin.run, daemon=True)

        self.vad_plugin = vad_plugin
        self.recorder = recorder

        self.full_audio_writer = full_audio_writer or DummyAudioWriter()
        self.label_writer = label_writer or DummyLabelWriter()
        self.speech_audio_writer = speech_audio_writer or DummySpeechAudioWriter()

        self.on_asr_event = on_asr_event or (lambda d: None)
        self.on_audio_level = on_audio_level or (lambda r: None)
        self.on_wakeword_score = on_wakeword_score or (lambda s: None)
        self.on_status = on_status or (lambda d: None)

        self.processed_size: int = 0
        self.current_time: float = 0.0
        self.vad_start: float = 0.0

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
        self.on_status({'event_type': 'listening_started'})

        try:
            while not self.stop_event.is_set():
                frame = self.recorder.read_frame()
                self.full_audio_writer.write(frame)

                if frame:
                    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
                    rms = float(np.sqrt(np.mean(samples ** 2))) / INT16_MAX
                    self.on_audio_level(rms)

                if not frame:
                    if isinstance(self.recorder, WavAudioRecorder):
                        if self.vad_plugin.in_speech:
                            self.logger.info('WAV 終端: 発話セッションを強制終了')
                            self.audio_queue.put(
                                (ASRCommand.STOP, str(self.current_time).encode())
                            )
                            self.speech_audio_writer.close()
                            self.vad_plugin.in_speech = False
                        self.on_status({
                            'event_type': 'wav_finished',
                            'duration': self.current_time,
                        })
                        self.stop_event.set()
                        break
                    time.sleep(0.01)
                    continue

                vad_result = self.vad_plugin.process_frame(frame)

                # ウェイクワードスコア通知（OpenWakeWordPlugin のみ）
                if isinstance(self.vad_plugin, OpenWakeWordPlugin):
                    oww_score = 0.0
                    for scores in self.vad_plugin._oww_model.prediction_buffer.values():
                        oww_score = scores[-1] if scores else 0.0
                    self.on_wakeword_score(oww_score)

                if vad_result.event == VADEvent.SPEECH_START:
                    self.logger.info("VAD 発話検出 → 'start'")
                    self.audio_queue.put((ASRCommand.START, str(self.current_time).encode()))
                    self.speech_audio_writer.open()
                    for f in vad_result.frames:
                        self.audio_queue.put((ASRCommand.AUDIO, f))
                        self.speech_audio_writer.write(f)
                    self.vad_start = self.current_time

                    if isinstance(self.vad_plugin, OpenWakeWordPlugin):
                        self.on_asr_event({
                            'event_type': 'wakeword_detected',
                            'start': self.current_time,
                            'end': self.current_time,
                            'text': self.vad_plugin._model_name,
                            'score': round(oww_score, 4),
                        })

                elif vad_result.event == VADEvent.SPEECH_CONT:
                    for f in vad_result.frames:
                        self.audio_queue.put((ASRCommand.AUDIO, f))
                        self.speech_audio_writer.write(f)

                elif vad_result.event == VADEvent.SPEECH_STOP:
                    for f in vad_result.frames:
                        self.audio_queue.put((ASRCommand.AUDIO, f))
                        self.speech_audio_writer.write(f)
                    self.logger.info("VAD 終話検出 → 'stop'")
                    self.audio_queue.put((ASRCommand.STOP, str(self.current_time).encode()))
                    self.speech_audio_writer.close()
                    self.label_writer.write_segment(
                        self.vad_start, self.current_time, 'speech'
                    )

                elif vad_result.event == VADEvent.SPEECH_TIMEOUT:
                    self.logger.info("VAD タイムアウト → 'stop'")
                    self.audio_queue.put((ASRCommand.STOP, str(self.current_time).encode()))
                    self.speech_audio_writer.close()
                    self.on_asr_event({
                        'event_type': 'timeout',
                        'start': self.vad_start,
                        'end': self.current_time,
                        'reason': 'speech_duration_exceeded',
                    })

                elif vad_result.event != VADEvent.SILENCE:
                    raise ValueError(f'未知のイベント: {vad_result.event}')

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
                self.on_asr_event({
                    'event_type': 'final_result',
                    'start': result.start,
                    'end': result.end if result.end is not None else self.current_time,
                    'text': result.text,
                })
                if result.end is not None:
                    self.label_writer.write_segment(result.start, result.end, result.text)
            else:
                self.logger.info(f'[Partial] {result.text}')
                self.on_asr_event({
                    'event_type': 'partial_result',
                    'start': result.start,
                    'end': None,
                    'text': result.text,
                })
