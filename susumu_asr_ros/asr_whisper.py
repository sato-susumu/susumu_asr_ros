"""faster-whisper バッチ ASR プラグイン."""
import queue
import threading

import numpy as np
import torch
from faster_whisper import WhisperModel
from rclpy.logging import get_logger

from susumu_asr_ros.constants import INT16_MAX
from susumu_asr_ros.plugin_base import ASRCommand, ASRPluginBase, PluginParam


class WhisperASRPlugin(ASRPluginBase):
    """faster-whisper を用いた ASR プラグイン（発話終了時にまとめて認識）."""

    plugin_name = "whisper"

    def get_param_declarations(self) -> list[PluginParam]:
        return [
            PluginParam("model_name", "large-v2", "Whisper モデル名"),
            PluginParam("language_code", "auto", "認識言語コード (auto / ja / en ...)"),
            PluginParam("device", "auto", "推論デバイス (auto / cpu / cuda)"),
        ]

    def load_params(self, params: dict) -> None:
        self._model_name = params.get("model_name", "large-v2")
        self._language_code = params.get("language_code", "auto")
        self._device = params.get("device", "auto")

    def setup(
        self,
        audio_queue: queue.Queue,
        result_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> "WhisperASRPlugin":  # noqa: F821
        self.audio_queue = audio_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.logger = get_logger("whisper_asr")

        device = self._device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._resolved_device = device

        self._model = WhisperModel(
            self._model_name,
            device=self._resolved_device,
            compute_type="auto",
        )
        self.logger.info(
            f"model={self._model_name}, device={self._resolved_device}"
        )

        self.call_active = False
        self.audio_buffer = bytearray()
        self._start_time = None
        self._stop_time = None
        return self

    def run(self) -> None:
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
                self._handle_stop_all()
                return

    def _handle_start(self, data: bytes) -> None:
        self._start_time = float(data.decode())
        self._stop_time = None
        self.audio_buffer.clear()
        self.call_active = True
        self.logger.info("音声認識セッション開始")

    def _handle_audio(self, data: bytes) -> None:
        if self.call_active:
            self.audio_buffer.extend(data)

    def _handle_stop(self, data: bytes) -> None:
        self._stop_time = float(data.decode())
        if self.call_active:
            self.logger.info("発話終了 → まとめてデコードを実行")
            text = self._run_inference(self.audio_buffer)
            if text:
                self.result_queue.put((True, text, self._start_time, self._stop_time))
            self.call_active = False
            self.audio_buffer.clear()

    def _handle_stop_all(self) -> None:
        self.logger.info("stop_all受信 → ワーカー終了")
        if self.call_active and len(self.audio_buffer) > 0:
            text = self._run_inference(self.audio_buffer)
            if text:
                self.result_queue.put((True, text, self._start_time, None))

    def _run_inference(self, audio_data: bytes) -> str:
        if not audio_data:
            return ""
        samples = (
            np.frombuffer(audio_data, dtype=np.int16).astype(np.float16) / INT16_MAX
        )
        lang = None if self._language_code == "auto" else self._language_code
        segments, _info = self._model.transcribe(
            samples, language=lang, beam_size=5, vad_filter=False,
        )
        return "".join(seg.text for seg in segments).strip()
