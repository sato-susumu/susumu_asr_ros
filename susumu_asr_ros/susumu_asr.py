import collections
import math
import os
import queue
import signal
import sys
import threading
import time
import wave
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

import click
import numpy as np
import openwakeword
import pyaudio
import torch
from google.cloud import speech
from openwakeword.model import Model
from faster_whisper import WhisperModel
from rclpy.logging import get_logger

VAD_SILERO_VAD = "silero_vad"
VAD_OPENWAKEWORD = "openwakeword"

ASR_GOOGLE_CLOUD = "google_cloud"
ASR_WHISPER = "whisper"


class VADBase(ABC):
    @abstractmethod
    def process_frame(self, frame: bytes) -> (str, List[bytes]):
        pass


class ASRBase(ABC):
    @abstractmethod
    def run(self):
        pass


# 定数定義(元からある部分はそのまま)
SAMPLE_RATE = 16000  # サンプリングレート(Hz)
SAMPLE_WIDTH = 2  # サンプル幅(byte)
CHANNELS = 1  # チャンネル数
FRAME_DURATION_MS = 30  # 1フレームあたりの音声長さ(ms)
SILENCE_SECONDS = 1.0  # 発話終了とみなす無音秒数
PRE_SPEECH_TIME_MS = 300  # 発話開始検知時に遡って送るバッファ時間(ms)
SILERO_VAD_THRESHOLD = 0.5  # Silero VAD のしきい値(0.0-1.0)
SILERO_VAD_SILENCE_THRESHOLD_MS = 1000  # SileroVadProcessor用の発話終了とみなす無音時間(ms)
OPENWAKEWORD_SILENCE_THRESHOLD_MS = 2000  # OpenWakeWordProcessor用の発話終了とみなす無音時間(ms)

OPEN_WAKEWORD_THRESHOLD = 0.5
# WakeWord 検出後に認める発話開始からの最小時間(秒)
OPEN_WAKEWORD_SPEECH_END_MIN_DURATION = 2.0
# WakeWord 検出後に認める発話継続時間(秒)
OPEN_WAKEWORD_SPEECH_TIMEOUT_SECONDS = 8.0

# 1秒あたり何フレームか
FRAMES_PER_SECOND = int(1000 / FRAME_DURATION_MS)
# SILENCE_SECONDS 秒分の無音フレームしきい値
SILENCE_FRAMES_THRESHOLD = int(math.ceil(SILENCE_SECONDS * FRAMES_PER_SECOND))


class WhisperASR(ASRBase):
    """faster-whisper を用いた ASR（発話終了時にまとめて認識）."""

    def __init__(
        self,
        model_name: str,
        whisper_language_code: str,
        whisper_device: str,
        audio_queue: queue.Queue,
        result_queue: queue.Queue,
        stop_event: threading.Event,
    ):
        self.model_name = model_name
        self.whisper_language_code = whisper_language_code
        self.whisper_device = whisper_device
        self.audio_queue = audio_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.logger = get_logger('whisper_asr')

        # デバイス自動判別 ("auto") → CUDA が使えるなら "cuda"、そうでなければ "cpu"
        if self.whisper_device == "auto":
            self.whisper_device = "cuda" if torch.cuda.is_available() else "cpu"

        # WhisperModel の初期化
        self.model = WhisperModel(
            model_name,
            device=self.whisper_device,
            compute_type="auto",
        )

        self.logger.info(f"model={model_name}, device={self.whisper_device}")

        self.call_active = False
        self.audio_buffer = bytearray()  # 発話中に貯める PCM バッファ
        self._start_time = None
        self._stop_time = None

    def run(self):
        while not self.stop_event.is_set():
            try:
                command, data = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if command == "start":
                self._handle_start(data)
            elif command == "audio":
                self._handle_audio(data)
            elif command == "stop":
                self._handle_stop(data)
            elif command == "stop_all":
                self._handle_stop_all()
                return

    def _handle_start(self, data: bytes):
        self._start_time = float(data.decode())
        self._stop_time = None
        self.audio_buffer.clear()
        self.call_active = True
        self.logger.info("音声認識セッション開始")

    def _handle_audio(self, data: bytes):
        if self.call_active:
            self.audio_buffer.extend(data)

    def _handle_stop(self, data: bytes):
        self._stop_time = float(data.decode())
        if self.call_active:
            self.logger.info("発話終了 → まとめてデコードを実行")
            text = self._run_inference(self.audio_buffer)
            if text:
                self.result_queue.put((True, text, self._start_time, self._stop_time))
            self.call_active = False
            self.audio_buffer.clear()

    def _handle_stop_all(self):
        self.logger.info("stop_all受信 → ワーカー終了")
        if self.call_active and len(self.audio_buffer) > 0:
            text = self._run_inference(self.audio_buffer)
            if text:
                self.result_queue.put((True, text, self._start_time, None))

    def _run_inference(self, audio_data: bytes) -> str:
        if not audio_data:
            return ""

        # 16bit PCM → numpy int16 -> numpy float16 -> normalize
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float16) / 32768.0

        # transcribe時に language="xx" / None で自動判別も可
        lang = None if self.whisper_language_code == "auto" else self.whisper_language_code

        segments, info = self.model.transcribe(
            samples,
            language=lang,
            beam_size=5,
            vad_filter=False,
        )

        text_list = []
        for seg in segments:
            text_list.append(seg.text)
        full_text = "".join(text_list).strip()
        return full_text


class GoogleCloudASR(ASRBase):
    """
    Google Cloud Speech-to-Text (ストリーミング認識) 用 ASR.

    single_utterance=True で1回の発話が終わると自動的にストリーム終了.
    interim_results=True で途中経過(Partial)を取得.
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        result_queue: queue.Queue,
        stop_event: threading.Event,
        language_code: str = "ja-JP",
    ):
        self.audio_queue = audio_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.language_code = language_code
        self.logger = get_logger('google_cloud_asr')

        self.client = speech.SpeechClient()

        self.call_active = False  # ストリーミング中かどうか
        self._audio_buffer_queue = queue.Queue()
        self._response_thread = None
        self._start_time = None
        self._stop_time = None

    def run(self):
        self.logger.info("スレッド起動: Streaming Speech API (single_utterance=True)")
        while not self.stop_event.is_set():
            try:
                command, data = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if command == "start":
                self._handle_start(data)
            elif command == "audio":
                self._handle_audio(data)
            elif command == "stop":
                self._handle_stop(data)
            elif command == "stop_all":
                self._handle_stop_all()
                return  # スレッド終了

    def _handle_start(self, data: bytes):
        self._start_time = float(data.decode())
        self._stop_time = None

        if not self.call_active:
            self.logger.info("ストリーミング認識 開始")
            self.call_active = True
            # 音声チャンクを受け取る queue を空に
            while not self._audio_buffer_queue.empty():
                self._audio_buffer_queue.get_nowait()

            # レスポンス受信用スレッドを立ち上げる
            self._response_thread = threading.Thread(
                target=self._streaming_recognize_loop, daemon=True
            )
            self._response_thread.start()
        else:
            self.logger.info("既にストリーミング中 → 'start' を無視")

    def _handle_audio(self, data: bytes):
        # ストリーミング中のみ、音声チャンクを queue に積む.
        if self.call_active:
            self._audio_buffer_queue.put(data)

    def _handle_stop(self, data: bytes):
        # VAD 側で発話終了を検知した際に呼ばれる想定。
        # ただし single_utterance=True の場合、GCP 側が自動終了するため
        # 必ずしもここで終わるわけではない。
        #
        # 一応、明示的に「もう終わり！」という場合にも対応できるように実装。
        self._stop_time = float(data.decode())
        if self.call_active:
            self.logger.info("_handle_stop: 明示的にストリーミング終了を要求")
            self.call_active = False
            # ストリームへの送信を止めるため、None などを入れてジェネレータを終了させる
            self._audio_buffer_queue.put(None)

            # レスポンススレッドが終了するのを待つ
            if self._response_thread:
                self._response_thread.join()
                self._response_thread = None
        else:
            self.logger.info("ストリーミング中ではない → 'stop' を無視")

    def _handle_stop_all(self):
        self.logger.info("stop_all受信 → スレッド終了処理")
        # まだストリーミング中なら終了
        if self.call_active:
            self._handle_stop(b"0.0")

    def _streaming_recognize_loop(self):
        # GCP ストリーミング API と対話するメソッド。
        # _request_stream() で音声をジェネレータとして送り、
        # 返ってくる認識レスポンスを逐次処理する。
        # single_utterance=True のため、
        # 1回の発話が終わるとサーバー側で自動的にストリームは終了。
        streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=SAMPLE_RATE,
                language_code=self.language_code,
                enable_automatic_punctuation=True,
            ),
            interim_results=True,  # 途中経過を取得
            single_utterance=True,  # 発話終了で自動的にストリーム終了
        )

        requests_generator = self._request_stream()

        # streaming_recognize(...) はレスポンス(StreamingRecognizeResponse)を逐次返してくる
        try:
            for response in self.client.streaming_recognize(
                config=streaming_config,
                requests=requests_generator,
            ):
                # response.results に複数の認識結果が入る場合がある(途中結果 or final)
                for result in response.results:
                    if not result.alternatives:
                        continue

                    transcript = result.alternatives[0].transcript
                    if result.is_final:
                        # 最終結果
                        self.result_queue.put(
                            (True, transcript, self._start_time, self._stop_time)
                        )

                        # single_utterance=True の場合、ここでストリーミングが終了するはず
                        # 念のため終了処理
                        self.call_active = False
                        # スレッド安全に Queue に None を入れてジェネレータを終了
                        self._audio_buffer_queue.put(None)
                        return

                    else:
                        # 途中経過 (Partial)
                        self.result_queue.put(
                            (False, transcript, self._start_time, None)
                        )

            # for ループが自然に抜けたらストリーム終了 or エラー
        except Exception as e:
            self.logger.error(f"ストリーミング認識中に例外: {e}")
        finally:
            self.call_active = False
            self.logger.info("ストリーミング終了")

    def _request_stream(self):
        """
        Queue から音声チャンクを受け取って、StreamingRecognizeRequest に包んで yield する.

        None が来たら終了。
        """
        while True:
            try:
                chunk = self._audio_buffer_queue.get(timeout=0.1)
            except queue.Empty:
                if not self.call_active:
                    # ストリーミングが終わっているなら終了
                    return
                continue
            if chunk is None:
                # 終了指示
                return
            yield speech.StreamingRecognizeRequest(audio_content=chunk)


class SilenceAwareVADIterator:
    """
    VADIteratorのラッパークラス.
    発話終了判定を指定された無音継続時間に基づいて行う.
    """
    MODEL_NAME = "silero_vad"
    REPO = "snakers4/silero-vad:v4.0"

    def __init__(self, silence_threshold_ms=SILERO_VAD_SILENCE_THRESHOLD_MS,
                 threshold=SILERO_VAD_THRESHOLD):
        self.logger = get_logger('silence_aware_vad')
        self.logger.info("Torch Hubからモデルをロードします...")
        # SileroVADモデルのロード
        self.model, self.utils = torch.hub.load(
            repo_or_dir=self.REPO,
            model=self.MODEL_NAME,
            force_reload=False,
        )
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = self.utils

        # 元のVADIteratorを初期化
        self.vad_iterator = self.VADIterator(self.model, threshold=threshold)
        # 無音検出用の状態
        self.silence_frame_count = 0
        self.silence_threshold_frames = int(math.ceil(silence_threshold_ms / FRAME_DURATION_MS))
        self.in_speech = False
        self.last_result = None

    def __call__(self, audio_float32, return_seconds=False):
        # 元のVADIteratorで判定
        result = self.vad_iterator(audio_float32, return_seconds=return_seconds)
        self.last_result = result

        if result and "start" in result:
            # 発話開始
            self.in_speech = True
            self.silence_frame_count = 0
            return result
        elif result and "end" in result:
            # 発話終了検知 → 無音カウント開始
            if self.in_speech:
                self.silence_frame_count = 1
                self.in_speech = False
                return None  # まだendは返さない
            else:
                # すでに無音中ならカウントアップ
                self.silence_frame_count += 1
                if self.silence_frame_count >= self.silence_threshold_frames:
                    # 1秒間無音が続いたら発話終了
                    self.silence_frame_count = 0
                    return result
                else:
                    return None
        else:
            if self.in_speech:
                # 発話継続
                self.silence_frame_count = 0
                return result
            else:
                # 無音中
                if self.silence_frame_count > 0:
                    self.silence_frame_count += 1
                    if self.silence_frame_count >= self.silence_threshold_frames:
                        self.silence_frame_count = 0
                        return {"end": True}  # 1秒間無音が続いたら発話終了
                    else:
                        return None
                else:
                    return None


class SileroVadProcessor(VADBase):
    """
    Silero VAD を用いた発話検知を行うクラス.

    - フレーム単位の音声を受け取り、(model, VADIterator)で判定.
    - 連続無音フレーム数などの判定は Silero 方式に任せるが、
      発話開始時は過去数フレーム(PRE_SPEECH_TIME_MS分)もまとめて返す.
    """

    def __init__(self):
        self.logger = get_logger('silero_vad')
        self.logger.info("SilenceAwareVADIteratorを初期化します...")

        # SilenceAwareVADIteratorを初期化
        self.vad_it = SilenceAwareVADIterator()

        # フレーム関連
        self.samples_per_frame = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
        self.pre_speech_frames = PRE_SPEECH_TIME_MS // FRAME_DURATION_MS
        self.pre_speech_buffer = collections.deque(maxlen=self.pre_speech_frames)
        self.in_speech = False

    def process_frame(self, frame: bytes):
        # フレームを Silero VAD にかけて音声状態を判断し、
        # 以下のいずれかを返す:
        #   - ("speech_start", [過去分 + 今フレーム])
        #   - ("speech_cont", [今フレーム])
        #   - ("speech_stop", [今フレーム])
        #   - (None, [])
        self.pre_speech_buffer.append(frame)

        # メモリ割り当てを最適化: copyを避ける
        data_np = np.frombuffer(frame, dtype=np.int16)
        audio_float32 = torch.from_numpy(data_np).float() / 32768.0

        result_dic = self.vad_it(audio_float32, return_seconds=False)

        if result_dic and "start" in result_dic:
            # 発話開始
            self.in_speech = True
            return "speech_start", list(self.pre_speech_buffer)
        elif result_dic and "end" in result_dic:
            # 発話終了
            self.in_speech = False
            return "speech_stop", [frame]
        else:
            if self.in_speech:
                # 発話継続
                return "speech_cont", [frame]
            else:
                # 無音
                return None, []


class OpenWakeWordProcessor(VADBase):
    """
    OpenWakeWordProcessor.

    - SileroVAD のモデルを利用して「発話終了」を検出。
    - OpenWakeWord でウェイクワードを検出したら "speech_start" を返す。
    - "speech_start" から 8秒経過、または 3秒後以降に Silero VAD で「発話終了」を検出したら "speech_stop" を返す。
    - それ以外のフレーム中は、発話中であれば "speech_cont"、無音であれば (None, []) を返す。
    """

    def __init__(self, model_folder: str, model_name: str):
        self.logger = get_logger('open_wake_word')

        # Silero VAD (終了検出にのみ使う)
        self.vad_it = SilenceAwareVADIterator(
            silence_threshold_ms=OPENWAKEWORD_SILENCE_THRESHOLD_MS,
            threshold=SILERO_VAD_THRESHOLD
        )

        self.logger.info("OpenWakeWordのモデルをロードします...")
        # OpenWakeWord のモデル (ダウンロード(動作に必要な基本的なモデル用)
        openwakeword.utils.download_models()
        # OpenWakeWord のモデル ( 使用するウェイクワード用モデルを指定するために、上記とは別にモデルをダウンロード
        openwakeword.utils.download_models(target_directory=model_folder)

        model_path = os.path.join(model_folder, model_name)
        self.oww_model = Model(wakeword_models=[model_path])

        self.in_speech = False  # 今、OpenWakeWordProcessor的に「発話中」かどうか
        self.speech_start_time = (
            None  # 発話を開始したフレームの時刻（フレーム数ベース）
        )
        self.frame_count_since_start = 0  # 発話開始後に経過したフレーム数

        # 累積フレーム数（時間計算用）
        self.total_frame_count = 0

    def process_frame(self, frame: bytes):
        # 1フレーム(30ms)ごとに呼び出される想定。
        # 戻り値:
        #   - ("speech_start", [過去分 + 今フレーム])
        #   - ("speech_cont", [今フレーム])
        #   - ("speech_stop", [今フレーム])
        #   - (None, [])
        self.total_frame_count += 1

        # OpenWakeWord でウェイクワード検知
        data_np = np.frombuffer(frame, dtype=np.int16)
        self.oww_model.predict(data_np)
        # 直近のスコアを取り出し
        oww_score = 0.0
        for scores in self.oww_model.prediction_buffer.values():
            # 最後のスコアを参照
            oww_score = scores[-1] if len(scores) > 0 else 0.0

        # Silero VAD (終了判定に使う)
        audio_float32 = torch.from_numpy(data_np.copy()).float() / 32768.0
        silero_result = self.vad_it(audio_float32, return_seconds=False)
        silero_end_detected = (silero_result is not None) and ("end" in silero_result)

        # イベント判定
        if not self.in_speech:
            # まだ発話中ではない
            if oww_score > OPEN_WAKEWORD_THRESHOLD:
                # ウェイクワードを検出 → 発話開始
                self.in_speech = True
                self.speech_start_time = self.total_frame_count
                self.frame_count_since_start = 0
                # 今フレームのみ
                return "speech_start", [frame]
            else:
                return None, []
        else:
            # in_speech = True → ウェイクワード検出後
            self.frame_count_since_start += 1
            elapsed_sec = (self.frame_count_since_start * FRAME_DURATION_MS) / 1000.0

            # speech_stopの判定
            if (
                silero_end_detected
                and elapsed_sec > OPEN_WAKEWORD_SPEECH_END_MIN_DURATION
            ) or (elapsed_sec >= OPEN_WAKEWORD_SPEECH_TIMEOUT_SECONDS):
                self.in_speech = False
                return "speech_stop", [frame]
            else:
                return "speech_cont", [frame]


class AudioRecorderBase(ABC):
    """音声入力のための抽象基底クラス."""

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def read_frame(self) -> bytes:
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def get_device_info(self) -> dict:
        pass


class MicAudioRecorder(AudioRecorderBase):
    """マイク入力専用のAudioRecorderクラス."""

    def __init__(self, read_frame_size, input_device_index=None):
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.read_frame_size = read_frame_size
        self.input_device_index = input_device_index
        self.logger = get_logger('mic_audio_recorder')

    def get_device_info(self) -> dict:
        """デバイス情報を取得して返す."""
        device_count = self.pa.get_device_count()
        device_info = {}
        for i in range(device_count):
            info = self.pa.get_device_info_by_index(i)
            device_info[i] = info
        return device_info

    def open(self):
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.read_frame_size,
            input_device_index=self.input_device_index,
            stream_callback=None,
        )
        self.stream.start_stream()
        self.logger.info(f"input_device_index: {self.input_device_index}")
        self.logger.info("マイク入力開始")

    def read_frame(self) -> bytes:
        if self.stream is not None:
            frame = self.stream.read(self.read_frame_size, exception_on_overflow=False)
            return frame
        return b""

    def close(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()
        self.logger.info("マイク入力終了")


class WavAudioRecorder(AudioRecorderBase):
    """WAVファイル入力専用のAudioRecorderクラス."""

    def __init__(
        self, read_frame_size, input_file: str, simulate_realtime: bool = False
    ):
        self.read_frame_size = read_frame_size
        self.input_file = input_file
        self.simulate_realtime = simulate_realtime
        self.wav_handle = None
        self.logger = get_logger('wav_audio_recorder')

    def get_device_info(self) -> dict:
        # デバイス情報はマイクと異なるため、空の辞書を返す。
        return {}

    def open(self):
        if not self.input_file:
            raise ValueError("input_file が指定されていません。")
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"WAVファイルが見つかりません: {self.input_file}")
        self.wav_handle = wave.open(self.input_file, "rb")
        # チャンネル数、サンプル幅、サンプリングレートのチェック
        if self.wav_handle.getnchannels() != CHANNELS:
            raise ValueError(f"WAVファイルのチャンネル数が {CHANNELS} ではありません。")
        if self.wav_handle.getsampwidth() != SAMPLE_WIDTH:
            raise ValueError(
                f"WAVファイルのサンプル幅が {SAMPLE_WIDTH} バイトではありません。"
            )
        if self.wav_handle.getframerate() != SAMPLE_RATE:
            raise ValueError(
                f"WAVファイルのサンプリングレートが {SAMPLE_RATE} Hz ではありません。"
            )
        self.logger.info(f"WAVファイル入力開始: {self.input_file}")

    def read_frame(self) -> bytes:
        if self.wav_handle is None:
            return b""

        data = self.wav_handle.readframes(self.read_frame_size)
        if not data:
            return b""  # WAVファイルの終わり

        if self.simulate_realtime:
            time.sleep(FRAME_DURATION_MS / 1000.0)  # フレーム間に遅延を挿入

        return data

    def close(self):
        if self.wav_handle is not None:
            self.wav_handle.close()
        self.logger.info("WAVファイル入力終了")


class AudioWriterBase(ABC):
    @abstractmethod
    def open(self) -> None:
        pass

    @abstractmethod
    def write(self, data: bytes) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class LabelWriterBase(ABC):
    @abstractmethod
    def write_segment(self, start: float, end: float, label: str) -> None:
        pass


# ダミー実装
class DummyAudioWriter(AudioWriterBase):
    def open(self) -> None:
        pass

    def write(self, data: bytes) -> None:
        pass

    def close(self) -> None:
        pass


class DummyLabelWriter(LabelWriterBase):
    def write_segment(self, start: float, end: float, label: str) -> None:
        pass


# 実際の実装
class FullAudioWriter(AudioWriterBase):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.full_wav_handle = None
        # フォルダの存在を確認し、なければ作成
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def open(self) -> None:
        # WAVファイルを"書き込みモード"で開き、チャンネル数、サンプル幅、サンプリングレートを設定
        self.full_wav_handle = wave.open(self.file_path, "wb")
        self.full_wav_handle.setnchannels(1)  # チャンネル数: モノラル
        self.full_wav_handle.setsampwidth(2)  # サンプル幅: 16bit = 2 bytes
        self.full_wav_handle.setframerate(16000)  # サンプリングレート: 16kHz

    def write(self, data: bytes) -> None:
        if self.full_wav_handle:
            self.full_wav_handle.writeframes(data)

    def close(self) -> None:
        if self.full_wav_handle:
            self.full_wav_handle.close()
            self.full_wav_handle = None


class LabelWriter(LabelWriterBase):
    def __init__(self, vad_text_file_path: str):
        self.file_path = vad_text_file_path
        # フォルダの存在を確認し、なければ作成
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def write_segment(self, start: float, end: float, label: str) -> None:
        with open(self.file_path, mode="a", encoding="utf-8") as f:
            f.write(f"{start:.2f}\t{end:.2f}\t{label}\n")


class DummySpeechAudioWriter(AudioWriterBase):
    """音声認識1回分の音声を出力しないダミークラス"""
    def open(self) -> None:
        pass

    def write(self, data: bytes) -> None:
        pass

    def close(self) -> None:
        pass


class SpeechAudioWriter(AudioWriterBase):
    """音声認識1回分の音声を出力するクラス"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.current_wav_handle = None
        self.current_file_path = None
        self.logger = get_logger('speech_audio_writer')
        # 出力ディレクトリの作成
        os.makedirs(self.output_dir, exist_ok=True)

    def open(self) -> None:
        # 新しいセッションの開始時に呼ばれる
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_file_path = os.path.join(self.output_dir, f"speech_{timestamp}.wav")
        self.current_wav_handle = wave.open(self.current_file_path, "wb")
        self.current_wav_handle.setnchannels(1)  # モノラル
        self.current_wav_handle.setsampwidth(2)  # 16bit = 2 bytes
        self.current_wav_handle.setframerate(SAMPLE_RATE)  # 16kHz
        self.logger.info(f"音声認識セッションの音声出力開始: {self.current_file_path}")

    def write(self, data: bytes) -> None:
        if self.current_wav_handle:
            self.current_wav_handle.writeframes(data)

    def close(self) -> None:
        if self.current_wav_handle:
            self.current_wav_handle.close()
            self.current_wav_handle = None
            self.logger.info(f"音声認識セッションの音声出力終了: {self.current_file_path}")


class SpeechRecognitionSystem:
    """
    システム全体の制御を行うクラス.

    - スレッド生成（音声認識ワーカー）
    - メインループで AudioRecorder → VAD → ASRへコマンド送信 → 結果出力
    - シグナルハンドラ登録
    """

    def __init__(
        self,
        vad_processor: VADBase,  # VADモジュールを直接受け取る
        asr: ASRBase,  # ASRモジュールを直接受け取る
        recorder: AudioRecorderBase,  # AudioRecorderモジュールを直接受け取る
        full_audio_writer: Optional[FullAudioWriter],
        label_writer: Optional[LabelWriter],
        speech_audio_writer: Optional[SpeechAudioWriter],
        on_asr_event=None,
    ):
        # vad_processor: VADモジュール (VAD_SILERO_VAD または VAD_OPENWAKEWORD)
        # asr: ASRモジュール (ASR_GOOGLE_CLOUD または ASR_WHISPER)
        # recorder: AudioRecorder モジュール
        # full_audio_writer: デバッグ用の全音声出力クラス
        # label_writer: デバッグ用のラベル出力クラス
        # on_asr_event: 音声認識イベント用のコールバック関数

        # 終了フラグ
        self.stop_event = threading.Event()

        # シグナルハンドラ登録
        signal.signal(signal.SIGINT, self._signal_handler)

        # ワーカーとのやりとり用のキュー
        self.audio_queue = asr.audio_queue  # ASRモジュールからキューを取得
        self.result_queue = asr.result_queue  # ASRモジュールからキューを取得

        self.asr = asr
        self.asr_thread = threading.Thread(target=self.asr.run, daemon=True)

        # VAD処理(VADBase)
        self.vad_processor = vad_processor

        # オーディオレコーダ
        self.recorder = recorder

        self.full_audio_writer = full_audio_writer
        self.label_writer = label_writer
        self.speech_audio_writer = speech_audio_writer

        # コールバック登録
        self.on_asr_event = on_asr_event if on_asr_event else lambda d: None

        # 時間計測用
        self.processed_size: int = 0
        self.current_time: float = 0
        self.vad_start: float = 0

        self.logger = get_logger('speech_recognition_system')

    def _signal_handler(self, sig, frame):
        self.logger.info("捕捉: Ctrl+C で停止要求")
        self.stop_event.set()
        sys.exit(0)

    def start(self):
        self.logger.info("システム起動。Ctrl+Cで終了")
        device_info = self.recorder.get_device_info()
        for idx, info in device_info.items():
            if info["maxInputChannels"] > 0:
                self.logger.info(f"マイクデバイス {idx}: {info['name']}")

        self.asr_thread.start()
        self.recorder.open()

        try:
            while not self.stop_event.is_set():
                frame = self.recorder.read_frame()
                self.full_audio_writer.write(frame)
                if not frame:
                    # 何らかの理由でフレームが取れなかった → 適宜休止
                    time.sleep(0.01)
                    continue

                # VAD判定
                event, frames_to_send = self.vad_processor.process_frame(frame)

                # イベントに応じてワーカーへコマンド送信
                if event == "speech_start":
                    self.logger.info("VAD発話検出 → 'start'")
                    self.audio_queue.put(("start", str(self.current_time).encode()))
                    self.speech_audio_writer.open()
                    for f in frames_to_send:
                        self.audio_queue.put(("audio", f))
                        self.speech_audio_writer.write(f)
                    self.vad_start = self.current_time

                    # ウェイクワード検出時にコールバックを呼び出す
                    if isinstance(self.vad_processor, OpenWakeWordProcessor):
                        event_dict = {
                            "event_type": "wakeword_detected",
                            "start": self.current_time,
                            "end": self.current_time,
                            "text": "hey_mycroft",  # 実際のウェイクワード名を設定
                            "score": 0.95,  # 実際のスコアを設定
                        }
                        self.on_asr_event(event_dict)
                elif event == "speech_cont":
                    for f in frames_to_send:
                        self.audio_queue.put(("audio", f))
                        self.speech_audio_writer.write(f)

                elif event == "speech_stop":
                    for f in frames_to_send:
                        self.audio_queue.put(("audio", f))
                        self.speech_audio_writer.write(f)
                    self.logger.info("VAD終話検出 → 'stop'")
                    self.audio_queue.put(("stop", str(self.current_time).encode()))
                    self.speech_audio_writer.close()
                    self.label_writer.write_segment(
                        self.vad_start, self.current_time, "speech"
                    )
                elif event == "speech_timeout":
                    self.logger.info("VADタイムアウト検出 → 'stop'")
                    self.audio_queue.put(("stop", str(self.current_time).encode()))
                    self.speech_audio_writer.close()
                    event_dict = {
                        "event_type": "timeout",
                        "start": self.vad_start,
                        "end": self.current_time,
                        "reason": "speech_duration_exceeded",
                    }
                    self.on_asr_event(event_dict)
                elif event is None:
                    pass
                else:
                    raise ValueError(f"未知のイベント: {event}")

                # ワーカーからの結果を取り出して出力
                self._fetch_results()

                self._update_current_time(frame)

                # CPU負荷を下げるために少し待つ (30msフレーム間隔に合わせる)
                time.sleep(0.03)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            self.logger.error(f"エラー: {e}")
            raise
        finally:
            self.logger.info("終了処理。stop_allをワーカーへ送信")
            self.audio_queue.put(("stop_all", None))
            self.stop_event.set()

            self.recorder.close()
            self.asr_thread.join()

            self.full_audio_writer.close()
            self.speech_audio_writer.close()

            self.logger.info("プログラム終了")

    def _update_current_time(self, frame: bytes) -> None:
        self.processed_size += len(frame)
        self.current_time = (
            self.processed_size / SAMPLE_RATE / SAMPLE_WIDTH
        )  # 2 bytes per sample

    def _fetch_results(self):
        """ワーカーからの認識結果を受け取って出力."""
        while True:
            try:
                is_final, text, start, end = self.result_queue.get_nowait()
            except queue.Empty:
                break
            if text:
                if is_final:
                    self.logger.info(f"[Final] {text}")
                    event_dict = {
                        "event_type": "final_result",
                        "start": start,
                        "end": end if end else self.current_time,
                        "text": text,
                    }
                    self.on_asr_event(event_dict)
                    if end is not None:
                        self.label_writer.write_segment(start, end, text)
                else:
                    self.logger.info(f"[Partial] {text}")
                    event_dict = {
                        "event_type": "partial_result",
                        "start": start,
                        "end": None,
                        "text": text,
                    }
                    self.on_asr_event(event_dict)


def list_microphone_devices():
    """利用可能なマイクデバイスを一覧表示する関数."""
    pa = pyaudio.PyAudio()
    device_count = pa.get_device_count()
    print("利用可能なマイクデバイス一覧:")
    for i in range(device_count):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"デバイス {i}: {info['name']}")
    pa.terminate()


@click.command()
@click.option(
    "-l",
    "--list-mic-devices",
    is_flag=True,
    default=False,
    show_default=True,
    help="利用可能なマイクデバイスを一覧表示します。",
)
@click.option(
    "--vad-type",
    type=click.Choice([VAD_SILERO_VAD, VAD_OPENWAKEWORD], case_sensitive=False),
    default=VAD_OPENWAKEWORD,
    show_default=True,
    help='タイプを指定します。 "silero_vad" または "openwakeword"。',
)
@click.option(
    "--asr-type",
    type=click.Choice([ASR_GOOGLE_CLOUD, ASR_WHISPER], case_sensitive=False),
    default=ASR_GOOGLE_CLOUD,
    show_default=True,
    help='ASRのタイプを指定します。 "google_cloud" または "whisper"。',
)
@click.option(
    "--input-device-index",
    type=int,
    default=None,
    show_default=True,
    help="使用する入力デバイスのインデックスを指定します。デフォルトはシステムのデフォルトデバイス。",
)
@click.option(
    "--language-code",
    type=str,
    default="ja-JP",
    show_default=True,
    help="ASRの言語コードを指定します。",
)
@click.option(
    "--oww-model-folder",
    type=str,
    default="models",
    show_default=True,
    help="OpenWakeWordのモデルフォルダを指定します。",
)
@click.option(
    "--oww-model-name",
    type=str,
    default="hey_mycroft_v0.1.tflite",
    show_default=True,
    help="OpenWakeWordのモデル名を指定します。",
)
@click.option(
    "--input-file",
    type=str,
    default=None,
    show_default=True,
    help="WAVファイルのパスを指定します。指定しない場合はマイク入力を使用します。",
)
@click.option(
    "--simulate-realtime",
    is_flag=True,
    default=False,
    show_default=True,
    help="WAVファイル入力時にリアルタイム入力をシミュレートします。",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    show_default=True,
    help="デバッグモードを有効にします。",
)
def main(
    list_mic_devices,
    vad_type,
    asr_type,
    input_device_index,
    language_code,
    oww_model_folder,
    oww_model_name,
    input_file,
    simulate_realtime,
    debug,
):
    """音声認識システムを起動します."""
    if list_mic_devices:
        list_microphone_devices()
        return

    # 入力モードの決定
    if input_file:
        input_mode = "wav"
    else:
        input_mode = "mic"

    if debug:
        base_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 出力ファイルパスやフォーマットを定数で定義
        full_audio_path = f"./debug/{base_time_str}_audio_full.wav"
        label_text_path = f"./debug/{base_time_str}_label.txt"

        # デバッグ用コンポーネントの初期化
        full_audio_writer = FullAudioWriter(full_audio_path)
        full_audio_writer.open()

        speech_audio_writer = SpeechAudioWriter(output_dir="debug")

        label_writer = LabelWriter(label_text_path)
        print(f"デバッグモード: 全音声の出力開始: path={full_audio_path}")
        print(f"デバッグモード: ラベルの出力開始: path={label_text_path}")
    else:
        full_audio_writer = DummyAudioWriter()
        speech_audio_writer = DummySpeechAudioWriter()
        label_writer = DummyLabelWriter()

    # AudioRecorderの生成
    if input_mode == "mic":
        recorder = MicAudioRecorder(
            read_frame_size=512,  # 固定値。必要に応じて調整可能
            input_device_index=input_device_index,
        )
    elif input_mode == "wav":
        recorder = WavAudioRecorder(
            read_frame_size=512,  # 固定値。必要に応じて調整可能
            input_file=input_file,
            simulate_realtime=simulate_realtime,
        )
    else:
        raise ValueError(f"未知の input_mode: {input_mode}")

    # VADモジュールの生成
    if vad_type == VAD_SILERO_VAD:
        print("Silero VAD を使用します。")
        vad_processor = SileroVadProcessor()
    elif vad_type == VAD_OPENWAKEWORD:
        print("OpenWakeWordProcessor を使用します。")
        vad_processor = OpenWakeWordProcessor(
            model_folder=oww_model_folder, model_name=oww_model_name
        )
    else:
        raise ValueError(f"未知のVADタイプ: {vad_type}")

    # ASRモジュールの生成
    if asr_type == ASR_GOOGLE_CLOUD:
        print("Google Cloud Speech-to-Text を使用します。")
        asr = GoogleCloudASR(
            audio_queue=queue.Queue(),
            result_queue=queue.Queue(),
            stop_event=threading.Event(),
            language_code=language_code,
        )
    elif asr_type == ASR_WHISPER:
        print("faster-whisper を使用します。")
        asr = WhisperASR(
            model_name="large-v2",
            whisper_language_code="ja",
            whisper_device="auto",
            audio_queue=queue.Queue(),
            result_queue=queue.Queue(),
            stop_event=threading.Event(),
        )
    else:
        raise ValueError(f"未知のASRタイプ: {asr_type}")

    # SpeechRecognitionSystemにモジュールを渡して生成
    system = SpeechRecognitionSystem(
        vad_processor=vad_processor,  # 生成済みのVADモジュールを渡す
        asr=asr,  # 生成済みのASRモジュールを渡す
        recorder=recorder,  # 生成済みのAudioRecorderモジュールを渡す
        full_audio_writer=full_audio_writer,
        label_writer=label_writer,
        speech_audio_writer=speech_audio_writer,
        on_asr_event=lambda d: publish_asr_event(d),
        # ここでコールバックを設定
    )
    system.start()

    # ROS 2 との連携部分（必要に応じて適宜調整）
    # ここではコールバック関数内で ROS 2 ノードの Publisher にアクセスする必要があるため、
    # 実際の実装では ROS 2 ノードと音声認識システムの連携方法を再検討する必要があります。
    # 例えば、コールバック関数を ROS 2 ノード内で定義し、SpeechRecognitionSystem に渡すなど。

    # 以下は単純化した例です（実際には ROS 2 ノード内での実装を推奨）
    def publish_asr_event(event_dict: dict):
        # これは実際の ROS 2 ノードではなく、単なる関数です。
        # 実際には ROS 2 ノードの Publisher を使用する必要があります。
        print(f"Publishing event: {event_dict}")


if __name__ == "__main__":
    main()
