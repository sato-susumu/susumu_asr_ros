import os
import queue
import threading
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from susumu_asr_ros.susumu_asr import (
    SpeechRecognitionSystem,
    SileroVadProcessor,
    OpenWakeWordProcessor,
    VoskASR,
    GoogleCloudASR,
    MicAudioRecorder,
    WavAudioRecorder,
    FullAudioWriter,
    LabelWriter,
    DummyAudioWriter,
    DummyLabelWriter,
    list_microphone_devices,
    WhisperASR,
)


class SusumuAsrNode(Node):
    def __init__(self):
        super().__init__("susumu_asr_node")

        # ===================================
        # (A) Publisher は1つ。トピック名を "stt_event" にする
        # ===================================
        self.pub_stt_event = self.create_publisher(
            String, "stt_event", 10  # 単一のトピック
        )

        # ============================
        # 1) パラメータ宣言
        # ============================
        # 端末から --ros-args -p param_name:=value で上書き可能

        self.declare_parameter("list_mic_devices", True)
        # VADタイプ: "silero_vad" or "openwakeword"
        self.declare_parameter("vad_type", "openwakeword")
        # ASRタイプ: "google_cloud" or "vosk"
        self.declare_parameter("asr_type", "google_cloud")
        # Google Cloud の言語コードなどを指定
        self.declare_parameter("language_code", "ja-JP")
        # Voskモデル名
        self.declare_parameter("vosk_model_name", "vosk-model-ja-0.22")
        # OpenWakeWordモデルフォルダ＆モデル名
        self.declare_parameter("oww_model_folder", "models")
        self.declare_parameter("oww_model_name", "hey_mycroft_v0.1.tflite")

        # 入力デバイス (マイク) のインデックス
        self.declare_parameter("input_device_index", None)

        # Whisper専用パラメータ
        self.declare_parameter("whisper_model_name", "large-v2")
        self.declare_parameter("whisper_language_code", "auto")  # "auto", "ja", "en", ...
        self.declare_parameter("whisper_device", "auto")  # "auto", "cpu", "cuda"

        # ============================
        # デバッグ用
        # ============================
        # WAVファイルを使う場合 (指定があればWav, なければMic)
        self.declare_parameter("input_file", None)
        self.declare_parameter("simulate_realtime", False)

        # デバッグモード (音声やラベルをファイル出力)
        self.declare_parameter("debug", False)

        self.get_logger().info("SusumuAsrNode: パラメータの宣言完了")

        # ============================
        # 2) パラメータ取得
        # ============================
        list_mic = self.get_parameter("list_mic_devices").value
        vad_type = self.get_parameter("vad_type").value
        asr_type = self.get_parameter("asr_type").value
        language_code = self.get_parameter("language_code").value
        vosk_model_name = self.get_parameter("vosk_model_name").value
        oww_model_folder = self.get_parameter("oww_model_folder").value
        oww_model_name = self.get_parameter("oww_model_name").value
        input_device_index = self.get_parameter("input_device_index").value
        input_file = self.get_parameter("input_file").value
        simulate_realtime = self.get_parameter("simulate_realtime").value
        debug = self.get_parameter("debug").value

        # Whisper用のパラメータ取得
        whisper_model_name = self.get_parameter("whisper_model_name").value
        whisper_language_code = self.get_parameter("whisper_language_code").value
        whisper_device = self.get_parameter("whisper_device").value

        # ログ出力例
        self.get_logger().info(
             f"[Whisper Params] model={whisper_model_name}, "
             f"lang={whisper_language_code}, device={whisper_device}"
        )

        # ============================
        # 3) マイクデバイス一覧表示 (パラメータで指定された場合)
        # ============================
        if list_mic:
            # マイクの一覧を表示する関数
            self.get_logger().info("利用可能なマイクデバイス一覧を表示します。")
            list_microphone_devices()

        # ============================
        # 4) デバッグ出力の準備
        # ============================
        if debug:
            base_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_dir = "./debug"
            os.makedirs(debug_dir, exist_ok=True)

            full_audio_path = f"{debug_dir}/{base_time_str}_audio_full.wav"
            label_text_path = f"{debug_dir}/{base_time_str}_label.txt"

            self.full_audio_writer = FullAudioWriter(full_audio_path)
            self.full_audio_writer.open()

            self.label_writer = LabelWriter(label_text_path)

            self.get_logger().info(
                f"デバッグモード: 全音声={full_audio_path}, ラベル={label_text_path}"
            )
        else:
            self.full_audio_writer = DummyAudioWriter()
            self.label_writer = DummyLabelWriter()

        # ============================
        # 5) AudioRecorder の生成
        # ============================
        if input_file:
            # WAVファイルから入力
            self.get_logger().info(f"WAV入力モード: file={input_file}")
            self.recorder = WavAudioRecorder(
                read_frame_size=512,
                input_file=input_file,
                simulate_realtime=simulate_realtime,
            )
        else:
            # マイク入力
            self.get_logger().info("マイク入力モード")
            self.recorder = MicAudioRecorder(
                read_frame_size=512, input_device_index=input_device_index
            )

        # ============================
        # 6) VAD モジュールの生成
        # ============================
        if vad_type == "silero_vad":
            self.get_logger().info("VAD: SileroVadProcessor を使用します。")
            self.vad_processor = SileroVadProcessor()
        elif vad_type == "openwakeword":
            self.get_logger().info("VAD: OpenWakeWordProcessor を使用します。")

            if not os.path.exists(oww_model_folder):
                os.makedirs(oww_model_folder)

            self.vad_processor = OpenWakeWordProcessor(
                model_folder=oww_model_folder, model_name=oww_model_name
            )
        else:
            self.get_logger().warn(
                f"未知のVADタイプ={vad_type} → silero_vad とみなします。"
            )
            self.vad_processor = SileroVadProcessor()

        # ============================
        # 7) ASR モジュールの生成
        # ============================
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()

        if asr_type == "google_cloud":
            self.get_logger().info("ASR: GoogleCloudASR を使用します。")
            self.asr = GoogleCloudASR(
                audio_queue=self.audio_queue,
                result_queue=self.result_queue,
                stop_event=self.stop_event,
                language_code=language_code,
            )
        elif asr_type == "vosk":
            self.get_logger().info("ASR: VoskASR を使用します。")
            self.asr = VoskASR(
                model_name=vosk_model_name,
                audio_queue=self.audio_queue,
                result_queue=self.result_queue,
                stop_event=self.stop_event,
            )
        elif asr_type == "whisper":
            self.get_logger().info("ASR: WhisperASR (faster-whisper) を使用します。")
            self.asr = WhisperASR(
                model_name=whisper_model_name,
                whisper_language_code=whisper_language_code,
                whisper_device=whisper_device,
                audio_queue=self.audio_queue,
                result_queue=self.result_queue,
                stop_event=self.stop_event,
            )
        else:
            self.get_logger().warn(f"未知のASRタイプ={asr_type} → vosk とみなします。")
            self.asr = VoskASR(
                model_name=vosk_model_name,
                audio_queue=self.audio_queue,
                result_queue=self.result_queue,
                stop_event=self.stop_event,
            )

        # ============================
        # 8) SpeechRecognitionSystem の生成
        # ============================
        self.system = SpeechRecognitionSystem(
            vad_processor=self.vad_processor,
            asr=self.asr,
            recorder=self.recorder,
            full_audio_writer=self.full_audio_writer,
            label_writer=self.label_writer,
            # ===================================
            # (B) イベントを受け取るコールバックを登録
            # ===================================
            on_asr_event=self.on_asr_event,
        )

        # スレッドで音声ループを開始
        self.thread = threading.Thread(target=self.system.start, daemon=True)
        self.thread.start()

        self.get_logger().info("SusumuAsrNode: 初期化完了")

    def on_asr_event(self, event_dict: dict):
        """音声認識システムからのイベントを受け取り、JSON 文字列にして単一トピック (stt_event) へパブリッシュする."""
        import json

        msg = String()
        msg.data = json.dumps(event_dict, ensure_ascii=False)
        self.pub_stt_event.publish(msg)

    def destroy_node(self):
        # ノード破棄時に、システム側の終了処理を進めたい場合はここで何かしてもOK
        self.get_logger().info("SusumuAsrNode: destroy_node called")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SusumuAsrNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt: shutdown.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
