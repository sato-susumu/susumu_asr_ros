"""ROS2 音声認識ノード（プラグインベース）."""
from dataclasses import asdict
from datetime import datetime
import json
import os
import queue
import threading

from dotenv import load_dotenv
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from susumu_asr_ros.audio_io import (
    DummyAudioWriter,
    DummyLabelWriter,
    DummySpeechAudioWriter,
    FullAudioWriter,
    LabelWriter,
    MicAudioRecorder,
    SpeechAudioWriter,
    WavAudioRecorder,
    WaveformImageWriter,
)
from susumu_asr_ros.constants import AUDIO_FRAME_SAMPLES
from susumu_asr_ros.plugin_base import ASREventUnion, AsrFinalResultEvent
from susumu_asr_ros.plugin_loader import PluginLoader
from susumu_asr_ros.susumu_asr import SpeechRecognitionSystem


class SusumuAsrNode(Node):

    def __init__(self):
        super().__init__('susumu_asr_node')

        self.pub_stt_event = self.create_publisher(String, 'stt_event', 10)
        self.pub_stt = self.create_publisher(String, 'stt', 10)

        # -------------------------------------------------------
        # フレームワーク共通パラメータ
        # -------------------------------------------------------
        self.declare_parameter('vad_plugin', 'silero_vad')
        self.declare_parameter('wakeword_plugin', 'passthrough')
        self.declare_parameter('asr_plugin', 'google_cloud')
        self.declare_parameter('input_device_index', -1)
        self.declare_parameter('input_file', '')
        self.declare_parameter('simulate_realtime', True)
        self.declare_parameter('debug', False)
        self.declare_parameter('debug_dir', './debug')
        self.declare_parameter('list_mic_devices', False)

        vad_name = self.get_parameter('vad_plugin').value
        wakeword_name = self.get_parameter('wakeword_plugin').value
        asr_name = self.get_parameter('asr_plugin').value
        input_device_index = self.get_parameter('input_device_index').value
        input_file = self.get_parameter('input_file').value or None
        simulate_realtime = self.get_parameter('simulate_realtime').value
        debug = self.get_parameter('debug').value
        debug_dir = self.get_parameter('debug_dir').value
        list_mic = self.get_parameter('list_mic_devices').value

        self.get_logger().info(
            f'プラグイン: vad={vad_name}, wakeword={wakeword_name}, asr={asr_name}'
        )

        if list_mic:
            MicAudioRecorder.list_devices()

        # -------------------------------------------------------
        # VAD プラグイン: ロード → パラメータ宣言・注入 → setup()
        # -------------------------------------------------------
        vad_cls = PluginLoader.load_vad(vad_name)
        self._vad_plugin = vad_cls()
        vad_params = self._declare_plugin_params(
            vad_name, self._vad_plugin.get_param_declarations()
        )
        self.get_logger().info(f'VAD パラメータ ({vad_name}): {vad_params}')
        self._vad_plugin.load_params(vad_params)
        self._vad_plugin.setup()

        # -------------------------------------------------------
        # Wakeword プラグイン: ロード → パラメータ宣言・注入 → setup()
        # -------------------------------------------------------
        wakeword_cls = PluginLoader.load_wakeword(wakeword_name)
        self._wakeword_plugin = wakeword_cls()
        wakeword_params = self._declare_plugin_params(
            wakeword_name, self._wakeword_plugin.get_param_declarations()
        )
        self.get_logger().info(f'Wakeword パラメータ ({wakeword_name}): {wakeword_params}')
        self._wakeword_plugin.load_params(wakeword_params)
        self._wakeword_plugin.setup()

        # -------------------------------------------------------
        # ASR プラグイン: ロード → パラメータ宣言・注入 → setup()
        # -------------------------------------------------------
        asr_cls = PluginLoader.load_asr(asr_name)
        self._asr_plugin = asr_cls()
        asr_params = self._declare_plugin_params(
            asr_name, self._asr_plugin.get_param_declarations()
        )
        self.get_logger().info(f'ASR パラメータ ({asr_name}): {asr_params}')
        self._asr_plugin.load_params(asr_params)
        self._asr_plugin.setup(queue.Queue(), queue.Queue(), threading.Event())

        # -------------------------------------------------------
        # デバッグ用ライター
        # -------------------------------------------------------
        if debug:
            base_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            os.makedirs(debug_dir, exist_ok=True)
            full_audio_path = f'{debug_dir}/{base_time_str}_audio_full.wav'
            label_text_path = f'{debug_dir}/{base_time_str}_label.txt'
            image_path = f'{debug_dir}/{base_time_str}_waveform.png'
            log_path = f'{debug_dir}/{base_time_str}_log.txt'
            full_audio_writer = FullAudioWriter(full_audio_path)
            full_audio_writer.open()
            speech_audio_writer = SpeechAudioWriter(output_dir=debug_dir)
            label_writer = LabelWriter(label_text_path)
            self._waveform_image_writer = WaveformImageWriter(
                wav_path=full_audio_path,
                label_path=label_text_path,
                image_path=image_path,
            )
            self._setup_file_logger(log_path)
            self.get_logger().info(
                f'デバッグモード: 全音声={full_audio_path}, ラベル={label_text_path}'
                f', 画像={image_path}, ログ={log_path}'
            )
        else:
            full_audio_writer = DummyAudioWriter()
            speech_audio_writer = DummySpeechAudioWriter()
            label_writer = DummyLabelWriter()
            self._waveform_image_writer = None

        # -------------------------------------------------------
        # AudioRecorder
        # -------------------------------------------------------
        if input_file:
            self.get_logger().info(f'WAV 入力モード: file={input_file}')
            recorder = WavAudioRecorder(
                read_frame_size=AUDIO_FRAME_SAMPLES,
                input_file=input_file,
                simulate_realtime=simulate_realtime,
            )
        else:
            idx = input_device_index if input_device_index >= 0 else None
            self.get_logger().info(f'マイク入力モード: device_index={idx}')
            recorder = MicAudioRecorder(
                read_frame_size=AUDIO_FRAME_SAMPLES, input_device_index=idx
            )

        # -------------------------------------------------------
        # SpeechRecognitionSystem
        # -------------------------------------------------------
        self._system = SpeechRecognitionSystem(
            vad_plugin=self._vad_plugin,
            wakeword_plugin=self._wakeword_plugin,
            asr_plugin=self._asr_plugin,
            recorder=recorder,
            full_audio_writer=full_audio_writer,
            label_writer=label_writer,
            speech_audio_writer=speech_audio_writer,
            on_asr_event=self._on_asr_event,
            on_stop=self._on_system_stop,
        )

        self._thread = threading.Thread(target=self._system.start, daemon=True)
        self._thread.start()
        self.get_logger().info('SusumuAsrNode: 初期化完了')

    def _declare_plugin_params(self, prefix: str, declarations) -> dict:
        """
        プラグインのパラメータ宣言をノードに登録し、値を返す.

        ROS2 パラメータ名は "{prefix}.{param_name}" 形式。
        launch ファイルや --ros-args から上書き可能。
        """
        values = {}
        for decl in declarations:
            ros_name = f'{prefix}.{decl.name}'
            self.declare_parameter(ros_name, decl.default)
            values[decl.name] = self.get_parameter(ros_name).value
        return values

    def _on_asr_event(self, event: ASREventUnion):
        if not rclpy.ok():
            return
        msg = String()
        msg.data = json.dumps(asdict(event), ensure_ascii=False)
        self.pub_stt_event.publish(msg)

        if isinstance(event, AsrFinalResultEvent) and event.text:
            msg2 = String()
            msg2.data = event.text
            self.pub_stt.publish(msg2)

    def _setup_file_logger(self, log_path: str) -> None:
        """rclpy の全ロガーにファイル出力を追加する."""
        import rclpy.impl.rcutils_logger as rcutils_logger_module

        log_file = open(log_path, 'w', encoding='utf-8')
        self._log_file = log_file

        original_log = rcutils_logger_module.RcutilsLogger.log

        def patched_log(self_logger, message, severity, *args, **kwargs):
            original_log(self_logger, message, severity, *args, **kwargs)
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            log_file.write(f'{ts} [{severity.name}] [{self_logger.name}]: {message}\n')
            log_file.flush()

        rcutils_logger_module.RcutilsLogger.log = patched_log
        self._original_log = original_log
        self._rcutils_logger_module = rcutils_logger_module

    def _on_system_stop(self):
        if self._waveform_image_writer is not None:
            self._waveform_image_writer.generate()
        if rclpy.ok():
            rclpy.shutdown()

    def destroy_node(self):
        self.get_logger().info('SusumuAsrNode: destroy_node called')
        self._system.stop_event.set()
        self._thread.join(timeout=10.0)
        if hasattr(self, '_original_log'):
            self._rcutils_logger_module.RcutilsLogger.log = self._original_log
            self._log_file.close()
        super().destroy_node()


def main(args=None):
    load_dotenv(os.environ.get('SUSUMU_ASR_ENV_FILE') or None)
    rclpy.init(args=args)
    node = SusumuAsrNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
