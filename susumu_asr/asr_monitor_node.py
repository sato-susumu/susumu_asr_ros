"""ASR モニターノード — /stt_event を購読してリアルタイムGUIに表示."""
import json
import sys
import threading

import pyaudio
from PyQt5 import QtWidgets  # noqa: I100,I201
import rclpy  # noqa: I201
from rclpy.node import Node
from std_msgs.msg import String  # noqa: I201

from susumu_asr.asr_monitor_gui import ASRMonitorWidget
from susumu_asr.constants import AUDIO_FRAME_SAMPLES, CHANNELS, SAMPLE_RATE

_SAMPLE_WIDTH = 2  # int16


class ASRMonitorNode(Node):
    """stt_event 購読 + マイク入力 → GUI へ転送する ROS2 ノード."""

    def __init__(self, widget: ASRMonitorWidget):  # noqa: D107
        super().__init__('asr_monitor_node')
        self._widget = widget

        self.declare_parameter('mic_device_index', -1)

        self._sub = self.create_subscription(
            String, 'stt_event', self._on_stt_event, 10
        )
        self.get_logger().info('ASR モニター起動。/stt_event を購読中')

        mic_idx = self.get_parameter('mic_device_index').value
        if mic_idx == -2:
            self.get_logger().info('マイク入力無効（WAVデバッグモード）')
        else:
            self._start_mic(None if mic_idx < 0 else mic_idx)

    def _on_stt_event(self, msg: String):
        try:
            ev = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._widget.push_event(ev)

    def _start_mic(self, device_index):
        def _run():
            pa = pyaudio.PyAudio()
            try:
                stream = pa.open(
                    rate=SAMPLE_RATE,
                    channels=CHANNELS,
                    format=pyaudio.paInt16,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=AUDIO_FRAME_SAMPLES,
                )
                self.get_logger().info('マイク入力開始')
                while rclpy.ok():
                    data = stream.read(
                        AUDIO_FRAME_SAMPLES, exception_on_overflow=False
                    )
                    self._widget.push_audio(data)
                stream.stop_stream()
                stream.close()
            except Exception as e:  # noqa: B902
                self.get_logger().error(f'マイク入力エラー: {e}')
            finally:
                pa.terminate()

        t = threading.Thread(target=_run, daemon=True)
        t.start()


def _is_wav_mode() -> bool:
    """コマンドライン引数から mic_device_index=-2 を検出する."""
    for i, a in enumerate(sys.argv):
        if a == '-p' and i + 1 < len(sys.argv):
            if 'mic_device_index:=-2' in sys.argv[i + 1]:
                return True
        if 'mic_device_index:=-2' in a:
            return True
    return False


def main():  # noqa: D103
    rclpy.init()
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('ASR Monitor')

    win = QtWidgets.QMainWindow()
    win.setWindowTitle('ASR Monitor')
    win.resize(1200, 600)

    wav_mode = _is_wav_mode()

    widget = ASRMonitorWidget(wav_mode=wav_mode)
    win.setCentralWidget(widget)
    win.show()

    node = ASRMonitorNode(widget)

    spin_thread = threading.Thread(
        target=lambda: rclpy.spin(node), daemon=True
    )
    spin_thread.start()

    try:
        sys.exit(app.exec_())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
