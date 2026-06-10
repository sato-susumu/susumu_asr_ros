"""ASR モニターノード — /stt_event と /audio_raw を購読してリアルタイムGUIに表示."""
import json
import sys
import threading

import numpy as np
from PyQt5 import QtWidgets  # noqa: I100,I201
import rclpy  # noqa: I201
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, String  # noqa: I201

from susumu_asr.asr_monitor_gui import ASRMonitorWidget


class ASRMonitorNode(Node):
    """stt_event + audio_raw 購読 → GUI へ転送する ROS2 ノード."""

    def __init__(self, widget: ASRMonitorWidget):  # noqa: D107
        super().__init__('asr_monitor_node')
        self._widget = widget

        self._sub_event = self.create_subscription(
            String, 'stt_event', self._on_stt_event, 10
        )
        self._sub_audio = self.create_subscription(
            Int16MultiArray, 'audio_raw', self._on_audio_raw, 100
        )
        self.get_logger().info('ASR モニター起動。/stt_event, /audio_raw を購読中')

    def _on_stt_event(self, msg: String):
        try:
            ev = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._widget.push_event(ev)

    def _on_audio_raw(self, msg: Int16MultiArray):
        frame = np.array(msg.data, dtype=np.int16).tobytes()
        self._widget.push_audio(frame)


def main():  # noqa: D103
    rclpy.init()
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('ASR Monitor')

    win = QtWidgets.QMainWindow()
    win.setWindowTitle('ASR Monitor')
    win.resize(1200, 600)

    widget = ASRMonitorWidget()
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
