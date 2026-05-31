"""
ASR リアルタイムモニター.

/stt_event・/stt・/audio_level・/wakeword_score をサブスクライブし、
イベントが届くたびにターミナルへ表示する。

使い方:
    ros2 run susumu_asr_ros susumu_asr_monitor
"""
from datetime import datetime
import json
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String

from susumu_asr_ros.plugin_base import ASREventType

_COLORS = {
    ASREventType.LISTENING_STARTED: '\033[96m',   # シアン
    ASREventType.WAV_FINISHED:      '\033[96m',
    ASREventType.WAKEWORD_DETECTED: '\033[93m',   # 黄
    ASREventType.PARTIAL_RESULT:    '\033[90m',   # グレー
    ASREventType.FINAL_RESULT:      '\033[92m',   # 緑
    ASREventType.TIMEOUT:           '\033[91m',   # 赤
}
_RESET = '\033[0m'

_BAR_WIDTH = 30
_METER_INTERVAL = 0.1  # 秒


def _format_event(event: dict) -> str:
    ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    raw_type = event.get('event_type', 'unknown')

    try:
        etype = ASREventType(raw_type)
    except ValueError:
        return f'[{ts}] {raw_type}: {event}'

    color = _COLORS.get(etype, '')

    if etype == ASREventType.LISTENING_STARTED:
        line = f'[{ts}] 録音開始'
    elif etype == ASREventType.WAV_FINISHED:
        dur = event.get('duration', 0)
        line = f'[{ts}] WAV 再生完了 ({dur:.2f}s)'
    elif etype == ASREventType.WAKEWORD_DETECTED:
        text = event.get('text', '')
        score = event.get('score', 0)
        line = f'[{ts}] *** WAKEWORD: {text}  score={score:.4f} ***'
    elif etype == ASREventType.PARTIAL_RESULT:
        text = event.get('text', '')
        line = f'[{ts}] partial: {text}'
    elif etype == ASREventType.FINAL_RESULT:
        text = event.get('text', '')
        start = event.get('start', 0)
        end = event.get('end', 0)
        dur = f'{end - start:.1f}s' if end else '?'
        line = f'[{ts}] FINAL ({dur}): {text}'
    elif etype == ASREventType.TIMEOUT:
        reason = event.get('reason', '')
        line = f'[{ts}] TIMEOUT: {reason}'
    else:
        line = f'[{ts}] {raw_type}: {event}'

    return f'{color}{line}{_RESET}'


def _bar(value: float, width: int = _BAR_WIDTH, color: str = '') -> str:
    fill = min(int(value * width), width)
    bar = '█' * fill + '░' * (width - fill)
    return f'{color}{bar}{_RESET}' if color else bar


class AsrMonitorNode(Node):

    def __init__(self):
        super().__init__('asr_monitor')
        self._audio_level: float = 0.0
        self._wakeword_score: float = 0.0
        self._lock = threading.Lock()

        self.create_subscription(String, 'stt_event', self._cb_event, 10)
        self.create_subscription(String, 'stt', self._cb_stt, 10)
        self.create_subscription(Float32, 'audio_level', self._cb_audio_level, 10)
        self.create_subscription(Float32, 'wakeword_score', self._cb_wakeword_score, 10)

        self.get_logger().info(
            'ASR Monitor 起動 — /stt_event /stt /audio_level /wakeword_score 購読中'
        )

    def _cb_event(self, msg: String):
        try:
            event = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        # メーター行を消してイベントを出力し、改行後メーターが次の tick で再描画される
        print(f'\r{" " * 80}\r{_format_event(event)}', flush=True)

    def _cb_stt(self, msg: String):
        pass  # stt_event の final_result で表示済み

    def _cb_audio_level(self, msg: Float32):
        with self._lock:
            self._audio_level = msg.data

    def _cb_wakeword_score(self, msg: Float32):
        with self._lock:
            self._wakeword_score = msg.data

    def print_meter(self):
        """音量・ウェイクワードスコアのバーを1行で上書き表示する."""
        with self._lock:
            level = self._audio_level
            score = self._wakeword_score

        ww_color = '\033[91m' if score >= 0.5 else ''
        line = (
            f'\r  音量[{_bar(level * 6)}]{level:.3f}'
            f'  WW[{_bar(score, color=ww_color)}]{score:.3f}'
        )
        print(line, end='', flush=True)


def _meter_loop(node: AsrMonitorNode):
    while rclpy.ok():
        node.print_meter()
        time.sleep(_METER_INTERVAL)


def main(args=None):
    rclpy.init(args=args)
    node = AsrMonitorNode()

    meter_thread = threading.Thread(target=_meter_loop, args=(node,), daemon=True)
    meter_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
