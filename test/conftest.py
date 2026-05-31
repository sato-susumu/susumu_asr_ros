"""テスト共通ユーティリティ."""
import os
import wave

AUDIO_DIR = os.path.join(os.path.dirname(__file__), 'audio')
SAMPLE_RATE = 16000
# Silero VAD は 512 サンプル未満を受け付けないため 512 サンプル (32ms) を使用
FRAME_SAMPLES = 512
FRAME_BYTES = FRAME_SAMPLES * 2  # 16bit = 2 bytes/sample


def load_frames(filename: str) -> list[bytes]:
    """WAVファイルを FRAME_SAMPLES 単位のフレームリストに分割して返す."""
    path = os.path.join(AUDIO_DIR, filename)
    with wave.open(path, 'rb') as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == SAMPLE_RATE
        raw = wf.readframes(wf.getnframes())
    return [raw[i:i + FRAME_BYTES] for i in range(0, len(raw) - FRAME_BYTES + 1, FRAME_BYTES)]


def feed_all(plugin, frames: list[bytes]) -> list:
    """全フレームをプラグインに送り込み、None でないイベントのみ収集して返す."""
    return [ev for f in frames for ev in [plugin.process_frame(f)[0]] if ev is not None]


def feed_all_with_timing(plugin, frames: list[bytes]) -> list[tuple]:
    """全フレームをプラグインに送り込み、(event, 秒) のリストを返す."""
    frame_sec = FRAME_SAMPLES / SAMPLE_RATE
    result = []
    for i, frame in enumerate(frames):
        ev, _ = plugin.process_frame(frame)
        if ev is not None:
            result.append((ev, round(i * frame_sec, 3)))
    return result
