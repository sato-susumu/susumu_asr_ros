"""テスト共通ユーティリティ."""
import os
import wave

import numpy as np

AUDIO_DIR = os.path.join(os.path.dirname(__file__), 'audio')
SAMPLE_RATE = 16000
# Silero VAD は 512 サンプル未満を受け付けないため 512 サンプル (32ms) を使用
FRAME_SAMPLES = 512
FRAME_DURATION_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE  # 32ms
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


def silence_frames(duration_sec: float) -> list[bytes]:
    """指定秒数分の無音フレームリストを生成する."""
    n = int(duration_sec * 1000 / FRAME_DURATION_MS)
    return [b'\x00' * FRAME_BYTES] * n


def white_noise_frames(duration_sec: float, amplitude: float = 0.01, seed: int = 42) -> list[bytes]:
    """指定秒数分の低振幅ホワイトノイズフレームリストを生成する."""
    rng = np.random.default_rng(seed)
    n_frames = int(duration_sec * 1000 / FRAME_DURATION_MS)
    noise = (rng.standard_normal(FRAME_SAMPLES * n_frames) * amplitude * 32767).astype(np.int16)
    raw = noise.tobytes()
    return [raw[i:i + FRAME_BYTES] for i in range(0, len(raw) - FRAME_BYTES + 1, FRAME_BYTES)]


def feed_all(plugin, frames: list[bytes]) -> list:
    """全フレームをプラグインに送り込み、None でないイベントのみ収集して返す."""
    return [ev for f in frames for ev in [plugin.process_frame(f)[0]] if ev is not None]
