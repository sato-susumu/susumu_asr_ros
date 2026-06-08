"""
LivekitWakewordPlugin のユニットテスト.

計測済みタイミング（threshold=0.5）:
  hey_mycroft*.wav:            DETECTED @ 1.984s (score≈0.691)
  hey_mycroft_with_speech*.wav: DETECTED @ 1.984s (score=1.000)
  silence / white_noise:       未検出

livekit と openwakeword の検出タイミングの差:
  hey_mycroft*.wav:              両プラグイン共に 1.984s で一致
  hey_mycroft_with_speech*.wav:  livekit=1.984s、openwakeword=1.408s（livekitが遅い）
  → livekitは2秒リングバッファ使用のため、openwakewordより遅延が大きくなる場合がある

使い方:
  pytest test/test_wakeword_livekit.py -v
"""
from conftest import load_frames
import pytest

from susumu_asr.plugin_base import WakewordEvent

FRAME_SEC = 512 / 16000


def _fresh_plugin(**overrides):
    """毎回クリーンな LivekitWakewordPlugin インスタンスを返す."""
    pytest.importorskip('livekit.wakeword', reason='livekit-wakeword が未インストール')
    from susumu_asr.wakeword_livekit import LivekitWakewordPlugin
    plugin = LivekitWakewordPlugin()
    params = {
        'model_folder': 'models',
        'model_name': 'hey_mycroft_v0.1.onnx',
        'threshold': 0.5,
    }
    params.update(overrides)
    plugin.load_params(params)
    plugin.setup()
    return plugin


def _run(plugin, frames):
    """全フレームを渡し、最初に DETECTED になったフレーム番号と秒数を返す。なければ None."""
    plugin.reset()
    for i, f in enumerate(frames):
        r = plugin.process_frame(f)
        if r.event == WakewordEvent.DETECTED:
            return round(i * FRAME_SEC, 3), round(r.score, 3)
    return None, None


class TestLivekitWakewordPluginDetection:
    """ウェイクワード検出のタイミングを検証する."""

    def test_hey_mycroft(self):
        """hey_mycroft.wav: DETECTED @ 1.984s."""
        t, score = _run(_fresh_plugin(), load_frames('hey_mycroft.wav'))
        assert t == 1.984
        assert score >= 0.5

    def test_hey_mycroft_with_silence_2s(self):
        """hey_mycroft_with_silence_2s.wav: DETECTED @ 1.984s."""
        t, _ = _run(_fresh_plugin(), load_frames('hey_mycroft_with_silence_2s.wav'))
        assert t == 1.984

    def test_hey_mycroft_with_silence_4s(self):
        """hey_mycroft_with_silence_4s.wav: DETECTED @ 1.984s."""
        t, _ = _run(_fresh_plugin(), load_frames('hey_mycroft_with_silence_4s.wav'))
        assert t == 1.984

    def test_hey_mycroft_with_speech(self):
        """hey_mycroft_with_speech.wav: DETECTED @ 1.984s（openwakewordより遅い）."""
        t, score = _run(_fresh_plugin(), load_frames('hey_mycroft_with_speech.wav'))
        assert t == 1.984
        assert score >= 0.5

    def test_hey_mycroft_with_speech_with_silence_2s(self):
        """hey_mycroft_with_speech_with_silence_2s.wav: DETECTED @ 1.984s."""
        t, _ = _run(_fresh_plugin(), load_frames('hey_mycroft_with_speech_with_silence_2s.wav'))
        assert t == 1.984


class TestLivekitWakewordPluginNegative:
    """無音・ノイズでは未検出であることを検証する."""

    def test_silence_2s_no_detection(self):
        """2秒無音: 未検出."""
        t, _ = _run(_fresh_plugin(), load_frames('silence_2s.wav'))
        assert t is None

    def test_silence_3s_no_detection(self):
        """3秒無音: 未検出."""
        t, _ = _run(_fresh_plugin(), load_frames('silence_3s.wav'))
        assert t is None

    def test_silence_4s_no_detection(self):
        """4秒無音: 未検出."""
        t, _ = _run(_fresh_plugin(), load_frames('silence_4s.wav'))
        assert t is None

    def test_white_noise_no_detection(self):
        """ホワイトノイズ: 未検出."""
        t, _ = _run(_fresh_plugin(), load_frames('white_noise_3s.wav'))
        assert t is None


class TestLivekitWakewordPluginReset:

    def test_reset_keeps_buffer(self):
        """reset() はリングバッファを保持し推論カウンタのみリセットする."""
        plugin = _fresh_plugin()
        frames = load_frames('hey_mycroft.wav')
        t1, _ = _run(plugin, frames)
        # バッファを保持したまま reset() → 直後のフレームで再検出できる
        t2, _ = _run(plugin, frames)
        assert t1 is not None
        assert t2 is not None

    def test_default_params(self):
        """デフォルトパラメータが正しく設定されること."""
        pytest.importorskip('livekit.wakeword', reason='livekit-wakeword が未インストール')
        from susumu_asr.wakeword_livekit import LivekitWakewordPlugin
        plugin = LivekitWakewordPlugin()
        plugin.load_params({})
        assert plugin._model_name == 'hey_mycroft_v0.1.onnx'
        assert plugin._threshold == 0.5
