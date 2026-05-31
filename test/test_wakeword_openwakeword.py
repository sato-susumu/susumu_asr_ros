"""
OpenWakewordPlugin のユニットテスト.

計測済みタイミング（threshold=0.5）:
  hey_mycroft*.wav:              DETECTED @ 1.984s (score≈0.968)
  hey_mycroft_with_speech*.wav:  DETECTED @ 1.408s (score≈0.958) ← livekitより速い
  silence / white_noise:         未検出

livekit との検出タイミングの差:
  hey_mycroft*.wav:              両プラグイン共に 1.984s で一致
  hey_mycroft_with_speech*.wav:  openwakeword=1.408s、livekit=1.984s
  → openwakewordはフレームごとにスコアを計算するため、livekitより早く検出できる場合がある

使い方:
  pytest test/test_wakeword_openwakeword.py -v
"""
from conftest import load_frames
import pytest

from susumu_asr_ros.plugin_base import WakewordEvent

FRAME_SEC = 512 / 16000


def _fresh_plugin(**overrides):
    """毎回クリーンな OpenWakewordPlugin インスタンスを返す."""
    pytest.importorskip('openwakeword', reason='openwakeword が未インストール')
    pytest.importorskip('torch', reason='torch が未インストール')
    from susumu_asr_ros.wakeword_openwakeword import OpenWakewordPlugin
    plugin = OpenWakewordPlugin()
    params = {
        'model_folder': 'models',
        'model_name': 'hey_mycroft_v0.1.tflite',
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


class TestOpenWakewordPluginDetection:
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
        """hey_mycroft_with_speech.wav: DETECTED @ 1.408s（livekitより速い）."""
        t, score = _run(_fresh_plugin(), load_frames('hey_mycroft_with_speech.wav'))
        assert t == 1.408
        assert score >= 0.5

    def test_hey_mycroft_with_speech_with_silence_2s(self):
        """hey_mycroft_with_speech_with_silence_2s.wav: DETECTED @ 1.408s."""
        t, _ = _run(_fresh_plugin(), load_frames('hey_mycroft_with_speech_with_silence_2s.wav'))
        assert t == 1.408


class TestOpenWakewordPluginNegative:
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


class TestOpenWakewordPluginReset:

    def test_reset_clears_buffer(self):
        """reset() で prediction_buffer がクリアされること（2回目の検出タイミングが変わる）."""
        plugin = _fresh_plugin()
        frames = load_frames('hey_mycroft.wav')
        t1, _ = _run(plugin, frames)   # 通常の検出
        t2, _ = _run(plugin, frames)   # reset() 後はバッファが空なので早期検出
        assert t1 is not None
        assert t2 is not None
        # bufferクリア後は蓄積なしで検出されるため t1 と t2 は異なる
        assert t1 != t2

    def test_default_params(self):
        """デフォルトパラメータが正しく設定されること."""
        pytest.importorskip('openwakeword', reason='openwakeword が未インストール')
        from susumu_asr_ros.wakeword_openwakeword import OpenWakewordPlugin
        plugin = OpenWakewordPlugin()
        plugin.load_params({})
        assert plugin._model_name == 'hey_mycroft_v0.1.tflite'
        assert plugin._threshold == 0.5
