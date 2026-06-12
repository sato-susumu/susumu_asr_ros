"""
PassthroughWakewordPlugin のユニットテスト.

passthrough は VAD_START 後 delay_sec 秒後に即 DETECTED を返す。
デフォルト delay_sec=0.5s = 512samples/16000Hz * n frames → 約16フレーム後。

使い方:
  pytest test/test_wakeword_passthrough.py -v
"""
from conftest import load_frames
import pytest
from susumu_asr.plugin_base import WakewordEvent

FRAME_SEC = 512 / 16000


def _fresh_plugin(**overrides):
    """毎回クリーンな PassthroughWakewordPlugin インスタンスを返す."""
    from susumu_asr.wakeword_passthrough import PassthroughWakewordPlugin
    plugin = PassthroughWakewordPlugin()
    params = {'delay_sec': 0.5}
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


class TestPassthroughWakewordPlugin:

    def test_detected_after_delay(self):
        """delay_sec 後に DETECTED を返す（どのWAVでも同じ）."""
        plugin = _fresh_plugin(delay_sec=0.5)
        frames = load_frames('hey_mycroft.wav')
        t, score = _run(plugin, frames)
        assert t == pytest.approx(0.480, abs=FRAME_SEC)
        assert score == 1.0

    def test_detected_on_silence(self):
        """無音 WAV でも delay_sec 後に DETECTED を返す（VADに関係なく動作する）."""
        plugin = _fresh_plugin(delay_sec=0.5)
        frames = load_frames('silence_3s.wav')
        t, score = _run(plugin, frames)
        assert t is not None
        assert score == 1.0

    def test_custom_delay(self):
        """delay_sec=1.0s: 約1秒後に DETECTED を返す."""
        plugin = _fresh_plugin(delay_sec=1.0)
        frames = load_frames('hey_mycroft_with_silence_4s.wav')
        t, score = _run(plugin, frames)
        assert t == pytest.approx(1.024, abs=FRAME_SEC)

    def test_reset_restarts_delay(self):
        """reset() 後は再びゼロカウントから遅延が始まる."""
        plugin = _fresh_plugin(delay_sec=0.5)
        frames = load_frames('hey_mycroft.wav')
        _run(plugin, frames)   # 1回目
        t2, _ = _run(plugin, frames)   # reset() 込みで2回目
        assert t2 == pytest.approx(0.480, abs=FRAME_SEC)

    def test_default_params(self):
        """デフォルトパラメータが正しく設定されること."""
        from susumu_asr.wakeword_passthrough import PassthroughWakewordPlugin
        plugin = PassthroughWakewordPlugin()
        plugin.load_params({})
        assert plugin._delay_sec == 0.5


class TestPassthroughWakewordPluginScore:

    def test_score_is_always_one(self):
        """DETECTED 時のスコアは常に 1.0."""
        plugin = _fresh_plugin()
        frames = load_frames('white_noise_3s.wav')
        _, score = _run(plugin, frames)
        assert score == 1.0
