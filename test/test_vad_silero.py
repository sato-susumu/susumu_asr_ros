"""
SileroVADPlugin のユニットテスト.

計測済みタイミング（silence_threshold_ms=1000）:
  hey_mycroft_with_silence_2s.wav: start=1.312s, stop=3.360s
  hey_mycroft_with_speech_with_silence_2s.wav: start=0.736s, stop=3.488s
  silence_3s.wav / white_noise_3s.wav: イベントなし

使い方:
  pytest test/test_vad_silero.py -v
"""
from conftest import feed_all_with_timing, load_frames
import pytest

from susumu_asr_ros.plugin_base import VADEvent


def _fresh_plugin(**params):
    """毎回クリーンな SileroVADPlugin インスタンスを返す."""
    pytest.importorskip('torch', reason='torch が未インストール')
    from susumu_asr_ros.vad_silero import SileroVADPlugin
    plugin = SileroVADPlugin()
    plugin.load_params(params)
    plugin.setup()
    return plugin


class TestSileroVADPluginTiming:
    """各WAVファイルに対してイベントのタイミングをフレーム単位で検証する."""

    def test_hey_mycroft_with_silence_2s(self):
        """ウェイクワード＋2秒無音: start=1.312s, stop=4.416s（silence_threshold=2000ms）."""
        timed = feed_all_with_timing(
            _fresh_plugin(), load_frames('hey_mycroft_with_silence_2s.wav')
        )
        starts = [(ev, t) for ev, t in timed if ev == VADEvent.VAD_START]
        stops = [(ev, t) for ev, t in timed if ev == VADEvent.VAD_END]

        assert starts[0] == (VADEvent.VAD_START, 1.312)
        assert stops[0] == (VADEvent.VAD_END, 4.416)

    def test_hey_mycroft_with_speech_with_silence_2s(self):
        """ウェイクワード＋発話＋2秒無音: start=0.736s, stop=4.544s（silence_threshold=2000ms）."""
        timed = feed_all_with_timing(
            _fresh_plugin(), load_frames('hey_mycroft_with_speech_with_silence_2s.wav')
        )
        starts = [(ev, t) for ev, t in timed if ev == VADEvent.VAD_START]
        stops = [(ev, t) for ev, t in timed if ev == VADEvent.VAD_END]

        assert starts[0] == (VADEvent.VAD_START, 0.736)
        assert stops[0] == (VADEvent.VAD_END, 4.544)

    def test_silence_3s_no_events(self):
        """3秒無音: イベントなし."""
        timed = feed_all_with_timing(_fresh_plugin(), load_frames('silence_3s.wav'))
        assert timed == [], f'無音でイベントが発生しました: {timed}'

    def test_white_noise_3s_no_events(self):
        """低振幅ホワイトノイズ(振幅0.01): イベントなし."""
        timed = feed_all_with_timing(_fresh_plugin(), load_frames('white_noise_3s.wav'))
        assert timed == [], f'ホワイトノイズでイベントが発生しました: {timed}'


class TestSileroVADPluginParams:

    def test_default_params(self):
        """デフォルトパラメータが正しく設定されること."""
        pytest.importorskip('torch', reason='torch が未インストール')
        from susumu_asr_ros.vad_silero import SileroVADPlugin
        plugin = SileroVADPlugin()
        plugin.load_params({})
        plugin.setup()
        assert plugin._threshold == 0.5
        assert plugin._silence_ms == 2000
        assert plugin._pre_speech_ms == 300

    def test_custom_params(self):
        """カスタムパラメータが setup() 後に反映されること."""
        plugin = _fresh_plugin(threshold=0.9, silence_threshold_ms=500)
        assert plugin._threshold == 0.9
        assert plugin._silence_ms == 500
