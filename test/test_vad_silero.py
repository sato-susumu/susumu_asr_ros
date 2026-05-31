"""SileroVADPlugin のユニットテスト.

テスト音声と期待するイベント列（独立インスタンスで計測済み）:

  hey_mycroft_with_silence_2s.wav (5.39s):
    1.312s speech_start, 1.344〜3.328s speech_cont x63, 3.360s speech_stop

  hey_mycroft_with_speech_with_silence_2s.wav (6.23s):
    0.736s speech_start, 0.768〜3.648s speech_cont x92, 3.680s speech_stop

  silence_3s.wav:    イベントなし
  white_noise_3s.wav: イベントなし（振幅0.01）

使い方:
  pytest test/test_vad_silero.py -v
"""
import pytest

from conftest import feed_all_with_timing, load_frames
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
        """ウェイクワード＋2秒無音: speech_start=1.312s, speech_stop=3.360s."""
        timed = feed_all_with_timing(
            _fresh_plugin(), load_frames('hey_mycroft_with_silence_2s.wav')
        )
        events = [(ev, t) for ev, t in timed]

        assert events[0] == (VADEvent.SPEECH_START, 1.312)
        assert all(ev == VADEvent.SPEECH_CONT for ev, _ in events[1:-1])
        assert events[-1] == (VADEvent.SPEECH_STOP, 3.360)

    def test_hey_mycroft_with_speech_with_silence_2s(self):
        """ウェイクワード＋発話＋2秒無音: speech_start=0.736s, speech_stop=3.488s."""
        timed = feed_all_with_timing(
            _fresh_plugin(), load_frames('hey_mycroft_with_speech_with_silence_2s.wav')
        )
        events = [(ev, t) for ev, t in timed]

        assert events[0] == (VADEvent.SPEECH_START, 0.736)
        assert all(ev == VADEvent.SPEECH_CONT for ev, _ in events[1:-1])
        assert events[-1] == (VADEvent.SPEECH_STOP, 3.488)

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
        assert plugin._silence_ms == 1000
        assert plugin._pre_speech_ms == 300

    def test_custom_params(self):
        """カスタムパラメータが setup() 後に反映されること."""
        plugin = _fresh_plugin(threshold=0.9, silence_threshold_ms=500)
        assert plugin._threshold == 0.9
        assert plugin._silence_ms == 500
