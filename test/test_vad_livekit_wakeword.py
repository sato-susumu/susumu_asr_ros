"""
LivekitWakeWordPlugin のユニットテスト.

テスト音声と期待するイベント列（独立インスタンスで計測済み）:

  hey_mycroft_with_silence_4s.wav (7.39s):
    1.984s speech_start, 2.016〜4.384s speech_cont x76, 4.416s speech_stop

  hey_mycroft_with_speech.wav (4.23s):
    1.984s speech_start, 2.016〜4.192s speech_cont x70（ファイル終端のため stop なし）

  hey_mycroft_with_speech_with_silence_2s.wav (6.23s):
    1.984s speech_start, 4.544s speech_stop

  silence_3s.wav: イベントなし

  white_noise_3s.wav: イベントなし

使い方:
  pytest test/test_vad_livekit_wakeword.py -v
"""
from conftest import feed_all_with_timing, load_frames
import pytest

from susumu_asr_ros.plugin_base import VADEvent


def _fresh_plugin(**overrides):
    """毎回クリーンな LivekitWakeWordPlugin インスタンスを返す."""
    pytest.importorskip('livekit.wakeword', reason='livekit-wakeword が未インストール')
    pytest.importorskip('torch', reason='torch が未インストール')
    from susumu_asr_ros.vad_livekit_wakeword import LivekitWakeWordPlugin
    plugin = LivekitWakeWordPlugin()
    params = {
        'model_folder': 'models',
        'model_name': 'hey_mycroft_v0.1.onnx',
        'threshold': 0.5,
        'speech_timeout_sec': 8.0,
        'speech_end_min_sec': 2.0,
        'silence_threshold_ms': 2000,
        'vad_threshold': 0.5,
    }
    params.update(overrides)
    plugin.load_params(params)
    plugin.setup()
    return plugin


class TestLivekitWakeWordPluginTiming:
    """各WAVファイルに対してイベントのタイミングをフレーム単位で検証する."""

    def test_hey_mycroft_with_silence_4s(self):
        """ウェイクワード＋4秒無音: speech_start=1.984s, speech_stop=4.416s."""
        timed = feed_all_with_timing(
            _fresh_plugin(), load_frames('hey_mycroft_with_silence_4s.wav')
        )
        starts = [(ev, t) for ev, t in timed if ev == VADEvent.SPEECH_START]
        stops = [(ev, t) for ev, t in timed if ev == VADEvent.SPEECH_STOP]

        assert starts[0] == (VADEvent.SPEECH_START, 1.984)
        assert stops[0] == (VADEvent.SPEECH_STOP, 4.416)

    def test_hey_mycroft_with_speech_ends_in_cont(self):
        """ウェイクワード＋発話: speech_start=1.984s、ファイル終端まで speech_cont が続く."""
        timed = feed_all_with_timing(
            _fresh_plugin(), load_frames('hey_mycroft_with_speech.wav')
        )
        events = [(ev, t) for ev, t in timed]

        assert events[0] == (VADEvent.SPEECH_START, 1.984)
        assert all(ev == VADEvent.SPEECH_CONT for ev, _ in events[1:])
        assert events[-1][1] == pytest.approx(4.192, abs=0.032)

    def test_hey_mycroft_with_speech_with_silence_2s(self):
        """ウェイクワード＋発話＋2秒無音: speech_start=1.984s, speech_stop=4.544s."""
        timed = feed_all_with_timing(
            _fresh_plugin(), load_frames('hey_mycroft_with_speech_with_silence_2s.wav')
        )
        starts = [(ev, t) for ev, t in timed if ev == VADEvent.SPEECH_START]
        stops = [(ev, t) for ev, t in timed if ev == VADEvent.SPEECH_STOP]

        assert starts[0] == (VADEvent.SPEECH_START, 1.984)
        assert stops[0] == (VADEvent.SPEECH_STOP, 4.544)

    def test_silence_3s_no_events(self):
        """3秒無音: イベントなし."""
        timed = feed_all_with_timing(_fresh_plugin(), load_frames('silence_3s.wav'))
        assert timed == [], f'無音でイベントが発生しました: {timed}'

    def test_white_noise_no_events(self):
        """3秒ホワイトノイズ: イベントなし."""
        timed = feed_all_with_timing(_fresh_plugin(), load_frames('white_noise_3s.wav'))
        assert timed == [], f'ホワイトノイズでイベントが発生しました: {timed}'

    def test_timeout(self):
        """speech_timeout_sec=1.5s: speech_start=1.984s, speech_stop=3.584s (diff=1.600s)."""
        timed = feed_all_with_timing(
            _fresh_plugin(speech_timeout_sec=1.5, speech_end_min_sec=0.0),
            load_frames('hey_mycroft_with_silence_4s.wav'),
        )
        starts = [(ev, t) for ev, t in timed if ev == VADEvent.SPEECH_START]
        stops = [(ev, t) for ev, t in timed if ev == VADEvent.SPEECH_STOP]

        assert starts[0] == (VADEvent.SPEECH_START, 1.984)
        assert stops[0] == (VADEvent.SPEECH_STOP, 3.584)


class TestLivekitWakeWordPluginParams:

    def test_default_params(self):
        """デフォルトパラメータが正しく設定されること."""
        pytest.importorskip('livekit.wakeword', reason='livekit-wakeword が未インストール')
        from susumu_asr_ros.vad_livekit_wakeword import LivekitWakeWordPlugin
        plugin = LivekitWakeWordPlugin()
        plugin.load_params({})
        assert plugin._threshold == 0.5
        assert plugin._speech_timeout_sec == 8.0
        assert plugin._speech_end_min_sec == 2.0
        assert plugin._model_name == 'hey_mycroft_v0.1.onnx'

    def test_custom_params(self):
        """カスタムパラメータが正しく保持されること."""
        pytest.importorskip('livekit.wakeword', reason='livekit-wakeword が未インストール')
        from susumu_asr_ros.vad_livekit_wakeword import LivekitWakeWordPlugin
        plugin = LivekitWakeWordPlugin()
        plugin.load_params({
            'threshold': 0.7,
            'speech_timeout_sec': 5.0,
            'model_name': 'alexa_v0.1.onnx',
        })
        assert plugin._threshold == 0.7
        assert plugin._speech_timeout_sec == 5.0
        assert plugin._model_name == 'alexa_v0.1.onnx'
