"""SileroVADPlugin のユニットテスト.

テスト音声:
  test/audio/hey_mycroft.wav          - ウェイクワードのみ (3.39s)
  test/audio/hey_mycroft_with_speech.wav - ウェイクワード＋発話 (4.23s)

使い方:
  pytest test/test_vad_silero.py -v
"""
import pytest

from conftest import feed_all, load_frames, silence_frames, white_noise_frames


@pytest.fixture(scope='module')
def silero_plugin():
    pytest.importorskip('torch', reason='torch が未インストール')
    from susumu_asr_ros.vad_silero import SileroVADPlugin
    plugin = SileroVADPlugin()
    plugin.load_params({})
    plugin.setup()
    return plugin


class TestSileroVADPlugin:

    def test_speech_start_detected_on_wakeword(self, silero_plugin):
        """ウェイクワード音声で speech_start が検出されること."""
        events = feed_all(silero_plugin, load_frames('hey_mycroft.wav'))
        assert 'speech_start' in events, (
            f'speech_start が検出されませんでした。検出イベント: {events}'
        )

    def test_speech_stop_detected_on_wakeword(self, silero_plugin):
        """ウェイクワード音声＋無音で speech_stop が検出されること."""
        events = feed_all(silero_plugin, load_frames('hey_mycroft.wav') + silence_frames(2.0))
        assert 'speech_stop' in events, (
            f'speech_stop が検出されませんでした。検出イベント: {events}'
        )

    def test_speech_start_and_stop_detected_with_speech(self, silero_plugin):
        """ウェイクワード＋発話音声で speech_start と speech_stop が検出されること."""
        events = feed_all(
            silero_plugin,
            load_frames('hey_mycroft_with_speech.wav') + silence_frames(2.0),
        )
        assert 'speech_start' in events
        assert 'speech_stop' in events

    def test_silence_not_detected(self, silero_plugin):
        """無音では speech_start が検出されないこと."""
        events = feed_all(silero_plugin, silence_frames(3.0))
        assert 'speech_start' not in events, (
            f'無音で speech_start が誤検出されました。検出イベント: {events}'
        )

    def test_white_noise_not_detected(self):
        """低振幅ホワイトノイズで speech_start が誤検出されないこと."""
        pytest.importorskip('torch', reason='torch が未インストール')
        from susumu_asr_ros.vad_silero import SileroVADPlugin
        plugin = SileroVADPlugin()
        plugin.load_params({})
        plugin.setup()
        events = feed_all(plugin, white_noise_frames(3.0))
        assert 'speech_start' not in events, (
            f'ホワイトノイズで speech_start が誤検出されました。検出イベント: {events}'
        )

    def test_event_ordering(self, silero_plugin):
        """speech_start の前に speech_cont や speech_stop が来ないこと."""
        events = feed_all(silero_plugin, load_frames('hey_mycroft.wav') + silence_frames(2.0))
        if 'speech_start' in events:
            before = events[:events.index('speech_start')]
            assert 'speech_cont' not in before
            assert 'speech_stop' not in before

    def test_params_override_threshold(self):
        """load_params の値が setup() 後に反映されること."""
        pytest.importorskip('torch', reason='torch が未インストール')
        from susumu_asr_ros.vad_silero import SileroVADPlugin
        plugin = SileroVADPlugin()
        plugin.load_params({'threshold': 0.9, 'silence_threshold_ms': 500})
        plugin.setup()
        assert plugin._threshold == 0.9
        assert plugin._silence_ms == 500
