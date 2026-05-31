"""OpenWakeWordPlugin のユニットテスト.

テスト音声:
  test/audio/hey_mycroft.wav          - ウェイクワードのみ (3.39s)
  test/audio/hey_mycroft_with_speech.wav - ウェイクワード＋発話 (4.23s)

使い方:
  pytest test/test_vad_openwakeword.py -v
"""
import pytest

from conftest import feed_all, load_frames, silence_frames


def _make_plugin(**overrides):
    """テスト用に setup() 済みの OpenWakeWordPlugin を返す."""
    from susumu_asr_ros.vad_openwakeword import OpenWakeWordPlugin
    plugin = OpenWakeWordPlugin()
    params = {
        'model_folder': 'models',
        'model_name': 'hey_mycroft_v0.1.tflite',
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


@pytest.fixture(scope='module')
def oww_plugin():
    pytest.importorskip('openwakeword', reason='openwakeword が未インストール')
    pytest.importorskip('torch', reason='torch が未インストール')
    return _make_plugin()


class TestOpenWakeWordPlugin:

    def test_speech_start_detected_on_wakeword(self, oww_plugin):
        """ウェイクワード音声で speech_start が検出されること."""
        events = feed_all(oww_plugin, load_frames('hey_mycroft.wav'))
        assert 'speech_start' in events, (
            f'speech_start が検出されませんでした。検出イベント: {events}'
        )

    def test_speech_stop_after_wakeword(self, oww_plugin):
        """ウェイクワード検出後、無音継続で speech_stop が返ること."""
        events = feed_all(oww_plugin, load_frames('hey_mycroft.wav') + silence_frames(4.0))
        assert 'speech_start' in events
        assert 'speech_stop' in events

    def test_speech_start_detected_with_speech(self, oww_plugin):
        """ウェイクワード＋発話音声で speech_start が検出されること."""
        events = feed_all(oww_plugin, load_frames('hey_mycroft_with_speech.wav'))
        assert 'speech_start' in events, (
            f'speech_start が検出されませんでした。検出イベント: {events}'
        )

    def test_speech_cont_after_speech_start(self, oww_plugin):
        """speech_start 後は speech_cont が返ること."""
        events = feed_all(oww_plugin, load_frames('hey_mycroft_with_speech.wav'))
        if 'speech_start' in events:
            after = events[events.index('speech_start') + 1:]
            assert 'speech_cont' in after, (
                f'speech_start 後に speech_cont が返りませんでした。後続イベント: {after}'
            )

    def test_silence_not_detected(self, oww_plugin):
        """無音で speech_start が検出されないこと."""
        events = feed_all(oww_plugin, silence_frames(3.0))
        assert 'speech_start' not in events, (
            f'無音で speech_start が誤検出されました。検出イベント: {events}'
        )

    def test_timeout_stops_speech(self):
        """speech_timeout_sec を短く設定すると speech_stop が返ること."""
        pytest.importorskip('openwakeword', reason='openwakeword が未インストール')
        pytest.importorskip('torch', reason='torch が未インストール')
        plugin = _make_plugin(speech_timeout_sec=1.5, speech_end_min_sec=0.0)
        events = feed_all(plugin, load_frames('hey_mycroft.wav') + silence_frames(2.0))
        assert 'speech_start' in events
        assert 'speech_stop' in events

    def test_params_loaded_correctly(self):
        """load_params の値が正しく保持されること."""
        pytest.importorskip('openwakeword', reason='openwakeword が未インストール')
        from susumu_asr_ros.vad_openwakeword import OpenWakeWordPlugin
        plugin = OpenWakeWordPlugin()
        plugin.load_params({
            'threshold': 0.7,
            'speech_timeout_sec': 5.0,
            'model_name': 'alexa_v0.1.tflite',
        })
        assert plugin._threshold == 0.7
        assert plugin._speech_timeout_sec == 5.0
        assert plugin._model_name == 'alexa_v0.1.tflite'
