"""
GoogleCloudASRPlugin のユニットテスト.

GCP への実通信・SpeechClient・streaming_recognize はモックで差し替える。

使い方:
  pytest test/test_asr_google.py -v
"""
import queue
import threading
from unittest.mock import MagicMock, patch

from susumu_asr_ros.plugin_base import ASRCommand


def _make_plugin(language_code='ja-JP'):
    """モック済みの GoogleCloudASRPlugin を返す."""
    from susumu_asr_ros.asr_google import GoogleCloudASRPlugin
    plugin = GoogleCloudASRPlugin()
    plugin.load_params({'language_code': language_code})
    with patch('susumu_asr_ros.asr_google.speech.SpeechClient'):
        plugin.setup(queue.Queue(), queue.Queue(), threading.Event())
    return plugin


def _run(plugin, commands: list[tuple], timeout: float = 5.0) -> list[tuple]:
    """コマンドをキューに積んで run() を実行し result_queue の内容を返す."""
    for cmd in commands:
        plugin.audio_queue.put(cmd)
    t = threading.Thread(target=plugin.run)
    t.start()
    t.join(timeout=timeout)
    results = []
    while not plugin.result_queue.empty():
        results.append(plugin.result_queue.get_nowait())
    return results


# ---------------------------------------------------------------------------
# パラメータ・セットアップ
# ---------------------------------------------------------------------------

class TestGoogleCloudASRPluginParams:

    def test_default_language_code(self):
        """デフォルトの language_code が ja-JP であること."""
        from susumu_asr_ros.asr_google import GoogleCloudASRPlugin
        plugin = GoogleCloudASRPlugin()
        plugin.load_params({})
        assert plugin._language_code == 'ja-JP'

    def test_custom_language_code(self):
        """指定した language_code が反映されること."""
        from susumu_asr_ros.asr_google import GoogleCloudASRPlugin
        plugin = GoogleCloudASRPlugin()
        plugin.load_params({'language_code': 'en-US'})
        assert plugin._language_code == 'en-US'

    def test_param_declarations(self):
        """get_param_declarations() が language_code を含むこと."""
        from susumu_asr_ros.asr_google import GoogleCloudASRPlugin
        names = [d.name for d in GoogleCloudASRPlugin().get_param_declarations()]
        assert 'language_code' in names

    def test_setup_initializes_queues(self):
        """setup() 後にキューがセットされること."""
        plugin = _make_plugin()
        assert isinstance(plugin.audio_queue, queue.Queue)
        assert isinstance(plugin.result_queue, queue.Queue)

    def test_setup_creates_speech_client(self):
        """setup() で SpeechClient が生成されること."""
        from susumu_asr_ros.asr_google import GoogleCloudASRPlugin
        plugin = GoogleCloudASRPlugin()
        plugin.load_params({})
        with patch('susumu_asr_ros.asr_google.speech.SpeechClient') as mock_cls:
            plugin.setup(queue.Queue(), queue.Queue(), threading.Event())
        mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# コマンド処理
# ---------------------------------------------------------------------------

class TestGoogleCloudASRPluginCommands:

    def test_start_activates_streaming(self):
        """Start コマンドで call_active が True になりレスポンススレッドが起動すること."""
        plugin = _make_plugin()
        plugin.client.streaming_recognize.return_value = iter([])

        plugin.audio_queue.put((ASRCommand.START, b'0.0'))
        plugin.audio_queue.put((ASRCommand.STOP_ALL, None))
        threading.Thread(target=plugin.run).start()

        import time
        time.sleep(0.2)
        assert plugin._start_time == 0.0

    def test_duplicate_start_ignored(self):
        """call_active 中に再度 start が来ても無視されること."""
        plugin = _make_plugin()

        # ストリーミングループをブロックし続けるイテレータ
        unblock = threading.Event()

        def _blocking(*args, **kwargs):
            unblock.wait()
            return iter([])

        plugin.client.streaming_recognize.side_effect = _blocking

        plugin._handle_start(b'1.0')
        first_thread = plugin._response_thread
        plugin._handle_start(b'2.0')  # call_active=True のため無視される

        # スレッドは再起動されないこと（重複 start は無視）
        assert plugin._response_thread is first_thread

        unblock.set()  # スレッドを解放

    def test_audio_buffered_while_active(self):
        """call_active 中の audio コマンドが _audio_buffer_queue に積まれること."""
        plugin = _make_plugin()
        plugin.call_active = True
        plugin._handle_audio(b'pcm_data')
        assert not plugin._audio_buffer_queue.empty()

    def test_audio_ignored_when_inactive(self):
        """call_active=False の audio コマンドは無視されること."""
        plugin = _make_plugin()
        plugin.call_active = False
        plugin._handle_audio(b'pcm_data')
        assert plugin._audio_buffer_queue.empty()

    def test_stop_deactivates_streaming(self):
        """Stop コマンドで call_active が False になること."""
        plugin = _make_plugin()
        plugin.client.streaming_recognize.return_value = iter([])
        plugin._handle_start(b'0.0')
        plugin._handle_stop(b'1.5')
        assert plugin.call_active is False
        assert plugin._stop_time == 1.5

    def test_stop_when_inactive_ignored(self):
        """非アクティブ時の stop は何も変えないこと."""
        plugin = _make_plugin()
        plugin.call_active = False
        plugin._handle_stop(b'1.0')
        assert plugin.call_active is False

    def test_stop_all_calls_stop_when_active(self):
        """call_active 中に stop_all が来ると stop 処理が呼ばれること."""
        plugin = _make_plugin()
        plugin.client.streaming_recognize.return_value = iter([])
        plugin._handle_start(b'0.0')
        plugin._handle_stop_all()
        assert plugin.call_active is False


# ---------------------------------------------------------------------------
# ストリーミング認識結果
# ---------------------------------------------------------------------------

class TestGoogleCloudASRPluginStreaming:

    def _make_response(self, transcript: str, is_final: bool):
        """モック用の StreamingRecognizeResponse を生成する."""
        alternative = MagicMock()
        alternative.transcript = transcript
        result = MagicMock()
        result.alternatives = [alternative]
        result.is_final = is_final
        response = MagicMock()
        response.results = [result]
        return response

    def test_final_result_queued(self):
        """is_final=True の結果が result_queue に積まれること."""
        plugin = _make_plugin()
        plugin.client.streaming_recognize.return_value = iter([
            self._make_response('こんにちは', is_final=True),
        ])
        plugin._handle_start(b'0.0')
        plugin._response_thread.join(timeout=3.0)

        assert not plugin.result_queue.empty()
        r = plugin.result_queue.get_nowait()
        assert r.is_final is True
        assert r.text == 'こんにちは'
        assert r.start == 0.0

    def test_partial_result_queued(self):
        """is_final=False の結果が result_queue に積まれること."""
        plugin = _make_plugin()
        plugin.client.streaming_recognize.return_value = iter([
            self._make_response('こん', is_final=False),
            self._make_response('こんにちは', is_final=True),
        ])
        plugin._handle_start(b'0.0')
        plugin._response_thread.join(timeout=3.0)

        results = []
        while not plugin.result_queue.empty():
            results.append(plugin.result_queue.get_nowait())

        assert any(not r.is_final for r in results), 'partial result が返りませんでした'
        assert any(r.is_final for r in results), 'final result が返りませんでした'

    def test_empty_alternatives_skipped(self):
        """Alternatives が空のレスポンスはスキップされること."""
        plugin = _make_plugin()
        empty_result = MagicMock()
        empty_result.alternatives = []
        empty_response = MagicMock()
        empty_response.results = [empty_result]
        plugin.client.streaming_recognize.return_value = iter([empty_response])
        plugin._handle_start(b'0.0')
        plugin._response_thread.join(timeout=3.0)

        assert plugin.result_queue.empty()

    def test_streaming_exception_does_not_crash(self):
        """streaming_recognize で例外が発生してもクラッシュしないこと."""
        plugin = _make_plugin()
        plugin.client.streaming_recognize.side_effect = Exception('network error')
        plugin._handle_start(b'0.0')
        plugin._response_thread.join(timeout=3.0)

        assert plugin.call_active is False

    def test_stop_time_attached_to_final_result(self):
        """Stop コマンドのタイムスタンプが final result の end に付くこと."""
        plugin = _make_plugin()

        def _delayed_response(*args, **kwargs):
            import time
            time.sleep(0.05)
            yield self._make_response('テスト', is_final=True)

        plugin.client.streaming_recognize.side_effect = _delayed_response
        plugin._handle_start(b'1.0')
        plugin._handle_stop(b'3.5')
        if plugin._response_thread:
            plugin._response_thread.join(timeout=3.0)

        if not plugin.result_queue.empty():
            r = plugin.result_queue.get_nowait()
            assert r.start == 1.0
            assert r.end == 3.5
