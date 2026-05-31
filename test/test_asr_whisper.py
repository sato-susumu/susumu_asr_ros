"""WhisperASRPlugin のユニットテスト.

テスト音声:
  test/audio/hey_mycroft_with_speech.wav - ウェイクワード＋発話 (4.23s, ja)

使い方:
  pytest test/test_asr_whisper.py -v
"""
import queue
import threading
import wave
import os

import pytest

from susumu_asr_ros.plugin_base import ASRCommand

AUDIO_DIR = os.path.join(os.path.dirname(__file__), 'audio')
AUDIO_FILE = 'hey_mycroft_with_speech.wav'


def _load_raw(filename: str) -> bytes:
    """WAVファイルの PCM バイト列を返す."""
    with wave.open(os.path.join(AUDIO_DIR, filename), 'rb') as wf:
        return wf.readframes(wf.getnframes())


def _make_plugin(model_name='small', language_code='ja', device='cpu'):
    """テスト用に setup() 済みの WhisperASRPlugin を返す."""
    from susumu_asr_ros.asr_whisper import WhisperASRPlugin
    plugin = WhisperASRPlugin()
    plugin.load_params({
        'model_name': model_name,
        'language_code': language_code,
        'device': device,
    })
    plugin.setup(queue.Queue(), queue.Queue(), threading.Event())
    return plugin


def _run_session(plugin, raw: bytes) -> list[tuple]:
    """start → audio → stop → stop_all を送り込み、result_queue の結果を返す."""
    plugin.audio_queue.put((ASRCommand.START, b'0.0'))
    plugin.audio_queue.put((ASRCommand.AUDIO, raw))
    plugin.audio_queue.put((ASRCommand.STOP, b'4.23'))
    plugin.audio_queue.put((ASRCommand.STOP_ALL, None))

    thread = threading.Thread(target=plugin.run)
    thread.start()
    thread.join(timeout=60)

    results = []
    while not plugin.result_queue.empty():
        results.append(plugin.result_queue.get_nowait())
    return results


@pytest.fixture(scope='module')
def whisper_plugin():
    pytest.importorskip('faster_whisper', reason='faster-whisper が未インストール')
    return _make_plugin()


class TestWhisperASRPluginParams:

    def test_default_params(self):
        """デフォルトパラメータが正しく設定されること."""
        from susumu_asr_ros.asr_whisper import WhisperASRPlugin
        plugin = WhisperASRPlugin()
        plugin.load_params({})
        assert plugin._model_name == 'large-v2'
        assert plugin._language_code == 'auto'
        assert plugin._device == 'auto'

    def test_custom_params(self):
        """指定したパラメータが反映されること."""
        from susumu_asr_ros.asr_whisper import WhisperASRPlugin
        plugin = WhisperASRPlugin()
        plugin.load_params({'model_name': 'small', 'language_code': 'ja', 'device': 'cpu'})
        assert plugin._model_name == 'small'
        assert plugin._language_code == 'ja'
        assert plugin._device == 'cpu'

    def test_param_declarations(self):
        """get_param_declarations() が3件返すこと."""
        from susumu_asr_ros.asr_whisper import WhisperASRPlugin
        decls = WhisperASRPlugin().get_param_declarations()
        names = [d.name for d in decls]
        assert 'model_name' in names
        assert 'language_code' in names
        assert 'device' in names

    def test_setup_sets_queues(self):
        """setup() 後にキューがセットされること."""
        pytest.importorskip('faster_whisper', reason='faster-whisper が未インストール')
        plugin = _make_plugin()
        assert isinstance(plugin.audio_queue, queue.Queue)
        assert isinstance(plugin.result_queue, queue.Queue)


class TestWhisperASRPluginInference:

    def test_transcription_is_nonempty(self, whisper_plugin):
        """音声ファイルを認識して空でない文字列が返ること."""
        raw = _load_raw(AUDIO_FILE)
        results = _run_session(whisper_plugin, raw)
        assert len(results) > 0, '認識結果が1件も返りませんでした'
        is_final, text, start, end = results[-1]
        assert is_final is True
        assert len(text) > 0, '認識テキストが空です'

    def test_transcription_contains_expected_text(self, whisper_plugin):
        """認識結果にウェイクワード後の発話内容が含まれること."""
        raw = _load_raw(AUDIO_FILE)
        results = _run_session(whisper_plugin, raw)
        assert len(results) > 0
        text = results[-1][1]
        assert 'マイクロフト' in text or 'mycroft' in text.lower(), (
            f'期待するテキストが含まれていません: {repr(text)}'
        )

    def test_result_is_final(self, whisper_plugin):
        """stop コマンド後の結果が is_final=True であること."""
        raw = _load_raw(AUDIO_FILE)
        results = _run_session(whisper_plugin, raw)
        assert len(results) > 0
        is_final, _, _, _ = results[-1]
        assert is_final is True

    def test_timestamps_are_set(self, whisper_plugin):
        """start / end タイムスタンプが結果に含まれること."""
        raw = _load_raw(AUDIO_FILE)
        results = _run_session(whisper_plugin, raw)
        assert len(results) > 0
        _, _, start, end = results[-1]
        assert start == 0.0
        assert end == pytest.approx(4.23)

    def test_empty_audio_returns_no_result(self, whisper_plugin):
        """空の音声データでは結果が返らないこと."""
        whisper_plugin.audio_queue.put((ASRCommand.START, b'0.0'))
        whisper_plugin.audio_queue.put((ASRCommand.STOP, b'1.0'))
        whisper_plugin.audio_queue.put((ASRCommand.STOP_ALL, None))

        thread = threading.Thread(target=whisper_plugin.run)
        thread.start()
        thread.join(timeout=10)

        assert whisper_plugin.result_queue.empty(), '空音声で結果が返りました'

    def test_stop_all_mid_session_flushes_buffer(self):
        """stop_all を stop なしで送っても buffer が flush されること."""
        pytest.importorskip('faster_whisper', reason='faster-whisper が未インストール')
        plugin = _make_plugin()
        raw = _load_raw(AUDIO_FILE)

        plugin.audio_queue.put((ASRCommand.START, b'0.0'))
        plugin.audio_queue.put((ASRCommand.AUDIO, raw))
        plugin.audio_queue.put((ASRCommand.STOP_ALL, None))  # stop なしで終了

        thread = threading.Thread(target=plugin.run)
        thread.start()
        thread.join(timeout=60)

        results = []
        while not plugin.result_queue.empty():
            results.append(plugin.result_queue.get_nowait())

        assert len(results) > 0, 'stop_all で buffer が flush されませんでした'
        assert results[-1][0] is True
