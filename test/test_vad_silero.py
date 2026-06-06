"""
SileroVADPlugin のユニットテスト.

計測済みタイミング（silence_threshold_ms=2000、デフォルト）:
  hey_mycroft_with_silence_2s.wav:
    VAD_START frame_time=1.312s, speech_start_sec=1.314s
    VAD_END   frame_time=4.192s, speech_end_sec=2.238s
  hey_mycroft_with_speech_with_silence_2s.wav:
    VAD_START frame_time=0.736s, speech_start_sec=0.738s
    VAD_END   frame_time=4.320s, speech_end_sec=2.366s

speech_start_sec / speech_end_sec は VADIterator がサンプル数ベースで計算した
精確な発話区間タイムスタンプ（speech_pad_ms を考慮済み）。
speech_end_sec は無音が始まった位置 + speech_pad_ms を示し、
frame_time（ENDイベント到達時刻）より min_silence_duration_ms 分早い値になる。

使い方:
  pytest test/test_vad_silero.py -v
"""
from conftest import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    feed_all_with_timing,
    load_frames,
)
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


def _feed_with_vad_timestamps(plugin, frames):
    """全フレームを送り込み、VAD_START/VAD_END の (event, frame_time, vad_ts) を返す."""
    frame_sec = FRAME_SAMPLES / SAMPLE_RATE
    result = []
    for i, frame in enumerate(frames):
        r = plugin.process_frame(frame)
        if r.event == VADEvent.VAD_START:
            result.append(
                (r.event, round(i * frame_sec, 3), r.speech_start_sec)
            )
        elif r.event == VADEvent.VAD_END:
            result.append(
                (r.event, round(i * frame_sec, 3), r.speech_end_sec)
            )
    return result


class TestSileroVADPluginTiming:
    """各WAVファイルに対してイベントのタイミングを検証する."""

    def test_hey_mycroft_with_silence_2s(self):
        """ウェイクワード＋2秒無音: START=1.312s, END=4.192s（フレーム時刻）."""
        timed = feed_all_with_timing(
            _fresh_plugin(), load_frames('hey_mycroft_with_silence_2s.wav')
        )
        starts = [(ev, t) for ev, t in timed if ev == VADEvent.VAD_START]
        stops = [(ev, t) for ev, t in timed if ev == VADEvent.VAD_END]

        assert starts[0] == (VADEvent.VAD_START, 1.312)
        assert stops[0] == (VADEvent.VAD_END, 4.192)

    def test_hey_mycroft_with_silence_2s_vad_timestamps(self):
        """サンプルベースの精確タイムスタンプ（VADIterator 由来）を検証する."""
        events = _feed_with_vad_timestamps(
            _fresh_plugin(), load_frames('hey_mycroft_with_silence_2s.wav')
        )
        assert len(events) == 2
        ev_start, frame_t_start, vad_ts_start = events[0]
        ev_end, frame_t_end, vad_ts_end = events[1]

        assert ev_start == VADEvent.VAD_START
        assert ev_end == VADEvent.VAD_END
        # speech_start_sec はフレーム時刻に近い（speech_pad_ms 分以内のズレ）
        assert abs(vad_ts_start - frame_t_start) < 0.1
        # speech_end_sec は ENDフレーム時刻より min_silence_duration_ms 分以上早い
        assert vad_ts_end < frame_t_end - 1.0

    def test_hey_mycroft_with_speech_with_silence_2s(self):
        """ウェイクワード＋発話＋2秒無音: START=0.736s, END=4.320s（フレーム時刻）."""
        timed = feed_all_with_timing(
            _fresh_plugin(),
            load_frames('hey_mycroft_with_speech_with_silence_2s.wav'),
        )
        starts = [(ev, t) for ev, t in timed if ev == VADEvent.VAD_START]
        stops = [(ev, t) for ev, t in timed if ev == VADEvent.VAD_END]

        assert starts[0] == (VADEvent.VAD_START, 0.736)
        assert stops[0] == (VADEvent.VAD_END, 4.32)

    def test_hey_mycroft_with_speech_with_silence_2s_vad_timestamps(self):
        """サンプルベースの精確タイムスタンプ（VADIterator 由来）を検証する."""
        events = _feed_with_vad_timestamps(
            _fresh_plugin(),
            load_frames('hey_mycroft_with_speech_with_silence_2s.wav'),
        )
        assert len(events) == 2
        ev_start, frame_t_start, vad_ts_start = events[0]
        ev_end, frame_t_end, vad_ts_end = events[1]

        assert ev_start == VADEvent.VAD_START
        assert ev_end == VADEvent.VAD_END
        assert abs(vad_ts_start - frame_t_start) < 0.1
        assert vad_ts_end < frame_t_end - 1.0

    def test_silence_3s_no_events(self):
        """3秒無音: イベントなし."""
        timed = feed_all_with_timing(
            _fresh_plugin(), load_frames('silence_3s.wav')
        )
        assert timed == [], f'無音でイベントが発生しました: {timed}'

    def test_white_noise_3s_no_events(self):
        """低振幅ホワイトノイズ(振幅0.01): イベントなし."""
        timed = feed_all_with_timing(
            _fresh_plugin(), load_frames('white_noise_3s.wav')
        )
        assert timed == [], f'ホワイトノイズでイベントが発生しました: {timed}'


class TestSileroVADPluginParams:
    """パラメータ設定のテスト."""

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
        assert plugin._speech_pad_ms == 30

    def test_custom_params(self):
        """カスタムパラメータが setup() 後に反映されること."""
        plugin = _fresh_plugin(threshold=0.9, silence_threshold_ms=500)
        assert plugin._threshold == 0.9
        assert plugin._silence_ms == 500
