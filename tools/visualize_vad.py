#!/usr/bin/env python3
"""WAVファイルの波形と VAD/ASR イベントを図示するスクリプト."""
import argparse
from dataclasses import dataclass, field
from datetime import datetime
import os
import queue
import sys
import threading
import wave

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans CJK JP'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

SAMPLE_RATE = 16000

EVENT_STYLES = {
    'vad_speech_start':     {'color': '#2ecc71', 'lw': 2, 'ls': '-',  'label': 'vad_speech_start'},
    'vad_speech_stop':      {'color': '#e74c3c', 'lw': 2, 'ls': '-',  'label': 'vad_speech_stop'},
    'ww_listening_started': {
        'color': '#3498db', 'lw': 1, 'ls': '--', 'label': 'ww_listening_started'},
    'ww_detected':          {'color': '#9b59b6', 'lw': 2, 'ls': '-',  'label': 'ww_detected'},
    'asr_timeout':          {'color': '#e67e22', 'lw': 2, 'ls': '--', 'label': 'asr_timeout'},
}

SPAN_COLOR = '#2ecc71'
SPAN_ALPHA = 0.12

_WAKEWORD_CLASSES = {
    'passthrough':    'susumu_asr_ros.wakeword_passthrough:PassthroughWakewordPlugin',
    'livekit_wakeword': 'susumu_asr_ros.wakeword_livekit:LivekitWakewordPlugin',
    'openwakeword':   'susumu_asr_ros.wakeword_openwakeword:OpenWakewordPlugin',
}


@dataclass
class CollectedEvents:
    """SpeechRecognitionSystem から収集したイベント一覧."""

    vad_events: list[tuple[str, float]] = field(default_factory=list)
    final_results: list[tuple[float, float, str]] = field(default_factory=list)
    partial_results: list[tuple[float, str]] = field(default_factory=list)


def load_wav(path: str):
    """WAVを読み込み audio_array を返す."""
    with wave.open(path, 'rb') as wf:
        assert wf.getnchannels() == 1, 'モノラルのみ対応'
        assert wf.getsampwidth() == 2, '16bit のみ対応'
        assert wf.getframerate() == SAMPLE_RATE, f'サンプルレートは {SAMPLE_RATE}Hz のみ対応'
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def build_vad_plugin(threshold: float):
    """Silero VAD プラグインを構築して返す."""
    from susumu_asr_ros.vad_silero import SileroVADPlugin
    plugin = SileroVADPlugin()
    plugin.load_params({'threshold': threshold})
    plugin.setup()
    return plugin


def build_wakeword_plugin(plugin_name: str, model_folder: str,
                          model_name: str | None, threshold: float):
    """Wakeword プラグインを構築して返す."""
    if plugin_name not in _WAKEWORD_CLASSES:
        raise ValueError(
            f"プラグイン '{plugin_name}' が見つかりません。"
            f'利用可能: {list(_WAKEWORD_CLASSES.keys())}'
        )
    import importlib
    module_path, class_name = _WAKEWORD_CLASSES[plugin_name].split(':')
    cls = getattr(importlib.import_module(module_path), class_name)
    plugin = cls()
    params = {'threshold': threshold}
    if model_folder:
        params['model_folder'] = model_folder
    if model_name:
        params['model_name'] = model_name
    plugin.load_params(params)
    plugin.setup()
    return plugin


def build_asr_plugin(whisper_model: str, language: str, device: str):
    """Whisper ASR プラグインを構築して返す."""
    from susumu_asr_ros.asr_whisper import WhisperASRPlugin
    plugin = WhisperASRPlugin()
    plugin.load_params({
        'model_name': whisper_model,
        'language_code': language,
        'device': device,
    })
    plugin.setup(queue.Queue(), queue.Queue(), threading.Event())
    return plugin


def run_pipeline(wav_path: str, vad_plugin, wakeword_plugin,
                 asr_plugin) -> CollectedEvents:
    """パイプラインを実行してイベントを収集する."""
    from susumu_asr_ros.audio_io import WavAudioRecorder
    from susumu_asr_ros.constants import AUDIO_FRAME_SAMPLES
    from susumu_asr_ros.plugin_base import (
        ASREventType, AsrFinalResultEvent, AsrPartialResultEvent,
        VadStartEvent, VadStopEvent,
        WakewordDetectedEvent, WakewordListeningStartedEvent,
    )
    from susumu_asr_ros.susumu_asr import SpeechRecognitionSystem

    collected = CollectedEvents()

    def on_event(event):
        et = event.event_type
        if et == ASREventType.VAD_START:
            assert isinstance(event, VadStartEvent)
            collected.vad_events.append(('vad_speech_start', event.start))
        elif et == ASREventType.VAD_STOP:
            assert isinstance(event, VadStopEvent)
            collected.vad_events.append(('vad_speech_stop', event.end))
        elif et == ASREventType.WAKEWORD_LISTENING_STARTED:
            assert isinstance(event, WakewordListeningStartedEvent)
            collected.vad_events.append(('ww_listening_started', event.start))
        elif et == ASREventType.WAKEWORD_DETECTED:
            assert isinstance(event, WakewordDetectedEvent)
            collected.vad_events.append(('ww_detected', event.start))
        elif et == ASREventType.ASR_FINAL_RESULT:
            assert isinstance(event, AsrFinalResultEvent)
            collected.final_results.append((event.start, event.end, event.text))
        elif et == ASREventType.ASR_PARTIAL_RESULT:
            assert isinstance(event, AsrPartialResultEvent)
            collected.partial_results.append((event.start, event.text))

    recorder = WavAudioRecorder(
        read_frame_size=AUDIO_FRAME_SAMPLES,
        input_file=wav_path,
        simulate_realtime=False,
    )
    system = SpeechRecognitionSystem(
        vad_plugin=vad_plugin,
        wakeword_plugin=wakeword_plugin,
        asr_plugin=asr_plugin,
        recorder=recorder,
        on_asr_event=on_event,
    )
    system.start()
    return collected


def plot(audio: np.ndarray, collected: CollectedEvents,
         wav_path: str, wakeword_name: str, whisper_model: str, out_path: str):
    """波形とイベントを描画して保存する."""
    duration = len(audio) / SAMPLE_RATE
    t_audio = np.linspace(0, duration, len(audio))

    has_asr = bool(
        collected.final_results or collected.partial_results
        or any(ev == 'asr_timeout' for ev, _ in collected.vad_events)
    )
    fig_h = 6 if has_asr else 4
    fig, axes = plt.subplots(
        2 if has_asr else 1, 1,
        figsize=(14, fig_h),
        gridspec_kw={'height_ratios': [3, 1]} if has_asr else {},
        sharex=True,
    )
    ax = axes[0] if has_asr else axes
    ax_asr = axes[1] if has_asr else None

    ax.plot(t_audio, audio, color='#3498db', lw=0.5, alpha=0.8)
    ax.set_xlim(0, duration)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel('Amplitude')
    title = (
        f'{os.path.basename(wav_path)}'
        f'  [WW: {wakeword_name} / ASR: whisper-{whisper_model}]'
    )
    ax.set_title(title)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(axis='x', which='major', lw=0.4, alpha=0.5)
    ax.grid(axis='x', which='minor', lw=0.2, alpha=0.3)

    # vad_speech_start / vad_speech_stop ペアをスパンで塗る
    starts = [t for ev, t in collected.vad_events if ev == 'vad_speech_start']
    stops = [t for ev, t in collected.vad_events if ev == 'vad_speech_stop']
    for s in starts:
        e = next((t for t in stops if t > s), duration)
        for a in ([ax, ax_asr] if ax_asr else [ax]):
            a.axvspan(s, e, color=SPAN_COLOR, alpha=SPAN_ALPHA)

    # イベント垂直線
    plotted = set()
    for ev, t in collected.vad_events:
        style = EVENT_STYLES.get(ev, {'color': 'gray', 'lw': 1.5, 'ls': ':', 'label': ev})
        label = style['label'] if ev not in plotted else None
        for a in ([ax, ax_asr] if ax_asr else [ax]):
            a.axvline(t, color=style['color'], lw=style['lw'],
                      ls=style['ls'], label=label if a is ax else None, alpha=0.9)
        short = ev.replace('vad_', '').replace('asr_', '').replace('ww_', '')
        ax.text(t + 0.02, 0.92, short,
                transform=ax.get_xaxis_transform(),
                fontsize=7, color=style['color'], rotation=90, va='top')
        plotted.add(ev)

    ax.legend(loc='upper right', fontsize=8)

    if ax_asr is not None:
        ax_asr.set_ylim(0, 1)
        ax_asr.set_ylabel('ASR')
        ax_asr.set_xlabel('Time (s)')
        ax_asr.set_yticks([])
        ax_asr.grid(axis='x', which='major', lw=0.4, alpha=0.5)
        ax_asr.grid(axis='x', which='minor', lw=0.2, alpha=0.3)

        for start, text in collected.partial_results:
            ax_asr.text(start, 0.75, text,
                        fontsize=7, color='#95a5a6', ha='left', va='bottom',
                        style='italic', clip_on=True)

        for start, end, text in collected.final_results:
            mid = (start + end) / 2
            ax_asr.annotate('', xy=(end, 0.5), xytext=(start, 0.5),
                            arrowprops={'arrowstyle': '<->', 'color': '#7f8c8d', 'lw': 1.2})
            ax_asr.text(mid, 0.55, text,
                        fontsize=9, color='#2c3e50', ha='center', va='bottom',
                        bbox={'boxstyle': 'round,pad=0.3', 'fc': '#ecf0f1',
                              'ec': '#bdc3c7', 'lw': 0.8})

        for ev, t in collected.vad_events:
            if ev == 'asr_timeout':
                ax_asr.text(t, 0.20, 'TIMEOUT',
                            fontsize=8, color='#e67e22', ha='left', va='bottom',
                            bbox={'boxstyle': 'round,pad=0.2', 'fc': '#fef9e7',
                                  'ec': '#e67e22', 'lw': 0.8})
    else:
        ax.set_xlabel('Time (s)')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'保存しました: {out_path}')


def main():
    """メインエントリポイント."""
    parser = argparse.ArgumentParser(description='WAV波形とVAD/ASRイベントを図示する')
    parser.add_argument('wav', help='入力WAVファイルのパス')
    parser.add_argument('--wakeword', default='livekit_wakeword',
                        help=f'Wakewordプラグイン名 (default: livekit_wakeword)'
                             f' [{"/".join(_WAKEWORD_CLASSES)}]')
    parser.add_argument('--vad-threshold', type=float, default=0.5,
                        help='VAD検出しきい値 (default: 0.5)')
    parser.add_argument('--model-folder', default='models',
                        help='Wakewordモデルフォルダ (default: models)')
    parser.add_argument('--model-name', default=None,
                        help='Wakewordモデルファイル名 (省略時はプラグインのデフォルト)')
    parser.add_argument('--ww-threshold', type=float, default=0.5,
                        help='Wakeword検出しきい値 (default: 0.5)')
    parser.add_argument('--whisper-model', default='large-v2',
                        help='Whisperモデル名 (default: large-v2)')
    parser.add_argument('--language', default='ja',
                        help='認識言語コード (default: ja)')
    parser.add_argument('--device', default='auto',
                        help='推論デバイス auto/cpu/cuda (default: auto)')
    default_out = os.path.join(os.path.dirname(__file__), 'output')
    parser.add_argument('--out-dir', default=default_out,
                        help='出力ディレクトリ (default: tools/output)')
    args = parser.parse_args()

    print(f'WAV 読み込み中: {args.wav}')
    audio = load_wav(args.wav)
    print(f'  {len(audio)/SAMPLE_RATE:.2f}s')

    print('VAD プラグイン初期化中: silero_vad')
    vad_plugin = build_vad_plugin(args.vad_threshold)

    print(f'Wakeword プラグイン初期化中: {args.wakeword}')
    wakeword_plugin = build_wakeword_plugin(
        args.wakeword, args.model_folder, args.model_name, args.ww_threshold
    )

    print(f'Whisper 初期化中: {args.whisper_model}')
    asr_plugin = build_asr_plugin(args.whisper_model, args.language, args.device)

    print('パイプライン実行中...')
    collected = run_pipeline(args.wav, vad_plugin, wakeword_plugin, asr_plugin)

    print(f'  イベント: {len(collected.vad_events)} 件')
    for ev, t in collected.vad_events:
        print(f'    {t:.3f}s  {ev}')
    print(f'  ASR 最終結果: {len(collected.final_results)} 件')
    for start, end, text in collected.final_results:
        print(f'    {start:.3f}s - {end:.3f}s  "{text}"')

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    wav_stem = os.path.splitext(os.path.basename(args.wav))[0]
    out_path = os.path.join(args.out_dir, f'{ts}_{wav_stem}.png')
    plot(audio, collected, args.wav, args.wakeword, args.whisper_model, out_path)


if __name__ == '__main__':
    main()
