#!/usr/bin/env python3
"""test/audio 内の全WAVを全Wakewordプラグインで処理してPNGを一括出力するスクリプト."""
import argparse
from datetime import datetime
import glob
import os
import queue
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

AUDIO_DIR = os.path.join(os.path.dirname(__file__), '..', 'test', 'audio')

WAKEWORD_PLUGINS = ['passthrough', 'livekit_wakeword', 'openwakeword']


def main():
    """メインエントリポイント."""
    parser = argparse.ArgumentParser(description='全WAV×全Wakewordプラグインを一括処理してPNG出力する')
    parser.add_argument('--audio-dir', default=AUDIO_DIR,
                        help='WAVファイルのディレクトリ')
    parser.add_argument('--model-folder', default='models',
                        help='Wakewordモデルフォルダ (default: models)')
    parser.add_argument('--vad-threshold', type=float, default=0.5,
                        help='VAD検出しきい値 (default: 0.5)')
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

    from tools.visualize_vad import (
        build_asr_plugin,
        build_vad_plugin,
        build_wakeword_plugin,
        load_wav,
        plot,
        run_pipeline,
    )

    wav_files = sorted(glob.glob(os.path.join(args.audio_dir, '*.wav')))
    if not wav_files:
        print(f'WAVファイルが見つかりません: {args.audio_dir}')
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.out_dir, ts)
    os.makedirs(out_dir, exist_ok=True)

    print(f'WAVファイル: {len(wav_files)} 件')
    print(f'Wakewordプラグイン: {WAKEWORD_PLUGINS}')
    print(f'Whisperモデル: {args.whisper_model}')
    print(f'出力先: {out_dir}')
    print()

    print(f'Whisper 初期化中: {args.whisper_model} ...')
    asr_plugin = build_asr_plugin(args.whisper_model, args.language, args.device)
    print('Whisper 初期化完了\n')

    print('VAD 初期化中: silero_vad ...')
    vad_plugin = build_vad_plugin(args.vad_threshold)
    print('VAD 初期化完了\n')

    total = len(wav_files) * len(WAKEWORD_PLUGINS)
    done = 0

    for wav_path in wav_files:
        wav_stem = os.path.splitext(os.path.basename(wav_path))[0]
        audio = load_wav(wav_path)
        duration = len(audio) / 16000

        for ww_name in WAKEWORD_PLUGINS:
            done += 1
            print(f'[{done}/{total}] {wav_stem} × {ww_name}')

            try:
                wakeword_plugin = build_wakeword_plugin(
                    ww_name, args.model_folder, None, args.ww_threshold
                )
                asr_plugin.audio_queue = queue.Queue()
                asr_plugin.result_queue = queue.Queue()
                asr_plugin.stop_event = threading.Event()

                collected = run_pipeline(wav_path, vad_plugin, wakeword_plugin, asr_plugin)

                for ev, t in collected.vad_events:
                    print(f'    {t:.3f}s  {ev}')
                for start, end, text in collected.final_results:
                    print(f'    {start:.3f}s-{end:.3f}s  "{text}"')
                if not collected.vad_events and not collected.final_results:
                    print(f'    ({duration:.2f}s 無イベント)')

                out_path = os.path.join(out_dir, f'{wav_stem}__{ww_name}.png')
                plot(audio, collected, wav_path, ww_name, args.whisper_model, out_path)

            except Exception as e:
                print(f'    エラー: {e}')

            print()

    print(f'完了。{done} 件の PNG を {out_dir} に保存しました。')


if __name__ == '__main__':
    main()
