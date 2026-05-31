# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

ROS2上で動作する音声認識パッケージ。VAD（音声区間検出）→ ウェイクワード検出 → ASR（音声認識）のパイプラインをマルチスレッドで実装し、長時間安定動作を目的とする。

## ビルドとテスト

```bash
# ビルド
cd ~/ros2_ws
colcon build --packages-select susumu_asr_ros
source install/setup.bash

# テスト（全件）
pytest test/

# コードスタイルチェック（コミット前に必ず実行）
python -m pytest test/test_flake8.py -v

# 単一テスト実行
python -m pytest test/test_asr.py::test_1 -v
```

> `test_copyright.py` は `@pytest.mark.skip` で意図的にスキップされている。

## アーキテクチャ

### データフロー

```
AudioRecorder → VADPlugin → WakewordPlugin → audio_queue → ASR → result_queue → SusumuAsrNode
(MicAudioRecorder / WavAudioRecorder)
```

### SpeechRecognitionSystem の状態機械

```
IDLE
  → [SPEECH_START] → BUFFERING
      VAD音声をバッファに蓄積しながら WakewordPlugin にフレームを渡す
  → [WAKEWORD_DETECTED] → IN_SPEECH
      VADの silence_threshold を延長（8秒）、バッファ先頭から ASR に送信開始
  → [SPEECH_STOP in BUFFERING] → IDLE
      ウェイクワード未検出のまま発話終了 → 音声を捨てる
  → [SPEECH_STOP in IN_SPEECH] → IDLE
      ASR に STOP 送信
```

### 主要クラス

| クラス | ファイル | 役割 |
|--------|----------|------|
| `VADPluginBase` | `plugin_base.py` | VAD抽象基底クラス。`process_frame(bytes) -> VADResult`、`extend_silence_threshold()` を定義 |
| `SileroVADPlugin` | `vad_silero.py` | PyTorch + Silero VADで発話区間検出 |
| `WakewordPluginBase` | `plugin_base.py` | ウェイクワード検出抽象基底クラス。`process_frame(bytes) -> WakewordResult`、`reset()` を定義 |
| `PassthroughWakewordPlugin` | `wakeword_passthrough.py` | ウェイクワードをスキップ。delay_sec 後に DETECTED を返す（SileroVAD単体モード用） |
| `LivekitWakewordPlugin` | `wakeword_livekit.py` | livekit-wakeword（ONNX）でウェイクワード検出 |
| `OpenWakewordPlugin` | `wakeword_openwakeword.py` | OpenWakeWord（tflite）でウェイクワード検出 |
| `ASRPluginBase` | `plugin_base.py` | ASR抽象基底クラス。`run()` を定義 |
| `GoogleCloudASRPlugin` | `asr_google.py` | ストリーミング認識（`single_utterance=True`）。別スレッドでレスポンス処理 |
| `WhisperASRPlugin` | `asr_whisper.py` | バッチ認識。発話終了まで音声を蓄積してまとめてデコード |
| `SpeechRecognitionSystem` | `susumu_asr.py` | メインループ。VAD・WakewordPlugin・ASRを協調させ `on_asr_event` コールバックで通知 |

### プラグイン組み合わせ

| vad_plugin | wakeword_plugin | 用途 |
|---|---|---|
| `silero_vad` | `passthrough` | ウェイクワードなし常時認識 |
| `silero_vad` | `livekit_wakeword` | livekit-wakewordでウェイクワード検出 |
| `silero_vad` | `openwakeword` | OpenWakeWordでウェイクワード検出 |

### スレッド構成

- **VADスレッド（メインスレッド内ループ）**: フレームを読み取り、VAD → Wakeword → ASRへ通す
- **ASRスレッド（`asr.run()`）**: `audio_queue` からコマンド（`start/audio/stop/stop_all`）を受け取って認識処理
- **Google ASR内部スレッド（`_streaming_recognize_loop`）**: ストリーミングAPIのレスポンスを非同期処理

### キュー通信プロトコル

`audio_queue` に渡すメッセージ形式：`(ASRCommand, data: bytes)`
- `(ASRCommand.START, timestamp_bytes)` — 発話開始
- `(ASRCommand.AUDIO, pcm_bytes)` — 音声データ
- `(ASRCommand.STOP, timestamp_bytes)` — 発話終了
- `(ASRCommand.STOP_ALL, None)` — シャットダウン

`result_queue` から返るメッセージ形式：`ASRResult(is_final, text, start, end)`（`end` は partial では `None`）

### `/stt_event` イベント一覧

| `event_type` | 型 | フィールド |
|---|---|---|
| `vad_speech_start` | `VadStartEvent` | `start` |
| `vad_speech_stop` | `VadStopEvent` | `start`, `end` |
| `ww_listening_started` | `WakewordListeningStartedEvent` | `start` |
| `ww_detected` | `WakewordDetectedEvent` | `start`, `score` |
| `asr_partial_result` | `AsrPartialResultEvent` | `start`, `text` |
| `asr_final_result` | `AsrFinalResultEvent` | `start`, `end`, `text` |

### ROS2ノード（`susumu_asr_ros/susumu_asr_node.py`）

`SusumuAsrNode.__init__` でVAD・Wakeword・ASR・録音の各モジュールを組み立て、`system.start()` を別スレッドで起動する。認識イベントは `on_asr_event` コールバック経由で受け取り、以下のトピックに配信する。

| トピック | 型 | 内容 |
|----------|----|------|
| `/stt_event` | `String` | JSON形式の全イベント（VAD由来: `vad_speech_start`, `vad_speech_stop` / ASR由来: `asr_partial_result`, `asr_final_result`, `asr_timeout`） |
| `/stt` | `String` | `asr_final_result` 時のテキストのみ |


## 音声設定の定数

`constants.py` で定義。変更する場合はここを編集する。

- `SAMPLE_RATE = 16000` — WAVファイル入力時もこのレートを強制チェック
- `FRAME_LENGTH_MS = 30` — PyAudioのread_frame_sizeと連動
- `AUDIO_FRAME_SAMPLES = 512` — Silero VAD が要求する最小サンプル数

プラグイン固有のしきい値（VAD検出感度・タイムアウト等）は launch ファイルまたは `--ros-args` のパラメータで上書きする。

## デバッグモード

`debug=True` で起動すると `./debug/` ディレクトリに以下を出力：
- `{timestamp}_audio_full.wav` — 全音声
- `speech_{timestamp}.wav` — 認識セッション単位の音声
- `{timestamp}_label.txt` — VADラベル（タブ区切り：start, end, label）

## livekit-wakeword のインストール

`livekit-wakeword` は `Requires-Python: >=3.11` と宣言されているが、推論に使う部分は Python 3.10 でも動作する（pure Python wheel）。ROS2 Humble（Python 3.10）へのインストールは以下で行う：

```bash
pip install livekit-wakeword --ignore-requires-python
```

`setup.py` の `install_requires` には含めない（通常の `pip install` でバージョン制約エラーになるため）。

## ウェイクワードモデル

`models/` ディレクトリに ONNX 形式で配置。デフォルトは `models/hey_mycroft_v0.1.onnx`。
利用可能モデル: `alexa`, `hey_jarvis`, `hey_mycroft`, `hey_rhasspy`, `timer`, `weather`。
モデルが存在しない場合は起動時に openWakeWord の GitHub リリース（v0.5.1）から自動ダウンロードされる。
livekit-wakeword と openWakeWord は同じ embedding モデル（Google Speech Embedding）を使うため、openWakeWord 形式の ONNX モデルをそのまま livekit-wakeword で使用できる。

## 環境変数

| 変数 | 用途 |
|------|------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Google Cloud認証JSONのパス |
| `CUDA_VISIBLE_DEVICES=""` | WhisperをCPUモードで実行 |
