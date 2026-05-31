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
AudioRecorder → VADProcessor → audio_queue → ASR → result_queue → SusumuAsrNode
(MicAudioRecorder / WavAudioRecorder)
```

### 主要クラス（`susumu_asr_ros/susumu_asr.py`）

| クラス | 役割 |
|--------|------|
| `VADBase` | VAD抽象基底クラス。`process_frame(bytes) -> (event, frames)` を定義 |
| `SileroVadProcessor` | PyTorch + Silero VADで発話区間検出。`speech_start/speech_cont/speech_stop` を返す |
| `LiveKitWakeWordProcessor` | livekit-wakewordでウェイクワード検出後、Silero VADで発話終了を検出 |
| `ASRBase` | ASR抽象基底クラス。`run()` を定義 |
| `GoogleCloudASR` | ストリーミング認識（`single_utterance=True`）。`_audio_buffer_queue` 経由で音声を受け取り別スレッドで応答処理 |
| `WhisperASR` | バッチ認識。発話終了まで音声を `audio_buffer` に蓄積してまとめてデコード |
| `SpeechRecognitionSystem` | メインループ。`on_asr_event` コールバックで認識結果をROS2ノードへ通知 |

### スレッド構成

- **VADスレッド（メインスレッド内ループ）**: フレームを読み取り、VADに通す
- **ASRスレッド（`asr.run()`）**: `audio_queue` からコマンド（`start/audio/stop/stop_all`）を受け取って認識処理
- **Google ASR内部スレッド（`_streaming_recognize_loop`）**: ストリーミングAPIのレスポンスを非同期処理

### キュー通信プロトコル

`audio_queue` に渡すメッセージ形式：`(command: str, data: bytes)`
- `("start", timestamp_bytes)` — 発話開始
- `("audio", pcm_bytes)` — 音声データ
- `("stop", timestamp_bytes)` — 発話終了
- `("stop_all", None)` — シャットダウン

`result_queue` から返るメッセージ形式：`(is_final: bool, text: str, start: float, end: float)`

### ROS2ノード（`susumu_asr_ros/susumu_asr_node.py`）

`SusumuAsrNode.__init__` でVAD・ASR・録音の各モジュールを組み立て、`system.start()` を別スレッドで起動する。認識イベントは `on_asr_event` コールバック経由で受け取り、以下のトピックに配信する。

| トピック | 型 | 内容 |
|----------|----|------|
| `/stt_event` | `String` | JSON形式の全イベント（`wakeword_detected`, `partial_result`, `final_result`, `timeout`） |
| `/stt` | `String` | `final_result` 時のテキストのみ |

### モニタリングノード（`susumu_asr_ros/asr_monitor.py`）

`/stt_event` をサブスクライブして統計（検出回数・成功率・平均処理時間）をターミナル表示する独立ノード。

## 音声設定の定数

`susumu_asr.py` 内の定数を変える場合：

- `SAMPLE_RATE = 16000` — WAVファイル入力時もこのレートを強制チェック
- `FRAME_DURATION_MS = 30` — PyAudioのread_frame_sizeと連動
- `SILERO_VAD_THRESHOLD = 0.5` — SileroVADの検出感度
- `LIVEKIT_WAKEWORD_SPEECH_TIMEOUT_SECONDS = 8.0` — ウェイクワード後の最大録音時間

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
