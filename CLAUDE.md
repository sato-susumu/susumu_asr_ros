# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Claude Code Context for susumu_asr_ros

## プロジェクト概要
ROS2上で動作する音声認識パッケージで、長時間安定動作を実現するメモリ管理機能を持つ。

### 主要コンポーネント
- **VAD (音声区間検出)**: Silero VAD
- **ウェイクワード検出**: OpenWakeWord  
- **ASR (音声認識)**: Google Cloud Speech-to-Text または faster-whisper
- **メモリ管理**: 自動メモリリーク防止機能
- **モニタリング**: リアルタイムASRイベント・メモリ監視

## アーキテクチャ

### クラス階層
```
VAD (抽象基底クラス)
├── SileroVAD - PyTorchベースの高精度VAD
└── OpenWakeWord - TFLiteベースのウェイクワード検出

ASR (抽象基底クラス)
├── GoogleCloudASR - ストリーミング音声認識
└── WhisperASR - バッチ処理音声認識

AudioIO (抽象基底クラス)
├── MicInput - マイク入力（PyAudio）
├── FileInput - WAVファイル入力
└── FileOutput - デバッグ用WAV出力
```

### スレッドアーキテクチャ
1. **メインスレッド**: ROS2ノード、全体制御
2. **VADスレッド**: 音声区間検出、ウェイクワード検出
3. **ASRスレッド**: 音声認識処理
4. **モニタースレッド**: メモリプロファイリング（オプション）

### キューベース通信
- `audio_queue`: VAD → ASR への音声データ転送
- `result_queue`: ASR → メイン への認識結果転送

## ビルドコマンド
```bash
cd ~/ros2_ws
colcon build --packages-select susumu_asr_ros
source install/setup.bash
```

## テストコマンド
```bash
# 全テスト実行
pytest test/

# コードスタイルチェック（最重要）
python -m pytest test/test_flake8.py -v

# 個別テスト実行
python -m pytest test/test_pep257.py  # docstringチェック
python -m pytest test/test_copyright.py  # 著作権表記チェック
python -m pytest test/test_asr.py  # 機能テスト

# 特定のテストのみ実行
python -m pytest test/test_asr.py::test_google_cloud_asr -v
```

## 起動コマンド

### 基本起動
```bash
# ウェイクワード + Google ASR
ros2 launch susumu_asr_ros openwakeword_google.launch.py

# ウェイクワード + Whisper ASR  
ros2 launch susumu_asr_ros openwakeword_whisper.launch.py

# Silero VAD + Google ASR
ros2 launch susumu_asr_ros silerovad_google.launch.py

# Silero VAD + Whisper ASR
ros2 launch susumu_asr_ros silerovad_whisper.launch.py
```

### モニタリング付き起動
```bash
# ASRノード + リアルタイムモニター
ros2 launch susumu_asr_ros openwakeword_google_monitor.launch.py
ros2 launch susumu_asr_ros silerovad_google_monitor.launch.py
```

### ノード単体起動
```bash
# 基本起動
ros2 run susumu_asr_ros susumu_asr_node

# パラメータ指定
ros2 run susumu_asr_ros susumu_asr_node --ros-args \
  -p vad_type:=silero_vad \
  -p asr_type:=google_cloud \
  -p enable_memory_profiling:=true
```

## ASR監視ツール

### 基本使用法
```bash
# リアルタイム監視
ros2 run susumu_asr_ros susumu_asr_monitor

# メモリのみ監視
ros2 run susumu_asr_ros susumu_asr_monitor --memory-only

# 詳細イベント表示
ros2 run susumu_asr_ros susumu_asr_monitor --show-details

# ワンタイム統計
ros2 run susumu_asr_ros susumu_asr_monitor --once
```

### 監視内容
- ウェイクワード検出回数
- 音声認識成功率
- 平均処理時間
- メモリ使用量（現在/ピーク/成長率）
- リアルタイムイベント履歴

## 依存関係
- pyaudio, torch, torchaudio
- google-cloud-speech
- openwakeword
- tflite_runtime==2.14.0
- faster-whisper
- numpy>=1.26,<2
- click (CLIツール用)
- psutil (メモリ監視用)

## 主要パラメータ
- `vad_type`: "openwakeword" or "silero_vad"
- `asr_type`: "google_cloud" or "whisper"
- `language_code`: "ja-JP" (デフォルト)
- `debug`: デバッグモード（WAV出力）
- `input_file`: WAVファイル入力パス
- `simulate_realtime`: リアルタイムシミュレーション
- `enable_memory_profiling`: メモリプロファイリング有効化

## ROS2トピック

### パブリッシュ
- `/stt_event` (String): JSONフォーマットのASRイベント
- `/stt` (String): 確定した音声認識テキスト

### イベントタイプ
```json
// ウェイクワード検出
{"type": "wakeword_detected", "timestamp": 1234567890.123}

// 部分認識結果
{"type": "partial_result", "text": "こんにちは", "timestamp": 1234567890.456}

// 最終認識結果
{"type": "final_result", "text": "こんにちは、今日はいい天気ですね", "timestamp": 1234567891.789}

// タイムアウト
{"type": "timeout", "reason": "speech_timeout", "timestamp": 1234567892.012}
```

## 開発時のヒント

### マイクデバイス確認
```bash
ros2 run susumu_asr_ros susumu_asr_node --ros-args -p list_mic_devices:=true
```

### デバッグモード
```bash
# 音声ファイルと認識結果を保存
ros2 run susumu_asr_ros susumu_asr_node --ros-args -p debug:=true
# 出力: debug_audio_*.wav, debug_labels_*.txt
```

### WAVファイルテスト
```bash
# リアルタイムシミュレーション
ros2 run susumu_asr_ros susumu_asr_node --ros-args \
  -p input_file:="/path/to/test.wav" \
  -p simulate_realtime:=true
```

### メモリプロファイリング
```bash
# メモリ使用量の詳細追跡
ros2 run susumu_asr_ros susumu_asr_node --ros-args \
  -p enable_memory_profiling:=true
```

## トラブルシューティング

### Google Cloud認証
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

### PyAudioインストール
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

### CUDAメモリ不足（Whisper使用時）
```bash
export CUDA_VISIBLE_DEVICES=""  # CPUモードに切り替え
```

### ウェイクワード検出調整
OpenWakeWordのモデルファイルを`models/`ディレクトリに配置し、`oww_model_name`パラメータで指定。

## パッケージタイプ
- ROS2 ament_pythonパッケージ
- ライセンス: MIT
- コンソールスクリプトエントリーポイント付きPythonパッケージ