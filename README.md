# susumu_asr_ros
作りかけのROS 2 向けの音声認識パッケージです。永遠に作りかけ？
VAD (音声区間検出)、ウェイクワード検出、ASR (音声認識) を組み合わせて、ROS2上で動作させることを目的としています。

---

## 概要

- **VAD (Voice Activity Detection)**  
  - [Silero VAD](https://github.com/snakers4/silero-vad)  
- **ウェイクワード検出**
  - [OpenWakeWord](https://github.com/dsoto/openwakeword)  
- **ASR (Automatic Speech Recognition)**  
  - [Google Cloud Speech-to-Text](https://cloud.google.com/speech-to-text)  
  - [faster-whisper](https://github.com/SYSTRAN/faster-whisper)

---

## インストール & ビルド手順

本パッケージは [ROS 2](https://docs.ros.org/en) のワークスペース (例: `ros2_ws/src/`) に配置し、依存をインストールしてからビルドします。

### 1. ワークスペースに配置

```bash
cd ~/ros2_ws/src
git clone https://github.com/sato-susumu/susumu_asr_ros.git
```

### 2. 依存パッケージのインストール

手動でインストールする場合:

```bash
# 例: Python仮想環境下で
pip install pyaudio torch torchaudio google-cloud-speech openwakeword click "numpy<2.0" tflite_runtime==2.14.0 faster-whisper
```

### 3. ビルド

```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

---

## 実行方法

### ランチファイルから起動

```bash
ros2 launch susumu_asr_ros openwakeword_google_debug.launch.py
```

標準設定では下記が実行されます。

- VAD: OpenWakeWord
- ASR: Google Cloud Speech-to-Text
- マイク入力で音声を取得

### ノード単体で起動

```bash
ros2 run susumu_asr_ros susumu_asr_node
```

（パラメータを指定する方法は後述）

---

## パラメータによるカスタマイズ

`susumu_asr_node.py` では、ROS 2 のパラメータを使って以下の項目を切り替えられます。

| パラメータ名           | 型      | 既定値               | 説明                                         |
|------------------------|---------|----------------------|--------------------------------------------|
| `list_mic_devices`     | bool    | `False`             | `True` にすると起動時にマイクデバイス一覧を表示                |
| `vad_type`             | string  | `"openwakeword"`     | `silero_vad` or `openwakeword`             |
| `asr_type`             | string  | `"google_cloud"`     | `google_cloud` or `whisper`      |
| `language_code`        | string  | `"ja-JP"`           | Google Cloud Speech-to-Text の言語コード         |
| `oww_model_folder`     | string  | `"models"`          | OpenWakeWord のモデルを配置するフォルダ                 |
| `oww_model_name`       | string  | `"hey_mycroft_v0.1.tflite"` | OpenWakeWord のモデルファイル名                     |
| `input_device_index`   | int     | `None`              | マイク入力のデバイスインデックス (未指定の場合、デフォルトマイク)           |
| `input_file`           | string  | `None`              | WAV ファイルのパスを指定するとファイル入力に切り替わる              |
| `simulate_realtime`    | bool    | `False`             | WAV ファイル入力時にフレーム単位で遅延を挿入し、リアルタイムっぽく動かす     |
| `debug`                | bool    | `False`             | `True` にするとデバッグモードでの全音声 WAV 出力 & ラベル出力を有効化 |

### パラメータ指定例

たとえば、VADを"Silero VAD"、ASRを"Google Cloud"にする場合:

```bash
ros2 run susumu_asr_ros susumu_asr_node \
  --ros-args \
    -p vad_type:=silero_vad \
    -p asr_type:=google_cloud
```

WAVファイルから入力し、リアルタイムシミュレーションを ON にする:

```bash
ros2 run susumu_asr_ros susumu_asr_node \
  --ros-args \
    -p input_file:="path/to/sample.wav" \
    -p simulate_realtime:=True
```
