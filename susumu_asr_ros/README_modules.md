# モジュール責務一覧

## 目次

- [パイプライン概要](#パイプライン概要)
- [定数・基盤](#定数基盤)
  - [constants.py](#constantspy)
  - [plugin_base.py](#plugin_basepy)
  - [plugin_loader.py](#plugin_loaderpy)
  - [asr_base.py](#asr_basepy)
- [プラグイン共通ルール](#プラグイン共通ルール)
  - [ライフサイクル](#ライフサイクル)
  - [パラメータ宣言](#パラメータ宣言)
  - [エントリポイント登録](#エントリポイント登録)
  - [VADプラグインのイベント仕様](#vadプラグインのイベント仕様)
  - [ASRプラグインのキュープロトコル](#asrプラグインのキュープロトコル)
- [VADプラグイン](#vadプラグイン)
  - [vad_silero.py](#vad_sileropy)
  - [vad_openwakeword.py](#vad_openwakewordpy)
- [ASRプラグイン](#asrプラグイン)
  - [asr_google.py](#asr_googlepy)
  - [asr_whisper.py](#asr_whisperpy)
- [音声入出力](#音声入出力)
  - [audio_io.py](#audio_iopy)
- [パイプライン制御](#パイプライン制御)
  - [susumu_asr.py](#susumu_asrpy)
- [ROS2ノード](#ros2ノード)
  - [susumu_asr_node.py](#susumu_asr_nodepy)
  - [asr_monitor.py](#asr_monitorpy)

---

## パイプライン概要

```mermaid
flowchart LR
    AR([AudioRecorder])
    VAD([VADPlugin])
    AQ[[audio_queue]]
    ASR([ASRPlugin])
    RQ[[result_queue]]
    SRS([SpeechRecognitionSystem])
    NODE([SusumuAsrNode\nROS2トピック配信])

    AR -->|frame| VAD
    VAD -->|speech_start/cont/stop| SRS
    SRS -->|start/audio/stop| AQ
    AQ --> ASR
    ASR -->|result| RQ
    RQ --> SRS
    SRS -->|on_asr_event| NODE
```

---

## 定数・基盤

### `constants.py`
パイプライン全体で共有する定数を定義する。サンプリングレート・フレーム長・VAD/ASRのデフォルト閾値など。他モジュールはここからimportし、直接マジックナンバーを書かない。

### `plugin_base.py`
プラグインの抽象基底クラスと、パラメータ宣言型を定義する。

| クラス | 責務 |
|---|---|
| `PluginParam` | プラグインが宣言するパラメータ1件（名前・デフォルト値・説明）を保持するデータクラス |
| `ASRPluginBase` | ASRプラグインのインタフェース |
| `VADPluginBase` | VADプラグインのインタフェース |

### `plugin_loader.py`
Pythonエントリポイント（`importlib.metadata`）を使い、プラグイン名からクラスを動的にロードする。`setup.py` に登録されたプラグインのみ発見対象となるため、サードパーティが独自プラグインを追加する際も本体コードの変更は不要。

### `asr_base.py`
`ASRBase` 抽象クラスのみを定義する薄いモジュール。後方互換のために残しており、新規コードは `ASRPluginBase` を使う。

---

## プラグイン共通ルール

### ライフサイクル

プラグインは以下の順序でメソッドが呼ばれる。

```
__init__()  →  load_params()  →  setup()  →  run() / process_frame()
```

- `__init__()` では重い処理（モデルロード等）を行わない
- `load_params()` でパラメータ値を受け取り、インスタンス変数に保存する
- モデルロード等の重い初期化は `setup()` で行う
- ASR は `setup()` でキューも受け取り、その後 `run()` をスレッドで実行する
- VAD は `setup()` 後、フレームごとに `process_frame()` を呼ばれる

### パラメータ宣言

各プラグインは `get_param_declarations()` で使用するパラメータを `PluginParam` のリストとして返す。`PluginParam` には名前・デフォルト値・説明を記載する。

```python
def get_param_declarations(self) -> list[PluginParam]:
    return [
        PluginParam("param_name", default_value, "説明"),
    ]
```

ノードはこのリストをもとにROS2パラメータを `{plugin_name}.{param_name}` 形式で宣言する。`--ros-args` から上書き可能。

### エントリポイント登録

新しいプラグインは `setup.py` の対応グループにエントリポイントを追加することで利用可能になる。

```python
"susumu_asr_ros.asr_plugins": [
    "my_asr = my_package.my_asr:MyASRPlugin",
],
"susumu_asr_ros.vad_plugins": [
    "my_vad = my_package.my_vad:MyVADPlugin",
],
```

### VADプラグインのイベント仕様

`process_frame(frame: bytes)` は `(event, frames)` のタプルを返す。`in_speech` フラグを持ち、`SpeechRecognitionSystem` から参照される。

```mermaid
flowchart LR
    S([Silence])
    P([Speaking])

    S -->|"(None, [])"| S
    S -->|"speech_start"| P
    P -->|"speech_cont"| P
    P -->|"speech_stop"| S
```

| event | frames の内容 |
|---|---|
| `"speech_start"` | 発話開始前のバッファ＋現フレーム |
| `"speech_cont"` | 現フレーム |
| `"speech_stop"` | 現フレーム |
| `None` | `[]` |

### ASRプラグインのキュープロトコル

`audio_queue` に渡すメッセージ形式は `(command: str, data: bytes)` 。`result_queue` から返すメッセージ形式は `(is_final: bool, text: str, start: float, end: float)` 。

```mermaid
sequenceDiagram
    participant S as SpeechRecognitionSystem
    participant Q as audio_queue
    participant A as ASRPlugin
    participant R as result_queue

    S ->> Q : ("start", timestamp)
    loop 発話中
        S ->> Q : ("audio", pcm_bytes)
        A -->> R : (False, partial_text, start, None)
    end
    S ->> Q : ("stop", timestamp)
    A -->> R : (True, final_text, start, end)

    note over S,A : 終了時
    S ->> Q : ("stop_all", None)
```

---

## VADプラグイン

### `vad_silero.py`
Silero VAD を用いた発話区間検出プラグイン。

### `vad_openwakeword.py`
OpenWakeWord によるウェイクワード検出 + Silero VAD による発話終了検出プラグイン。

---

## ASRプラグイン

### `asr_google.py`
Google Cloud Speech-to-Text ストリーミング認識プラグイン。

### `asr_whisper.py`
faster-whisper によるバッチ認識プラグイン。

---

## 音声入出力

### `audio_io.py`
音声の録音（入力）とファイル書き込み（デバッグ出力）に関するクラス群をまとめる。

**録音クラス（`AudioRecorderBase` 派生）**

| クラス | 責務 |
|---|---|
| `MicAudioRecorder` | PyAudio 経由でマイクからフレームを読み取る |
| `WavAudioRecorder` | WAVファイルからフレームを読み取る |

**音声書き込みクラス（`AudioWriterBase` 派生）**

| クラス | 責務 |
|---|---|
| `FullAudioWriter` | 全音声を1つのWAVファイルに書き出す（デバッグ用） |
| `SpeechAudioWriter` | 発話セッション単位でWAVファイルを書き出す（デバッグ用） |
| `DummyAudioWriter` | 何もしないダミー実装。デバッグ無効時に使用 |

**ラベル書き込みクラス（`LabelWriterBase` 派生）**

| クラス | 責務 |
|---|---|
| `LabelWriter` | 発話区間（開始・終了・ラベル）をタブ区切りテキストに書き出す（デバッグ用） |
| `DummyLabelWriter` | 何もしないダミー実装 |

---

## パイプライン制御

### `susumu_asr.py`
**音声認識パイプラインのメインループ。**

`SpeechRecognitionSystem` が VADPlugin・ASRPlugin・AudioRecorder・各Writerを受け取り、以下を担う。

- AudioRecorder からフレームを読み取り VADPlugin に渡す
- VADPlugin のイベントに応じて `audio_queue` にコマンドを送信する
- `result_queue` から認識結果を受け取り `on_asr_event` コールバックで通知する
- WAVファイル終端・Ctrl+C・シグナルによる終了処理を行う
- `on_audio_level`・`on_wakeword_score`・`on_status` コールバックで補助情報を通知する

---

## ROS2ノード

### `susumu_asr_node.py`
**ROS2ノードのエントリポイント。**

`SusumuAsrNode` が以下を担う。

1. ROS2パラメータ `vad_plugin` / `asr_plugin` でプラグインを選択する
2. `plugin_loader` でクラスをロード後、`{plugin_name}.{param_name}` 形式でROS2パラメータを宣言・取得し、プラグインに注入する
3. デバッグ用ライター・AudioRecorder を生成して `SpeechRecognitionSystem` を組み立てる
4. 認識イベントを以下のトピックに配信する

| トピック | 型 | 内容 |
|---|---|---|
| `/stt_event` | `String` | 全イベントのJSON（`listening_started`・`wakeword_detected`・`partial_result`・`final_result`・`timeout`・`wav_finished`） |
| `/stt` | `String` | `final_result` 時のテキストのみ |
| `/audio_level` | `Float32` | フレームごとのRMS音量 |
| `/wakeword_score` | `Float32` | ウェイクワード検出スコア（OpenWakeWordPlugin使用時） |

### `asr_monitor.py`
**モニタリング用の独立ROS2ノード。**

`/stt_event`・`/stt`・`/audio_level`・`/wakeword_score` をサブスクライブし、イベントログと音量バーをターミナルにリアルタイム表示する。`--stop-on` オプションで特定イベントを受信したら自動終了できるため、スクリプトやテストからの利用にも対応する。
