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
  - [vad_livekit_wakeword.py](#vad_livekit_wakewordpy)
- [ASRプラグイン](#asrプラグイン)
  - [asr_google.py](#asr_googlepy)
  - [asr_whisper.py](#asr_whisperpy)
- [音声入出力](#音声入出力)
  - [audio_io.py](#audio_iopy)
- [パイプライン制御](#パイプライン制御)
  - [susumu_asr.py](#susumu_asrpy)
  - [SpeechRecognitionSystem コールバックプロトコル](#speechrecognitionsystem-コールバックプロトコル)
- [ROS2ノード](#ros2ノード)
  - [susumu_asr_node.py](#susumu_asr_nodepy)

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
    VAD -->|VADEvent| SRS
    SRS -->|ASRCommand| AQ
    AQ --> ASR
    ASR -->|result| RQ
    RQ --> SRS
    SRS -->|on_asr_event| NODE
```

以下のシーケンスで共通する participant は省略表記を統一する。

| 省略名 | 実体 |
|---|---|
| AR | AudioRecorder |
| SRS | SpeechRecognitionSystem |
| VAD | VADPlugin |
| ASR | ASRPlugin |
| NODE | SusumuAsrNode |
| EVT | stt_event topic |
| STT | stt topic |

#### ケース1: 正常系（発話検出 → ASR認識完了）

```mermaid
sequenceDiagram
    participant AR as AR
    participant SRS as SRS
    participant VAD as VAD
    participant ASR as ASR
    participant NODE as NODE
    participant EVT as EVT
    participant STT as STT

    loop 待機中
        AR ->> SRS : frame
        SRS ->> VAD : process_frame(frame)
        VAD -->> SRS : SILENCE
    end

    AR ->> SRS : frame
    SRS ->> VAD : process_frame(frame)
    VAD -->> SRS : SPEECH_START
    SRS ->> ASR : (START, timestamp)
    SRS ->> NODE : SpeechStartEvent
    NODE -->> EVT : vad_speech_start

    loop 発話中
        AR ->> SRS : frame
        SRS ->> VAD : process_frame(frame)
        VAD -->> SRS : SPEECH_CONT
        SRS ->> ASR : (AUDIO, pcm_bytes)
        ASR -->> NODE : PartialResultEvent
        NODE -->> EVT : asr_partial_result
    end

    AR ->> SRS : frame
    SRS ->> VAD : process_frame(frame)
    VAD -->> SRS : SPEECH_STOP
    SRS ->> ASR : (STOP, timestamp)
    SRS ->> NODE : SpeechStopEvent
    NODE -->> EVT : vad_speech_stop

    ASR -->> SRS : ASRResult(is_final=True, text=...)
    SRS ->> NODE : FinalResultEvent
    NODE -->> EVT : asr_final_result
    NODE -->> STT : テキスト
```

#### ケース2: タイムアウト（発話が長すぎて強制終了）

発話開始後、`speech_timeout_sec` を超えても無音にならない場合。`SPEECH_STOP` の代わりに `SPEECH_TIMEOUT` が返り、ASR 結果は捨てられる。

```mermaid
sequenceDiagram
    participant SRS as SRS
    participant VAD as VAD
    participant ASR as ASR
    participant NODE as NODE
    participant EVT as EVT

    note over SRS,VAD : SPEECH_START〜発話中は ケース1 と同じ

    SRS ->> VAD : process_frame(frame)
    VAD -->> SRS : SPEECH_TIMEOUT
    SRS ->> ASR : (STOP, timestamp)
    SRS ->> NODE : TimeoutEvent
    NODE -->> EVT : asr_timeout
```

#### ケース3: ASR が空文字を返す（認識結果なし）

VAD は正常に `SPEECH_STOP` を返すが、ASR が空文字 or 無音判定した場合。`FinalResultEvent` は発火しない。

```mermaid
sequenceDiagram
    participant SRS as SRS
    participant ASR as ASR
    participant NODE as NODE
    participant EVT as EVT

    note over SRS,EVT : SPEECH_START〜SPEECH_STOP は ケース1 と同じ

    ASR -->> SRS : ASRResult(is_final=True, text="")
    note over SRS : text が空のため on_asr_event を呼ばない
```

#### ケース4: WAVファイル終端で発話セッションを強制終了

`input_file` 指定時、発話中にファイルが終わった場合。`SPEECH_STOP` は返らず SRS が直接 STOP を送る。

```mermaid
sequenceDiagram
    participant AR as AR
    participant SRS as SRS
    participant ASR as ASR
    participant NODE as NODE
    participant EVT as EVT
    participant STT as STT

    note over AR,EVT : SPEECH_START〜発話中は ケース1 と同じ

    AR -->> SRS : frame=None (EOF)
    SRS ->> ASR : (STOP, timestamp)
    note over SRS : stop_event をセットしてループ終了

    ASR -->> SRS : ASRResult(is_final=True, text=...)
    SRS ->> NODE : FinalResultEvent
    NODE -->> EVT : asr_final_result
    NODE -->> STT : テキスト
```

---

## 定数・基盤

### `constants.py`
パイプライン全体で共有する定数を定義する。他モジュールはここからimportし、直接マジックナンバーを書かない。

| 定数 | 値 | 意味 |
|---|---|---|
| `SAMPLE_RATE` | `16000` | サンプリングレート (Hz) |
| `SAMPLE_WIDTH` | `2` | サンプル幅 (bytes) |
| `CHANNELS` | `1` | チャンネル数 |
| `FRAME_LENGTH_MS` | `30` | 1フレームの長さ (ms) |
| `AUDIO_FRAME_SAMPLES` | `512` | AudioRecorder フレームサイズ / Silero VAD 最小サンプル数 |
| `INT16_MAX` | `32768.0` | int16 PCM を -1.0〜1.0 に正規化する係数（2^15） |
| `MS_PER_SEC` | `1000.0` | ms を秒に変換する係数 |
| `VAD_SILERO_VAD` | `"silero_vad"` | VADプラグイン識別名 |
| `VAD_LIVEKIT_WAKEWORD` | `"livekit_wakeword"` | VADプラグイン識別名 |
| `ASR_GOOGLE_CLOUD` | `"google_cloud"` | ASRプラグイン識別名 |
| `ASR_WHISPER` | `"whisper"` | ASRプラグイン識別名 |

### `plugin_base.py`
プラグインのインタフェース・列挙型・パラメータ宣言型を定義する。

| クラス/型 | 責務 |
|---|---|
| `VADEvent` | `process_frame()` が返すイベント名の列挙型（`StrEnum`） |
| `ASRCommand` | `audio_queue` に送るコマンド名の列挙型（`StrEnum`） |
| `ASREventType` | `on_asr_event` コールバックのイベント種別の列挙型（`StrEnum`） |
| `VADResult` | `process_frame()` の戻り値（`event: VADEvent`, `frames: list[bytes]`） |
| `ASRResult` | `result_queue` から返る認識結果（`is_final`, `text`, `start`, `end`） |
| `SpeechStartEvent` | VAD 発話開始イベント（`start`, `score`） |
| `SpeechStopEvent` | VAD 発話終了イベント（`start`, `end`） |
| `PartialResultEvent` | 途中認識結果イベント（`start`, `text`） |
| `FinalResultEvent` | 確定認識結果イベント（`start`, `end`, `text`） |
| `TimeoutEvent` | 発話タイムアウトイベント（`start`, `end`, `reason`） |
| `PluginParam` | プラグインが宣言するパラメータ1件（名前・デフォルト値・説明）を保持するデータクラス |
| `ASRPluginBase` | ASRプラグインの抽象基底クラス |
| `VADPluginBase` | VADプラグインの抽象基底クラス。`last_score: float` でウェイクワードスコアを公開する |

### `plugin_loader.py`
`PluginLoader` クラスがPythonエントリポイント（`importlib.metadata`）を使い、プラグイン名からクラスを動的にロードする。`setup.py` に登録されたプラグインのみ発見対象となるため、サードパーティが独自プラグインを追加する際も本体コードの変更は不要。

| メソッド | 責務 |
|---|---|
| `PluginLoader.load_asr(name)` | 名前で ASR プラグインクラスを返す |
| `PluginLoader.load_vad(name)` | 名前で VAD プラグインクラスを返す |
| `PluginLoader.list_asr_plugins()` | 登録済み ASR プラグイン名一覧を返す |
| `PluginLoader.list_vad_plugins()` | 登録済み VAD プラグイン名一覧を返す |

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

`process_frame(frame: bytes)` は `(VADEvent | None, list[bytes])` のタプルを返す。`in_speech` フラグを持ち、`SpeechRecognitionSystem` から参照される。

```mermaid
flowchart LR
    S([Silence])
    P([Speaking])

    S -->|"VADEvent.SILENCE"| S
    S -->|"VADEvent.SPEECH_START"| P
    P -->|"VADEvent.SPEECH_CONT"| P
    P -->|"VADEvent.SPEECH_STOP"| S
```

`process_frame()` は常に `VADResult(event, frames)` を返す。

| VADEvent | frames の内容 |
|---|---|
| `SILENCE` | `[]`（処理不要） |
| `SPEECH_START` | 発話開始前のバッファ＋現フレーム |
| `SPEECH_CONT` | 現フレーム |
| `SPEECH_STOP` | 現フレーム |
| `SPEECH_TIMEOUT` | 現フレーム（タイムアウト時） |

### ASRプラグインのキュープロトコル

`audio_queue` に渡すメッセージ形式は `(ASRCommand, data: bytes)` 。`result_queue` から返すメッセージ形式は `ASRResult`（`is_final`, `text`, `start`, `end`）。`end` は partial結果では `None`。

```mermaid
sequenceDiagram
    participant S as SpeechRecognitionSystem
    participant Q as audio_queue
    participant A as ASRPlugin
    participant R as result_queue

    S ->> Q : (ASRCommand.START, timestamp)
    loop 発話中
        S ->> Q : (ASRCommand.AUDIO, pcm_bytes)
        A -->> R : (False, partial_text, start, None)
    end
    S ->> Q : (ASRCommand.STOP, timestamp)
    A -->> R : (True, final_text, start, end)

    note over S,A : 終了時
    S ->> Q : (ASRCommand.STOP_ALL, None)
```

---

## VADプラグイン

### `vad_silero.py`
Silero VAD を用いた発話区間検出プラグイン。`SilenceAwareVADIterator`（内部クラス）が Silero の `VADIterator` をラップし、無音継続時間ベースの発話終了判定を行う。512サンプル未満のフレームは `ValueError` を送出する。

### `vad_livekit_wakeword.py`
livekit-wakeword による ONNX ウェイクワード検出 + Silero VAD による発話終了検出プラグイン。2秒のリングバッファに音声を蓄積し 0.2秒ごとに推論する。`last_score` に最新スコアを保持し、検出時は `SpeechStartEvent.score` として `/stt_event` に配信される。

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
| `MicAudioRecorder` | PyAudio 経由でマイクからフレームを読み取る。`list_devices()` staticmethod でデバイス一覧を表示できる |
| `WavAudioRecorder` | WAVファイルからフレームを読み取る。`simulate_realtime=True` でリアルタイム入力を模倣できる |

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
- `VADEvent` に応じて `ASRCommand` を `audio_queue` に送信する
- VAD イベント（`vad_speech_start` / `vad_speech_stop` 等）と ASR 結果（`asr_final_result` 等）をまとめて `on_asr_event` コールバックで通知する
- WAVファイル終端・Ctrl+C・シグナルによる終了処理を行う

### `SpeechRecognitionSystem` コールバックプロトコル

コンストラクタに渡すコールバックの型と呼ばれるタイミング。

| コールバック | シグネチャ | 呼ばれるタイミング |
|---|---|---|
| `on_asr_event` | `(ASREventUnion) -> None` | 全イベント発生時（下記イベント型を参照） |

#### `on_asr_event` のイベント型

各イベントは `plugin_base.py` で定義された dataclass。`event_type` フィールド（`ASREventType`）でイベント種別を識別する。

| 型 | `event_type` | フィールド |
|---|---|---|
| `SpeechStartEvent` | `vad_speech_start` | `start: float`, `score: float`（ウェイクワードスコア。非対応プラグインは 0.0） |
| `SpeechStopEvent` | `vad_speech_stop` | `start: float`, `end: float` |
| `PartialResultEvent` | `asr_partial_result` | `start: float`, `text: str` |
| `FinalResultEvent` | `asr_final_result` | `start: float`, `end: float`, `text: str` |
| `TimeoutEvent` | `asr_timeout` | `start: float`, `end: float`, `reason: str` |

`SusumuAsrNode` はこれらを `dataclasses.asdict()` でシリアライズし、`/stt_event` トピックに JSON として配信する。

---

## ROS2ノード

### `susumu_asr_node.py`
**ROS2ノードのエントリポイント。**

`SusumuAsrNode` が以下を担う。

1. ROS2パラメータ `vad_plugin` / `asr_plugin` でプラグインを選択する
2. `PluginLoader` でクラスをロード後、`_declare_plugin_params()` で `{plugin_name}.{param_name}` 形式のROS2パラメータを宣言・取得し、プラグインに注入する
3. デバッグ用ライター・AudioRecorder を生成して `SpeechRecognitionSystem` を組み立てる
4. 認識イベントを以下のトピックに配信する

| トピック | 型 | 内容 |
|---|---|---|
| `/stt_event` | `String` | 全イベントの JSON。`event_type` フィールドでイベント種別を識別する（詳細は上記ペイロード表を参照） |
| `/stt` | `String` | `asr_final_result` 時のテキストのみ（`text` フィールドの値） |

