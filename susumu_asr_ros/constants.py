"""音声処理パイプライン全体で共通する定数."""

VAD_SILERO_VAD = 'silero_vad'
VAD_OPENWAKEWORD = 'openwakeword'

ASR_GOOGLE_CLOUD = 'google_cloud'
ASR_WHISPER = 'whisper'

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1
FRAME_LENGTH_MS = 30

# int16 PCM を -1.0〜1.0 の float に正規化するための係数（2^15）
INT16_MAX = 32768.0

# ms を秒に変換する係数
MS_PER_SEC = 1000.0

# Silero VAD が要求する最小サンプル数 / AudioRecorder のフレームサイズ
AUDIO_FRAME_SAMPLES = 512
