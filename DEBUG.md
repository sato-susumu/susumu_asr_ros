## デバッグ実行方法
### 前提

### ランチファイルから起動 livekit-wakeword＋google音声認識

通常モード:
```bash
ros2 launch ./launch/livekit_wakeword_google.launch.py
```

デバッグモード:
```bash
ros2 launch ./launch/livekit_wakeword_google.launch.py debug:=true
```

### ランチファイルから起動 livekit-wakeword＋whisper音声認識

通常モード:
```bash
ros2 launch ./launch/livekit_wakeword_whisper.launch.py
```

デバッグモード:
```bash
ros2 launch ./launch/livekit_wakeword_whisper.launch.py debug:=true
```

### ランチファイルから起動 SileroVAD＋google音声認識

通常モード:
```bash
ros2 launch ./launch/silerovad_google.launch.py
```

デバッグモード:
```bash
ros2 launch ./launch/silerovad_google.launch.py debug:=true
```

### ランチファイルから起動 SileroVAD＋whisper音声認識

通常モード:
```bash
ros2 launch ./launch/silerovad_whisper.launch.py
```

デバッグモード:
```bash
ros2 launch ./launch/silerovad_whisper.launch.py debug:=true
```

