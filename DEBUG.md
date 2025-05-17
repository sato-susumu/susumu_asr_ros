## デバッグ実行方法
### 前提

### ランチファイルから起動 ウェイクワード認識＋google音声認識

通常モード:
```bash
ros2 launch ./launch/openwakeword_google.launch.py
```

デバッグモード:
```bash
ros2 launch ./launch/openwakeword_google.launch.py debug:=true
```

### ランチファイルから起動 ウェイクワード認識＋whisper音声認識

通常モード:
```bash
ros2 launch ./launch/openwakeword_whisper.launch.py
```

デバッグモード:
```bash
ros2 launch ./launch/openwakeword_whisper.launch.py debug:=true
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

