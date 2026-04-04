# pysoniq

Minimal, pure-Python cross-platform audio analysis and playback library.

- **Pure Python** - No compiled extensions
- **Cross-platform** - Windows, macOS, Linux
- **Minimal dependencies** - numpy, wave
- **Simple API** - Play, pause, stop, loop

## Installation
```bash
pip install pysoniq
```

## Use in context
```python
import pysoniq
import numpy as np

# Play WAV file
pysoniq.play('audio.wav')

# Play as numpy array
sr = 44100
t = np.linspace(0, 1.0, sr)
audio = 0.3 * np.sin(2 * np.pi * 440 * t)
pysoniq.play(audio, samplerate=sr)

# Loop playback
pysoniq.set_loop(True)
pysoniq.play(audio, sr)

# Stop
pysoniq.stop()
```

## Features

**Playback**
```python
pysoniq.play(data, samplerate)  # Play audio
pysoniq.stop()                   # Stop playback
pysoniq.pause()                  # Pause
pysoniq.resume()                 # Resume
```

**Looping**
```python
pysoniq.set_loop(True)   # Enable loop
pysoniq.is_looping()     # Check status
```

**Gain Control**
```python
pysoniq.set_gain(0.5)           # 50% volume
pysoniq.set_volume_db(-6.0)     # Set dB
audio = pysoniq.adjust_gain_level(audio, 1.5)
```

**Audio I/O**
```python
audio, sr = pysoniq.load('file.wav')
pysoniq.save('output.wav', audio, sr)
```

## Platform Requirements

- **Windows**: Built-in (winsound)
- **macOS**: Built-in (afplay)
- **Linux**: ALSA (aplay) - usually pre-installed

## Limitations

- WAV format only (for now)
- Pause/resume uses time-based estimation
- Gain changes apply on next loop iteration

## License

MIT

## Author

laelume