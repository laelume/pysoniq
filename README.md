# pysoniq

Minimal Pythonic cross-platform audio analysis and playback library.

- **Cross-platform** - Windows, macOS, Linux
- **Minimal dependencies** - numpy, scipy, wave, ffmpeg
- **Simple API** - Play, pause, stop, loop

## Installation
```bash
pip install pysoniq
```

WIP: For MP3 support, install ffmpeg: https://ffmpeg.org/

## Use in context
```python
import pysoniq
import numpy as np
import wave

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

# Make audio blobs for model input
X, y = make_audioblobs(
    source       = "path/to/wav_dir",
    n_fft        = 512,
    metric       = "euclidean",
    n_classes    = 10,
    random_state = 42,
    verbose      = True,
)
```

## Features

**Playback**
```python
pysoniq.play(data, samplerate)   # Play audio
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
audio, sr = pysoniq.load_audio('file.wav')  # or .mp3
pysoniq.save_audio('output.wav', audio, sr)
```

**Fourier Analysis**
```python
import pysoniq.fourier as pf

S = pf.magnitude_stft(audio, n_fft=2048)
S_db = pf.amplitude_to_db(S)
```

## Platform Requirements

- **Windows**: Built-in (winsound)
- **macOS**: Built-in (afplay)
- **Linux**: ALSA (aplay) - usually pre-installed

- **MP3**: ffmpeg (install separately)

## Limitations

- mp3 support is WIP
- Pause/resume uses time-based estimation
- Gain changes apply on next loop iteration

## License

MIT

## Author

laelume
