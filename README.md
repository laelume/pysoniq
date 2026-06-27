# pysoniq

Minimal Pythonic cross-platform audio analysis, synthesis, and playback library.

- **Cross-platform** - Windows, macOS, Linux
- **Minimal dependencies** - numpy, scipy, wave, ffmpeg
- **Simple API** - Play, pause, stop, loop

[![Documentation Status](https://readthedocs.org/projects/pysoniq/badge/?version=latest)](https://pysoniq.readthedocs.io/en/latest/?badge=latest)


## Installation
```bash
pip install pysoniq
```

WIP: For MP3 support, install ffmpeg: https://ffmpeg.org/

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

import pysoniq.fourier as pf

S = pf.magnitude_stft(audio, n_fft=2048)
S_db = pf.amplitude_to_db(S)

import pysoniq.audioblobs

# Make audio blobs for model input
X, y = make_audioblobs(
    source       = "path/to/wav_dir", # path to source audio (files or subdirs); output mirrors input tree
    n_fft        = 512,               # number of fft bins (frequency resolution)
    metric       = "euclidean",       # distance metric between samples
    n_classes    = 10,                # arbitrary data-driven class prototype
    aggregation  = "frame",           # segmentation resolution; supports frame and syllable
    random_state = 42,                # random seed for repeatability
    verbose      = True,              # include logging
)
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


## Author & License

Copyright 2025-2026 laelume. Licensed under MIT. 
