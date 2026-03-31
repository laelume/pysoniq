"""Audio file I/O"""

import numpy as np
import wave
from pathlib import Path

class Signal:
    def __init__(self, y, sr):
        self.y = y
        self.sr = sr

def load_signal(path): 
    y, sr = load_audio(path)
    return Signal(y, sr) 

def load_audio(filepath):
    """
    Load audio file
    
    Args:
        filepath: str or Path to audio file
    
    Returns:
        audio: numpy array (normalized float32, -1 to 1)
        samplerate: int, sample rate in Hz
    
    Supports: .wav, .mp3
    """
    filepath = Path(filepath)
    
    if filepath.suffix.lower() == '.wav':
        return _load_wav(filepath)
    elif filepath.suffix.lower() == '.mp3':
        return _load_mp3(filepath)
    else:
        raise ValueError(f"Unsupported format: {filepath.suffix}")

def _load_wav(filepath):
    """Load WAV file using wave module"""
    with wave.open(str(filepath), 'rb') as wav:
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        framerate = wav.getframerate()
        n_frames = wav.getnframes()
        
        # Read raw data
        raw_data = wav.readframes(n_frames)
    
    # Convert to numpy array based on sample width
    if sampwidth == 1:  # 8-bit unsigned
        audio = np.frombuffer(raw_data, dtype=np.uint8)
        audio = (audio.astype(np.float32) - 128) / 128.0
    elif sampwidth == 2:  # 16-bit signed
        audio = np.frombuffer(raw_data, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0
    elif sampwidth == 3:  # 24-bit signed
        # Expand 24-bit to 32-bit
        audio_bytes = np.frombuffer(raw_data, dtype=np.uint8)
        audio_int32 = np.zeros(len(audio_bytes) // 3, dtype=np.int32)
        for i in range(len(audio_int32)):
            audio_int32[i] = (audio_bytes[i*3] | 
                             (audio_bytes[i*3+1] << 8) | 
                             (audio_bytes[i*3+2] << 16))
            # Sign extend
            if audio_int32[i] & 0x800000:
                audio_int32[i] |= 0xFF000000
        audio = audio_int32.astype(np.float32) / 8388608.0
    elif sampwidth == 4:  # 32-bit signed
        audio = np.frombuffer(raw_data, dtype=np.int32)
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
    
    # Reshape for multi-channel
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)
    
    return audio, framerate

# Adding support for mp3 files using ffmpeg

def _load_mp3(filepath):
    """Load MP3 file using ffmpeg (mono output)"""
    import subprocess
    from pathlib import Path
    
    filepath = Path(filepath)
    
    # Check ffmpeg availability
    try:
        subprocess.run(["ffmpeg", "-version"], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL,
                      check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError(
            "ffmpeg not found. Install from https://ffmpeg.org/"
        )
    
    # Probe for sample rate
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(filepath)
    ]
    
    probe_result = subprocess.run(
        probe_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if probe_result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {probe_result.stderr}")
    
    samplerate = int(probe_result.stdout.strip())
    
    # Decode audio to mono PCM
    decode_cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", str(filepath),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", "1",  # Force mono
        "-"
    ]
    
    decode_result = subprocess.run(
        decode_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    if decode_result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {decode_result.stderr.decode()}")
    
    # Convert to numpy (already mono, no reshape needed)
    audio = np.frombuffer(decode_result.stdout, dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0
    
    return audio, samplerate


def save_audio(filepath, audio, samplerate):
    """
    Save audio to file
    
    Args:
        filepath: str or Path to output file
        audio: numpy array (float32, -1 to 1)
        samplerate: int, sample rate in Hz
    
    Supports: .wav
    """
    filepath = Path(filepath)
    
    if filepath.suffix.lower() == '.wav':
        _save_wav(filepath, audio, samplerate)
    else:
        raise ValueError(f"Unsupported format: {filepath.suffix}")

def _save_wav(filepath, audio, samplerate):
    """Save WAV file using wave module"""
    from .utils import to_int16
    
    # Convert to mono if needed
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # Convert to 16-bit PCM
    audio_int16 = to_int16(audio)
    
    with wave.open(str(filepath), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(samplerate)
        wav.writeframes(audio_int16.tobytes())