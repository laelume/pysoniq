"""Audio file I/O"""

import numpy as np
import wave
from pathlib import Path

class Signal:
    def __init__(self, y, sr):
        self.y = y
        self.sr = sr


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
    samplerate = _sample_rate(filepath)


    # bind native sr for silent default in fourier sr-dependent functions
    from . import fourier
    fourier.set_native_sr(samplerate)

    if filepath.suffix.lower() == '.wav':
        return _load_wav(filepath, samplerate)
    elif filepath.suffix.lower() == '.mp3':
        return _load_mp3(filepath, samplerate)
    else:
        raise ValueError(f"Unsupported format: {filepath.suffix}")


def load_signal(path): 
    y, sr = load_audio(path)
    return Signal(y, sr) 


def _sample_rate(filepath):
    """Extract native sample rate from audio file without decoding.
    
    Args:
        filepath: str or Path to audio file
    
    Returns:
        int: sample rate in Hz
    
    Raises:
        ValueError: if format unsupported
        RuntimeError: if probe fails
    """
    filepath = Path(filepath)
    
    if filepath.suffix.lower() == '.wav':
        return _sample_rate_wav(filepath)
    elif filepath.suffix.lower() == '.mp3':
        return _sample_rate_mp3(filepath)
    else:
        raise ValueError(f"Unsupported format: {filepath.suffix}")


def _sample_rate_wav(filepath):
    """Extract sample rate from WAV file metadata.
    
    Args:
        filepath: Path to WAV file
    
    Returns:
        int: sample rate in Hz
    """
    with wave.open(str(filepath), 'rb') as wav:
        return wav.getframerate()


def _sample_rate_mp3(filepath):
    """Extract sample rate from MP3 file via ffprobe.
    
    Args:
        filepath: Path to MP3 file
    
    Returns:
        int: sample rate in Hz
    
    Raises:
        RuntimeError: if ffprobe unavailable or fails
    """
    import subprocess
    
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError(
            "ffmpeg not found. Install from https://ffmpeg.org/"
        )
    
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
    
    return int(probe_result.stdout.strip())


def _load_wav(filepath, samplerate):
    """Load WAV file using wave module

    Args:
    filepath: Path object
    samplerate: int, pre-extracted sample rate in Hz
    
    """
    with wave.open(str(filepath), 'rb') as wav:
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        # framerate = wav.getframerate()
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
    
    return audio, samplerate

# Adding support for mp3 files using ffmpeg

def _load_mp3(filepath, samplerate):
    """Load MP3 file using ffmpeg (mono output)
            
    Args:
        filepath: Path object
        samplerate: int, pre-extracted sample rate in Hz
    """
    import subprocess
    # from pathlib import Path
    
    # filepath = Path(filepath)
    
    # # Check ffmpeg availability
    # try:
    #     subprocess.run(["ffmpeg", "-version"], 
    #                   stdout=subprocess.DEVNULL, 
    #                   stderr=subprocess.DEVNULL,
    #                   check=True)
    # except (FileNotFoundError, subprocess.CalledProcessError):
    #     raise RuntimeError(
    #         "ffmpeg not found. Install from https://ffmpeg.org/"
    #     )
    
    # # Probe for sample rate
    # probe_cmd = [
    #     "ffprobe",
    #     "-v", "error",
    #     "-select_streams", "a:0",
    #     "-show_entries", "stream=sample_rate",
    #     "-of", "default=noprint_wrappers=1:nokey=1",
    #     str(filepath)
    # ]
    
    # probe_result = subprocess.run(
    #     probe_cmd,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    #     text=True
    # )
    
    # if probe_result.returncode != 0:
    #     raise RuntimeError(f"ffprobe failed: {probe_result.stderr}")
    
    # samplerate = int(probe_result.stdout.strip())
    
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


# (つ -' o '- )つ                 (つ -' ~ '- )つ
#                 (つ -' x '- )つ                 (つ -' 3 '- )つ


# SEGMENTATION and various things for YAAAT and other applications

# TODO: Refactor load_segment (and load_audio) to return a Signal object instead
# of a raw (audio, samplerate) tuple once BaseLayer is refactored to accept Signal
# objects as its audio state. This requires:
#   1. BaseLayer.y and BaseLayer.sr replaced by BaseLayer.signal: Signal
#   2. All tab/layer subclasses updated to reference signal.y and signal.sr
#   3. pysoniq.play(), pysoniq.set_gain() etc. updated to accept Signal directly
# Coordinate with the overall yaaat refactor.

def load_segment(filepath, start_sec, end_sec):
    """Load a time-bounded segment from an audio file without reading the full file.

    Implements lazy loading for long recordings (10-20+ minutes). Only the
    frame range corresponding to [start_sec, end_sec] is read from disk.
    No normalization is applied beyond the bit-depth conversion in the
    underlying loader. Caller is responsible for mono conversion if needed.

    Args:
        filepath: str or Path to the audio file (.wav or .mp3).
        start_sec: float, segment start time in seconds.
        end_sec: float, segment end time in seconds.

    Returns:
        audio: numpy array (float32, shape (n_samples,) or (n_samples, n_channels))
        samplerate: int, sample rate in Hz.

    Raises:
        ValueError: if end_sec <= start_sec or file format is unsupported.
        RuntimeError: if ffmpeg is unavailable for MP3 files.
    """
    if end_sec <= start_sec:
        raise ValueError(
            f"end_sec ({end_sec:.4f}) must be greater than "
            f"start_sec ({start_sec:.4f})"
        )

    filepath = Path(filepath)
    samplerate = _sample_rate(filepath)

    if filepath.suffix.lower() == '.wav':
        return _load_wav_segment(filepath, start_sec, end_sec, samplerate)
    elif filepath.suffix.lower() == '.mp3':
        return _load_mp3_segment(filepath, start_sec, end_sec, samplerate)
    else:
        raise ValueError(f"Unsupported format: {filepath.suffix}")


##    <(''<)    <( ' ' )>    (> '')>


def _load_wav_segment(filepath, start_sec, end_sec, samplerate):
    """Load a frame-range segment from a WAV file using wave module seeking.

    Uses wave.setpos() to seek directly to the start frame, then reads only
    the required frames. No full file load occurs at any point.

    Args:
        filepath: Path to the WAV file.
        start_sec: float, segment start in seconds.
        end_sec: float, segment end in seconds.

    Returns:
        audio: numpy float32 array, shape (n_samples,) mono or (n_samples, n_channels).
        samplerate: int.
    """
    with wave.open(str(filepath), 'rb') as wav:
        n_channels = wav.getnchannels()
        sampwidth  = wav.getsampwidth()
        n_frames   = wav.getnframes()

        # Convert time bounds to frame indices, clamped to file length
        start_frame = max(0, int(start_sec * samplerate))
        end_frame   = min(n_frames, int(end_sec * samplerate))
        n_read      = end_frame - start_frame

        if n_read <= 0:
            raise ValueError(
                f"Segment [{start_sec:.4f}, {end_sec:.4f}] yields 0 frames "
                f"in file {filepath.name} (samplerate={samplerate}, "
                f"n_frames={n_frames})"
            )

        # Seek to start frame — no preceding audio is read
        wav.setpos(start_frame)
        raw_data = wav.readframes(n_read)

    # Bit-depth conversion — mirrors _load_wav exactly
    if sampwidth == 1:
        audio = np.frombuffer(raw_data, dtype=np.uint8)
        audio = (audio.astype(np.float32) - 128) / 128.0
    elif sampwidth == 2:
        audio = np.frombuffer(raw_data, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0
    elif sampwidth == 3:
        # 24-bit: expand to 32-bit with sign extension
        audio_bytes = np.frombuffer(raw_data, dtype=np.uint8)
        audio_int32 = np.zeros(len(audio_bytes) // 3, dtype=np.int32)
        for i in range(len(audio_int32)):
            audio_int32[i] = (
                audio_bytes[i * 3] |
                (audio_bytes[i * 3 + 1] << 8) |
                (audio_bytes[i * 3 + 2] << 16)
            )
            # Sign extend from 24-bit
            if audio_int32[i] & 0x800000:
                audio_int32[i] |= 0xFF000000
        audio = audio_int32.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw_data, dtype=np.int32)
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    # Reshape for multi-channel — consistent with _load_wav
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)

    return audio, samplerate


# (つ -' _ '- )つ    (つ -' _ '- )つ


def _load_mp3_segment(filepath, start_sec, end_sec):
    """Load a time-bounded segment from an MP3 file via ffmpeg subprocess.

    Uses ffmpeg -ss and -t flags for seeking and duration. Only the requested
    segment is decoded — no full file load occurs. Output is forced to mono
    float32, consistent with _load_mp3.

    Args:
        filepath: Path to the MP3 file.
        start_sec: float, segment start in seconds.
        end_sec: float, segment end in seconds.

    Returns:
        audio: numpy float32 array, shape (n_samples,), mono.
        samplerate: int.

    Raises:
        RuntimeError: if ffmpeg or ffprobe is unavailable or fails.
    """
    import subprocess

    duration_sec = end_sec - start_sec

    # Check ffmpeg availability — same guard as _load_mp3
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError(
            "ffmpeg not found. Install from https://ffmpeg.org/"
        )

    # # Probe sample rate — same as _load_mp3
    # probe_cmd = [
    #     "ffprobe",
    #     "-v", "error",
    #     "-select_streams", "a:0",
    #     "-show_entries", "stream=sample_rate",
    #     "-of", "default=noprint_wrappers=1:nokey=1",
    #     str(filepath)
    # ]

    # probe_result = subprocess.run(
    #     probe_cmd,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    #     text=True
    # )

    # if probe_result.returncode != 0:
    #     raise RuntimeError(f"ffprobe failed: {probe_result.stderr}")

    # samplerate = int(probe_result.stdout.strip())

    # Decode only the requested segment via -ss (seek) and -t (duration)
    decode_cmd = [
        "ffmpeg",
        "-v", "error",
        "-ss", str(start_sec),       # seek before input for efficiency
        "-t",  str(duration_sec),    # duration to decode
        "-i",  str(filepath),
        "-f",      "s16le",
        "-acodec", "pcm_s16le",
        "-ac", "1",                  # force mono
        "-"
    ]

    decode_result = subprocess.run(
        decode_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if decode_result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg segment decode failed: "
            f"{decode_result.stderr.decode()}"
        )

    # Convert PCM bytes to float32 — consistent with _load_mp3
    audio = np.frombuffer(decode_result.stdout, dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0

    return audio, samplerate


# U S A G I
# from pysoniq.io import load_segment
# y, sr = load_segment('/path/to/recording.wav', start_sec=12.5, end_sec=14.0)
# y, sr = load_segment('/path/to/recording.mp3', start_sec=60.0, end_sec=61.5)