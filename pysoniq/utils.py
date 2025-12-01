"""Utility functions"""

import numpy as np


def to_pcm(data, target_bits=16):
    """
    Convert float audio to PCM with auto bit depth detection
    
    Args:
        data: numpy array, float
        target_bits: 8, 16, 24, or 32
    
    Returns:
        sampwidth: bytes per sample
        data_int: integer array
    """
    data_range = np.max(np.abs(data))
    
    if data_range <= 1.0:
        # Normalized float - convert to PCM
        data = np.clip(data, -1.0, 1.0)
        
        if target_bits == 8:
            data_int = ((data + 1.0) * 127.5).astype(np.uint8)
            sampwidth = 1
        elif target_bits == 16:
            data_int = (data * 32767).astype(np.int16)
            sampwidth = 2
        elif target_bits == 24:
            data_int = (data * 8388607).astype(np.int32)
            sampwidth = 3
        elif target_bits == 32:
            data_int = (data * 2147483647).astype(np.int32)
            sampwidth = 4
        else:
            raise ValueError(f"Unsupported bit depth: {target_bits}")
    else:
        # Already integer - preserve
        if data.dtype in [np.int16, np.uint8]:
            sampwidth = 2 if data.dtype == np.int16 else 1
            data_int = data.astype(np.int16) if data.dtype != np.int16 else data
        elif data.dtype == np.int32:
            sampwidth = 4
            data_int = data
        else:
            # Unknown - normalize and convert
            data_int = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
            sampwidth = 2
    
    return sampwidth, data_int


def to_int16(audio):
    """
    Convert float audio to 16-bit PCM
    
    Args:
        audio: numpy array, float, -1 to 1
    
    Returns:
        int16 array
    """
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)


def int32_to_24bit_bytes(data_int32):
    """Convert int32 array to 24-bit byte array"""
    data_bytes = bytearray()
    for sample in data_int32:
        data_bytes.extend(sample.to_bytes(4, byteorder='little', signed=True)[:3])
    return bytes(data_bytes)


def normalize_audio(audio, target_level=-3.0):
    """
    Normalize audio to target dB level
    
    Args:
        audio: numpy array
        target_level: float, target dB level
    
    Returns:
        normalized audio
    """
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        target_amplitude = 10**(target_level / 20.0)
        audio = audio * (target_amplitude / rms)
    return np.clip(audio, -1.0, 1.0)


