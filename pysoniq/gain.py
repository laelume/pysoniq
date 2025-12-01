"""Gain and volume control"""

import numpy as np

# Module state
_global_gain = 1.0  # Linear gain (0.0 to 2.0+)
_global_volume_db = 0.0  # dB (-inf to +6 dB typical)


def set_gain(gain):
    """
    Set global gain (linear)
    
    Args:
        gain: float, 0.0 to 2.0+ (1.0 = unity)
    """
    global _global_gain
    _global_gain = max(0.0, gain)


def get_gain():
    """Get current global gain (linear)"""
    return _global_gain


def set_volume_db(db):
    """
    Set global volume in dB
    
    Args:
        db: float, typically -60 to +6 dB (0 dB = unity)
    """
    global _global_volume_db, _global_gain
    _global_volume_db = db
    _global_gain = db_to_linear(db)


def get_volume_db():
    """Get current global volume in dB"""
    return _global_volume_db


def adjust_gain_level(audio, gain=None):
    """
    Adjust audio gain level
    
    Args:
        audio: numpy array
        gain: float, if None uses global gain (1.0 = unity, <1.0 = reduce, >1.0 = boost)
    
    Returns:
        audio with gain adjusted (clipped to -1, 1)
    """
    if gain is None:
        gain = _global_gain
    
    return np.clip(audio * gain, -1.0, 1.0)


def db_to_linear(db):
    """
    Convert dB to linear gain
    
    Args:
        db: float, decibels
    
    Returns:
        linear gain
    """
    return 10.0 ** (db / 20.0)


def linear_to_db(gain):
    """
    Convert linear gain to dB
    
    Args:
        gain: float, linear gain
    
    Returns:
        dB value
    """
    if gain <= 0:
        return -np.inf
    return 20.0 * np.log10(gain)


def normalize(audio, target_db=-3.0):
    """
    Normalize audio to target dB level
    
    Args:
        audio: numpy array
        target_db: float, target peak level in dB (e.g., -3.0)
    
    Returns:
        normalized audio
    """
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    
    target_linear = db_to_linear(target_db)
    current_db = linear_to_db(peak)
    gain_needed = target_linear / peak
    
    return audio * gain_needed


def compress(audio, threshold_db=-20.0, ratio=4.0, attack_ms=5.0, release_ms=50.0, sr=44100):
    """
    Simple dynamics compression
    
    Args:
        audio: numpy array (mono)
        threshold_db: float, compression threshold
        ratio: float, compression ratio (e.g., 4.0 = 4:1)
        attack_ms: float, attack time in milliseconds
        release_ms: float, release time in milliseconds
        sr: int, sample rate
    
    Returns:
        compressed audio
    """
    # Convert threshold to linear
    threshold = db_to_linear(threshold_db)
    
    # Calculate attack/release coefficients
    attack_samples = int(attack_ms * sr / 1000.0)
    release_samples = int(release_ms * sr / 1000.0)
    
    # Simple envelope follower with compression
    envelope = np.abs(audio)
    gain_reduction = np.ones_like(audio)
    
    for i in range(len(audio)):
        # If above threshold, calculate gain reduction
        if envelope[i] > threshold:
            # Calculate how much over threshold
            over = envelope[i] / threshold
            # Apply ratio
            gain_reduction[i] = 1.0 / (1.0 + (over - 1.0) * (ratio - 1.0) / ratio)
    
    # Apply gain reduction with smoothing
    smoothed_gain = np.copy(gain_reduction)
    for i in range(1, len(smoothed_gain)):
        if gain_reduction[i] < smoothed_gain[i-1]:
            # Attack
            alpha = 1.0 - np.exp(-1.0 / attack_samples)
        else:
            # Release
            alpha = 1.0 - np.exp(-1.0 / release_samples)
        
        smoothed_gain[i] = alpha * gain_reduction[i] + (1.0 - alpha) * smoothed_gain[i-1]
    
    return audio * smoothed_gain


def limiter(audio, threshold_db=-1.0, ceiling_db=-0.1):
    """
    Hard limiter to prevent clipping
    
    Args:
        audio: numpy array
        threshold_db: float, where limiting starts
        ceiling_db: float, absolute maximum level
    
    Returns:
        limited audio
    """
    threshold = db_to_linear(threshold_db)
    ceiling = db_to_linear(ceiling_db)
    
    # Hard clip at ceiling
    audio = np.clip(audio, -ceiling, ceiling)
    
    # Soft limiting above threshold
    mask = np.abs(audio) > threshold
    if np.any(mask):
        # Soft knee compression above threshold
        over = np.abs(audio[mask]) - threshold
        audio[mask] = np.sign(audio[mask]) * (threshold + over / (1.0 + over))
    
    return audio