"""Minimal-dependency audio processing and fourier transform utilities"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
from .utils import hz_to_mel, mel_to_hz, linear_to_db

def stft(y, n_fft=1024, hop_length=None, window='hann'):
    """
    Complex-valued short-time Fourier transform (retains magnitude and phase information)
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    n_fft : int
        FFT window size
    hop_length : int or None
        Number of samples between frames
    window : str
        Window type ('hann', 'hamming', 'blackman')
        
    Returns
    -------
    stft_matrix : np.ndarray (complex)
        STFT matrix
    """
    if hop_length is None:
        hop_length = n_fft // 4
    
    # Create window
    if window == 'hann':
        win = np.hanning(n_fft)
    elif window == 'hamming':
        win = np.hamming(n_fft)
    elif window == 'blackman':
        win = np.blackman(n_fft)
    else:
        win = np.ones(n_fft)
    
    # Compute STFT
    f, t, Zxx = signal.stft(y, 
                            nperseg=n_fft, 
                            noverlap=n_fft - hop_length,
                            window=win,
                            return_onesided=True,
                            boundary=None,
                            padded=False)
    
    return Zxx

def magnitude_stft(y, n_fft=1024, hop_length=None, window='hann'):
    """
    Real-valued magnitude of short-time Fourier transform (discards phase information)
    
    Args:
        y: numpy array, audio time series
        n_fft: int, FFT window size
        hop_length: int or None, samples between frames
        window: str, window type ('hann', 'hamming', 'blackman')
        
    Returns:
        magnitude STFT matrix (real-valued, non-negative)
    """
    return np.abs(stft(y, n_fft=n_fft, hop_length=hop_length, window=window))


def fft_frequencies(sr, n_fft):
    """
    Frequencies corresponding to linear Hz bins
    
    Parameters
    ----------
    sr : int
        Sample rate
    n_fft : int
        FFT size
        
    Returns
    -------
    freqs : np.ndarray
        Frequency array
    """
    return np.linspace(0, sr / 2, n_fft // 2 + 1)

def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0):
    """
    Frequencies corresponding to mel-spaced bins
    """
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mels = np.linspace(mel_min, mel_max, n_mels)
    return mel_to_hz(mels)

def frames_to_time(frames, sr, hop_length):
    """
    Convert frame indices to time in seconds
    
    Parameters
    ----------
    frames : np.ndarray or int
        Frame indices
    sr : int
        Sample rate
    hop_length : int
        Hop length in samples
        
    Returns
    -------
    times : np.ndarray or float
        Time in seconds
    """
    return frames * hop_length / sr

def amplitude_to_db(S, ref=1.0, amin=1e-10):
    """
    Convert amplitude spectrogram to dB scale
    
    Parameters
    ----------
    S : np.ndarray
        Amplitude spectrogram
    ref : float
        Reference amplitude (default: 1.0)
    amin : float
        Minimum amplitude threshold
        
    Returns
    -------
    db : np.ndarray
        dB spectrogram
    """
    magnitude = np.abs(S)
    magnitude = np.maximum(amin, magnitude)
    db = 20.0 * np.log10(magnitude / ref)
    return db

def power_to_db(S, ref=1.0, amin=1e-12):
    """
    Convert power spectrogram to dB scale
    
    Args:
        S: numpy array, power spectrogram
        ref: float, reference power (default 1.0)
        amin: float, minimum power threshold
        
    Returns:
        dB spectrogram
    """
    S_safe = np.maximum(amin, S)
    return 10.0 * np.log10(S_safe / ref)