import numpy as np
from scipy import signal
from scipy.io import wavfile

"""
Audio processing utilities using scipy/numpy only
"""

# def load(filepath, sr=None):
#     """
#     Load audio file and optionally resample
    
#     Parameters
#     ----------
#     filepath : str
#         Path to WAV file
#     sr : int or None
#         Target sample rate. If None, uses native sample rate
        
#     Returns
#     -------
#     y : np.ndarray
#         Audio time series (float32, normalized to [-1, 1])
#     sr : int
#         Sample rate
#     """
#     native_sr, y = wavfile.read(filepath)
    
#     # Convert to float and normalize
#     if y.dtype == np.int16:
#         y = y.astype(np.float32) / 32768.0
#     elif y.dtype == np.int32:
#         y = y.astype(np.float32) / 2147483648.0
#     elif y.dtype == np.uint8:
#         y = (y.astype(np.float32) - 128) / 128.0
#     elif y.dtype == np.float32 or y.dtype == np.float64:
#         y = y.astype(np.float32)
    
#     # Convert stereo to mono
#     if len(y.shape) > 1:
#         y = np.mean(y, axis=1)
    
#     # Resample if needed
#     if sr is not None and sr != native_sr:
#         num_samples = int(len(y) * sr / native_sr)
#         y = signal.resample(y, num_samples)
#         return y, sr
    
#     return y, native_sr


def stft(y, n_fft=1024, hop_length=None, window='hann'):
    """
    Short-time Fourier transform
    
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


def fft_frequencies(sr, n_fft):
    """
    Frequencies corresponding to FFT bins
    
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

