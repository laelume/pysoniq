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


def mel_filterbank(
    sr:     int,
    n_fft:  int,
    n_mels: int   = 128,
    fmin:   float = 0.0,
    fmax:   float = None,
) -> np.ndarray:
    """
    Construct a triangular mel filterbank matrix.

    Returns a matrix of shape (n_mels, n_fft // 2 + 1) mapping linear
    FFT bins to mel-spaced filter outputs.
    Reference: O'Shaughnessy (1987). Speech Communication.
    DOI: https://doi.org/10.1016/0167-6393(87)90050-3
    """
    if fmax is None:
        fmax = sr / 2.0

    # linear frequency bins
    linear_freqs = fft_frequencies(sr, n_fft)           # (n_fft // 2 + 1,)
    n_bins       = len(linear_freqs)

    # mel center frequencies including boundary bins
    mel_min  = hz_to_mel(fmin)
    mel_max  = hz_to_mel(fmax)
    mel_pts  = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts   = mel_to_hz(mel_pts)                        # (n_mels + 2,)

    # construct triangular filters
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_left   = hz_pts[m - 1]
        f_center = hz_pts[m]
        f_right  = hz_pts[m + 1]

        for k in range(n_bins):
            f = linear_freqs[k]
            if f_left <= f <= f_center:
                fb[m - 1, k] = (f - f_left) / (f_center - f_left + 1e-10)
            elif f_center < f <= f_right:
                fb[m - 1, k] = (f_right - f) / (f_right - f_center + 1e-10)

    return fb


def mfcc(
    y:          np.ndarray,
    sr:         int,
    n_mfcc:     int   = 40,
    n_mels:     int   = 128,
    n_fft:      int   = 1024,
    hop_length: int   = None,
    fmin:       float = 0.0,
    fmax:       float = None,
    window:     str   = "hann",
) -> np.ndarray:
    """
    Compute Mel-Frequency Cepstral Coefficients (MFCCs) from a waveform.

    Pipeline: STFT -> power spectrogram -> mel filterbank -> log -> DCT -> n_mfcc coefficients.
    Returns array of shape (n_mfcc, n_frames).
    """
    from scipy.fft import dct

    if hop_length is None:
        hop_length = n_fft // 4

    if fmax is None:
        fmax = sr / 2.0

    # === === === === === === === === 
    # P O W E R   S P E C T R O G R A M
    # === === === === === === === ===
    Zxx      = stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
    power    = np.abs(Zxx) ** 2                          # (n_fft//2+1, n_frames)

    # === === === === === === === === 
    # M E L   F I L T E R B A N K
    # === === === === === === === ===
    fb       = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_spec = fb @ power                                 # (n_mels, n_frames)

    # log compression
    mel_spec = np.log(mel_spec + 1e-10)

    # === === === === === === === === 
    # D C T
    # === === === === === === === ===
    # type-II DCT along mel axis, keep first n_mfcc coefficients
    coeffs   = dct(mel_spec, type=2, axis=0, norm="ortho")
    return coeffs[:n_mfcc, :]                            # (n_mfcc, n_frames)



def mel_spectrogram(
    y:          np.ndarray,
    sr:         int,
    n_mels:     int   = 128,
    n_fft:      int   = 1024,
    hop_length: int   = None,
    fmin:       float = 0.0,
    fmax:       float = None,
    window:     str   = "hann",
) -> np.ndarray:
    """
    Compute a mel-scale power spectrogram from a waveform.

    Returns linear power shape (n_mels, n_frames) — dB conversion left to caller.
    Reuses STFT -> power -> mel filterbank pipeline from mfcc, stops before log and DCT.
    """
    if hop_length is None:
        hop_length = n_fft // 4

    if fmax is None:
        fmax = sr / 2.0

    Zxx      = stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
    power    = np.abs(Zxx) ** 2
    fb       = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    return (fb @ power).astype(np.float32)              # (n_mels, n_frames)






















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