# pysoniq/fourier.py

"""Minimal-dependency audio processing and fourier transform utilities"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
from .utils import hz_to_mel, mel_to_hz, linear_to_db


# === === === === === === === ===
# module-level native sample rate state
# populated by set_native_sr(); read silently by sr-dependent functions
# === === === === === === === ===

_NATIVE_SR = None


def set_native_sr(sr):
    """Set module-level native sample rate for silent default in sr-dependent functions.
    
    Parameters
    ----------
    sr : int
        Sample rate in Hz to bind as the native default.
    """
    global _NATIVE_SR
    _NATIVE_SR = sr


def get_native_sr():
    """Return current module-level native sample rate, or None if unset.
    
    Returns
    -------
    int or None
        Bound native sample rate.
    """
    return _NATIVE_SR


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


def fft_frequencies(sr=None, n_fft=None, filepath=None):
    """
    Frequencies corresponding to linear Hz bins.

    Parameters
    ----------
    sr : int, optional
        Sample rate in Hz. If None, resolves in order: filepath probe, then module native sr.
    n_fft : int
        FFT size.
    filepath : str or Path, optional
        Audio file path. If provided and sr is None, sample rate is extracted.

    Returns
    -------
    freqs : np.ndarray
        Frequency array.

    Raises
    ------
    ValueError
        If sr unresolvable (no sr, no filepath, no module native sr), or n_fft is None.
    """
    if n_fft is None:
        raise ValueError("n_fft is required")
    
    if sr is None:
        if filepath is not None:
            from pysoniq.io import _sample_rate
            sr = _sample_rate(filepath)
        elif _NATIVE_SR is not None:
            sr = _NATIVE_SR
        else:
            raise ValueError("sr unresolvable: provide sr, filepath, or call set_native_sr()")
    
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


def psd(y, sr, n_fft=1024, nperseg=None):
    """
    Compute power spectral density via Welch method.

    Parameters
    ----------
    y : np.ndarray
        Audio time series.
    sr : int
        Sample rate in Hz.
    n_fft : int
        FFT size for Welch segments.
    nperseg : int or None
        Length of each segment. If None, defaults to n_fft.

    Returns
    -------
    np.ndarray
        PSD vector of shape (n_fft // 2 + 1,).
    """
    if nperseg is None:
        nperseg = n_fft
    
    freqs, psd_vals = signal.welch(y, fs=sr, nperseg=nperseg, nfft=n_fft)
    return psd_vals.astype(np.float32)


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


def frames_to_time(frames, sr=None, hop_length=None, filepath=None):
    """
    Convert frame indices to time in seconds.

    Parameters
    ----------
    frames : np.ndarray or int
        Frame indices.
    sr : int, optional
        Sample rate in Hz. If None, filepath must be provided.
    hop_length : int
        Hop length in samples.
    filepath : str or Path, optional
        Audio file path. If provided and sr is None, sample rate is extracted.

    Returns
    -------
    times : np.ndarray or float
        Time in seconds.

    Raises
    ------
    ValueError
        If both sr and filepath are None, or hop_length is None.
    """
    if sr is None:
        if filepath is not None:
            from .io import _sample_rate
            sr = _sample_rate(filepath)
        elif _NATIVE_SR is not None:
            sr = _NATIVE_SR
        else:
            raise ValueError("sr unresolvable: provide sr, filepath, or call set_native_sr()")
    
    if hop_length is None:
        raise ValueError("hop_length is required")
    
    return frames * hop_length / sr



def frames_to_time_positions(segments, sr=None, hop_length=None, filepath=None, verbose=False):
    """
    Enrich segment dicts with frame-index bounds derived from time bounds.

    Converts 'start_s' and 'end_s' keys in segment dicts to 'start_frame' and 'end_frame'.
    Used in UNSEGMENTED mode annotation pipelines to map temporal positions to STFT frame indices
    for feature slicing.

    Parameters
    ----------
    segments : list of dict
        Each dict must contain 'start_s', 'end_s', and 'index' keys.
    sr : int, optional
        Sample rate in Hz. If None, filepath must be provided.
    hop_length : int
        STFT hop length in samples.
    filepath : str or Path, optional
        Audio file path. If provided and sr is None, sample rate is extracted.
    verbose : bool
        Enable verbose logging.

    Returns
    -------
    list of dict
        Input dicts augmented with 'start_frame' and 'end_frame' keys.

    Raises
    ------
    ValueError
        If both sr and filepath are None, or hop_length is None.
    """
    if sr is None:
        if filepath is not None:
            from .io import _sample_rate
            sr = _sample_rate(filepath)
        elif _NATIVE_SR is not None:
            sr = _NATIVE_SR
        else:
            raise ValueError("sr unresolvable: provide sr, filepath, or call set_native_sr()")
    
    if hop_length is None:
        raise ValueError("hop_length is required")

    
    updated = []
    for seg in segments:
        entry = dict(seg)
        entry["start_frame"] = int(seg["start_s"] * sr / hop_length)
        entry["end_frame"] = int(seg["end_s"] * sr / hop_length)
        if verbose:
            print(f"[fourier] [frames_to_time_positions] index={seg['index']} | "
                  f"start_frame={entry['start_frame']} | end_frame={entry['end_frame']}")
        updated.append(entry)
    return updated


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