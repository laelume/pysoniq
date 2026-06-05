# Copyright © 2026 laelume All Rights Reserved.
# ==== ==== ====
# pysoniq/audioblobs.py
# Audio blob generator — mirrors sklearn.datasets.make_blobs API.
# Supports Euclidean, cosine, Wasserstein, and SRVF distance metrics.
# Frame segmentation: non-overlapping, fixed-length, derived from audio SR.
# ==== ==== ====

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional, Union
import numpy as np
from joblib import Parallel, delayed

from .io import load_audio

try:
    import fdasrsf.utility_functions as fda_uf
    _SRVF_AVAILABLE = True
except ImportError:
    _SRVF_AVAILABLE = False

try:
    from scipy.stats import wasserstein_distance
    _WASSERSTEIN_AVAILABLE = True
except ImportError:
    _WASSERSTEIN_AVAILABLE = False

try:
    from tslearn.metrics import dtw as _dtw
    _DTW_AVAILABLE = True
except ImportError:
    _DTW_AVAILABLE = False



# === === === === === === === === 
# L O G G E R
# === === === === === === === ===
log = logging.getLogger(__name__)

# === === === === === === === === 
# T Y P E S
# === === === === === === === ===
MetricLiteral = Literal["euclidean", "cosine", "wasserstein", "srvf", "dtw"]
PathLike      = Union[str, Path]


# ==== ==== ====
# I N T E R N A L   U T I L I T I E S
# ==== ==== ====

def _discover_wavs(root: Path) -> dict[str, list[Path]]:
    """
    Recursively discover WAV files under root and mirror directory tree as class labels.

    Returns a dict mapping class label strings to lists of WAV Paths.
    Single-file input is mapped to a single class '0'.
    """
    root = Path(root)

    if root.is_file():
        if root.suffix.lower() != ".wav":
            raise ValueError(f"Single-file input must be a WAV file, got: {root.suffix}")
        log.debug("Single-file mode: %s", root)
        return {"0": [root]}

    if not root.is_dir():
        raise ValueError(f"Input path is neither a file nor a directory: {root}")

    wav_map: dict[str, list[Path]] = {}
    for wav_path in sorted(root.rglob("*.wav")):
        # relative parent path used as class label to mirror input tree
        rel = wav_path.parent.relative_to(root)
        label = str(rel) if str(rel) != "." else wav_path.stem
        wav_map.setdefault(label, []).append(wav_path)

    if not wav_map:
        raise ValueError(f"No WAV files found under: {root}")

    log.debug(
        "Discovered %d class(es), %d total WAV files.",
        len(wav_map),
        sum(len(v) for v in wav_map.values()),
    )
    return wav_map


def _load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """
    Load a WAV file as a mono float32 array.

    Returns (waveform, sample_rate). Multi-channel audio is averaged to mono.
    """
    # pysoniq.load_audio returns float32 already; shape is (n,) for mono or (n, n_channels) for multi-channel
    data, sr = load_audio(str(path))
    if data.ndim == 2:
        if data.shape[1] > 1:
            log.debug("Multi-channel audio detected (%d ch), averaging to mono: %s", data.shape[1], path.name)
            data = data.mean(axis=1)
        else:
            # single-channel stored as (n, 1) → squeeze to (n,)
            data = data[:, 0]
    # cast applied unconditionally so the return dtype is stable regardless of upstream loader changes
    data = np.ascontiguousarray(data, dtype=np.float32)

    return data, sr


# === === === === === === === === 
# P H O N E M E   L E V E L   A N A L Y S I S
# frame segmentation produces one row per fixed-length time slice —
# this operates at sub-syllable resolution and is suited for
# phoneme-level discovery and within-syllable structure analysis.
# for syllable-level analysis, use aggregation='mean' in make_audioblobs
# to pool frames into a single vector per file.
# === === === === === === === ===

def _segment_frames(waveform: np.ndarray, sr: int, n_fft: int, hop_length: Optional[int]) -> np.ndarray:
    """
    Segment a waveform into non-overlapping fixed-length frames.

    Frame length is n_fft samples. hop_length is reserved for future use
    but must equal n_fft (non-overlapping constraint enforced here).
    Returns array of shape (n_frames, n_fft).
    """
    # non-overlapping: hop == frame length
    frame_len = n_fft
    n_frames  = len(waveform) // frame_len
    if n_frames == 0:
        log.warning(
            "Waveform length %d is shorter than frame length %d — skipping.",
            len(waveform), frame_len
        )
        return np.empty((0, frame_len), dtype=np.float32)

    trimmed = waveform[: n_frames * frame_len]
    frames  = trimmed.reshape(n_frames, frame_len)
    log.debug("Segmented %d frames of length %d from waveform of %d samples.", n_frames, frame_len, len(waveform))
    return frames


# ==== ==== ====
# D I S T A N C E   P R O J E C T I O N
# ==== ==== ====

def _project_euclidean(frames: np.ndarray) -> np.ndarray:
    """
    Return frames as-is for Euclidean metric — no projection required.

    Identity pass-through; included for API uniformity.
    """
    return frames


def _project_cosine(frames: np.ndarray) -> np.ndarray:
    """
    L2-normalise frames so that Euclidean distance in projected space equals cosine distance.

    Frames with zero norm are left as-is to avoid division by zero.
    Reference: Cunningham & Ghahramani (2015), Linear Dimensionality Reduction.
    """
    norms = np.linalg.norm(frames, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return frames / norms


def _project_wasserstein(frames: np.ndarray) -> np.ndarray:
    """
    Project frames to CDF representation for Wasserstein-1 embedding.

    Each frame is treated as a 1D distribution; its sorted CDF is the
    quantile function. L2 distance between quantile functions approximates
    the 1D Wasserstein-1 distance.
    Reference: Kolouri et al. (2017), Optimal Mass Transport.
    DOI: 10.1109/MSP.2017.2695801
    """
    # sort each frame to obtain empirical quantile function
    return np.sort(frames, axis=1)


def _project_srvf(frames: np.ndarray) -> np.ndarray:
    """
    Project frames to SRVF (Square Root Velocity Function) representation.

    Uses fdasrsf elastic_distance indirectly by converting each frame to
    its SRVF. L2 distance in SRVF space approximates elastic shape distance.
    method='DP2', lam=0.0 per pipeline convention.
    Reference: Srivastava et al. (2011), Shape Analysis of Elastic Curves.
    DOI: 10.1109/TPAMI.2011.49
    """
    if not _SRVF_AVAILABLE:
        raise ImportError("fdasrsf is required for SRVF metric.")

    srvf_frames = np.zeros_like(frames)
    for i, f in enumerate(frames):
        # finite difference approximation of derivative
        q       = np.gradient(f)
        # SRVF: sign(q) * sqrt(|q|)
        srvf    = np.sign(q) * np.sqrt(np.abs(q))
        srvf_frames[i] = srvf
    log.debug("SRVF projection applied to %d frames.", len(frames))
    return srvf_frames


_METRIC_PROJECTORS = {
    "euclidean":   _project_euclidean,
    "cosine":      _project_cosine,
    "wasserstein": _project_wasserstein,
    "srvf":        _project_srvf,
    # DTW is not a projection — handled via precomputed distance matrix in _compute_distance_matrix
    "dtw":         _project_euclidean,  # identity passthrough — DTW uses precomputed D
}


# === === === === === === === === 
# Frame-level features
# === === === === === === === ===

def _extract_frame_features(
    waveform: np.ndarray,
    sr:       int,
    n_fft:    int,
    feature:  str,
    n_mfcc:   int,
) -> np.ndarray:
    """
    Extract feature matrix from a waveform according to the feature type.

    Supported feature types:
      - 'raw':         non-overlapping frames of n_fft samples → (n_frames, n_fft)
      - 'mfcc':        Mel-frequency cepstral coefficients → (n_frames, n_mfcc)
      - 'psd':         Welch power spectral density → (n_frames, n_fft//2 + 1)
      - 'stft':        Short-time Fourier transform magnitude → (n_frames, n_fft//2 + 1)
      - 'spectrogram': log-power spectrogram (dB) → (n_frames, n_fft//2 + 1)
      - 'mel_spectrogram': mel-scale log-power spectrogram (dB) → (n_frames, n_mels)
    """
    # RAW: time-domain frames
    if feature == "raw":
        return _segment_frames(waveform, sr, n_fft, hop_length=None)

    # MFCC: cepstral coefficients
    elif feature == "mfcc":
        from .fourier import mfcc as _mfcc
        coeffs = _mfcc(
            y          = waveform,
            sr         = sr,
            n_mfcc     = n_mfcc,
            n_fft      = n_fft,
            hop_length = n_fft,
        )
        # coeffs shape: (n_mfcc, n_frames) -> transpose to (n_frames, n_mfcc)
        return coeffs.T.astype(np.float32)

    # PSD: Welch power spectral density per frame
    elif feature == "psd":
        from scipy.signal import welch
        frames     = _segment_frames(waveform, sr, n_fft, hop_length=None)
        if len(frames) == 0:
            return np.empty((0, n_fft // 2 + 1), dtype=np.float32)
        # nperseg matches frame length to obtain one PSD vector per frame
        psd_frames = np.zeros((len(frames), n_fft // 2 + 1), dtype=np.float32)
        for i, frame in enumerate(frames):
            _, psd     = welch(frame, fs=sr, nperseg=n_fft, scaling="density")
            psd_frames[i] = psd.astype(np.float32)
        log.debug("PSD features extracted: shape=%s", psd_frames.shape)
        return psd_frames

    # STFT: short-time Fourier transform magnitude
    elif feature == "stft":
        from .fourier import stft as _stft
        # non-overlapping window — hop equals frame length
        S          = _stft(y=waveform, n_fft=n_fft, hop_length=n_fft)
        # S shape: (n_freq, n_frames) -> magnitude -> transpose to (n_frames, n_freq)
        mag        = np.abs(S).T.astype(np.float32)
        log.debug("STFT magnitude features extracted: shape=%s", mag.shape)
        return mag

    # SPECTROGRAM: log-power spectrogram (dB)
    elif feature == "spectrogram":
        from .fourier import stft as _stft
        S          = _stft(y=waveform, n_fft=n_fft, hop_length=n_fft)
        # power |S|^2 then dB conversion with epsilon for numerical stability
        eps        = 1e-10
        power      = np.abs(S) ** 2
        spec_db    = (10.0 * np.log10(power + eps)).T.astype(np.float32)
        log.debug("Log-power spectrogram extracted: shape=%s", spec_db.shape)
        return spec_db

    # MEL SPECTROGRAM: perceptually-weighted log-power spectrogram
    elif feature == "mel_spectrogram":
        from .fourier import mel_spectrogram as _mel_spec
        # n_mels reuses n_mfcc parameter as the mel filterbank dimension
        # FALLBACK: n_mels defaults to 128 when n_mfcc unset — replace when n_mels
        # is added as a first-class parameter to the signature
        n_mels     = n_mfcc if n_mfcc and n_mfcc > 13 else 128
        mel_spec   = _mel_spec(
            y          = waveform,
            sr         = sr,
            n_fft      = n_fft,
            hop_length = n_fft,
            n_mels     = n_mels,
        )
        # convert power to dB with epsilon floor for numerical stability
        eps        = 1e-10
        mel_db     = (10.0 * np.log10(mel_spec + eps)).T.astype(np.float32)
        log.debug("Mel spectrogram extracted: shape=%s, n_mels=%d", mel_db.shape, n_mels)
        return mel_db

    # UNSUPPORTED FEATURE
    else:
        raise ValueError(
            f"Unsupported feature='{feature}'. "
            f"Choose from: ['raw', 'mfcc', 'psd', 'stft', 'spectrogram', 'mel_spectrogram']"
        )






# === === === === === === === === 
# S Y L L A B L E   L E V E L   F E A T U R E S
# computes feature sequence over full waveform as one unit —
# no fixed-length frame segmentation. each syllable yields a
# (n_frames, n_features) sequence for DTW-based distance computation.
# === === === === === === === ===

def _extract_syllable_features(
    waveform: np.ndarray,
    sr:       int,
    n_fft:    int,
    feature:  str,
    n_mfcc:   int,
) -> np.ndarray:
    """
    Extract a feature sequence from a full waveform for syllable-level analysis.

    Unlike _extract_features, no fixed-length segmentation is applied —
    the full waveform is analyzed as one unit with hop_length=n_fft//4
    to produce a temporal sequence of feature vectors.
    Returns (n_frames, n_features) where n_frames varies per syllable.
    Intended for use with DTW-based distance metrics.
    Reference: Sakoe & Chiba (1978). DOI: https://doi.org/10.1109/TASSP.1978.1163055
    """
    if feature == "raw":
        # full waveform as one frame — shape (1, n_samples)
        return waveform.reshape(1, -1).astype(np.float32)    
    
    elif feature == "mfcc":
        from .fourier import mfcc as _mfcc
        coeffs = _mfcc(
            y          = waveform,
            sr         = sr,
            n_mfcc     = n_mfcc,
            n_fft      = n_fft,
            hop_length = n_fft // 4,   # overlapping for temporal resolution
        )
        # (n_mfcc, n_frames) -> (n_frames, n_mfcc)
        return coeffs.T.astype(np.float32)

    elif feature == "psd":
        from scipy.signal import welch
        _, psd = welch(waveform, fs=sr, nperseg=n_fft, scaling="density")
        # single PSD vector for full waveform — shape (1, n_fft//2+1)
        return psd.reshape(1, -1).astype(np.float32)

    elif feature == "stft":
        from .fourier import stft as _stft
        S   = _stft(y=waveform, n_fft=n_fft, hop_length=n_fft // 4)
        return np.abs(S).T.astype(np.float32)

    elif feature == "spectrogram":
        from .fourier import stft as _stft
        S      = _stft(y=waveform, n_fft=n_fft, hop_length=n_fft // 4)
        eps    = 1e-10
        return (10.0 * np.log10(np.abs(S) ** 2 + eps)).T.astype(np.float32)

    elif feature == "mel_spectrogram":
        from .fourier import mel_spectrogram as _mel_spec
        n_mels   = n_mfcc if n_mfcc and n_mfcc > 13 else 128
        mel_spec = _mel_spec(
            y          = waveform,
            sr         = sr,
            n_fft      = n_fft,
            hop_length = n_fft // 4,
            n_mels     = n_mels,
        )
        eps    = 1e-10
        mel_db = (10.0 * np.log10(mel_spec + eps)).T.astype(np.float32)
        log.debug("Mel spectrogram (syllable) extracted: shape=%s", mel_db.shape)
        return mel_db

    else:
        raise ValueError(
            f"_extract_syllable_features: unsupported feature='{feature}'. "
            f"Choose from: ['mfcc', 'psd', 'stft', 'spectrogram']"
        )






# Shared distance matrix computation across all clustering algos

def _compute_distance_matrix(
    X:      np.ndarray,
    metric: MetricLiteral,
) -> np.ndarray:
    """
    Compute a pairwise distance matrix for non-native metrics.

    Wasserstein uses scipy.stats.wasserstein_distance (1D, row-wise).
    SRVF uses fdasrsf elastic_distance with method='DP2', lam=0.0.
    Reference (Wasserstein): Kolouri et al. (2017). DOI: https://doi.org/10.1109/MSP.2017.2695801
    Reference (SRVF): Srivastava et al. (2011). DOI: https://doi.org/10.1109/TPAMI.2011.49
    """
    n = len(X)
    D = np.zeros((n, n), dtype=np.float64)
    
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    log.debug("%s: computing %d pairs in parallel.", metric.upper(), len(pairs))

    if metric == "wasserstein":
        if not _WASSERSTEIN_AVAILABLE:
            raise ImportError("scipy is required for Wasserstein distance.")
        from scipy.stats import wasserstein_distance as _wass

        def _pair_fn(i: int, j: int) -> tuple[int, int, float]:
            """Compute Wasserstein-1 distance for a single pair of 1D distributions."""
            return i, j, float(_wass(X[i], X[j]))

        prefer = "threads"

    elif metric == "srvf":
        if not _SRVF_AVAILABLE:
            raise ImportError("fdasrsf is required for SRVF distance.")
        from fdasrsf.utility_functions import elastic_distance as _elastic

        def _pair_fn(i: int, j: int) -> tuple[int, int, float]:
            """Compute SRVF elastic distance for a single pair. method=DP2, lam=0.0."""
            # FALLBACK: scalar elastic_distance — replace with parallel
            # batch kernel from srvf_core.py when integrated
            return i, j, float(_elastic(
                X[i].astype(np.float64),
                X[j].astype(np.float64),
                method="DP2",
                lam=0.0,
            ))

        prefer = "threads"

    elif metric == "dtw":
        if not _DTW_AVAILABLE:
            raise ImportError("tslearn is required for DTW metric.")
        X_seq = [X[i].reshape(-1, 1).astype(np.float64) for i in range(n)]

        def _pair_fn(i: int, j: int) -> tuple[int, int, float]:
            """Compute DTW distance for a single pair of sequences."""
            return i, j, float(_dtw(X_seq[i], X_seq[j]))

        prefer = "threads"

    else:
        raise ValueError(
            f"_compute_distance_matrix: unsupported metric='{metric}'. "
            f"Choose from: ['wasserstein', 'srvf', 'dtw']"
        )

    results = Parallel(n_jobs=-1, prefer=prefer)(
        delayed(_pair_fn)(i, j) for i, j in pairs
    )

    for i, j, d in results:
        D[i, j] = d
        D[j, i] = d

    log.debug("%s distance matrix computed: shape=%s", metric.upper(), D.shape)


    return D











# === === === === === === === === 
# C L U S T E R I N G  M E T H O D S
# === === === === === === === ===

def _cluster_kmedoids(
    X:        np.ndarray,
    n_classes: int,
    metric:   MetricLiteral,
) -> np.ndarray:
    """
    Assign cluster labels to X using kmedoids.

    For Euclidean and cosine metrics, metric is forwarded directly.
    For Wasserstein and SRVF, a precomputed distance matrix is computed
    first and passed as metric='precomputed'.
    Reference: Kaufman & Rousseeuw (1990). DOI: https://doi.org/10.1002/9780470316801
    """
    try:
        from kmedoids import KMedoids
    except ImportError as e:
        raise ImportError("kmedoids is required for kmedoids clustering.") from e

    if metric in ("euclidean", "cosine"):
        log.debug("kmedoids: forwarding metric='%s' directly.", metric)
        km = KMedoids(n_clusters=n_classes, metric=metric, random_state=0)
        return km.fit_predict(X)

    # === === === === === === === === 
    # P R E C O M P U T E D   D I S T A N C E
    # === === === === === === === ===
    log.debug("kmedoids: computing precomputed distance matrix for metric='%s'.", metric)
    D = _compute_distance_matrix(X, metric)
    km = KMedoids(n_clusters=n_classes, metric="precomputed", random_state=0)
    return km.fit_predict(D)



def _cluster_hdbscan(
    X:          np.ndarray,
    metric:     MetricLiteral,
    min_cluster_size: int = 5,
) -> np.ndarray:
    """
    Assign cluster labels to X using HDBSCAN.

    Noise points (label -1) are retained as a dedicated class.
    n_classes is not applicable for HDBSCAN — cluster count is data-driven.
    For Wasserstein and SRVF, a precomputed distance matrix is used.
    Reference: Campello et al. (2013). DOI: https://doi.org/10.1007/978-3-642-37456-2_14
    """
    try:
        import hdbscan
    except ImportError as e:
        raise ImportError("hdbscan is required for HDBSCAN clustering.") from e

    if metric in ("euclidean", "cosine"):
        log.debug("HDBSCAN: forwarding metric='%s' directly.", metric)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size = min_cluster_size,
            metric           = metric,
        )
        labels = clusterer.fit_predict(X)

    else:
        # === === === === === === === === 
        # P R E C O M P U T E D   D I S T A N C E
        # === === === === === === === ===
        log.debug("HDBSCAN: computing precomputed distance matrix for metric='%s'.", metric)
        D = _compute_distance_matrix(X, metric)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size = min_cluster_size,
            metric           = "precomputed",
        )
        labels = clusterer.fit_predict(D)

    n_noise    = np.sum(labels == -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    log.debug(
        "HDBSCAN: found %d clusters, %d noise points (label=-1 retained).",
        n_clusters, n_noise,
    )
    return labels


def _cluster_gmm(
    X:               np.ndarray,
    n_classes:       int,
    covariance_type: str = "full",
    random_state:    Optional[Union[int, np.random.RandomState]] = None,
) -> np.ndarray:
    """
    Assign cluster labels to X using a Gaussian Mixture Model.

    Soft posterior probabilities are computed then collapsed to hard
    integer labels via argmax to preserve the (X, y) contract.
    GMM operates in Euclidean space only — metric projection is applied
    upstream via _METRIC_PROJECTORS before this call.
    Reference: Reynolds (2009). https://www.semanticscholar.org/paper/Gaussian-Mixture-Models-Reynolds/2b6d8e5e61ef407f9e14571419d2c3e5cac9fabc
    """
    from sklearn.mixture import GaussianMixture

    log.debug(
        "GMM: n_classes=%d, covariance_type='%s'.",
        n_classes, covariance_type,
    )
    gmm = GaussianMixture(
        n_components    = n_classes,
        covariance_type = covariance_type,
        random_state    = random_state,
    )
    gmm.fit(X)
    labels = gmm.predict(X)
    log.debug("GMM: label distribution=%s", np.bincount(labels).tolist())
    return labels



def _cluster_fuzzy_cmeans(
    X:         np.ndarray,
    n_classes: int,
    m:         float = 2.0,
) -> np.ndarray:
    """
    Assign cluster labels to X using Fuzzy C-Means.

    Soft membership matrix is computed then collapsed to hard integer
    labels via argmax to preserve the (X, y) contract.
    Fuzziness exponent m controls cluster overlap — m=2 is standard.
    Operates in Euclidean space only — metric projection applied upstream.
    Reference: Bezdek (1981). DOI: https://doi.org/10.1007/978-1-4757-0450-1
    """
    try:
        import skfuzzy as fuzz
    except ImportError as e:
        raise ImportError("scikit-fuzzy is required for Fuzzy C-Means clustering.") from e

    log.debug("FuzzyCMeans: n_classes=%d, m=%.2f.", n_classes, m)

    # skfuzzy expects (n_features, n_samples) — transpose required
    X_T = X.T

    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data     = X_T,
        c        = n_classes,
        m        = m,
        error    = 1e-6,
        maxiter  = 1000,
    )

    # u shape: (n_classes, n_samples) — argmax over class axis
    labels = np.argmax(u, axis=0).astype(np.int64)
    log.debug("FuzzyCMeans: label distribution=%s", np.bincount(labels).tolist())
    return labels



def _cluster_spectral(
    X:           np.ndarray,
    n_classes:   int,
    metric:      MetricLiteral,
    gamma:       Optional[float] = None,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> np.ndarray:
    """
    Assign cluster labels to X using Spectral Clustering.

    For Euclidean and cosine, uses RBF kernel affinity (sklearn default).
    For Wasserstein and SRVF, computes precomputed distance matrix D,
    converts to affinity via A = exp(-gamma * D), passes affinity='precomputed'.
    gamma defaults to 1.0 / n_features if not specified.
    Reference: Ng, Jordan & Weiss (2001).
    https://proceedings.neurips.cc/paper/2001/hash/801272ee79cfde7fa5960571fee36b9b-Abstract.html
    """
    from sklearn.cluster import SpectralClustering

    # default gamma mirrors sklearn internal RBF default
    _gamma = gamma if gamma is not None else 1.0 / X.shape[1]
    log.debug("Spectral: n_classes=%d, metric='%s', gamma=%.6f.", n_classes, metric, _gamma)

    if metric in ("euclidean", "cosine"):
        # === === === === === === === === 
        # S T A N D A R D   R B F   K E R N E L
        # default affinity='rbf' computes exp(-gamma * ||x-y||^2) internally
        # to incorporate a custom distance matrix instead:
        #   D = _compute_distance_matrix(X, metric)
        #   A = np.exp(-_gamma * D)
        #   sc = SpectralClustering(affinity="precomputed", ...)
        #   return sc.fit_predict(A)
        # === === === === === === === ===
        sc = SpectralClustering(
            n_clusters   = n_classes,
            affinity     = "rbf",
            gamma        = _gamma,
            random_state = random_state,
        )
        return sc.fit_predict(X)

    # === === === === === === === === 
    # P R E C O M P U T E D   A F F I N I T Y
    # for Wasserstein and SRVF: distance -> affinity via RBF transform
    # === === === === === === === ===
    D = _compute_distance_matrix(X, metric)
    A = np.exp(-_gamma * D)
    log.debug("Spectral: affinity matrix computed from precomputed D, shape=%s.", A.shape)

    sc = SpectralClustering(
        n_clusters   = n_classes,
        affinity     = "precomputed",
        random_state = random_state,
    )
    return sc.fit_predict(A)





# === === === === === === === === 
# C L U S T E R I N G   R E G I S T R Y
# === === === === === === === ===

_CLUSTERING_ALGORITHMS = {
    "kmedoids": "kmedoids.kmedoids",
    "hdbscan":  "hdbscan.HDBSCAN",
    "gmm":      "sklearn.mixture.GaussianMixture",
    "fuzzy":    "skfuzzy.cluster.cmeans",
    "spectral": "sklearn.cluster.SpectralClustering",
}


def _dispatch_clustering(
    X:                np.ndarray,
    clustering:       str,
    metric:           MetricLiteral,
    n_classes:        Optional[int],
    min_cluster_size: int,
    covariance_type:  str,
    m:                float,
    gamma:            Optional[float],
    random_state:     Optional[Union[int, np.random.RandomState]],
) -> np.ndarray:
    """
    Dispatch clustering algorithm by name and return integer label array.

    n_classes is not applicable for HDBSCAN — pass None to indicate
    data-driven cluster count. All other algorithms require n_classes.
    """
    if clustering != "hdbscan" and n_classes is None:
        raise ValueError(
            f"n_classes=None is only valid for clustering='hdbscan'. "
            f"Got clustering='{clustering}'."
        )

    if clustering == "kmedoids":
        return _cluster_kmedoids(
            X         = X,
            n_classes = n_classes,
            metric    = metric,
        )

    elif clustering == "hdbscan":
        if n_classes is not None:
            log.warning(
                "n_classes=%d passed to HDBSCAN — ignored. "
                "Cluster count is data-driven. Pass n_classes=None to suppress.",
                n_classes,
            )
        return _cluster_hdbscan(
            X                = X,
            metric           = metric,
            min_cluster_size = min_cluster_size,
        )

    elif clustering == "gmm":
        return _cluster_gmm(
            X               = X,
            n_classes       = n_classes,
            covariance_type = covariance_type,
            random_state    = random_state,
        )

    elif clustering == "fuzzy":
        return _cluster_fuzzy_cmeans(
            X         = X,
            n_classes = n_classes,
            m         = m,
        )

    elif clustering == "spectral":
        return _cluster_spectral(
            X            = X,
            n_classes    = n_classes,
            metric       = metric,
            gamma        = gamma,
            random_state = random_state,
        )

    else:
        raise ValueError(
            f"Unknown clustering='{clustering}'. "
            f"Choose from: {list(_CLUSTERING_ALGORITHMS.keys())}"
        )















# ==== ==== ====
# P U B L I C   A P I
# ==== ==== ====

"""
Inspired by make_blobs, which synthesizes a point cloud of gaussians clustered around prededfined centers. 
This adapts the concept to read audio samples and cluster them into a defined number of classes based on 
a defined clustering algorithm and distance metric. 
Support is initialized for clustering algorithms: k-medioids, hdbscan, fuzzy c-means, 
and metrics: euclidea, cosine, wasserstein, srvf. 
"""

def make_audioblobs(
    source:           PathLike,
    n_fft:            int                  = 512,
    metric:           MetricLiteral        = "srvf", # or "dtw"
    n_classes:        Optional[int]        = None,
    clustering:       str                  = "kmedoids",
    min_cluster_size: int                  = 5,
    covariance_type:  str                  = "full",
    m:                float                = 2.0,
    gamma:            Optional[float]      = None,
    feature:          str                  = "raw",
    n_mfcc:           int                  = 40,
    aggregation:      Optional[str]        = "syllable",
    random_state:     Optional[Union[int, np.random.RandomState]] = None,
    return_metadata:  bool                 = False,
    verbose:          bool                 = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict] :
    """
    Build a (X, y) dataset from WAV audio in the shape expected by make_blobs consumers.

    Discovers WAV files at source (file or directory), segments each into
    non-overlapping frames of n_fft samples, applies metric projection,
    and returns (X, y) where X is (n_samples, n_fft) and y is integer class labels.
    Class labels are derived from the directory tree under source.
    Optionally subsample to n_classes if fewer classes are needed.

    Parameters
    ----------
    source : path to a single WAV file or a directory of WAV files.
    n_fft : frame length in samples. SR is read natively from each file.
    metric : one of 'euclidean', 'cosine', 'wasserstein', 'srvf'.
    n_classes : if set, subsample to this many classes randomly.
    random_state : seed or RandomState for reproducibility.
    verbose : enable verbose debug logging.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_fft)
    y : np.ndarray, shape (n_samples,), integer class labels
    """
    # === === === === === === === === 
    # L O G G I N G   S E T U P
    # === === === === === === === ===
    if verbose:
        log.setLevel(logging.DEBUG)
        if not log.handlers:
            _h = logging.StreamHandler()
            _h.setLevel(logging.DEBUG)
            log.addHandler(_h)
    else:
        log.setLevel(logging.WARNING)

    if metric not in _METRIC_PROJECTORS:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from: {list(_METRIC_PROJECTORS)}")

    rng = (
        random_state if isinstance(random_state, np.random.RandomState)
        else np.random.RandomState(random_state)
    )

    # === === === === === === === === 
    # D I S C O V E R Y
    # === === === === === === === ===
    wav_map = _discover_wavs(Path(source))
    log.debug("WAV map keys (classes): %s", list(wav_map.keys()))

    # optionally subsample classes
    all_labels = sorted(wav_map.keys())
    if n_classes is not None:
        if n_classes > len(all_labels):
            log.warning(
                "n_classes=%d exceeds discovered sequence count=%d — clamping.",
                n_classes, len(all_labels),
            )
            n_classes = len(all_labels)
        
        selected = rng.choice(all_labels, size=n_classes, replace=False).tolist()
        
        wav_map  = {k: wav_map[k] for k in selected}
        all_labels = selected
        log.debug("Subsampled to %d classes: %s", n_classes, selected)

    label_encoder = {lbl: idx for idx, lbl in enumerate(all_labels)}
    projector     = _METRIC_PROJECTORS[metric]

    # === === === === === === === === 
    # F R A M E   E X T R A C T I O N
    # === === === === === === === ===
    X_parts: list[np.ndarray] = []
    meta_individ: list[str] = []
    meta_paths:   list[str] = []    
    y_parts: list[np.ndarray] = []

    for label, paths in wav_map.items():
        class_idx = label_encoder[label]
        for wav_path in paths:
            waveform, sr = _load_wav_mono(wav_path)
            seq_id        = wav_path.parent.name
            individ_name  = seq_id.split("_")[0] if "_" in seq_id else seq_id
            log.debug("Loaded %s | SR=%d | samples=%d", wav_path.name, sr, len(waveform))

            # # frames = _segment_frames(waveform, sr, n_fft, hop_length=None)
            # frames = _extract_frame_features(waveform, sr, n_fft, feature, n_mfcc)
            # if frames.shape[0] == 0:
            #     continue

            # # metric projection applied to raw frames only —
            # # MFCC features are already in a meaningful metric space
            # projected = projector(frames) if feature == "raw" else frames
            # # projected = projector(frames)


            # === === === === === === === === 
            # S Y L L A B L E   /   F R A M E   L E V E L   S P L I T
            # aggregation='syllable' + metric='dtw' → one sequence per file,
            # distance matrix computed via DTW — syllable-level default.
            # aggregation=None → frame-level (phoneme) segmentation.
            # === === === === === === === ===
            if aggregation == "syllable":
                # syllable level: full waveform as one temporal sequence
                frames = _extract_syllable_features(waveform, sr, n_fft, feature, n_mfcc)
            elif aggregation in ("frame", None):
                # PHONEME LEVEL: fixed-length frame segmentation
                frames = _extract_frame_features(waveform, sr, n_fft, feature, n_mfcc)
            else:
                raise ValueError(
                    f"Unsupported aggregation='{aggregation}'. "
                    f"Choose from: ['syllable', 'frame', None]"
                )

            if frames.shape[0] == 0:
                continue

            projected = projector(frames) if feature == "raw" else frames

            X_parts.append(projected)

            n_rows = projected.shape[0]
            meta_individ.extend([individ_name] * n_rows)
            meta_paths.extend([str(wav_path)] * n_rows)
            
            # y_parts.append(np.full(len(projected), fill_value=class_idx, dtype=np.int64))
            # ^^ label assignment deferred to clustering dispatch — no y accumulation here

    if not X_parts:
        raise RuntimeError("No frames could be extracted from the provided audio source.")

    
    
    # === === === === === === === === 
    # C L U S T E R I N G   L A B E L   A S S I G N M E N T
    # directory-derived labels from _discover_wavs are discarded here —
    # semantic class labels are assigned by the clustering algorithm
    # operating on the feature space of extracted frames
    # === === === === === === === ===
    
    X = np.vstack(X_parts)
    # y = np.concatenate(y_parts)

    log.debug(
        "Dispatching clustering: algorithm='%s', metric='%s', n_classes=%s.",
        clustering, metric, n_classes,
    )

    metadata = {
        "individ_name": np.array(meta_individ),
        "source_path":  np.array(meta_paths),
    }

    # y = _dispatch_clustering(
    #     X                = X,
    #     clustering       = clustering,
    #     metric           = metric,
    #     n_classes        = n_classes,
    #     min_cluster_size = min_cluster_size,
    #     covariance_type  = covariance_type,
    #     m                = m,
    #     gamma            = gamma,
    #     random_state     = random_state,
    # )

    # === === === === === === === === 
    # D T W   P R E C O M P U T E D   D I S T A N C E
    # for syllable-level DTW, X rows are flattened sequences —
    # pairwise DTW distance matrix computed before clustering dispatch
    # === === === === === === === ===
    if metric == "dtw" and aggregation not in ("frame", None):
        log.debug("DTW syllable-level: computing pairwise distance matrix.")
        D = _compute_distance_matrix(X, metric="dtw")
        y = _dispatch_clustering(
            X                = D,
            clustering       = clustering,
            metric           = "precomputed",
            n_classes        = n_classes,
            min_cluster_size = min_cluster_size,
            covariance_type  = covariance_type,
            m                = m,
            gamma            = gamma,
            random_state     = random_state,
        )
    else:
        y = _dispatch_clustering(
            X                = X,
            clustering       = clustering,
            metric           = metric,
            n_classes        = n_classes,
            min_cluster_size = min_cluster_size,
            covariance_type  = covariance_type,
            m                = m,
            gamma            = gamma,
            random_state     = random_state,
        )


    log.debug(
        "Final dataset: X=%s, y=%s, classes=%d, metric=%s",
        X.shape, y.shape, len(all_labels), metric
    )

    # === === === === === === === === 
    # S U M M A R Y
    # === === === === === === === ===
    if verbose:
        print(
            f"\n- - - - - - - - - - - - - - - - - - - -\n"
            f"  M A K E _ A U D I O B L O B S   S U M M A R Y\n"
            f"- - - - - - - - - - - - - - - - - - - -\n"
            f"  source      : {source}\n"
            f"  n_fft       : {n_fft}\n"
            f"  feature     : {feature}\n"
            f"  n_mfcc      : {n_mfcc if feature == 'mfcc' else 'n/a'}\n"
            f"  metric      : {metric}\n"
            f"  classes     : {len(all_labels)}\n"
            f"  X shape     : {X.shape}\n"
            f"  y shape     : {y.shape}\n"
            f"- - - - - - - - - - - - - - - - - - - -\n"
        )


    if return_metadata:
        return X, y, metadata
    return X, y
    


# ==== ==== ====
# U S A G I
# ==== ==== ====
# from pysoniq import make_audioblobs
#
# X, y = make_audioblobs(
#     source       = "path/to/wav_dir",
#     n_fft        = 512,
#     metric       = "euclidean",
#     n_classes    = 10,
#     random_state = 42,
#     verbose      = True,
# )


# from pysoniq import make_audioblobs

# # === === === === === === === === 
# # D A T A   S O U R C E
# # === === === === === === === ===
# wav_dir = Path("wav_dir")  # replace with actual syllable directory

# X_true, y_true = make_audioblobs(
#     source       = wav_dir,
#     n_fft        = 512,
#     metric       = "euclidean",
#     n_classes    = 10,
#     random_state = 0,
#     verbose      = True,
# )