# Copyright © 2026 laelume All Rights Reserved.
# ==== ==== ====
# pysoniq/datasets/blobs.py
# Synthetic audio blob generator — mirrors sklearn.datasets.make_blobs API.
# Supports Euclidean, cosine, Wasserstein, and SRVF distance metrics.
# Frame segmentation: non-overlapping, fixed-length, derived from audio SR.
# ==== ==== ====

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np

# === === === === === === === === 
# T H I R D - P A R T Y   I M P O R T S
# === === === === === === === ===
try:
    import pysoniq
    from pysoniq.utils import load_segment
except ImportError as e:
    raise ImportError("pysoniq is required for make_audioblobs.") from e

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

# === === === === === === === === 
# L O G G E R
# === === === === === === === ===
log = logging.getLogger(__name__)

# === === === === === === === === 
# T Y P E S
# === === === === === === === ===
MetricLiteral = Literal["euclidean", "cosine", "wasserstein", "srvf"]
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
    Load a WAV file as a mono float32 array using pysoniq.

    Returns (waveform, sample_rate). Multi-channel audio is averaged to mono.
    """
    data, sr = pysoniq.load_audio(str(path), dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        log.debug("Multi-channel audio detected (%d ch), averaging to mono: %s", data.shape[1], path.name)
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    return data, sr


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
}


# ==== ==== ====
# P U B L I C   A P I
# ==== ==== ====

def make_audioblobs(
    source:      PathLike,
    n_fft:       int                  = 512,
    metric:      MetricLiteral        = "euclidean",
    n_classes:   Optional[int]        = None,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    verbose:     bool                 = False,
) -> tuple[np.ndarray, np.ndarray]:
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
        logging.basicConfig(level=logging.DEBUG)
        log.setLevel(logging.DEBUG)
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
            raise ValueError(
                f"n_classes={n_classes} exceeds discovered classes={len(all_labels)}"
            )
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
    y_parts: list[np.ndarray] = []

    for label, paths in wav_map.items():
        class_idx = label_encoder[label]
        for wav_path in paths:
            waveform, sr = _load_wav_mono(wav_path)
            log.debug("Loaded %s | SR=%d | samples=%d", wav_path.name, sr, len(waveform))

            frames = _segment_frames(waveform, sr, n_fft, hop_length=None)
            if frames.shape[0] == 0:
                continue

            projected = projector(frames)
            X_parts.append(projected)
            y_parts.append(np.full(len(projected), fill_value=class_idx, dtype=np.int64))

    if not X_parts:
        raise RuntimeError("No frames could be extracted from the provided audio source.")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

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
            f"  metric      : {metric}\n"
            f"  classes     : {len(all_labels)}\n"
            f"  X shape     : {X.shape}\n"
            f"  y shape     : {y.shape}\n"
            f"- - - - - - - - - - - - - - - - - - - -\n"
        )

    return X, y


# ==== ==== ====
# U S A G I
# ==== ==== ====
# from pysoniq.datasets.blobs import make_audioblobs
#
# X, y = make_audioblobs(
#     source       = "path/to/wav_dir",
#     n_fft        = 512,
#     metric       = "euclidean",
#     n_classes    = 10,
#     random_state = 42,
#     verbose      = True,
# )