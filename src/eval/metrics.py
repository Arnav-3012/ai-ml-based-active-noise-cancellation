"""Speech enhancement evaluation metrics: SNR, STOI, PESQ.

All three functions take matching-length, matching-sample-rate (16kHz per
Phase 1's output format) clean and enhanced/estimate waveforms as 1D numpy
arrays or torch tensors, and return a single scalar (float).
"""

import numpy as np
from pesq import pesq as _pesq
from pystoi import stoi as _stoi


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "numpy"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def snr(clean, enhanced, eps: float = 1e-8) -> float:
    """Plain-English: how much louder is the actual speech than what's left over (noise + distortion) -- higher dB means cleaner audio."""
    clean = _to_numpy(clean)
    enhanced = _to_numpy(enhanced)
    n = min(len(clean), len(enhanced))
    clean, enhanced = clean[:n], enhanced[:n]

    noise = enhanced - clean
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum(noise ** 2)
    return float(10 * np.log10((signal_power + eps) / (noise_power + eps)))


def stoi_score(clean, enhanced, sample_rate: int = 16000) -> float:
    """Plain-English: predicts how intelligible the speech would sound to a human listener, from 0 (unintelligible) to 1 (perfectly clear) -- it's about understanding words, not just sounding clean."""
    clean = _to_numpy(clean)
    enhanced = _to_numpy(enhanced)
    n = min(len(clean), len(enhanced))
    clean, enhanced = clean[:n], enhanced[:n]

    return float(_stoi(clean, enhanced, sample_rate, extended=False))


def pesq_score(clean, enhanced, sample_rate: int = 16000, mode: str = "wb") -> float:
    """Plain-English: an ITU-standard score (roughly 1-4.5) approximating how a human on a phone call would rate the audio quality -- the industry-standard 'does this sound good' number."""
    clean = _to_numpy(clean)
    enhanced = _to_numpy(enhanced)
    n = min(len(clean), len(enhanced))
    clean, enhanced = clean[:n], enhanced[:n]

    return float(_pesq(sample_rate, clean, enhanced, mode))
