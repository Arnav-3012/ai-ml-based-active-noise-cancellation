"""Triage tool: for each of finetuned_v2's worst-scoring test pairs
(results/finetuned_v2_worst_pairs.csv), estimate WHY the pair scored badly --
leftover noise (under-suppression) vs. damaged speech (over-suppression) --
as a fast automated signal to point the next training fix, ahead of manual
listening.

METHOD (rough heuristic, NOT a validated perceptual metric -- read the
limitations below before trusting a single number):

1. Residual-noise estimate (proxy for "does it still sound noisy"):
   - STFT-magnitude the clean, noisy, and enhanced signals with the same
     window/hop as configs/finetune.yaml's diagnose_failures.stft (kept
     separate from baseline.stft so tuning one never silently changes the
     other).
   - `noise_only_mag = max(noisy_mag - clean_mag, 0)` per time-frequency
     bin -- an estimate of the noise component actually present in the
     noisy mixture (not a ground-truth noise reference, since Phase 1 does
     not persist the raw noise clip alongside the mix; this is a mixture
     subtraction estimate, not a real noise-only recording).
   - `residual_noise_estimate = mean(min(enhanced_mag, noise_only_mag))` --
     the enhanced output's magnitude in exactly the bins/frames where the
     mixture's own noise estimate says noise was present, capped at that
     noise estimate so it does not also count real speech energy that
     happens to coincide there. Normalized by the same file's noisy-signal
     total magnitude so the estimate is comparable across pairs of
     different loudness.

2. Speech-distortion estimate (proxy for "does the speech sound damaged"):
   - Restricted to (a) the speech band (diagnose_failures.speech_band_hz)
     and (b) clean-signal bins whose magnitude exceeds
     diagnose_failures.clean_energy_relative_floor * that file's own peak
     clean magnitude -- i.e. only bins where clean actually has real
     speech energy, so silence/pauses don't get scored as "distorted".
   - `speech_distortion_estimate` = mean absolute log-spectral distance,
     `|log(enhanced_mag + eps) - log(clean_mag + eps)|`, averaged over
     exactly those isolated bins. This is a simple LSD variant, not
     perceptually weighted (STOI/PESQ already provide the perceptual
     read -- this is a coarse, explainable triage signal, not a
     replacement for them).

LIMITATIONS (documented per task, read before acting on a single pair):
   - `noise_only_mag` is a linear-magnitude mixture-subtraction estimate,
     not a true noise reference -- it can be wrong when clean and noise
     overlap heavily in time-frequency (energy cancellation/aliasing in
     the subtraction), which biases residual_noise_estimate low exactly in
     the hardest (most overlapping) cases.
   - Neither estimate accounts for phase, temporal masking, or auditory
     perceptual weighting the way STOI/PESQ do -- two pairs with the same
     estimate value are not guaranteed equally perceptually similar.
   - The under/over label thresholds (dominant_share_threshold,
     min_combined_estimate) are fixed heuristic cutoffs, not
     statistically fitted classification boundaries -- treat the label as
     a triage pointer, not a certified diagnosis.
   - STFT resolution (n_fft/hop_length) trades time vs. frequency
     precision the same way it does everywhere else in signal
     processing -- both estimates inherit that resolution's blind spots.

Reads results/finetuned_v2_worst_pairs.csv (dedupes pair_id -- some pairs
rank worst on both PESQ and STOI). Writes results/failure_diagnosis.csv.
Prints an aggregate summary (count per label) and a shortlist of pairs
most worth manually listening to (least-confident/most-disagreeing labels,
plus the single worst-overall pair by PESQ).

Run standalone: `python -m src.eval.diagnose_failures` from repo root.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

CONFIG_PATH = "configs/finetune.yaml"


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_wav(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data[:, 0]


def _stft_mag(x: np.ndarray, n_fft: int, hop_length: int, win_length: int) -> np.ndarray:
    """Returns magnitude spectrogram, shape (n_freq_bins, n_frames)."""
    window = np.hanning(win_length).astype(np.float32)
    if win_length < n_fft:
        pad = n_fft - win_length
        window = np.pad(window, (pad // 2, pad - pad // 2))
    n_frames = 1 + (len(x) - n_fft) // hop_length if len(x) >= n_fft else 0
    if n_frames <= 0:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.float32)
    frames = np.stack(
        [x[i * hop_length : i * hop_length + n_fft] * window for i in range(n_frames)],
        axis=1,
    )
    spec = np.fft.rfft(frames, n=n_fft, axis=0)
    return np.abs(spec).astype(np.float32)


def _freq_bin_range(n_fft: int, sample_rate: int, band_hz: list) -> tuple:
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    lo, hi = band_hz
    idx = np.where((freqs >= lo) & (freqs <= hi))[0]
    return int(idx[0]), int(idx[-1]) + 1


def diagnose_pair(clean_path: Path, noisy_path: Path, enhanced_path: Path, cfg: dict) -> dict:
    sample_rate = cfg["sample_rate"]
    dcfg = cfg["diagnose_failures"]
    n_fft = dcfg["stft"]["n_fft"]
    hop_length = dcfg["stft"]["hop_length"]
    win_length = dcfg["stft"]["win_length"]
    band_lo, band_hi = dcfg["speech_band_hz"]
    clean_floor_ratio = dcfg["clean_energy_relative_floor"]

    clean = _load_wav(clean_path)
    noisy = _load_wav(noisy_path)
    enhanced = _load_wav(enhanced_path)
    n = min(len(clean), len(noisy), len(enhanced))
    clean, noisy, enhanced = clean[:n], noisy[:n], enhanced[:n]

    clean_mag = _stft_mag(clean, n_fft, hop_length, win_length)
    noisy_mag = _stft_mag(noisy, n_fft, hop_length, win_length)
    enhanced_mag = _stft_mag(enhanced, n_fft, hop_length, win_length)
    n_frames = min(clean_mag.shape[1], noisy_mag.shape[1], enhanced_mag.shape[1])
    clean_mag, noisy_mag, enhanced_mag = clean_mag[:, :n_frames], noisy_mag[:, :n_frames], enhanced_mag[:, :n_frames]

    eps = 1e-8

    # 1. Residual noise estimate
    noise_only_mag = np.maximum(noisy_mag - clean_mag, 0.0)
    residual_energy = np.minimum(enhanced_mag, noise_only_mag)
    residual_noise_estimate = float(residual_energy.mean() / (noisy_mag.mean() + eps))

    # 2. Speech distortion estimate, isolated to speech-band + real-speech-energy bins
    band_start, band_end = _freq_bin_range(n_fft, sample_rate, [band_lo, band_hi])
    clean_band = clean_mag[band_start:band_end, :]
    enhanced_band = enhanced_mag[band_start:band_end, :]
    clean_peak = clean_band.max() if clean_band.size else 0.0
    speech_mask = clean_band > (clean_floor_ratio * clean_peak + eps)
    if speech_mask.sum() > 0:
        lsd = np.abs(np.log(enhanced_band[speech_mask] + eps) - np.log(clean_band[speech_mask] + eps))
        speech_distortion_estimate = float(lsd.mean())
    else:
        speech_distortion_estimate = 0.0

    return {
        "residual_noise_estimate": residual_noise_estimate,
        "speech_distortion_estimate": speech_distortion_estimate,
    }


def label_pair(residual_noise_estimate: float, speech_distortion_estimate: float, cfg: dict) -> str:
    dcfg = cfg["diagnose_failures"]
    min_combined = dcfg["min_combined_estimate"]
    dominant_share = dcfg["dominant_share_threshold"]

    total = residual_noise_estimate + speech_distortion_estimate
    if total < min_combined:
        return "unclear"

    under_share = residual_noise_estimate / total
    over_share = speech_distortion_estimate / total

    if under_share >= dominant_share and over_share >= dominant_share:
        return "likely both"
    if under_share >= dominant_share:
        return "likely under-suppression"
    if over_share >= dominant_share:
        return "likely over-suppression"
    if under_share >= 0.4 and over_share >= 0.4:
        return "likely both"
    return "unclear"


def _read_worst_pairs(input_csv: Path) -> list:
    """Dedupes by pair_id -- a pair can appear in both the pesq and stoi rankings.
    Keeps the first occurrence (pesq ranking is written first) plus tracks the
    best (lowest) rank seen under either ranking, for shortlist tie-breaking."""
    seen = {}
    with open(input_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["pair_id"]
            if pid not in seen:
                seen[pid] = row
    return list(seen.values())


def run() -> None:
    cfg = load_config()
    dcfg = cfg["diagnose_failures"]
    input_csv = Path(dcfg["input_csv"])
    output_csv = Path(dcfg["output_csv"])
    shortlist_count = dcfg["shortlist_count"]

    rows = _read_worst_pairs(input_csv)
    print(f"[diagnose_failures] {len(rows)} unique pairs (deduped from {input_csv})")

    records = []
    for row in rows:
        pair_id = row["pair_id"]
        clean_path = Path(row["clean_path"])
        noisy_path = Path(row["noisy_path"])
        enhanced_path = Path(row["enhanced_path"])

        estimates = diagnose_pair(clean_path, noisy_path, enhanced_path, cfg)
        label = label_pair(estimates["residual_noise_estimate"], estimates["speech_distortion_estimate"], cfg)

        records.append(
            {
                "pair_id": pair_id,
                "residual_noise_estimate": estimates["residual_noise_estimate"],
                "speech_distortion_estimate": estimates["speech_distortion_estimate"],
                "label": label,
                "snr": float(row["snr"]),
                "stoi": float(row["stoi"]),
                "pesq": float(row["pesq"]),
                "clean_path": row["clean_path"],
                "noisy_path": row["noisy_path"],
                "enhanced_path": row["enhanced_path"],
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pair_id",
                "residual_noise_estimate",
                "speech_distortion_estimate",
                "label",
                "snr",
                "stoi",
                "pesq",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r["pair_id"],
                    r["residual_noise_estimate"],
                    r["speech_distortion_estimate"],
                    r["label"],
                    r["snr"],
                    r["stoi"],
                    r["pesq"],
                ]
            )
    print(f"[diagnose_failures] wrote {output_csv}")

    # --- Summary ---
    label_counts = defaultdict(int)
    for r in records:
        label_counts[r["label"]] += 1

    print(f"\n=== Summary ({len(records)} unique pairs) ===")
    for label in ["likely under-suppression", "likely over-suppression", "likely both", "unclear"]:
        n = label_counts.get(label, 0)
        print(f"{n}/{len(records)} {label}")

    # --- Shortlist for manual listening ---
    # Confidence = how decisively one mechanism dominates (larger = more
    # confident); pairs with lowest confidence, or labeled "unclear"/"both",
    # are exactly where automated triage is least reliable.
    def _confidence(r):
        total = r["residual_noise_estimate"] + r["speech_distortion_estimate"]
        if total <= 0:
            return 0.0
        return abs(r["residual_noise_estimate"] - r["speech_distortion_estimate"]) / total

    least_confident = sorted(records, key=_confidence)
    shortlist = []
    shortlist_ids = set()
    for r in least_confident:
        if len(shortlist) >= shortlist_count - 1:
            break
        shortlist.append(r)
        shortlist_ids.add(r["pair_id"])

    worst_overall = min(records, key=lambda r: r["pesq"])
    if worst_overall["pair_id"] not in shortlist_ids:
        shortlist.append(worst_overall)

    print(f"\n=== Shortlist for manual listening ({len(shortlist)} pairs) ===")
    print("(least-confident/most-disagreeing automated labels, plus the single worst-overall pair by PESQ)")
    for r in shortlist:
        tag = " <- worst overall (PESQ)" if r["pair_id"] == worst_overall["pair_id"] else ""
        print(
            f"  {r['pair_id']}: label={r['label']}, "
            f"residual_noise={r['residual_noise_estimate']:.4f}, "
            f"speech_distortion={r['speech_distortion_estimate']:.4f}, "
            f"pesq={r['pesq']:.3f}, stoi={r['stoi']:.3f}{tag}\n"
            f"    clean:    {r['clean_path']}\n"
            f"    noisy:    {r['noisy_path']}\n"
            f"    enhanced: {r['enhanced_path']}"
        )


if __name__ == "__main__":
    run()
