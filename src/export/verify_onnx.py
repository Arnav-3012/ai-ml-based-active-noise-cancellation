"""Phase 5a verification: runs the exported ONNX model (via onnxruntime) over
the full test split, computes SNR/STOI/PESQ the same way evaluate.py does,
and reports the exact delta against v5's known PyTorch numbers.

This is the concrete verification step -- export succeeding (no exception)
is NOT proof of quality parity. LSTM ONNX exports in particular can produce
a graph that runs but diverges numerically from the PyTorch model. This
script does not assume a clean export; it measures one.

Uses the FIXED-length ONNX export (checkpoints/onnx/dns48_finetuned_v5.onnx
by default) since that is the known-working path per to_onnx.py. Because
the model was exported at a fixed length (export.fixed_length_seconds), test
clips are center-cropped/padded to that same fixed length before ONNX
inference -- this must be reported explicitly, since it means this
verification is not identical-input to v5's original PyTorch evaluation
(which ran on full variable-length clips). If the dynamic-length export also
succeeded, this script will note that a variable-length verification is
possible as a follow-up, but does not run it automatically.

Reuses src/eval/metrics.py's snr/stoi_score/pesq_score (no duplicated metric
logic) and src/eval/evaluate.py's _snr_bucket/_summarize for the breakdown
structure.

Writes results/onnx_verification.csv and prints:
  - overall/per-category ONNX-model metrics
  - the known v5 PyTorch numbers (hardcoded from the real, verified,
    already-reported evaluate.py run: STOI 0.674, PESQ 1.786, SNR 8.75dB
    overall -- update these three constants if v5's official PyTorch numbers
    are later relocated to config)
  - the exact delta (onnx - pytorch) per metric

Run standalone: `python -m src.export.verify_onnx` from repo root. Requires
`onnx`/`onnxruntime` installed and checkpoints/onnx/dns48_finetuned_v5.onnx
to exist (run `python -m src.export.to_onnx` first).
"""

import csv
import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from src.eval.evaluate import CATEGORIES, TEST_MANIFEST, load_config
from src.eval.metrics import pesq_score, snr, stoi_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ONNX_MODEL_PATH = Path("checkpoints/onnx/dns48_finetuned_v5.onnx")
RESULTS_CSV = Path("results/onnx_verification.csv")

# v5 PyTorch numbers, overall (n=299) -- from the already-reported,
# user-verified v5 evaluate.py run. Hardcoded per explicit task instruction
# ("compare directly against v5's PyTorch numbers: STOI 0.674, PESQ 1.786,
# SNR 8.75dB overall"); not re-derived here since re-running the PyTorch
# eval is not part of this script's scope.
V5_PYTORCH_OVERALL = {"snr": 8.75, "stoi": 0.674, "pesq": 1.786}


def _load_wav(path: Path):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data[:, 0], sr


def _fit_to_fixed_length(audio: np.ndarray, length: int) -> np.ndarray:
    """Center-crop or zero-pad to the ONNX model's fixed export length.
    Explicitly logged per-call-site (not silent) since this changes the
    input relative to v5's original variable-length PyTorch evaluation."""
    n = audio.shape[0]
    if n == length:
        return audio
    if n > length:
        start = (n - length) // 2
        return audio[start : start + length]
    pad_total = length - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(audio, (pad_left, pad_right), mode="constant")


def run(manifest_path: Path = TEST_MANIFEST) -> dict:
    cfg = load_config()
    sample_rate = cfg["sample_rate"]
    pesq_mode = cfg["eval"]["pesq_mode"]
    snr_edges = cfg["eval"]["snr_buckets_db"]
    fixed_length = int(round(sample_rate * cfg["export"]["fixed_length_seconds"]))

    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{ONNX_MODEL_PATH} not found -- run `python -m src.export.to_onnx` first."
        )

    logger.info(f"Loading ONNX model {ONNX_MODEL_PATH} via onnxruntime...")
    session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    logger.warning(
        f"ONNX model was exported at a FIXED length ({fixed_length} samples = "
        f"{cfg['export']['fixed_length_seconds']}s). All test clips are being "
        "center-cropped/zero-padded to this length before inference -- this is "
        "NOT identical input to v5's original PyTorch evaluation, which ran on "
        "full variable-length clips. This delta computation includes that effect, "
        "not export-only numerical drift. Reported explicitly, not hidden."
    )

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    results = {"onnx": {cat: [] for cat in CATEGORIES + ["overall"]}}

    for i, entry in enumerate(manifest):
        clean_path = Path(entry["clean_path"])
        noisy_path = Path(entry["noisy_path"])
        category = entry["noise_category"]

        clean, sr_c = _load_wav(clean_path)
        noisy, sr_n = _load_wav(noisy_path)
        if sr_c != sample_rate or sr_n != sample_rate:
            raise RuntimeError(f"sample rate mismatch: {clean_path}={sr_c}, {noisy_path}={sr_n}, expected {sample_rate}")

        noisy_fixed = _fit_to_fixed_length(noisy, fixed_length)
        onnx_input = noisy_fixed[np.newaxis, np.newaxis, :].astype(np.float32)
        enhanced = session.run(None, {input_name: onnx_input})[0][0, 0]

        clean_fixed = _fit_to_fixed_length(clean, fixed_length)

        scores = {
            "snr": snr(clean_fixed, enhanced),
            "stoi": stoi_score(clean_fixed, enhanced, sample_rate),
            "pesq": pesq_score(clean_fixed, enhanced, sample_rate, pesq_mode),
        }
        results["onnx"][category].append(scores)
        results["onnx"]["overall"].append(scores)

        if (i + 1) % 50 == 0 or (i + 1) == len(manifest):
            print(f"[verify_onnx] {i + 1}/{len(manifest)} pairs scored")

    summary = _summarize_onnx(results)

    _write_csv(summary)
    _print_report(summary)

    return summary


def _summarize_onnx(results: dict) -> dict:
    summary = {"onnx": {}}
    for cat in CATEGORIES + ["overall"]:
        scores = results["onnx"][cat]
        if not scores:
            summary["onnx"][cat] = {"n": 0, "snr": None, "stoi": None, "pesq": None}
            continue
        n = len(scores)
        summary["onnx"][cat] = {
            "n": n,
            "snr": sum(s["snr"] for s in scores) / n,
            "stoi": sum(s["stoi"] for s in scores) / n,
            "pesq": sum(s["pesq"] for s in scores) / n,
        }
    return summary


def _write_csv(summary: dict) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "n", "snr_db", "stoi", "pesq"])
        for cat in CATEGORIES + ["overall"]:
            row = summary["onnx"][cat]
            writer.writerow([cat, row["n"], row["snr"], row["stoi"], row["pesq"]])
        overall = summary["onnx"]["overall"]
        writer.writerow([])
        writer.writerow(["metric", "onnx_overall", "pytorch_v5_overall", "delta_onnx_minus_pytorch"])
        for metric in ("snr", "stoi", "pesq"):
            writer.writerow([metric, overall[metric], V5_PYTORCH_OVERALL[metric], overall[metric] - V5_PYTORCH_OVERALL[metric]])
    print(f"\nWrote {RESULTS_CSV}")


def _print_report(summary: dict) -> None:
    print("\n=== ONNX model results (test split) ===")
    header = f"{'category':<11} {'n':>5} {'SNR(dB)':>9} {'STOI':>7} {'PESQ':>7}"
    print(header)
    print("-" * len(header))
    for cat in CATEGORIES + ["overall"]:
        row = summary["onnx"][cat]
        if row["n"] == 0:
            print(f"{cat:<11} {row['n']:>5} {'--':>9} {'--':>7} {'--':>7}")
        else:
            print(f"{cat:<11} {row['n']:>5} {row['snr']:>9.2f} {row['stoi']:>7.3f} {row['pesq']:>7.3f}")

    overall = summary["onnx"]["overall"]
    print("\n=== ONNX vs. v5 PyTorch (overall, n=299) -- exact delta ===")
    print(f"{'metric':<8} {'onnx':>10} {'pytorch_v5':>12} {'delta':>10}")
    for metric, label in (("snr", "SNR(dB)"), ("stoi", "STOI"), ("pesq", "PESQ")):
        onnx_val = overall[metric]
        pt_val = V5_PYTORCH_OVERALL[metric]
        delta = onnx_val - pt_val if onnx_val is not None else None
        print(f"{label:<8} {onnx_val:>10.4f} {pt_val:>12.4f} {delta:>+10.4f}")

    print(
        "\nIf any |delta| is more than a small rounding-level gap, STOP -- do not "
        "proceed to Core ML/TFLite conversion on a broken export. Report the gap "
        "instead (and note the fixed-length-crop caveat above as a possible "
        "confound, not just export drift)."
    )


if __name__ == "__main__":
    run()
