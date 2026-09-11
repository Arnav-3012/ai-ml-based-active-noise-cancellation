"""Phase 5a verification, isolated: separates true ONNX export drift from
the fixed-length crop/pad confound identified in verify_onnx.py's real run.

verify_onnx.py's delta (ONNX-on-cropped-input vs. PyTorch-on-full-length-
input) conflates two effects:
  1. Export drift -- does the ONNX graph compute the same thing as PyTorch?
  2. Crop effect -- 184/299 test clips (61.5%, per real manifest inspection)
     are longer than the 4.0s fixed export length, so center-cropping alone
     discards real content regardless of export quality.

This script isolates them by running BOTH PyTorch and ONNX on the exact
SAME cropped/padded input (same waveform array, same length, same clean
target) for every test pair, then reports two separate deltas:
  - delta(onnx, pytorch_cropped)   -- true export-only drift, same input both sides
  - delta(pytorch_cropped, pytorch_full_v5) -- the crop effect alone, isolated
    from export (pytorch_full_v5 = the already-reported real v5 PyTorch numbers)

Reuses src/model/inference.py's load_finetuned_model (same checkpoint-load
path used everywhere else in this project, no reimplementation) and
src/eval/metrics.py's snr/stoi_score/pesq_score (no duplicated metric
logic). Reuses verify_onnx.py's _fit_to_fixed_length so both scripts crop
identically.

Writes results/onnx_verification_isolated.csv and prints both deltas
side by side, plus an explicit verdict: if delta(onnx, pytorch_cropped) is
small (rounding-level) while delta(pytorch_cropped, pytorch_full_v5)
accounts for most of verify_onnx.py's original gap, that confirms the
crop effect -- not the export -- was the dominant cause, and it is safe to
treat the export as clean for Core ML/TFLite purposes. If delta(onnx,
pytorch_cropped) is itself large, that is real export drift and this
script says so explicitly -- STOP, do not proceed to Core ML/TFLite.

Run standalone: `python -m src.export.verify_onnx_isolated` from repo
root. Requires the same ONNX model as verify_onnx.py
(checkpoints/onnx/dns48_finetuned_v5.onnx, run `python -m src.export.to_onnx`
first) and the v5 PyTorch checkpoint (checkpoints/dns48_finetuned_v5_best.pt,
already required elsewhere in this project).
"""

import csv
import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

from src.eval.evaluate import CATEGORIES, TEST_MANIFEST, load_config
from src.eval.metrics import pesq_score, snr, stoi_score
from src.export.verify_onnx import V5_PYTORCH_OVERALL, ONNX_MODEL_PATH, _fit_to_fixed_length
from src.model.inference import load_finetuned_model, verify_finetuned_weights_differ

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_CSV = Path("results/onnx_verification_isolated.csv")
CHECKPOINT_PATH = Path("checkpoints/dns48_finetuned_v5_best.pt")


def _load_wav(path: Path):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data[:, 0], sr


def run(manifest_path: Path = TEST_MANIFEST) -> dict:
    cfg = load_config()
    sample_rate = cfg["sample_rate"]
    pesq_mode = cfg["eval"]["pesq_mode"]
    fixed_length = int(round(sample_rate * cfg["export"]["fixed_length_seconds"]))

    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(f"{ONNX_MODEL_PATH} not found -- run `python -m src.export.to_onnx` first.")

    logger.info(f"Loading ONNX model {ONNX_MODEL_PATH} via onnxruntime...")
    session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    logger.info(f"Loading PyTorch v5 checkpoint {CHECKPOINT_PATH}...")
    model, device, metadata = load_finetuned_model(cfg, CHECKPOINT_PATH)
    verify_finetuned_weights_differ(model, cfg, device)
    if device.type != "cpu":
        logger.info(f"Moving PyTorch model to CPU (matches ONNX's CPUExecutionProvider, apples-to-apples).")
        model = model.to("cpu")
        device = torch.device("cpu")
    logger.info(f"Loaded: {metadata}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    results = {"onnx": {cat: [] for cat in CATEGORIES + ["overall"]}, "pytorch_cropped": {cat: [] for cat in CATEGORIES + ["overall"]}}

    with torch.no_grad():
        for i, entry in enumerate(manifest):
            clean_path = Path(entry["clean_path"])
            noisy_path = Path(entry["noisy_path"])
            category = entry["noise_category"]

            clean, sr_c = _load_wav(clean_path)
            noisy, sr_n = _load_wav(noisy_path)
            if sr_c != sample_rate or sr_n != sample_rate:
                raise RuntimeError(f"sample rate mismatch: {clean_path}={sr_c}, {noisy_path}={sr_n}, expected {sample_rate}")

            noisy_fixed = _fit_to_fixed_length(noisy, fixed_length)
            clean_fixed = _fit_to_fixed_length(clean, fixed_length)

            # Same cropped input fed to both runtimes -- this is the entire point.
            onnx_input = noisy_fixed[np.newaxis, np.newaxis, :].astype(np.float32)
            onnx_enhanced = session.run(None, {input_name: onnx_input})[0][0, 0]

            pt_input = torch.from_numpy(noisy_fixed).to(device).unsqueeze(0).unsqueeze(0)
            pt_enhanced = model(pt_input)[0, 0].cpu().numpy()

            onnx_scores = {
                "snr": snr(clean_fixed, onnx_enhanced),
                "stoi": stoi_score(clean_fixed, onnx_enhanced, sample_rate),
                "pesq": pesq_score(clean_fixed, onnx_enhanced, sample_rate, pesq_mode),
            }
            pt_scores = {
                "snr": snr(clean_fixed, pt_enhanced),
                "stoi": stoi_score(clean_fixed, pt_enhanced, sample_rate),
                "pesq": pesq_score(clean_fixed, pt_enhanced, sample_rate, pesq_mode),
            }
            results["onnx"][category].append(onnx_scores)
            results["onnx"]["overall"].append(onnx_scores)
            results["pytorch_cropped"][category].append(pt_scores)
            results["pytorch_cropped"]["overall"].append(pt_scores)

            if (i + 1) % 50 == 0 or (i + 1) == len(manifest):
                print(f"[verify_onnx_isolated] {i + 1}/{len(manifest)} pairs scored")

    summary = _summarize(results)
    _write_csv(summary)
    _print_report(summary)
    return summary


def _summarize(results: dict) -> dict:
    summary = {}
    for col in ("onnx", "pytorch_cropped"):
        summary[col] = {}
        for cat in CATEGORIES + ["overall"]:
            scores = results[col][cat]
            n = len(scores)
            summary[col][cat] = {
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
        writer.writerow(["category", "column", "n", "snr_db", "stoi", "pesq"])
        for cat in CATEGORIES + ["overall"]:
            for col in ("onnx", "pytorch_cropped"):
                row = summary[col][cat]
                writer.writerow([cat, col, row["n"], row["snr"], row["stoi"], row["pesq"]])

        onnx_o = summary["onnx"]["overall"]
        pt_crop_o = summary["pytorch_cropped"]["overall"]
        writer.writerow([])
        writer.writerow(["metric", "onnx", "pytorch_cropped", "pytorch_v5_full", "export_drift(onnx-pytorch_cropped)", "crop_effect(pytorch_cropped-pytorch_v5_full)"])
        for metric in ("snr", "stoi", "pesq"):
            writer.writerow([
                metric,
                onnx_o[metric],
                pt_crop_o[metric],
                V5_PYTORCH_OVERALL[metric],
                onnx_o[metric] - pt_crop_o[metric],
                pt_crop_o[metric] - V5_PYTORCH_OVERALL[metric],
            ])
    print(f"\nWrote {RESULTS_CSV}")


def _print_report(summary: dict) -> None:
    print("\n=== Isolated comparison (test split, same cropped input both runtimes) ===")
    header = f"{'category':<11} {'column':<17} {'n':>5} {'SNR(dB)':>9} {'STOI':>7} {'PESQ':>7}"
    print(header)
    print("-" * len(header))
    for cat in CATEGORIES + ["overall"]:
        for col in ("onnx", "pytorch_cropped"):
            row = summary[col][cat]
            print(f"{cat:<11} {col:<17} {row['n']:>5} {row['snr']:>9.2f} {row['stoi']:>7.3f} {row['pesq']:>7.3f}")

    onnx_o = summary["onnx"]["overall"]
    pt_crop_o = summary["pytorch_cropped"]["overall"]

    print("\n=== Export drift: onnx vs. pytorch_cropped, SAME input both sides ===")
    print(f"{'metric':<8} {'onnx':>10} {'pytorch_cropped':>16} {'export_drift':>13}")
    export_drift = {}
    for metric, label in (("snr", "SNR(dB)"), ("stoi", "STOI"), ("pesq", "PESQ")):
        d = onnx_o[metric] - pt_crop_o[metric]
        export_drift[metric] = d
        print(f"{label:<8} {onnx_o[metric]:>10.4f} {pt_crop_o[metric]:>16.4f} {d:>+13.4f}")

    print("\n=== Crop effect: pytorch_cropped vs. v5's original full-length PyTorch eval ===")
    print(f"{'metric':<8} {'pytorch_cropped':>16} {'pytorch_v5_full':>16} {'crop_effect':>13}")
    crop_effect = {}
    for metric, label in (("snr", "SNR(dB)"), ("stoi", "STOI"), ("pesq", "PESQ")):
        d = pt_crop_o[metric] - V5_PYTORCH_OVERALL[metric]
        crop_effect[metric] = d
        print(f"{label:<8} {pt_crop_o[metric]:>16.4f} {V5_PYTORCH_OVERALL[metric]:>16.4f} {d:>+13.4f}")

    print("\n=== Verdict ===")
    max_export_drift = max(abs(v) for v in export_drift.values())
    if max_export_drift < 0.01 * max(abs(pt_crop_o["snr"]), 1.0) and abs(export_drift["stoi"]) < 0.005 and abs(export_drift["pesq"]) < 0.02:
        print(
            "Export drift is small/rounding-level on identical input -- the ONNX graph "
            "computes essentially the same thing as PyTorch. The gap verify_onnx.py "
            "originally reported vs. v5's full-length numbers is attributable to the "
            "crop-effect figures above, not the export itself. Safe to treat this export "
            "as clean for Core ML/TFLite conversion purposes (subject to the fixed-length "
            "input constraint carrying forward)."
        )
    else:
        print(
            "Export drift is NOT small on identical input -- this indicates a real "
            "numerical discrepancy between the ONNX graph and the PyTorch model, "
            "separate from the crop effect. STOP -- do not proceed to Core ML/TFLite "
            "conversion until this is resolved."
        )


if __name__ == "__main__":
    run()
