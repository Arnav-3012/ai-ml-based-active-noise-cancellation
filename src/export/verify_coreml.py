"""Phase 5b verification: compares the converted Core ML model's output
against the ONNX baseline (Phase 5a, already verified at zero export drift
vs. PyTorch) on the SAME isolated test set / methodology.

Runs the .mlpackage via coremltools' Python prediction API (predict()) --
this is an on-Mac numerical check, it does NOT require the physical iOS
device. Uses the identical cropped/padded input convention as
verify_onnx_isolated.py (same fixed_length, same manifest, same metrics
module) so any delta reported here is attributable to the ONNX->Core ML
conversion step specifically, not to a different test methodology.

Reports delta vs. a baseline -- by default the isolated ONNX baseline
(STOI 0.670, PESQ 1.751, SNR 8.45dB overall, per Phase 5a's verify_onnx.py
real run on the fixed-length export), used for the fp32 Core ML
conversion's own verification. For the fp16 compute-precision isolation
measurement (Phase 5b follow-up), pass `mode="fp16"` (or `--fp16` on the
CLI) to verify the fp16 .mlpackage against the NOW-VERIFIED fp32 Core ML
numbers instead -- this isolates pure precision effect, separate from
conversion-format drift (already proven zero) and from quantization. For
the INT8 weights-only quantization isolation measurement, pass
`mode="int8"` (or `--int8`) to verify the INT8 .mlpackage against the same
fp32 baseline, isolating pure quantization effect, separate from both
conversion-format drift and fp16 precision effect. Same rigor/reporting
shape in every mode: per-category table, per-SNR-bucket table (to check
whether degradation correlates with harder/lower-SNR clips -- relevant to
the LSTM quantization-error hypothesis), an explicit delta section, then a
verdict.

Run standalone: `python -m src.export.verify_coreml` from repo root
(verifies the fp32 .mlpackage against the ONNX baseline). For the fp16
isolation check: `python -m src.export.verify_coreml --fp16`. For the INT8
isolation check: `python -m src.export.verify_coreml --int8`. Requires
`coremltools` and the relevant .mlpackage already converted
(`python -m src.export.to_coreml` / `--fp16`, or
`python -m src.export.quantize_coreml` for int8, first).
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf

from src.eval.evaluate import CATEGORIES, TEST_MANIFEST, _snr_bucket, load_config
from src.eval.metrics import pesq_score, snr, stoi_score
from src.export.verify_onnx import _fit_to_fixed_length

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_CSV = Path("results/coreml_verification.csv")
RESULTS_CSV_FP16 = Path("results/coreml_fp16_verification.csv")
RESULTS_CSV_INT8 = Path("results/coreml_int8_verification.csv")

# Phase 5a's real isolated ONNX-on-cropped-input overall numbers
# (src/export/verify_onnx.py real run, fixed-length export).
ONNX_ISOLATED_OVERALL = {"snr": 8.4508, "stoi": 0.6700, "pesq": 1.7505}

# Phase 5b's real, now-verified fp32 Core ML overall numbers (this script's
# own prior real run, verified at zero drift vs. ONNX_ISOLATED_OVERALL above)
# -- the baseline for the fp16 compute-precision isolation measurement AND
# the INT8 weights-only quantization isolation measurement (both compare
# against fp32, not against each other or against ONNX, so each precision-
# reduction step's cost is isolated from the others).
COREML_FP32_OVERALL = {"snr": 8.4508, "stoi": 0.6700, "pesq": 1.7505}


def _load_wav(path: Path):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data[:, 0], sr


_MODE_CONFIG = {
    "fp32": {
        "filename_key": "mlpackage_filename",
        "baseline": ONNX_ISOLATED_OVERALL,
        "baseline_label": "onnx_isolated_baseline",
        "results_csv": RESULTS_CSV,
        "to_coreml_cmd": "python -m src.export.to_coreml",
    },
    "fp16": {
        "filename_key": "mlpackage_filename_fp16",
        "baseline": COREML_FP32_OVERALL,
        "baseline_label": "coreml_fp32_baseline",
        "results_csv": RESULTS_CSV_FP16,
        "to_coreml_cmd": "python -m src.export.to_coreml --fp16",
    },
    "int8": {
        "filename_key": "mlpackage_filename_int8",
        "baseline": COREML_FP32_OVERALL,
        "baseline_label": "coreml_fp32_baseline",
        "results_csv": RESULTS_CSV_INT8,
        "to_coreml_cmd": "python -m src.export.quantize_coreml",
    },
}


def run(manifest_path: Path = TEST_MANIFEST, mode: str = "fp32") -> dict:
    if mode not in _MODE_CONFIG:
        raise ValueError(f"mode must be one of {list(_MODE_CONFIG)}, got {mode!r}")
    mode_cfg = _MODE_CONFIG[mode]

    cfg = load_config()
    sample_rate = cfg["sample_rate"]
    pesq_mode = cfg["eval"]["pesq_mode"]
    fixed_length = int(round(sample_rate * cfg["export"]["fixed_length_seconds"]))
    snr_edges = cfg["eval"]["snr_buckets_db"]
    bucket_labels = [_snr_bucket((snr_edges[i] + snr_edges[i + 1]) / 2, snr_edges) for i in range(len(snr_edges) - 1)]

    mlpackage_path = Path(cfg["coreml"]["mlpackage_dir"]) / cfg["coreml"][mode_cfg["filename_key"]]
    if not mlpackage_path.exists():
        raise FileNotFoundError(f"{mlpackage_path} not found -- run `{mode_cfg['to_coreml_cmd']}` first.")

    baseline = mode_cfg["baseline"]
    baseline_label = mode_cfg["baseline_label"]
    results_csv = mode_cfg["results_csv"]

    logger.info(f"Loading Core ML model {mlpackage_path} via coremltools (mode={mode})...")
    mlmodel = ct.models.MLModel(str(mlpackage_path))
    input_name = mlmodel.get_spec().description.input[0].name
    output_name = mlmodel.get_spec().description.output[0].name

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    results = {cat: [] for cat in CATEGORIES + ["overall"]}
    bucket_results = {b: [] for b in bucket_labels}

    for i, entry in enumerate(manifest):
        clean_path = Path(entry["clean_path"])
        noisy_path = Path(entry["noisy_path"])
        category = entry["noise_category"]
        bucket = _snr_bucket(entry["snr_db"], snr_edges)

        clean, sr_c = _load_wav(clean_path)
        noisy, sr_n = _load_wav(noisy_path)
        if sr_c != sample_rate or sr_n != sample_rate:
            raise RuntimeError(f"sample rate mismatch: {clean_path}={sr_c}, {noisy_path}={sr_n}, expected {sample_rate}")

        noisy_fixed = _fit_to_fixed_length(noisy, fixed_length)
        clean_fixed = _fit_to_fixed_length(clean, fixed_length)

        coreml_input = noisy_fixed[np.newaxis, np.newaxis, :].astype(np.float32)
        prediction = mlmodel.predict({input_name: coreml_input})
        enhanced = np.asarray(prediction[output_name])[0, 0]

        scores = {
            "snr": snr(clean_fixed, enhanced),
            "stoi": stoi_score(clean_fixed, enhanced, sample_rate),
            "pesq": pesq_score(clean_fixed, enhanced, sample_rate, pesq_mode),
        }
        results[category].append(scores)
        results["overall"].append(scores)
        bucket_results[bucket].append(scores)

        if (i + 1) % 50 == 0 or (i + 1) == len(manifest):
            print(f"[verify_coreml] {i + 1}/{len(manifest)} pairs scored")

    summary = _summarize(results, CATEGORIES + ["overall"])
    bucket_summary = _summarize(bucket_results, bucket_labels)
    _write_csv(summary, bucket_summary, bucket_labels, baseline, baseline_label, results_csv)
    _print_report(summary, bucket_summary, bucket_labels, baseline, baseline_label, mode)
    return summary


def _summarize(results: dict, keys: list) -> dict:
    summary = {}
    for key in keys:
        scores = results[key]
        n = len(scores)
        if n == 0:
            summary[key] = {"n": 0, "snr": float("nan"), "stoi": float("nan"), "pesq": float("nan")}
            continue
        summary[key] = {
            "n": n,
            "snr": sum(s["snr"] for s in scores) / n,
            "stoi": sum(s["stoi"] for s in scores) / n,
            "pesq": sum(s["pesq"] for s in scores) / n,
        }
    return summary


def _write_csv(summary: dict, bucket_summary: dict, bucket_labels: list, baseline: dict, baseline_label: str, results_csv: Path) -> None:
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "n", "snr_db", "stoi", "pesq"])
        for cat in CATEGORIES + ["overall"]:
            row = summary[cat]
            writer.writerow([cat, row["n"], row["snr"], row["stoi"], row["pesq"]])

        writer.writerow([])
        writer.writerow(["snr_bucket", "n", "snr_db", "stoi", "pesq"])
        for bucket in bucket_labels:
            row = bucket_summary[bucket]
            writer.writerow([bucket, row["n"], row["snr"], row["stoi"], row["pesq"]])

        overall = summary["overall"]
        writer.writerow([])
        writer.writerow(["metric", "coreml", baseline_label, f"delta(coreml-{baseline_label})"])
        for metric in ("snr", "stoi", "pesq"):
            writer.writerow([
                metric,
                overall[metric],
                baseline[metric],
                overall[metric] - baseline[metric],
            ])
    print(f"\nWrote {results_csv}")


def _print_report(summary: dict, bucket_summary: dict, bucket_labels: list, baseline: dict, baseline_label: str, mode: str) -> None:
    print(f"\n=== Core ML verification, mode={mode} (test split, fixed-length cropped input) ===")
    header = f"{'category':<11} {'n':>5} {'SNR(dB)':>9} {'STOI':>7} {'PESQ':>7}"
    print(header)
    print("-" * len(header))
    for cat in CATEGORIES + ["overall"]:
        row = summary[cat]
        print(f"{cat:<11} {row['n']:>5} {row['snr']:>9.2f} {row['stoi']:>7.3f} {row['pesq']:>7.3f}")

    print(f"\n=== Per-SNR-bucket breakdown (mode={mode}) ===")
    print(
        "Not a per-metric delta vs. baseline (bucket-level baseline numbers were never "
        "separately recorded) -- this shows whether THIS mode's own scores degrade "
        "unevenly across input-SNR buckets. Relevant to the LSTM-quantization-error "
        "hypothesis: if error compounds over the LSTM's longer effective context, "
        "harder/lower-SNR buckets (where more temporal context matters for separation) "
        "would be expected to show comparatively larger drops than easier/high-SNR "
        "buckets, rather than a uniform drop across buckets."
    )
    bhdr = f"{'snr_bucket':<12} {'n':>5} {'SNR(dB)':>9} {'STOI':>7} {'PESQ':>7}"
    print(bhdr)
    print("-" * len(bhdr))
    for bucket in bucket_labels:
        row = bucket_summary[bucket]
        print(f"{bucket:<12} {row['n']:>5} {row['snr']:>9.2f} {row['stoi']:>7.3f} {row['pesq']:>7.3f}")

    overall = summary["overall"]
    print(f"\n=== Delta vs. {baseline_label} (same fixed-length methodology) ===")
    print(f"{'metric':<8} {'coreml':>10} {baseline_label:>18} {'delta':>10}")
    delta = {}
    for metric, label in (("snr", "SNR(dB)"), ("stoi", "STOI"), ("pesq", "PESQ")):
        d = overall[metric] - baseline[metric]
        delta[metric] = d
        print(f"{label:<8} {overall[metric]:>10.4f} {baseline[metric]:>18.4f} {d:>+10.4f}")

    print("\n=== Verdict ===")
    if abs(delta["snr"]) < 0.05 and abs(delta["stoi"]) < 0.005 and abs(delta["pesq"]) < 0.02:
        print(
            f"Delta is small/rounding-level vs. {baseline_label} -- no meaningful drift "
            "introduced at this step."
        )
    else:
        print(
            f"Delta vs. {baseline_label} is NOT small -- this indicates a real numerical "
            "discrepancy introduced at this step. Report this exactly, do not proceed "
            "further until it's understood."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fp16", action="store_true", help="Verify the fp16 .mlpackage against the fp32 Core ML baseline.")
    group.add_argument("--int8", action="store_true", help="Verify the INT8 weights-only-quantized .mlpackage against the fp32 Core ML baseline.")
    args = parser.parse_args()
    run_mode = "fp16" if args.fp16 else "int8" if args.int8 else "fp32"
    run(mode=run_mode)
