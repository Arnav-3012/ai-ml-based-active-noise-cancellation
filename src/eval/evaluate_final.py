"""Demo/PPT-focused comparison: noisy (floor) / baseline (classical) / finetuned_v5 only.

Reuses evaluate.py's metric-computation and summarization logic (_score_pair,
_snr_bucket, _summarize, load_config) rather than recomputing SNR/STOI/PESQ --
this script only restricts which columns are scored and changes the output
shape/destination. It does NOT replace evaluate.py's full run (v1/v2/v4/v5
all still get computed and written to results/baseline_results.* by that
script) -- this is an additional, narrower output for presentation use.

Writes results/final_comparison.csv (overall + per-category + per-SNR-bucket
rows, three columns only) and prints the same breakdown to console, plus a
HEADLINE section explicitly calling out the gunshot category row and the
-5 to 0 dB SNR-bucket row -- the two evidence points locked for the demo.

Requires results/finetuned_v5/*_enhanced.wav to exist (run
`python -m src.model.inference --checkpoint checkpoints/dns48_finetuned_v5_best.pt
--output-dir results/finetuned_v5` first if not already present).

Run standalone: `python -m src.eval.evaluate_final` from repo root.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

from src.eval.evaluate import (
    BASELINE_DIR,
    CATEGORIES,
    FINETUNED_V5_DIR,
    TEST_MANIFEST,
    _score_pair,
    _snr_bucket,
    _summarize,
    load_config,
)

FINAL_COLUMNS = ["noisy", "baseline", "finetuned_v5"]
FINAL_COLUMN_DIRS = {"baseline": BASELINE_DIR, "finetuned_v5": FINETUNED_V5_DIR}
FINAL_CSV = Path("results/final_comparison.csv")
HEADLINE_BUCKET = "-5 to 0 dB"


def run(manifest_path: Path = TEST_MANIFEST) -> dict:
    cfg = load_config()
    sample_rate = cfg["sample_rate"]
    pesq_mode = cfg["eval"]["pesq_mode"]
    snr_edges = cfg["eval"]["snr_buckets_db"]
    bucket_labels = [_snr_bucket((snr_edges[i] + snr_edges[i + 1]) / 2, snr_edges) for i in range(len(snr_edges) - 1)]

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    results = {col: defaultdict(list) for col in FINAL_COLUMNS}
    bucket_results = {col: defaultdict(list) for col in FINAL_COLUMNS}

    for i, entry in enumerate(manifest):
        clean_path = Path(entry["clean_path"])
        noisy_path = Path(entry["noisy_path"])
        category = entry["noise_category"]
        bucket = _snr_bucket(entry["snr_db"], snr_edges)

        noisy_scores = _score_pair(clean_path, noisy_path, sample_rate, pesq_mode)
        results["noisy"][category].append(noisy_scores)
        results["noisy"]["overall"].append(noisy_scores)
        bucket_results["noisy"][bucket].append(noisy_scores)

        for col in ("baseline", "finetuned_v5"):
            col_path = FINAL_COLUMN_DIRS[col] / f"{entry['pair_id']}_enhanced.wav"
            if col_path.exists():
                col_scores = _score_pair(clean_path, col_path, sample_rate, pesq_mode)
                results[col][category].append(col_scores)
                results[col]["overall"].append(col_scores)
                bucket_results[col][bucket].append(col_scores)

        if (i + 1) % 50 == 0 or (i + 1) == len(manifest):
            print(f"[evaluate_final] {i + 1}/{len(manifest)} pairs scored")

    for col in FINAL_COLUMNS:
        for cat in CATEGORIES + ["overall"]:
            results[col][cat] = results[col].get(cat, [])
        for bucket in bucket_labels:
            bucket_results[col][bucket] = bucket_results[col].get(bucket, [])

    summary = _summarize_final(results, CATEGORIES + ["overall"])
    bucket_summary = _summarize_final(bucket_results, bucket_labels)

    _write_csv(summary, bucket_summary, bucket_labels)
    _print_table(summary, "category", CATEGORIES + ["overall"])
    _print_table(bucket_summary, "snr_bucket", bucket_labels)
    _print_headline(summary, bucket_summary)

    return summary


def _summarize_final(results: dict, keys: list) -> dict:
    # Same aggregation as evaluate.py's _summarize, scoped to FINAL_COLUMNS.
    summary = {}
    for col in FINAL_COLUMNS:
        summary[col] = {}
        for key in keys:
            scores = results[col].get(key, [])
            if not scores:
                summary[col][key] = {"n": 0, "snr": None, "stoi": None, "pesq": None}
                continue
            n = len(scores)
            summary[col][key] = {
                "n": n,
                "snr": sum(s["snr"] for s in scores) / n,
                "stoi": sum(s["stoi"] for s in scores) / n,
                "pesq": sum(s["pesq"] for s in scores) / n,
            }
    return summary


def _write_csv(summary: dict, bucket_summary: dict, bucket_labels: list) -> None:
    FINAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["breakdown", "key", "column", "n", "snr_db", "stoi", "pesq"])
        for cat in CATEGORIES + ["overall"]:
            for col in FINAL_COLUMNS:
                row = summary[col][cat]
                writer.writerow(["category", cat, col, row["n"], row["snr"], row["stoi"], row["pesq"]])
        for bucket in bucket_labels:
            for col in FINAL_COLUMNS:
                row = bucket_summary[col][bucket]
                writer.writerow(["snr_bucket", bucket, col, row["n"], row["snr"], row["stoi"], row["pesq"]])
    print(f"\nWrote {FINAL_CSV}")


def _print_table(summary: dict, label: str, keys: list) -> None:
    print(f"\n=== Final comparison by {label} (test split) ===")
    header = f"{'column':<14} {label:<11} {'n':>5} {'SNR(dB)':>9} {'STOI':>7} {'PESQ':>7}"
    print(header)
    print("-" * len(header))
    for col in FINAL_COLUMNS:
        for key in keys:
            row = summary[col][key]
            if row["n"] == 0:
                print(f"{col:<14} {key:<11} {row['n']:>5} {'--':>9} {'--':>7} {'--':>7}")
            else:
                print(
                    f"{col:<14} {key:<11} {row['n']:>5} "
                    f"{row['snr']:>9.2f} {row['stoi']:>7.3f} {row['pesq']:>7.3f}"
                )


def _print_headline(summary: dict, bucket_summary: dict) -> None:
    print("\n=== HEADLINE (locked demo evidence points) ===")
    print("-- Gunshot category --")
    for col in FINAL_COLUMNS:
        row = summary[col]["gunshot"]
        if row["n"] == 0:
            print(f"{col:<14} n=0 -- (missing enhanced output)")
        else:
            print(f"{col:<14} n={row['n']:<4} SNR={row['snr']:.2f}dB STOI={row['stoi']:.3f} PESQ={row['pesq']:.3f}")

    print(f"\n-- {HEADLINE_BUCKET} SNR bucket --")
    if HEADLINE_BUCKET not in bucket_summary["noisy"]:
        print(f"Bucket '{HEADLINE_BUCKET}' not found among computed buckets: {list(bucket_summary['noisy'].keys())}")
        return
    for col in FINAL_COLUMNS:
        row = bucket_summary[col][HEADLINE_BUCKET]
        if row["n"] == 0:
            print(f"{col:<14} n=0 -- (missing enhanced output)")
        else:
            print(f"{col:<14} n={row['n']:<4} SNR={row['snr']:.2f}dB STOI={row['stoi']:.3f} PESQ={row['pesq']:.3f}")


if __name__ == "__main__":
    run()
