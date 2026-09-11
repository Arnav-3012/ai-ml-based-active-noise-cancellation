"""Evaluates enhancement output against the clean test set.

Computes SNR / STOI / PESQ for three columns:
  - "noisy"    : the raw noisy audio, no processing at all (reference floor).
  - "baseline" : spectral_subtraction.py's output.
  - "finetuned": placeholder, empty until Phase 3 (the fine-tuned model).

Reports overall averages AND a per-noise-category breakdown (gunshot /
stationary / general), matching the Tier-1 breakdown tagged in the Phase 1
manifest. Writes results/baseline_results.json (full nested numbers) and
results/baseline_results.csv (flat table, one row per category+column, for
direct use in the PPT/demo comparison table).

Run standalone: `python -m src.eval.evaluate` from repo root.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import soundfile as sf

from src.eval.metrics import pesq_score, snr, stoi_score

CONFIG_PATH = "configs/finetune.yaml"
TEST_MANIFEST = Path("manifests/test.json")
BASELINE_DIR = Path("results/baseline")
RESULTS_JSON = Path("results/baseline_results.json")
RESULTS_CSV = Path("results/baseline_results.csv")

CATEGORIES = ["gunshot", "stationary", "general"]
COLUMNS = ["noisy", "baseline", "finetuned"]


def load_config(path: str = CONFIG_PATH) -> dict:
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_wav(path: Path):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data[:, 0], sr  # mono


def _score_pair(clean_path: Path, estimate_path: Path, sample_rate: int, pesq_mode: str) -> dict:
    clean, sr_c = _load_wav(clean_path)
    estimate, sr_e = _load_wav(estimate_path)
    if sr_c != sample_rate or sr_e != sample_rate:
        raise RuntimeError(
            f"sample rate mismatch: {clean_path}={sr_c}, {estimate_path}={sr_e}, expected {sample_rate}"
        )
    return {
        "snr": snr(clean, estimate),
        "stoi": stoi_score(clean, estimate, sample_rate),
        "pesq": pesq_score(clean, estimate, sample_rate, pesq_mode),
    }


def run(manifest_path: Path = TEST_MANIFEST) -> dict:
    cfg = load_config()
    sample_rate = cfg["sample_rate"]
    pesq_mode = cfg["eval"]["pesq_mode"]

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # results[column][category] -> list of per-pair metric dicts
    results = {col: defaultdict(list) for col in COLUMNS}

    for i, entry in enumerate(manifest):
        clean_path = Path(entry["clean_path"])
        noisy_path = Path(entry["noisy_path"])
        category = entry["noise_category"]

        noisy_scores = _score_pair(clean_path, noisy_path, sample_rate, pesq_mode)
        results["noisy"][category].append(noisy_scores)
        results["noisy"]["overall"] = results["noisy"].get("overall", [])
        results["noisy"]["overall"].append(noisy_scores)

        baseline_path = BASELINE_DIR / f"{entry['pair_id']}_enhanced.wav"
        if baseline_path.exists():
            baseline_scores = _score_pair(clean_path, baseline_path, sample_rate, pesq_mode)
            results["baseline"][category].append(baseline_scores)
            results["baseline"]["overall"] = results["baseline"].get("overall", [])
            results["baseline"]["overall"].append(baseline_scores)

        if (i + 1) % 50 == 0 or (i + 1) == len(manifest):
            print(f"[evaluate] {i + 1}/{len(manifest)} pairs scored")

    # finetuned column stays empty (no per-pair scores) until Phase 3.
    for cat in CATEGORIES + ["overall"]:
        results["finetuned"][cat] = []

    summary = _summarize(results)
    _write_json(summary)
    _write_csv(summary)
    _print_table(summary)
    return summary


def _summarize(results: dict) -> dict:
    summary = {}
    for col in COLUMNS:
        summary[col] = {}
        for cat in CATEGORIES + ["overall"]:
            scores = results[col].get(cat, [])
            if not scores:
                summary[col][cat] = {"n": 0, "snr": None, "stoi": None, "pesq": None}
                continue
            n = len(scores)
            summary[col][cat] = {
                "n": n,
                "snr": sum(s["snr"] for s in scores) / n,
                "stoi": sum(s["stoi"] for s in scores) / n,
                "pesq": sum(s["pesq"] for s in scores) / n,
            }
    return summary


def _write_json(summary: dict) -> None:
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(summary, f, indent=2)


def _write_csv(summary: dict) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["column", "category", "n", "snr_db", "stoi", "pesq"])
        for col in COLUMNS:
            for cat in CATEGORIES + ["overall"]:
                row = summary[col][cat]
                writer.writerow([col, cat, row["n"], row["snr"], row["stoi"], row["pesq"]])


def _print_table(summary: dict) -> None:
    print("\n=== Results (test split) ===")
    header = f"{'column':<10} {'category':<11} {'n':>5} {'SNR(dB)':>9} {'STOI':>7} {'PESQ':>7}"
    print(header)
    print("-" * len(header))
    for col in COLUMNS:
        for cat in CATEGORIES + ["overall"]:
            row = summary[col][cat]
            if row["n"] == 0:
                print(f"{col:<10} {cat:<11} {row['n']:>5} {'--':>9} {'--':>7} {'--':>7}")
            else:
                print(
                    f"{col:<10} {cat:<11} {row['n']:>5} "
                    f"{row['snr']:>9.2f} {row['stoi']:>7.3f} {row['pesq']:>7.3f}"
                )
    print(f"\nWrote {RESULTS_JSON} and {RESULTS_CSV}")


if __name__ == "__main__":
    run()
