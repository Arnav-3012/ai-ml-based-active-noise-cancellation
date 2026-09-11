"""Evaluates enhancement output against the clean test set.

Computes SNR / STOI / PESQ for six columns:
  - "noisy"        : the raw noisy audio, no processing at all (reference floor).
  - "baseline"      : spectral_subtraction.py's output.
  - "finetuned"     : src/model/inference.py's output using the v1 checkpoint
    (Phase 3c checkpoint, epoch 7 best-val -- Phase 4).
  - "finetuned_v2"  : src/model/inference.py's output using the v2 checkpoint
    (Phase 3c-v2 checkpoint -- Phase 4 v2).
  - "finetuned_v4"  : src/model/inference.py's output using the v4 checkpoint
    (Phase 3c-v4 checkpoint, epoch 11 best-val -- Phase 4 v4; v4's run
    plateau-stopped at epoch 31, but epoch 11 was the last new-best before
    it regressed, so epoch 11 is what's scored here, not epoch 31).
  - "finetuned_v5"  : src/model/inference.py's output using the v5 checkpoint
    (dns48_finetuned_v5_best.pt -- dynamic mixing + expanded clean pool +
    reactive LR run).

Reports overall averages, a per-noise-category breakdown (gunshot /
stationary / general, matching the Tier-1 breakdown tagged in the Phase 1
manifest), and a per-input-SNR-bucket breakdown (edges from
configs/finetune.yaml eval.snr_buckets_db) for all six columns -- the
locked Tier-1 deliverable. Also lists the worst-scoring finetuned_v2 test
pairs by PESQ and by STOI separately (count from eval.worst_pairs_count),
and separately the worst-scoring finetuned_v4 pairs the same way.

Writes results/baseline_results.json (full nested numbers),
results/baseline_results.csv (flat table, one row per category+column),
results/snr_bucket_results.csv (one row per SNR-bucket+column),
results/finetuned_v2_worst_pairs.csv (v2 worst pairs by PESQ/STOI),
results/finetuned_v4_worst_pairs.csv (v4 worst pairs by PESQ/STOI), and
results/finetuned_v5_worst_pairs.csv (v5 worst pairs by PESQ/STOI).
Filenames kept as "baseline_results.*" (not renamed to something like
"full_results.*") to avoid breaking any existing reference to these paths
from Phase 2 -- they now simply contain all six columns instead of two.

Run standalone: `python -m src.eval.evaluate` from repo root. Requires
`results/finetuned/*_enhanced.wav` (run `python -m src.model.inference`),
`results/finetuned_v2/*_enhanced.wav` (run `python -m src.model.inference
--checkpoint checkpoints/dns48_finetuned_v2_best.pt --output-dir
results/finetuned_v2`), `results/finetuned_v4/*_enhanced.wav` (run
`python -m src.model.inference --checkpoint
checkpoints/dns48_finetuned_v4_best.pt --output-dir results/finetuned_v4`),
and `results/finetuned_v5/*_enhanced.wav` (run `python -m src.model.inference
--checkpoint checkpoints/dns48_finetuned_v5_best.pt --output-dir
results/finetuned_v5`)
to exist for those columns to be non-empty -- if a directory is
missing/empty, that column is scored as n=0, same placeholder behavior
Phase 2 used originally.
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
FINETUNED_DIR = Path("results/finetuned")
FINETUNED_V2_DIR = Path("results/finetuned_v2")
FINETUNED_V4_DIR = Path("results/finetuned_v4")
FINETUNED_V5_DIR = Path("results/finetuned_v5")
RESULTS_JSON = Path("results/baseline_results.json")
RESULTS_CSV = Path("results/baseline_results.csv")
SNR_BUCKET_CSV = Path("results/snr_bucket_results.csv")
WORST_PAIRS_CSV = Path("results/finetuned_v2_worst_pairs.csv")
WORST_PAIRS_V4_CSV = Path("results/finetuned_v4_worst_pairs.csv")
WORST_PAIRS_V5_CSV = Path("results/finetuned_v5_worst_pairs.csv")

CATEGORIES = ["gunshot", "stationary", "general"]
COLUMNS = ["noisy", "baseline", "finetuned", "finetuned_v2", "finetuned_v4", "finetuned_v5"]
COLUMN_DIRS = {
    "baseline": BASELINE_DIR,
    "finetuned": FINETUNED_DIR,
    "finetuned_v2": FINETUNED_V2_DIR,
    "finetuned_v4": FINETUNED_V4_DIR,
    "finetuned_v5": FINETUNED_V5_DIR,
}


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


def _snr_bucket(snr_db: float, edges: list) -> str:
    """Maps an input snr_db to a bucket label from consecutive `edges`.
    Every bucket is [low, high) except the last, which is [low, high] so
    pairs at exactly the max edge are still included."""
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        is_last = i == len(edges) - 2
        if (low <= snr_db < high) or (is_last and snr_db == high):
            return f"{low} to {high} dB"
    raise ValueError(f"snr_db {snr_db} falls outside bucket edges {edges}")


def run(manifest_path: Path = TEST_MANIFEST) -> dict:
    cfg = load_config()
    sample_rate = cfg["sample_rate"]
    pesq_mode = cfg["eval"]["pesq_mode"]
    snr_edges = cfg["eval"]["snr_buckets_db"]
    worst_pairs_count = cfg["eval"]["worst_pairs_count"]
    bucket_labels = [_snr_bucket((snr_edges[i] + snr_edges[i + 1]) / 2, snr_edges) for i in range(len(snr_edges) - 1)]

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # results[column][category] -> list of per-pair metric dicts
    results = {col: defaultdict(list) for col in COLUMNS}
    # bucket_results[column][bucket_label] -> list of per-pair metric dicts
    bucket_results = {col: defaultdict(list) for col in COLUMNS}
    # finetuned_v2 per-pair records, for the worst-pairs export
    v2_pair_records = []
    # finetuned_v4 per-pair records, for the worst-pairs export
    v4_pair_records = []
    # finetuned_v5 per-pair records, for the worst-pairs export
    v5_pair_records = []

    for i, entry in enumerate(manifest):
        clean_path = Path(entry["clean_path"])
        noisy_path = Path(entry["noisy_path"])
        category = entry["noise_category"]
        bucket = _snr_bucket(entry["snr_db"], snr_edges)

        noisy_scores = _score_pair(clean_path, noisy_path, sample_rate, pesq_mode)
        results["noisy"][category].append(noisy_scores)
        results["noisy"]["overall"].append(noisy_scores)
        bucket_results["noisy"][bucket].append(noisy_scores)

        for col in ("baseline", "finetuned", "finetuned_v2", "finetuned_v4", "finetuned_v5"):
            col_dir = COLUMN_DIRS[col]
            col_path = col_dir / f"{entry['pair_id']}_enhanced.wav"
            if col_path.exists():
                col_scores = _score_pair(clean_path, col_path, sample_rate, pesq_mode)
                results[col][category].append(col_scores)
                results[col]["overall"].append(col_scores)
                bucket_results[col][bucket].append(col_scores)
                if col == "finetuned_v2":
                    v2_pair_records.append(
                        {
                            "pair_id": entry["pair_id"],
                            "clean_path": str(clean_path),
                            "noisy_path": str(noisy_path),
                            "enhanced_path": str(col_path),
                            **col_scores,
                        }
                    )
                if col == "finetuned_v4":
                    v4_pair_records.append(
                        {
                            "pair_id": entry["pair_id"],
                            "clean_path": str(clean_path),
                            "noisy_path": str(noisy_path),
                            "enhanced_path": str(col_path),
                            **col_scores,
                        }
                    )
                if col == "finetuned_v5":
                    v5_pair_records.append(
                        {
                            "pair_id": entry["pair_id"],
                            "clean_path": str(clean_path),
                            "noisy_path": str(noisy_path),
                            "enhanced_path": str(col_path),
                            **col_scores,
                        }
                    )

        if (i + 1) % 50 == 0 or (i + 1) == len(manifest):
            print(f"[evaluate] {i + 1}/{len(manifest)} pairs scored")

    for col in COLUMNS:
        for cat in CATEGORIES + ["overall"]:
            results[col][cat] = results[col].get(cat, [])
        for bucket in bucket_labels:
            bucket_results[col][bucket] = bucket_results[col].get(bucket, [])

    summary = _summarize(results, CATEGORIES + ["overall"])
    bucket_summary = _summarize(bucket_results, bucket_labels)

    _write_json(summary)
    _write_csv(summary)
    _write_bucket_csv(bucket_summary, bucket_labels)
    _write_worst_pairs(v2_pair_records, worst_pairs_count)
    _write_worst_pairs_v4(v4_pair_records, worst_pairs_count)
    _write_worst_pairs_v5(v5_pair_records, worst_pairs_count)
    _print_table(summary)
    _print_bucket_table(bucket_summary, bucket_labels)

    return summary


def _summarize(results: dict, keys: list) -> dict:
    summary = {}
    for col in COLUMNS:
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


def _write_bucket_csv(bucket_summary: dict, bucket_labels: list) -> None:
    SNR_BUCKET_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SNR_BUCKET_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["column", "snr_bucket", "n", "snr_db", "stoi", "pesq"])
        for col in COLUMNS:
            for bucket in bucket_labels:
                row = bucket_summary[col][bucket]
                writer.writerow([col, bucket, row["n"], row["snr"], row["stoi"], row["pesq"]])


def _write_worst_pairs(v2_pair_records: list, worst_pairs_count: int) -> None:
    WORST_PAIRS_CSV.parent.mkdir(parents=True, exist_ok=True)
    worst_by_pesq = sorted(v2_pair_records, key=lambda r: r["pesq"])[:worst_pairs_count]
    worst_by_stoi = sorted(v2_pair_records, key=lambda r: r["stoi"])[:worst_pairs_count]

    with open(WORST_PAIRS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ranked_by", "rank", "pair_id", "clean_path", "noisy_path", "enhanced_path", "snr", "stoi", "pesq"])
        for rank, r in enumerate(worst_by_pesq, start=1):
            writer.writerow(["pesq", rank, r["pair_id"], r["clean_path"], r["noisy_path"], r["enhanced_path"], r["snr"], r["stoi"], r["pesq"]])
        for rank, r in enumerate(worst_by_stoi, start=1):
            writer.writerow(["stoi", rank, r["pair_id"], r["clean_path"], r["noisy_path"], r["enhanced_path"], r["snr"], r["stoi"], r["pesq"]])


def _write_worst_pairs_v4(v4_pair_records: list, worst_pairs_count: int) -> None:
    WORST_PAIRS_V4_CSV.parent.mkdir(parents=True, exist_ok=True)
    worst_by_pesq = sorted(v4_pair_records, key=lambda r: r["pesq"])[:worst_pairs_count]
    worst_by_stoi = sorted(v4_pair_records, key=lambda r: r["stoi"])[:worst_pairs_count]

    with open(WORST_PAIRS_V4_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ranked_by", "rank", "pair_id", "clean_path", "noisy_path", "enhanced_path", "snr", "stoi", "pesq"])
        for rank, r in enumerate(worst_by_pesq, start=1):
            writer.writerow(["pesq", rank, r["pair_id"], r["clean_path"], r["noisy_path"], r["enhanced_path"], r["snr"], r["stoi"], r["pesq"]])
        for rank, r in enumerate(worst_by_stoi, start=1):
            writer.writerow(["stoi", rank, r["pair_id"], r["clean_path"], r["noisy_path"], r["enhanced_path"], r["snr"], r["stoi"], r["pesq"]])


def _write_worst_pairs_v5(v5_pair_records: list, worst_pairs_count: int) -> None:
    WORST_PAIRS_V5_CSV.parent.mkdir(parents=True, exist_ok=True)
    worst_by_pesq = sorted(v5_pair_records, key=lambda r: r["pesq"])[:worst_pairs_count]
    worst_by_stoi = sorted(v5_pair_records, key=lambda r: r["stoi"])[:worst_pairs_count]

    with open(WORST_PAIRS_V5_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ranked_by", "rank", "pair_id", "clean_path", "noisy_path", "enhanced_path", "snr", "stoi", "pesq"])
        for rank, r in enumerate(worst_by_pesq, start=1):
            writer.writerow(["pesq", rank, r["pair_id"], r["clean_path"], r["noisy_path"], r["enhanced_path"], r["snr"], r["stoi"], r["pesq"]])
        for rank, r in enumerate(worst_by_stoi, start=1):
            writer.writerow(["stoi", rank, r["pair_id"], r["clean_path"], r["noisy_path"], r["enhanced_path"], r["snr"], r["stoi"], r["pesq"]])


def _print_table(summary: dict) -> None:
    print("\n=== Results (test split) ===")
    header = f"{'column':<14} {'category':<11} {'n':>5} {'SNR(dB)':>9} {'STOI':>7} {'PESQ':>7}"
    print(header)
    print("-" * len(header))
    for col in COLUMNS:
        for cat in CATEGORIES + ["overall"]:
            row = summary[col][cat]
            if row["n"] == 0:
                print(f"{col:<14} {cat:<11} {row['n']:>5} {'--':>9} {'--':>7} {'--':>7}")
            else:
                print(
                    f"{col:<14} {cat:<11} {row['n']:>5} "
                    f"{row['snr']:>9.2f} {row['stoi']:>7.3f} {row['pesq']:>7.3f}"
                )
    print(f"\nWrote {RESULTS_JSON} and {RESULTS_CSV}")


def _print_bucket_table(bucket_summary: dict, bucket_labels: list) -> None:
    print("\n=== Results by input SNR bucket (test split) ===")
    header = f"{'column':<14} {'snr_bucket':<14} {'n':>5} {'SNR(dB)':>9} {'STOI':>7} {'PESQ':>7}"
    print(header)
    print("-" * len(header))
    for col in COLUMNS:
        for bucket in bucket_labels:
            row = bucket_summary[col][bucket]
            if row["n"] == 0:
                print(f"{col:<14} {bucket:<14} {row['n']:>5} {'--':>9} {'--':>7} {'--':>7}")
            else:
                print(
                    f"{col:<14} {bucket:<14} {row['n']:>5} "
                    f"{row['snr']:>9.2f} {row['stoi']:>7.3f} {row['pesq']:>7.3f}"
                )
    print(f"\nWrote {SNR_BUCKET_CSV}")
    print(f"Wrote {WORST_PAIRS_CSV}")
    print(f"Wrote {WORST_PAIRS_V4_CSV}")
    print(f"Wrote {WORST_PAIRS_V5_CSV}")


if __name__ == "__main__":
    run()
