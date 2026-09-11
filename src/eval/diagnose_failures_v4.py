"""Re-runs the same over-suppression/under-suppression diagnostic used on
v2 (src/eval/diagnose_failures.py) against v4's worst-scoring test pairs,
to test whether v4's silence-penalty term (configs/finetune.yaml
training_run_v4.silence_penalty) actually reduced over-suppression.

Method: reuses diagnose_pair()/label_pair() from src.eval.diagnose_failures
unchanged (same STFT-based residual-noise / speech-distortion estimator,
same config-driven thresholds under diagnose_failures.* in
configs/finetune.yaml -- no new estimation logic here, only a second run
against v4's worst-pairs set). Reads
results/finetuned_v4_worst_pairs.csv (written by src/eval/evaluate.py's
v4 column) instead of v2's results/finetuned_v2_worst_pairs.csv, which is
why this is a separate script rather than repointing
diagnose_failures.py's config-driven input_csv/output_csv -- those remain
v2's locked record, untouched.

Inherits every limitation documented in diagnose_failures.py's docstring
(rough heuristic, not a validated perceptual metric).

Writes results/failure_diagnosis_v4.csv. Prints the same aggregate summary
(count per label) diagnose_failures.py prints for v2, so the two are
directly comparable against v2's already-reported 18/18 over-suppression
baseline.

Run standalone: `python -m src.eval.diagnose_failures_v4` from repo root.
Requires results/finetuned_v4_worst_pairs.csv to exist (run
`python -m src.eval.evaluate` first, after `results/finetuned_v4/` has
been populated via `python -m src.model.inference --checkpoint
checkpoints/dns48_finetuned_v4_best.pt --output-dir results/finetuned_v4`).
"""

import csv
from collections import defaultdict
from pathlib import Path

from src.eval.diagnose_failures import diagnose_pair, label_pair, load_config

INPUT_CSV = Path("results/finetuned_v4_worst_pairs.csv")
OUTPUT_CSV = Path("results/failure_diagnosis_v4.csv")


def _read_worst_pairs(input_csv: Path) -> list:
    """Dedupes by pair_id -- a pair can appear in both the pesq and stoi
    rankings. Keeps the first occurrence (pesq ranking is written first)."""
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
    shortlist_count = dcfg["shortlist_count"]

    rows = _read_worst_pairs(INPUT_CSV)
    print(f"[diagnose_failures_v4] {len(rows)} unique pairs (deduped from {INPUT_CSV})")

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

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
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
    print(f"[diagnose_failures_v4] wrote {OUTPUT_CSV}")

    # --- Summary ---
    label_counts = defaultdict(int)
    for r in records:
        label_counts[r["label"]] += 1

    print(f"\n=== Summary ({len(records)} unique pairs) ===")
    for label in ["likely under-suppression", "likely over-suppression", "likely both", "unclear"]:
        n = label_counts.get(label, 0)
        print(f"{n}/{len(records)} {label}")
    print("\n(compare against v2's already-reported 18/18 likely over-suppression baseline)")

    # --- Shortlist for manual listening ---
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
