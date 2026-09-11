"""Determines whether v2's over-suppression on its worst-scoring pairs is
something v2's changes (2x STFT loss weight, 3x hard-mixture oversampling)
INTRODUCED, or a PRE-EXISTING weakness v1 (flat loss weights, no
oversampling) already had on these same hard clips.

Method: takes the same unique pair_ids already diagnosed in
results/failure_diagnosis.csv (v2's worst pairs, per diagnose_failures.py),
locates v1's enhanced output for each pair_id in results/finetuned/
(the exact directory Phase 4's src/model/inference.py + src/eval/evaluate.py
wrote v1's per-pair outputs to -- confirmed present on disk before writing
this script, not assumed), and runs the SAME diagnostic (diagnose_pair() /
label_pair() from src.eval.diagnose_failures, imported and reused, not
reimplemented) on v1's outputs for this exact pair set.

Produces a side-by-side CSV and prints an aggregate verdict comparing v1's
vs. v2's average speech_distortion_estimate on this shared hard-pair set.

Inherits every limitation documented in diagnose_failures.py's docstring
(rough heuristic, not a validated perceptual metric) -- this script adds no
new estimation logic of its own, only a second run of the same method
against a second set of files plus a comparison layer.

Run standalone: `python -m src.eval.compare_v1_v2_worst_pairs` from repo
root. Requires results/failure_diagnosis.csv (v2's diagnosis, run
`python -m src.eval.diagnose_failures` first if missing) and v1's enhanced
wavs in results/finetuned/ (Phase 4 output) to already exist.
"""

import csv
from pathlib import Path

from src.eval.diagnose_failures import diagnose_pair, label_pair, load_config

V1_ENHANCED_DIR = Path("results/finetuned")


def _read_v2_diagnosis(diagnosis_csv: Path) -> list:
    with open(diagnosis_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def run() -> None:
    cfg = load_config()
    dcfg = cfg["diagnose_failures"]
    diagnosis_csv = Path(dcfg["output_csv"])
    output_csv = Path("results/v1_vs_v2_worst_pairs_comparison.csv")

    v2_rows = _read_v2_diagnosis(diagnosis_csv)
    print(f"[compare_v1_v2] {len(v2_rows)} unique pairs from {diagnosis_csv}")

    missing = [r["pair_id"] for r in v2_rows if not (V1_ENHANCED_DIR / f"{r['pair_id']}_enhanced.wav").exists()]
    if missing:
        raise FileNotFoundError(
            f"v1 enhanced output missing for {len(missing)} pair(s) in {V1_ENHANCED_DIR}: {missing}"
        )

    comparisons = []
    for row in v2_rows:
        pair_id = row["pair_id"]
        clean_path = Path("data/processed/test") / f"{pair_id}_clean.wav"
        noisy_path = Path("data/processed/test") / f"{pair_id}_noisy.wav"
        v1_enhanced_path = V1_ENHANCED_DIR / f"{pair_id}_enhanced.wav"

        v1_estimates = diagnose_pair(clean_path, noisy_path, v1_enhanced_path, cfg)
        v1_label = label_pair(
            v1_estimates["residual_noise_estimate"], v1_estimates["speech_distortion_estimate"], cfg
        )

        comparisons.append(
            {
                "pair_id": pair_id,
                "v1_residual_noise_estimate": v1_estimates["residual_noise_estimate"],
                "v1_speech_distortion_estimate": v1_estimates["speech_distortion_estimate"],
                "v1_label": v1_label,
                "v2_residual_noise_estimate": float(row["residual_noise_estimate"]),
                "v2_speech_distortion_estimate": float(row["speech_distortion_estimate"]),
                "v2_label": row["label"],
                "snr": float(row["snr"]),
                "stoi": float(row["stoi"]),
                "pesq": float(row["pesq"]),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pair_id",
                "v1_residual_noise_estimate",
                "v1_speech_distortion_estimate",
                "v1_label",
                "v2_residual_noise_estimate",
                "v2_speech_distortion_estimate",
                "v2_label",
                "snr",
                "stoi",
                "pesq",
            ]
        )
        for c in comparisons:
            writer.writerow(
                [
                    c["pair_id"],
                    c["v1_residual_noise_estimate"],
                    c["v1_speech_distortion_estimate"],
                    c["v1_label"],
                    c["v2_residual_noise_estimate"],
                    c["v2_speech_distortion_estimate"],
                    c["v2_label"],
                    c["snr"],
                    c["stoi"],
                    c["pesq"],
                ]
            )
    print(f"[compare_v1_v2] wrote {output_csv}")

    print(f"\n=== Side-by-side: v1 vs v2 on v2's {len(comparisons)} worst pairs ===")
    header = f"{'pair_id':<14} {'v1_dist':>9} {'v1_label':<24} {'v2_dist':>9} {'v2_label':<24} {'snr':>7} {'stoi':>6} {'pesq':>6}"
    print(header)
    print("-" * len(header))
    for c in comparisons:
        print(
            f"{c['pair_id']:<14} {c['v1_speech_distortion_estimate']:>9.4f} {c['v1_label']:<24} "
            f"{c['v2_speech_distortion_estimate']:>9.4f} {c['v2_label']:<24} "
            f"{c['snr']:>7.2f} {c['stoi']:>6.3f} {c['pesq']:>6.3f}"
        )

    n = len(comparisons)
    v1_avg_distortion = sum(c["v1_speech_distortion_estimate"] for c in comparisons) / n
    v2_avg_distortion = sum(c["v2_speech_distortion_estimate"] for c in comparisons) / n
    v1_avg_residual = sum(c["v1_residual_noise_estimate"] for c in comparisons) / n
    v2_avg_residual = sum(c["v2_residual_noise_estimate"] for c in comparisons) / n
    v1_over_count = sum(1 for c in comparisons if "over-suppression" in c["v1_label"] or c["v1_label"] == "likely both")
    v2_over_count = sum(1 for c in comparisons if "over-suppression" in c["v2_label"] or c["v2_label"] == "likely both")

    print(f"\n=== Verdict ===")
    print(f"v1 avg speech_distortion_estimate on these {n} pairs: {v1_avg_distortion:.4f}")
    print(f"v2 avg speech_distortion_estimate on these {n} pairs: {v2_avg_distortion:.4f}")
    print(f"v1 avg residual_noise_estimate on these {n} pairs:    {v1_avg_residual:.4f}")
    print(f"v2 avg residual_noise_estimate on these {n} pairs:    {v2_avg_residual:.4f}")
    print(f"v1 pairs labeled over-suppression/both: {v1_over_count}/{n}")
    print(f"v2 pairs labeled over-suppression/both: {v2_over_count}/{n}")

    if v2_avg_distortion <= 0:
        ratio_note = "n/a (v2 avg distortion is zero)"
    else:
        ratio_note = f"{v1_avg_distortion / v2_avg_distortion:.2f}x"

    print()
    if v1_avg_distortion >= 0.7 * v2_avg_distortion:
        print(
            f"READ: v1's speech-distortion on this exact hard-pair set is similarly high "
            f"(v1={v1_avg_distortion:.4f} vs v2={v2_avg_distortion:.4f}, ratio {ratio_note}) -- "
            "this supports a PRE-EXISTING model weakness on these hard/low-SNR clips, not "
            "something v2's 2x STFT weight + 3x oversampling introduced. v2's changes did not "
            "meaningfully worsen distortion on cases that were already this model's hardest, "
            "though they also did not fix it."
        )
    else:
        print(
            f"READ: v1's speech-distortion on this exact hard-pair set is meaningfully LOWER "
            f"than v2's (v1={v1_avg_distortion:.4f} vs v2={v2_avg_distortion:.4f}, ratio {ratio_note}) -- "
            "this supports v2's changes (2x STFT weight and/or 3x hard-mixture oversampling) "
            "having INTRODUCED or amplified over-suppression on these specific pairs, rather than "
            "this being a pre-existing weakness of the base fine-tune."
        )


if __name__ == "__main__":
    run()
