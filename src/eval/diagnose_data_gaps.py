"""Checks whether "gunshot data scarcity" is actually the right diagnosis for
finetuned_v2's worst-case failures, before committing to expanding the
gunshot dataset specifically. Uses only existing data -- no new
training/inference is run by this script.

Three independent checks, cross-referenced:

1. CROSS-TABULATION (category x input-SNR-bucket): for the FULL test set
   (n=299, not just the worst-18), re-scores every finetuned_v2 pair
   (reusing evaluate.py's exact _score_pair/_snr_bucket logic and config,
   so numbers are directly comparable to results/snr_bucket_results.csv
   and results/baseline_results.csv) and buckets by (noise_category,
   snr_bucket). If gunshot-at-low-SNR is uniquely bad relative to
   stationary/general at the SAME low-SNR bucket, that supports a
   gunshot-specific diagnosis. If ALL categories are similarly bad at
   low-SNR, the gap is a general low-SNR capability issue, not a
   category-specific one.

2. WORST-PAIRS COMPOSITION: reads the existing worst-18 pairs
   (results/finetuned_v2_worst_pairs.csv, already identified by Phase 4's
   evaluate.py), joins back to manifests/test.json for noise_category, and
   reports the category breakdown. If it's not skewed toward gunshot
   relative to gunshot's ~1/3 share of the test set, that's evidence
   against "gunshot specifically is the bottleneck."

3. TRAINING EXPOSURE AUDIT: counts actual manifests/train.json pairs per
   noise_category (real post-mixing pair counts, not source clip counts).
   Also independently counts MUSAN's actual clip inventory on disk
   (data/raw/musan) rather than assuming "we used all of it" means
   plentiful.

Prints all three analyses plus a plain-language interpretation. Does not
write any files -- this is a read-only diagnostic, not a results-producing
pipeline step.

Run standalone: `python -m src.eval.diagnose_data_gaps` from repo root.
Requires results/finetuned_v2/*_enhanced.wav to exist (Phase 4 v2 already
produced these).
"""

import json
from collections import defaultdict
from pathlib import Path

from src.eval.evaluate import _score_pair, _snr_bucket, load_config

CONFIG_PATH = "configs/finetune.yaml"
TEST_MANIFEST = Path("manifests/test.json")
TRAIN_MANIFEST = Path("manifests/train.json")
FINETUNED_V2_DIR = Path("results/finetuned_v2")
WORST_PAIRS_CSV = Path("results/finetuned_v2_worst_pairs.csv")
MUSAN_DIR = Path("data/raw/musan")

CATEGORIES = ["gunshot", "stationary", "general"]


def _load_manifest(path: Path) -> list:
    with open(path, "r") as f:
        return json.load(f)


def _read_worst_pairs(path: Path) -> list:
    import csv

    pair_ids = []
    seen = set()
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            pid = row["pair_id"]
            if pid not in seen:
                seen.add(pid)
                pair_ids.append(pid)
    return pair_ids


# ---------------------------------------------------------------------------
# 1. Cross-tabulation: category x SNR-bucket, full test set, finetuned_v2
# ---------------------------------------------------------------------------


def build_cross_tab(cfg: dict) -> dict:
    test_entries = _load_manifest(TEST_MANIFEST)
    snr_edges = cfg["eval"]["snr_buckets_db"]
    bucket_labels = [
        _snr_bucket((snr_edges[i] + snr_edges[i + 1]) / 2, snr_edges)
        for i in range(len(snr_edges) - 1)
    ]
    sample_rate = cfg["sample_rate"]
    pesq_mode = cfg["eval"]["pesq_mode"]

    # cross_tab[category][bucket] -> list of per-pair score dicts
    cross_tab = {cat: {b: [] for b in bucket_labels} for cat in CATEGORIES}

    missing = 0
    for entry in test_entries:
        pair_id = entry["pair_id"]
        category = entry["noise_category"]
        bucket = _snr_bucket(entry["snr_db"], snr_edges)
        enhanced_path = FINETUNED_V2_DIR / f"{pair_id}_enhanced.wav"
        if not enhanced_path.exists():
            missing += 1
            continue
        clean_path = Path(entry["clean_path"])
        scores = _score_pair(clean_path, enhanced_path, sample_rate, pesq_mode)
        cross_tab[category][bucket].append(scores)

    if missing:
        print(
            f"WARNING: {missing} test pairs had no results/finetuned_v2 enhanced "
            f"wav and were skipped from the cross-tab."
        )

    return cross_tab, bucket_labels


def _summarize_cell(scores: list) -> dict:
    n = len(scores)
    if n == 0:
        return {"n": 0, "stoi": None, "pesq": None, "snr": None}
    return {
        "n": n,
        "snr": sum(s["snr"] for s in scores) / n,
        "stoi": sum(s["stoi"] for s in scores) / n,
        "pesq": sum(s["pesq"] for s in scores) / n,
    }


def print_cross_tab(cross_tab: dict, bucket_labels: list) -> None:
    print("\n=== 1. CROSS-TABULATION: finetuned_v2, category x input-SNR-bucket (full test set) ===")
    header = f"{'category':<12}" + "".join(f"{b:>20}" for b in bucket_labels)
    print(header)
    summary = {}
    for cat in CATEGORIES:
        row_cells = []
        summary[cat] = {}
        for bucket in bucket_labels:
            cell = _summarize_cell(cross_tab[cat][bucket])
            summary[cat][bucket] = cell
            if cell["n"] == 0:
                row_cells.append(f"{'n=0':>20}")
            else:
                row_cells.append(
                    f"{'n='+str(cell['n'])+' stoi='+format(cell['stoi'], '.3f')+' pesq='+format(cell['pesq'], '.2f'):>20}"
                )
        print(f"{cat:<12}" + "".join(row_cells))
    return summary


# ---------------------------------------------------------------------------
# 2. Worst-pairs composition
# ---------------------------------------------------------------------------


def worst_pairs_composition(cfg: dict) -> dict:
    test_entries = {e["pair_id"]: e for e in _load_manifest(TEST_MANIFEST)}
    worst_ids = _read_worst_pairs(WORST_PAIRS_CSV)

    counts = defaultdict(int)
    for pid in worst_ids:
        cat = test_entries[pid]["noise_category"]
        counts[cat] += 1

    test_set_counts = defaultdict(int)
    for e in test_entries.values():
        test_set_counts[e["noise_category"]] += 1
    total_test = len(test_entries)

    print(f"\n=== 2. WORST-PAIRS COMPOSITION (worst-{len(worst_ids)} pairs, deduped from finetuned_v2_worst_pairs.csv) ===")
    print(f"{'category':<12}{'worst-18 count':>16}{'worst-18 share':>18}{'test-set share':>18}")
    for cat in CATEGORIES:
        n_worst = counts.get(cat, 0)
        worst_share = n_worst / len(worst_ids) if worst_ids else 0.0
        test_share = test_set_counts[cat] / total_test
        print(f"{cat:<12}{n_worst:>16}{worst_share:>17.1%}{test_share:>18.1%}")

    return {"worst_counts": dict(counts), "worst_total": len(worst_ids), "test_set_counts": dict(test_set_counts), "total_test": total_test}


# ---------------------------------------------------------------------------
# 3. Training exposure audit
# ---------------------------------------------------------------------------


def training_exposure_audit() -> dict:
    train_entries = _load_manifest(TRAIN_MANIFEST)
    counts = defaultdict(int)
    for e in train_entries:
        counts[e["noise_category"]] += 1
    total = len(train_entries)

    print(f"\n=== 3. TRAINING EXPOSURE AUDIT (manifests/train.json, real post-mixing pairs, n={total}) ===")
    print(f"{'category':<12}{'train pairs':>14}{'share':>10}")
    for cat in CATEGORIES:
        n = counts.get(cat, 0)
        print(f"{cat:<12}{n:>14}{n/total:>9.1%}")

    # MUSAN actual clip inventory on disk (feeds stationary + general categories)
    musan_counts = {}
    if MUSAN_DIR.exists():
        wav_files = list(MUSAN_DIR.rglob("*.wav"))
        musan_counts["total_wav_files"] = len(wav_files)
        # break down by immediate parent-of-parent (musan's noise/{subfolder} convention)
        by_subdir = defaultdict(int)
        for wf in wav_files:
            by_subdir[str(wf.parent.relative_to(MUSAN_DIR))] += 1
        musan_counts["by_subdir"] = dict(sorted(by_subdir.items()))
    else:
        musan_counts["total_wav_files"] = None
        musan_counts["by_subdir"] = {}
        print(f"WARNING: {MUSAN_DIR} not found on disk -- cannot verify actual MUSAN clip count.")

    print(f"\nMUSAN actual clip inventory on disk ({MUSAN_DIR}):")
    if musan_counts["total_wav_files"] is not None:
        print(f"  total .wav files found: {musan_counts['total_wav_files']}")
        for subdir, n in musan_counts["by_subdir"].items():
            print(f"    {subdir}: {n}")

    return {"train_counts": dict(counts), "total_train": total, "musan": musan_counts}


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------


def print_interpretation(cross_tab_summary: dict, bucket_labels: list, worst_comp: dict, exposure: dict) -> None:
    print("\n=== INTERPRETATION ===")

    # Lowest SNR bucket, compare categories directly
    low_bucket = bucket_labels[0]
    low_bucket_cells = {cat: cross_tab_summary[cat][low_bucket] for cat in CATEGORIES}
    valid = {c: v for c, v in low_bucket_cells.items() if v["n"] > 0}

    print(f"Lowest SNR bucket ({low_bucket}) by category:")
    for cat, cell in low_bucket_cells.items():
        if cell["n"] == 0:
            print(f"  {cat}: n=0 (no data in this cell)")
        else:
            print(f"  {cat}: n={cell['n']}, STOI={cell['stoi']:.3f}, PESQ={cell['pesq']:.2f}")

    gunshot_is_worst_stoi = False
    gunshot_is_worst_pesq = False
    if len(valid) >= 2:
        stois = {c: v["stoi"] for c, v in valid.items()}
        pesqs = {c: v["pesq"] for c, v in valid.items()}
        stoi_spread = max(stois.values()) - min(stois.values())
        pesq_spread = max(pesqs.values()) - min(pesqs.values())
        gunshot_is_worst_stoi = "gunshot" in stois and stois["gunshot"] == min(stois.values())
        gunshot_is_worst_pesq = "gunshot" in pesqs and pesqs["gunshot"] == min(pesqs.values())
        print(f"\nSpread across categories at {low_bucket}: STOI spread={stoi_spread:.3f}, PESQ spread={pesq_spread:.2f}")
        print(f"Is gunshot the worst category at this SNR bucket? STOI: {gunshot_is_worst_stoi}, PESQ: {gunshot_is_worst_pesq}")

    worst_counts = worst_comp["worst_counts"]
    worst_total = worst_comp["worst_total"]
    test_set_counts = worst_comp["test_set_counts"]
    total_test = worst_comp["total_test"]
    gunshot_worst_share = worst_counts.get("gunshot", 0) / worst_total if worst_total else 0
    gunshot_test_share = test_set_counts.get("gunshot", 0) / total_test
    skew = gunshot_worst_share - gunshot_test_share
    print(f"\nGunshot's share of worst-18: {gunshot_worst_share:.1%} vs. its share of the full test set: {gunshot_test_share:.1%} (skew: {skew:+.1%})")

    train_counts = exposure["train_counts"]
    total_train = exposure["total_train"]
    train_shares = {c: train_counts.get(c, 0) / total_train for c in CATEGORIES}
    min_cat = min(train_shares, key=train_shares.get)
    max_cat = max(train_shares, key=train_shares.get)
    train_imbalance = train_shares[max_cat] - train_shares[min_cat]
    print(f"\nTraining-set category balance: {', '.join(f'{c}={train_shares[c]:.1%}' for c in CATEGORIES)} (max-min spread: {train_imbalance:.1%})")

    print("\n--- Plain-language read (mechanical signals above, not a final judgment) ---")
    reasons_for_gunshot_specific = []
    reasons_against = []

    if len(valid) >= 2 and (gunshot_is_worst_stoi or gunshot_is_worst_pesq):
        reasons_for_gunshot_specific.append("gunshot is the worst-scoring category specifically at the lowest SNR bucket")
    else:
        reasons_against.append("gunshot is NOT uniquely worst at the lowest SNR bucket -- other categories score similarly or worse there")

    if skew > 0.10:
        reasons_for_gunshot_specific.append(f"gunshot is over-represented in the worst-18 relative to its test-set share (+{skew:.1%})")
    else:
        reasons_against.append(f"gunshot's share of the worst-18 ({gunshot_worst_share:.1%}) roughly matches its test-set share ({gunshot_test_share:.1%}) -- not skewed toward gunshot")

    if train_imbalance > 0.10 and min_cat == "gunshot":
        reasons_for_gunshot_specific.append(f"gunshot is genuinely under-represented in training pairs ({train_shares['gunshot']:.1%} vs. others)")
    else:
        reasons_against.append(f"training-set category balance is roughly even ({train_imbalance:.1%} spread) -- gunshot is not data-starved relative to the other categories")

    if reasons_for_gunshot_specific and not reasons_against:
        verdict = "(a) gunshot-specific data scarcity"
    elif reasons_against and not reasons_for_gunshot_specific:
        verdict = "(b) a general low-SNR capability gap, not specific to noise category"
    elif reasons_for_gunshot_specific and reasons_against:
        verdict = "(d) mixed/inconclusive -- some signals point toward gunshot, others don't"
    else:
        verdict = "(d) inconclusive"

    print(f"\nEvidence FOR gunshot-specific diagnosis:")
    for r in reasons_for_gunshot_specific:
        print(f"  - {r}")
    print(f"Evidence AGAINST gunshot-specific diagnosis:")
    for r in reasons_against:
        print(f"  - {r}")
    print(f"\nVerdict: {verdict}")
    print(
        "Note: this script does not evaluate option (c) (some other category "
        "under-represented) beyond the training-exposure counts printed above -- "
        "read the per-category train-pair counts and MUSAN inventory directly if "
        "(c) needs to be ruled in or out."
    )


def run() -> None:
    cfg = load_config(CONFIG_PATH)
    cross_tab, bucket_labels = build_cross_tab(cfg)
    cross_tab_summary = print_cross_tab(cross_tab, bucket_labels)
    worst_comp = worst_pairs_composition(cfg)
    exposure = training_exposure_audit()
    print_interpretation(cross_tab_summary, bucket_labels, worst_comp, exposure)


if __name__ == "__main__":
    run()
