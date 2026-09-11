"""v5 TASK 1: builds the expanded clean-speech TRAIN pool for dynamic mixing.

Combines two LibriSpeech subsets into one train-only speaker pool:
  - dev-clean's train speakers (the SAME 32 speakers Phase 1's
    build_dataset.py already assigned to "train" -- read directly from
    manifests/train.json's speaker_id column, not recomputed, so this can
    never silently drift from the locked v1 speaker-disjoint split).
  - A subset of train-clean-100, selected here (see select_subset() below)
    to target configs/finetune.yaml['dataset']['librispeech_train_clean_100']
    ['target_hours'] (~22.5hr) / target_speakers_min (100+) speakers.

CRITICAL (per task instruction): val/test speakers (from manifests/val.json,
manifests/test.json -- the 4+4=8 speakers Phase 1 already excluded from
train) must NEVER appear in the new pool. LibriSpeech speaker IDs are
globally unique across dev-clean/train-clean-100/test-clean/etc (verified:
OpenSLR's speaker numbering is a single global namespace, not per-subset),
so a train-clean-100 speaker sharing an ID with a val/test speaker would be
a real identity leak, not just a coincidence -- explicitly checked below,
not assumed impossible.

Output: manifests/dynamic_train_pool.json -- a flat list of
{speaker_id, utterance_path, duration_seconds, source} entries (source is
"dev_clean" or "train_clean_100"), used by src/data/dynamic_dataset.py at
training time. Also prints the concrete overlap-check result (must be zero)
required by the task before training can proceed.

Run standalone: `python -m src.data.build_dynamic_pool` from repo root.
Must be run AFTER src/data/download.py (train-clean-100 downloaded).
"""

import json
import random
from pathlib import Path

import soundfile as sf
import yaml

CONFIG_PATH = "configs/finetune.yaml"
RAW_DIR = Path("data/raw")
MANIFEST_DIR = Path("manifests")
OUTPUT_MANIFEST = MANIFEST_DIR / "dynamic_train_pool.json"


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _index_by_speaker(root: Path) -> dict:
    """Returns {speaker_id: [flac_path, ...]} for a LibriSpeech-layout root
    (<root>/.../<speaker>/<chapter>/*.flac, same convention build_dataset.py
    uses for dev-clean)."""
    by_speaker = {}
    for f in sorted(root.rglob("*.flac")):
        speaker_id = f.parts[-3]
        by_speaker.setdefault(speaker_id, []).append(f)
    return by_speaker


def get_locked_train_val_test_speakers() -> dict:
    """Reads the ALREADY-DECIDED v1 speaker-disjoint split directly from the
    Phase 1 manifests, rather than recomputing split_speakers() here --
    guarantees this can never drift from the locked split even if
    build_dataset.py's split logic or seed usage ever changes.
    """
    result = {}
    for split in ("train", "val", "test"):
        path = MANIFEST_DIR / f"{split}.json"
        with open(path, "r") as f:
            entries = json.load(f)
        result[split] = sorted({e["speaker_id"] for e in entries})
    return result


def select_train_clean_100_subset(cfg: dict, excluded_speakers: set, rng: random.Random) -> list:
    """Selects a subset of train-clean-100 speakers (whole speakers only,
    never partial) until cumulative duration crosses target_hours, per the
    config comment's documented method: shuffle speakers (seeded), take them
    in full one at a time. Excludes any speaker ID already used by
    val/test (or, defensively, train) in the v1 split.

    Returns a list of {speaker_id, utterance_path, duration_seconds, source}
    dicts for the selected subset.
    """
    pool_cfg = cfg["dataset"]["librispeech_train_clean_100"]
    target_hours = pool_cfg["target_hours"]
    target_seconds = target_hours * 3600.0

    root = RAW_DIR / "librispeech_train_clean_100"
    by_speaker = _index_by_speaker(root)

    candidate_speakers = [s for s in by_speaker if s not in excluded_speakers]
    rng.shuffle(candidate_speakers)

    selected_entries = []
    selected_speakers = []
    cumulative_seconds = 0.0

    for speaker_id in candidate_speakers:
        if cumulative_seconds >= target_seconds:
            break
        speaker_entries = []
        speaker_seconds = 0.0
        for flac_path in by_speaker[speaker_id]:
            info = sf.info(str(flac_path))
            duration = info.frames / info.samplerate
            speaker_entries.append({
                "speaker_id": speaker_id,
                "utterance_path": str(flac_path),
                "duration_seconds": duration,
                "source": "train_clean_100",
            })
            speaker_seconds += duration
        selected_entries.extend(speaker_entries)
        selected_speakers.append(speaker_id)
        cumulative_seconds += speaker_seconds

    print(f"[build_dynamic_pool] train-clean-100 subset: {len(selected_speakers)} speakers, "
          f"{len(selected_entries)} utterances, {cumulative_seconds / 3600.0:.2f}hr "
          f"(target was {target_hours}hr / {pool_cfg['target_speakers_min']}+ speakers).")

    if len(selected_speakers) < pool_cfg["target_speakers_min"]:
        print(f"[build_dynamic_pool] WARNING: only {len(selected_speakers)} speakers selected, "
              f"below target_speakers_min={pool_cfg['target_speakers_min']}. This can happen if "
              f"target_hours is small relative to average per-speaker duration -- not a bug, but "
              f"flagged since it falls short of the task's explicit '100+' ask.")

    return selected_entries


def get_dev_clean_train_entries() -> list:
    """The dev-clean train-speaker entries, read directly from
    manifests/train.json's source_utterance column (the exact clean-speech
    files v1/v2/v4 already used) -- reused as-is, not re-derived from
    data/raw/librispeech/, so the dynamic pool's dev-clean slice is
    byte-identical to what the static pipeline already validated.
    """
    with open(MANIFEST_DIR / "train.json", "r") as f:
        train_entries = json.load(f)

    seen_utterances = set()
    entries = []
    for e in train_entries:
        path = e["source_utterance"]
        if path in seen_utterances:
            continue
        seen_utterances.add(path)
        info = sf.info(path)
        duration = info.frames / info.samplerate
        entries.append({
            "speaker_id": e["speaker_id"],
            "utterance_path": path,
            "duration_seconds": duration,
            "source": "dev_clean",
        })
    return entries


def verify_no_val_test_overlap(pool_entries: list, val_speakers: list, test_speakers: list) -> None:
    """Concrete, printed overlap check -- per explicit task instruction
    ("Verify concretely post-download (print overlap check = zero)"), not a
    silent assumption that exclusion worked.
    """
    pool_speakers = {e["speaker_id"] for e in pool_entries}
    val_overlap = pool_speakers & set(val_speakers)
    test_overlap = pool_speakers & set(test_speakers)

    print(f"[build_dynamic_pool] OVERLAP CHECK -- pool speakers: {len(pool_speakers)}, "
          f"val speakers: {val_speakers}, test speakers: {test_speakers}")
    print(f"[build_dynamic_pool] pool ∩ val  = {sorted(val_overlap)} (count={len(val_overlap)})")
    print(f"[build_dynamic_pool] pool ∩ test = {sorted(test_overlap)} (count={len(test_overlap)})")

    if val_overlap or test_overlap:
        raise RuntimeError(
            f"SPEAKER LEAK DETECTED: {len(val_overlap)} val-speaker overlap(s), "
            f"{len(test_overlap)} test-speaker overlap(s) found in the dynamic train pool. "
            f"This must be exactly zero -- aborting rather than writing a leaking manifest."
        )
    print("[build_dynamic_pool] Overlap check PASSED: zero val/test speaker leakage.")


def main() -> None:
    cfg = load_config()
    rng = random.Random(cfg["seed"])

    locked_split = get_locked_train_val_test_speakers()
    train_speakers = set(locked_split["train"])
    val_speakers = locked_split["val"]
    test_speakers = locked_split["test"]
    excluded_speakers = train_speakers | set(val_speakers) | set(test_speakers)

    print(f"[build_dynamic_pool] locked v1 split -- train: {len(train_speakers)} speakers, "
          f"val: {val_speakers}, test: {test_speakers}")

    dev_clean_entries = get_dev_clean_train_entries()
    dev_clean_hours = sum(e["duration_seconds"] for e in dev_clean_entries) / 3600.0
    print(f"[build_dynamic_pool] dev-clean train slice (reused from v1): "
          f"{len(train_speakers)} speakers, {len(dev_clean_entries)} utterances, "
          f"{dev_clean_hours:.2f}hr.")

    train_clean_100_entries = select_train_clean_100_subset(cfg, excluded_speakers, rng)

    pool_entries = dev_clean_entries + train_clean_100_entries
    pool_speakers = sorted({e["speaker_id"] for e in pool_entries})
    total_hours = sum(e["duration_seconds"] for e in pool_entries) / 3600.0

    verify_no_val_test_overlap(pool_entries, val_speakers, test_speakers)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MANIFEST, "w") as f:
        json.dump(pool_entries, f, indent=2)

    print(f"[build_dynamic_pool] TOTAL dynamic train pool: {len(pool_speakers)} distinct speakers, "
          f"{len(pool_entries)} utterances, {total_hours:.2f}hr "
          f"(dev-clean: {len(train_speakers)} speakers/{dev_clean_hours:.2f}hr + "
          f"train-clean-100 subset above).")
    print(f"[build_dynamic_pool] wrote {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()
