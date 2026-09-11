"""Orchestrates the full Phase 1 data pipeline.

Loads clean speech (LibriSpeech dev-clean) + noise pools (MUSAN noise,
UrbanSound8K gun_shot, Kabealo Zenodo gunshots), splits LibriSpeech by SPEAKER
(not utterance) into train/val/test to avoid identity leakage, runs
SNR-targeted mixing + augmentation per pair, writes noisy/clean wav pairs
to data/processed/{train,val,test}/ and manifest JSONs to manifests/.

Assumes src/data/download.py has already been run (or the manual gunshots
step completed) so data/raw/{librispeech,musan,urbansound8k,gunshots}/ are populated.

Run standalone: `python -m src.data.build_dataset` from the repo root.
"""

import json
import random
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as taF
import yaml

from src.data.augment import augment
from src.data.mixing import mix_pair

CONFIG_PATH = "configs/finetune.yaml"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MANIFEST_DIR = Path("manifests")


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_wav(path: Path, target_sr: int) -> torch.Tensor:
    """Strict loader for clean speech (LibriSpeech). LibriSpeech is already
    16kHz natively, so a mismatch here means something is genuinely wrong
    (wrong file, corrupted download) rather than an expected raw-source
    sample rate difference -- fail loudly instead of silently resampling.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).transpose(0, 1).contiguous()  # [channels, samples]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # downmix to mono
    if sr != target_sr:
        raise RuntimeError(
            f"{path} has sample rate {sr}, expected {target_sr}. "
            "Resample during download/preprocessing before running build_dataset.py."
        )
    return wav


def _load_noise_wav(path: Path, target_sr: int) -> torch.Tensor:
    """Loader for noise sources (MUSAN, UrbanSound8K, Kabealo gunshots).

    Unlike clean speech, noise sources are pulled from several independent
    datasets that ship at different native sample rates (e.g. the Kabealo
    gunshot set is 44.1kHz, UrbanSound8K mixes 44.1/48/96kHz) -- resampling
    here is expected preprocessing, not a sign of a bug, so it's done
    on-the-fly rather than raising.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).transpose(0, 1).contiguous()  # [channels, samples]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # downmix to mono
    if sr != target_sr:
        wav = taF.resample(wav, orig_freq=sr, new_freq=target_sr)
    return wav


# --- Clean speech: LibriSpeech, indexed by speaker ---

def index_librispeech_by_speaker(cfg: dict) -> dict:
    """Returns {speaker_id: [flac_path, ...]}.

    LibriSpeech layout: data/raw/librispeech/LibriSpeech/dev-clean/<speaker>/<chapter>/*.flac
    """
    root = RAW_DIR / "librispeech"
    flac_files = sorted(root.rglob("*.flac"))
    by_speaker = {}
    for f in flac_files:
        speaker_id = f.parts[-3]  # .../<speaker>/<chapter>/<file>.flac
        by_speaker.setdefault(speaker_id, []).append(f)
    return by_speaker


def split_speakers(by_speaker: dict, cfg: dict, rng: random.Random) -> dict:
    """Speaker-disjoint split: each speaker's entire utterance set goes to
    exactly one split, so no speaker identity leaks across train/val/test.
    """
    speaker_ids = list(by_speaker.keys())
    rng.shuffle(speaker_ids)

    n = len(speaker_ids)
    train_ratio = cfg["split"]["train_ratio"]
    val_ratio = cfg["split"]["val_ratio"]

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_speakers = speaker_ids[:n_train]
    val_speakers = speaker_ids[n_train:n_train + n_val]
    test_speakers = speaker_ids[n_train + n_val:]

    return {"train": train_speakers, "val": val_speakers, "test": test_speakers}


# --- Noise pools ---

def load_noise_pool(cfg: dict) -> dict:
    """Returns {category: [wav_path, ...]} for gunshot / stationary / general.

    - gunshot: UrbanSound8K gun_shot clips + Kabealo Zenodo gunshot clips.
    - stationary / general: MUSAN noise clips, split by subfolder naming
      convention (MUSAN's noise set separates 'free-sound' (general) and
      'sound-bible' plus tone/hum-style clips; since MUSAN doesn't formally
      label "stationary" vs "general", we treat the sound-bible subset as
      stationary (mostly steady hums/tones/engines) and free-sound as general
      (mixed/non-stationary field recordings) as a practical, documented split.
    """
    pool = {"gunshot": [], "stationary": [], "general": []}

    urbansound8k_dir = RAW_DIR / "urbansound8k" / cfg["dataset"]["urbansound8k"]["target_class"]
    pool["gunshot"].extend(sorted(urbansound8k_dir.glob("*.wav")))

    gunshots_dir = RAW_DIR / "gunshots"
    pool["gunshot"].extend(sorted(gunshots_dir.rglob("*.wav")))

    musan_noise_dir = RAW_DIR / "musan" / "noise"
    for wav in sorted(musan_noise_dir.rglob("*.wav")):
        if "sound-bible" in wav.parts:
            pool["stationary"].append(wav)
        else:
            pool["general"].append(wav)

    for category, files in pool.items():
        if not files:
            raise RuntimeError(
                f"Noise category '{category}' has 0 clips. "
                "Run src/data/download.py (and complete manual gunshots step) first."
            )

    return pool


# --- Pipeline ---

def build_split(
    split_name: str,
    speakers: list,
    by_speaker: dict,
    noise_pool: dict,
    cfg: dict,
    rng: random.Random,
) -> list:
    """Mixes+augments every utterance for the given speakers, writes wavs,
    and returns the list of manifest entries for this split.
    """
    sample_rate = cfg["sample_rate"]
    categories = cfg["mixing"]["noise_categories"]
    out_dir = PROCESSED_DIR / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    pair_idx = 0

    for speaker_id in speakers:
        for clean_path in by_speaker[speaker_id]:
            clean = _load_wav(clean_path, sample_rate)

            noise_category = rng.choice(categories)
            noise_path = rng.choice(noise_pool[noise_category])
            noise = _load_noise_wav(noise_path, sample_rate)

            result = mix_pair(clean, noise, noise_category, cfg, rng)
            noisy = augment(result["noisy"], cfg, rng, sample_rate)

            pair_id = f"{split_name}_{pair_idx:06d}"
            clean_out = out_dir / f"{pair_id}_clean.wav"
            noisy_out = out_dir / f"{pair_id}_noisy.wav"

            sf.write(str(clean_out), result["clean"].transpose(0, 1).numpy(),
                      sample_rate, subtype=cfg["output"]["bit_depth"])
            sf.write(str(noisy_out), noisy.transpose(0, 1).numpy(),
                      sample_rate, subtype=cfg["output"]["bit_depth"])

            manifest_entries.append({
                "pair_id": pair_id,
                "clean_path": str(clean_out),
                "noisy_path": str(noisy_out),
                "speaker_id": speaker_id,
                "split": split_name,
                "snr_db": result["snr_db"],
                "noise_category": noise_category,
                "noise_source": str(noise_path),
                "source_utterance": str(clean_path),
            })
            pair_idx += 1

    return manifest_entries


def write_manifest(split_name: str, entries: list) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"{split_name}.json"
    with open(manifest_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"[manifest] wrote {len(entries)} entries to {manifest_path}")


def main() -> None:
    cfg = load_config()
    rng = random.Random(cfg["seed"])

    print("[build_dataset] indexing LibriSpeech by speaker...")
    by_speaker = index_librispeech_by_speaker(cfg)
    print(f"[build_dataset] found {len(by_speaker)} speakers, "
          f"{sum(len(v) for v in by_speaker.values())} utterances total.")

    splits = split_speakers(by_speaker, cfg, rng)
    for name, speakers in splits.items():
        n_utts = sum(len(by_speaker[s]) for s in speakers)
        print(f"[build_dataset] split '{name}': {len(speakers)} speakers, {n_utts} utterances.")

    print("[build_dataset] loading noise pool...")
    noise_pool = load_noise_pool(cfg)
    for category, files in noise_pool.items():
        print(f"[build_dataset] noise category '{category}': {len(files)} clips.")

    for split_name, speakers in splits.items():
        print(f"[build_dataset] processing split '{split_name}'...")
        entries = build_split(split_name, speakers, by_speaker, noise_pool, cfg, rng)
        write_manifest(split_name, entries)

    print("[build_dataset] done.")


if __name__ == "__main__":
    main()
