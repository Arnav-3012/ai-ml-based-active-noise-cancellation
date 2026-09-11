"""Download/verify all raw datasets for Phase 1.

Sources (also logged to info.md on a successful run of this script):
  - LibriSpeech dev-clean     : https://www.openslr.org/resources/12/
  - MUSAN (noise subset)      : https://www.openslr.org/resources/17/
  - UrbanSound8K (gun_shot cls): https://zenodo.org/records/1203745
    (ESC-50 was originally planned here but has no gun_shot class among its
    50 official categories -- verified directly, not assumed. UrbanSound8K
    has a real gun_shot class and, unlike Kabealo below, a direct download
    link with no manual click-through.)
  - Kabealo et al. gunshot dataset (Zenodo record 7004819) — NO stable
    programmatic API; Zenodo gates this record behind a browser download /
    access click-through. This script detects that data/raw/gunshots/ is
    empty and prints manual download instructions instead of failing
    silently or fabricating a download URL. This is expected behavior, not
    a bug — see `check_gunshots()` below.

Each dataset is checked under data/raw/{librispeech,musan,urbansound8k,gunshots}/
first; anything already present and verified (non-empty, expected file
count in range) is skipped rather than re-downloaded.

Run standalone: `python -m src.data.download` from the repo root.
"""

import shutil
import sys
import tarfile
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import yaml

CONFIG_PATH = "configs/finetune.yaml"
RAW_DIR = Path("data/raw")
INFO_MD_PATH = Path("info.md")


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _download_with_progress(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest}")

    def _report(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100.0, block_num * block_size / total_size * 100)
            print(f"\r  {pct:5.1f}%", end="", file=sys.stderr)

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest, reporthook=_report)
    print(file=sys.stderr)


def check_librispeech(cfg: dict) -> bool:
    target_dir = RAW_DIR / "librispeech"
    flac_files = list(target_dir.rglob("*.flac"))
    lo = cfg["dataset"]["librispeech"]["target_utterances_min"]
    hi = cfg["dataset"]["librispeech"]["target_utterances_max"]
    if len(flac_files) >= lo:
        print(f"[librispeech] already present: {len(flac_files)} utterances found "
              f"(target range {lo}-{hi}). Skipping download.")
        return True
    return False


def download_librispeech(cfg: dict) -> None:
    if check_librispeech(cfg):
        return
    target_dir = RAW_DIR / "librispeech"
    url = cfg["dataset"]["librispeech"]["url"]
    archive_path = target_dir / "dev-clean.tar.gz"

    _download_with_progress(url, archive_path)
    print("[librispeech] extracting...")
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(target_dir)
    archive_path.unlink()

    flac_files = list(target_dir.rglob("*.flac"))
    lo = cfg["dataset"]["librispeech"]["target_utterances_min"]
    hi = cfg["dataset"]["librispeech"]["target_utterances_max"]
    print(f"[librispeech] done: {len(flac_files)} utterances extracted "
          f"(target range was {lo}-{hi}).")


def check_musan(cfg: dict) -> bool:
    target_dir = RAW_DIR / "musan" / "noise"
    wav_files = list(target_dir.rglob("*.wav")) if target_dir.exists() else []
    if len(wav_files) > 0:
        print(f"[musan] already present: {len(wav_files)} noise clips found. Skipping download.")
        return True
    return False


def download_musan(cfg: dict) -> None:
    if check_musan(cfg):
        return
    target_dir = RAW_DIR / "musan"
    url = cfg["dataset"]["musan"]["url"]
    archive_path = target_dir / "musan.tar.gz"

    _download_with_progress(url, archive_path)
    print("[musan] extracting (noise/ subset only takes effect after extraction)...")
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(target_dir)
    archive_path.unlink()

    # openslr musan tarball extracts to musan/musan/{noise,speech,music}/ —
    # flatten one level so data/raw/musan/noise/ is the stable path callers expect.
    nested = target_dir / "musan"
    if nested.exists():
        for item in nested.iterdir():
            shutil.move(str(item), str(target_dir / item.name))
        nested.rmdir()

    wav_files = list((target_dir / "noise").rglob("*.wav"))
    print(f"[musan] done: {len(wav_files)} noise clips extracted.")


def check_urbansound8k(cfg: dict) -> bool:
    target_dir = RAW_DIR / "urbansound8k"
    target_class = cfg["dataset"]["urbansound8k"]["target_class"]
    gunshot_dir = target_dir / target_class
    wav_files = list(gunshot_dir.glob("*.wav")) if gunshot_dir.exists() else []
    expected = cfg["dataset"]["urbansound8k"]["expected_clip_count"]
    if len(wav_files) >= expected:
        print(f"[urbansound8k] already present: {len(wav_files)} '{target_class}' clips found. "
              "Skipping download.")
        return True
    return False


def download_urbansound8k(cfg: dict) -> None:
    """UrbanSound8K has a real gun_shot class (classID 6) and is directly
    downloadable from Zenodo (no manual click-through, unlike the Kabealo
    dataset below). Note: this is a large (~6GB) tarball since it contains
    all 10 urban sound classes, not just gun_shot -- we only keep the
    gun_shot clips after extraction and discard the rest.
    """
    if check_urbansound8k(cfg):
        return
    target_dir = RAW_DIR / "urbansound8k"
    url = cfg["dataset"]["urbansound8k"]["url"]
    target_class = cfg["dataset"]["urbansound8k"]["target_class"]
    archive_path = target_dir / "urbansound8k.tar.gz"

    _download_with_progress(url, archive_path)
    print("[urbansound8k] extracting (this is a large archive, may take a while)...")
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(target_dir)
    archive_path.unlink()

    extracted_root = target_dir / "UrbanSound8K"
    meta_csv = extracted_root / "metadata" / "UrbanSound8K.csv"
    audio_dir = extracted_root / "audio"

    gunshot_dir = target_dir / target_class
    gunshot_dir.mkdir(exist_ok=True)

    count = 0
    with open(meta_csv, "r") as f:
        header = f.readline().strip().split(",")
        filename_idx = header.index("slice_file_name")
        class_idx = header.index("class")
        fold_idx = header.index("fold")
        for line in f:
            fields = line.strip().split(",")
            if fields[class_idx] == target_class:
                fold = f"fold{fields[fold_idx]}"
                src = audio_dir / fold / fields[filename_idx]
                dst = gunshot_dir / fields[filename_idx]
                if src.exists():
                    shutil.copy(src, dst)
                    count += 1

    # Discard the other 9 classes' audio/folds -- we only need gun_shot,
    # and keeping the full ~6GB extraction around wastes disk space.
    shutil.rmtree(extracted_root)

    expected = cfg["dataset"]["urbansound8k"]["expected_clip_count"]
    print(f"[urbansound8k] done: {count} '{target_class}' clips extracted to {gunshot_dir} "
          f"(expected >= {expected}).")


def check_gunshots(cfg: dict) -> bool:
    target_dir = RAW_DIR / "gunshots"
    wav_files = list(target_dir.rglob("*.wav")) if target_dir.exists() else []
    lo = cfg["dataset"]["gunshots"]["sample_min"]
    if len(wav_files) >= lo:
        print(f"[gunshots] already present: {len(wav_files)} clips found. Skipping.")
        return True
    return False


def download_gunshots(cfg: dict) -> None:
    """Kabealo et al. gunshot dataset has no stable programmatic Zenodo API
    for this record (gated behind a browser click-through). Detect the
    missing data and print manual instructions rather than failing silently
    or guessing a direct-download URL that may not exist/may break.
    """
    if check_gunshots(cfg):
        return

    zenodo_url = cfg["dataset"]["gunshots"]["zenodo_url"]
    sample_min = cfg["dataset"]["gunshots"]["sample_min"]
    sample_max = cfg["dataset"]["gunshots"]["sample_max"]
    target_dir = RAW_DIR / "gunshots"

    print()
    print("=" * 70)
    print("[gunshots] MANUAL DOWNLOAD REQUIRED — Kabealo et al. Zenodo dataset")
    print("=" * 70)
    print(f"  1. Visit: {zenodo_url}")
    print("     (update configs/finetune.yaml `dataset.gunshots.zenodo_url` /")
    print("      `zenodo_record_id` first if this is still a placeholder)")
    print("  2. Download the dataset archive from the Zenodo record page.")
    print(f"  3. Extract/copy a sample of {sample_min}-{sample_max} .wav clips into:")
    print(f"       {target_dir.resolve()}/")
    print("  4. Re-run this script (or just build_dataset.py) — it will detect")
    print("     the files and skip this step next time.")
    print("=" * 70)
    print()


def log_sources_to_info_md(cfg: dict) -> None:
    """Append resolved source URLs into info.md's dataset-paths placeholders.

    Only called after a successful run — this script is what fills in the
    Phase 0 placeholders left in info.md, not the assistant editing it by hand.
    """
    if not INFO_MD_PATH.exists():
        print("[info.md] not found, skipping source logging.")
        return

    content = INFO_MD_PATH.read_text()
    replacements = {
        "- LibriSpeech dev-clean: `data/raw/librispeech/` — *(source URL / version TBD in Phase 1)*":
            f"- LibriSpeech dev-clean: `data/raw/librispeech/` — "
            f"{cfg['dataset']['librispeech']['url']}",
        "- MUSAN (noise/all): `data/raw/musan/` — *(source URL / version TBD in Phase 1)*":
            f"- MUSAN (noise/all): `data/raw/musan/` — {cfg['dataset']['musan']['url']}",
        "- ESC-50 (gunshot category): `data/raw/esc50/` — *(source URL / version TBD in Phase 1)*":
            f"- UrbanSound8K (gun_shot category): `data/raw/urbansound8k/` — "
            f"{cfg['dataset']['urbansound8k']['url']}",
        "- UrbanSound8K (gun_shot category): `data/raw/urbansound8k/` — *(source URL / version TBD in Phase 1)*":
            f"- UrbanSound8K (gun_shot category): `data/raw/urbansound8k/` — "
            f"{cfg['dataset']['urbansound8k']['url']}",
        "- Kabealo Zenodo gunshot dataset: `data/raw/gunshots/` — *(Zenodo record ID / DOI TBD in Phase 1)*":
            f"- Kabealo Zenodo gunshot dataset: `data/raw/gunshots/` — "
            f"{cfg['dataset']['gunshots']['zenodo_url']} (downloaded {date.today().isoformat()})",
    }
    for old, new in replacements.items():
        content = content.replace(old, new)
    INFO_MD_PATH.write_text(content)
    print("[info.md] source URLs logged.")


def main() -> None:
    cfg = load_config()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    download_librispeech(cfg)
    download_musan(cfg)
    download_urbansound8k(cfg)
    download_gunshots(cfg)  # may only print manual instructions

    log_sources_to_info_md(cfg)

    print()
    print("Done. If [gunshots] printed manual instructions above, complete that")
    print("step before running build_dataset.py.")


if __name__ == "__main__":
    main()
