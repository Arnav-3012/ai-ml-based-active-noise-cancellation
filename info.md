# info.md — Reference

Stable reference doc. Fill in placeholders as decisions are made in later phases.

## Dataset paths
- LibriSpeech dev-clean: `data/raw/librispeech/` — https://www.openslr.org/resources/12/dev-clean.tar.gz
- MUSAN (noise/all): `data/raw/musan/` — https://www.openslr.org/resources/17/musan.tar.gz
- UrbanSound8K (gun_shot category): `data/raw/urbansound8k/` — https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz *(replaces ESC-50, which was found to have no gun_shot class — not yet downloaded)*
- Kabealo Zenodo gunshot dataset: `data/raw/gunshots/` — https://zenodo.org/records/7004819 (Kabealo, Wyatt et al., "A multi-firearm, multi-orientation audio dataset of gunshots," Data in Brief 2023, DOI 10.1016/j.dib.2023.109091) — *(not yet downloaded; requires manual step, see logs.md)*

## dns48 checkpoint
- Source: *(TBD — facebookresearch/denoiser release URL / torch hub identifier, to be filled in Phase 3)*

## Metric definitions
- **SNR (Signal-to-Noise Ratio)**: *(one-liner TBD — formula/reference implementation used, Phase 4)*
- **STOI (Short-Time Objective Intelligibility)**: *(one-liner TBD — via `pystoi`, Phase 4)*
- **PESQ (Perceptual Evaluation of Speech Quality)**: *(one-liner TBD — via `pesq`, mode/version used, Phase 4)*

## Manifest schema
*(TBD — decided in Phase 1 once `build_dataset.py` is written. Will document the JSON fields for each entry in `manifests/*.json`.)*
