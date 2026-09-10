# info.md — Reference

Stable reference doc. Fill in placeholders as decisions are made in later phases.

## Dataset paths
- LibriSpeech dev-clean: `data/raw/librispeech/` — *(source URL / version TBD in Phase 1)*
- MUSAN (noise/all): `data/raw/musan/` — *(source URL / version TBD in Phase 1)*
- ESC-50 (gunshot category): `data/raw/esc50/` — *(source URL / version TBD in Phase 1)*
- Kabealo Zenodo gunshot dataset: `data/raw/gunshots/` — *(Zenodo record ID / DOI TBD in Phase 1)*

## dns48 checkpoint
- Source: *(TBD — facebookresearch/denoiser release URL / torch hub identifier, to be filled in Phase 3)*

## Metric definitions
- **SNR (Signal-to-Noise Ratio)**: *(one-liner TBD — formula/reference implementation used, Phase 4)*
- **STOI (Short-Time Objective Intelligibility)**: *(one-liner TBD — via `pystoi`, Phase 4)*
- **PESQ (Perceptual Evaluation of Speech Quality)**: *(one-liner TBD — via `pesq`, mode/version used, Phase 4)*

## Manifest schema
*(TBD — decided in Phase 1 once `build_dataset.py` is written. Will document the JSON fields for each entry in `manifests/*.json`.)*
