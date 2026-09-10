# context.md — Current State Snapshot

> This file is overwritten fully each session. It reflects *current* state only — history lives in `logs.md`.

## Project
SIH26052 — fine-tuning Facebook Research's Denoiser (dns48 checkpoint) for defence-noise speech enhancement (e.g. gunshots, battlefield/vehicle noise) on top of clean/noisy speech pairs.

## Model
- Base: **dns48** (Facebook Research `denoiser`), ~18.9M params.
- Verified warm inference latency: **~9ms/sec of audio** on MPS (M4 Pro).
- Fine-tuned via transfer learning only — no training from scratch.

## Datasets (planned, not yet downloaded)
- **LibriSpeech dev-clean** — ~2000–2500 utterances (clean speech source).
- **MUSAN** — noise/all subset (general noise).
- **ESC-50** — gunshot category, ~40 clips.
- **Kabealo Zenodo gunshot dataset** — ~800–1200 clips (defence-relevant gunshot noise).

## Data pipeline (planned)
- Mixing: SNR-targeted mixing, range **-5 dB to 15 dB**.
- Split: **speaker-disjoint** (no speaker overlap between train/val/test).
- Augmentation: reverb + clipping (post-mixing).

## Baseline
- Hand-built **spectral subtraction** (not `noisereduce` or any third-party library) — implemented from scratch in `src/baseline/spectral_subtraction.py`.

## Loss (planned)
- **L1 + multi-resolution STFT loss**.

## Evaluation (planned)
- Metrics: **SNR, STOI, PESQ**, via `pystoi` and `pesq` libraries.
- Runs both fine-tuned model and baseline, writes a results table.

## MVP scope
- End-to-end: mix defence-noise-corrupted speech → fine-tune dns48 → beat spectral-subtraction baseline on SNR/STOI/PESQ on a held-out speaker-disjoint test set.

## Explicitly out of scope
- No dashboard/UI (Streamlit/Gradio) — decided against.
- No training from scratch.
- No real-time streaming inference (batch/offline only, for now).
- No mobile/edge deployment until Export phase (Phase 5), and only ONNX → CoreML/TFLite.

## Environment / dependencies
- **torch==2.14.0**, **torchaudio==2.11.0** — exact-pinned in `requirements.txt`. This is the correct current pairing: torchaudio's release cadence lags torch's, so no `2.14.x` torchaudio exists on PyPI. Verified directly against PyPI's JSON API, not assumed.
- **denoiser==0.1.5** (unpinned in requirements — see below) has two confirmed runtime breakages against torch 2.14/torchaudio 2.11: `torchaudio.get_audio_backend()` removed, and `torch.stft()` requires `return_complex` now. Both patched locally.
- **Do not import `denoiser.audio` or `denoiser.stft_loss` directly** — use `src/vendor/denoiser_patched/audio.py` (`Audioset`, `find_audio_files`, `get_info`) and `src/vendor/denoiser_patched/stft_loss.py` (`MultiResolutionSTFTLoss`) instead. Everything else from `denoiser` (model architecture, pretrained checkpoint loading, `convert_audio`/resample) is used as-is from the pip package. Full patch rationale/diffs in `src/vendor/denoiser_patched/README.md`.
- **All audio file I/O goes through `soundfile`, not `torchaudio.load`/`torchaudio.save`.** torchaudio 2.11's I/O backend requires `torchcodec`, which requires native FFmpeg linking that failed to load on this machine (missing dylib on the linker search path) and was judged too fragile to depend on (this project may later move to Colab/Kaggle if MPS training hits limits). `torchaudio` stays installed for non-I/O ops; `torchcodec` is not a dependency.
- MPS backend confirmed working (`torch.backends.mps.is_available()` → `True`) after all of the above.

## Current phase
**Phase 0 — Scaffold.** Repo structure, gitignore, venv, and dependency install only. No data-mixing, model-loading, or training code exists yet.
