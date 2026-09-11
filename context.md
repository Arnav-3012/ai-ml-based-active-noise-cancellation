# context.md — Current State Snapshot

> This file is overwritten fully each session. It reflects *current* state only — history lives in `logs.md`.

## Project
SIH26052 — fine-tuning Facebook Research's Denoiser (dns48 checkpoint) for defence-noise speech enhancement (e.g. gunshots, battlefield/vehicle noise) on top of clean/noisy speech pairs.

## Model
- Base: **dns48** (Facebook Research `denoiser`), ~18.9M params.
- Verified warm inference latency: **~9ms/sec of audio** on MPS (M4 Pro).
- Fine-tuned via transfer learning only — no training from scratch.

## Datasets — ALL DOWNLOADED
- **LibriSpeech dev-clean** — 2703 utterances, 40 speakers (clean speech source). Native 16kHz.
- **MUSAN** — 930 noise clips (noise/all subset: 87 "stationary" from sound-bible subfolder, 843 "general"). Native 16kHz.
- **UrbanSound8K** — gun_shot category (classID 6). Replaces ESC-50 (originally planned but
  verified to have NO gun_shot class among its 50 official categories — real dataset-selection
  error caught when the ESC-50 step returned 0 clips on the user's actual run). Downloaded from
  Zenodo (https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz, ~6GB tarball, all 10
  classes — only gun_shot kept, rest discarded). Native rate mixed 44.1/48/96kHz.
- **Kabealo Zenodo gunshot dataset** — Zenodo record 7004819 (Kabealo, Wyatt et al., "A
  multi-firearm, multi-orientation audio dataset of gunshots," Data in Brief 2023). User had
  already manually downloaded all 2,148 clips (4 firearms: remington_870_12_gauge, glock_17_9mm,
  ruger_ar_556_dot223, 38s&ws_dot38) to `~/edge-collected-gunshot-audio/`; copied in full (not
  sampled — decision made to keep full diversity) into `data/raw/gunshots/`. Native rate 44.1kHz.
- Combined gunshot noise pool: 2522 clips (2148 Kabealo + ~374 UrbanSound8K).

## Data pipeline — PHASE 1 COMPLETE, verified with real output
- Mixing: SNR-targeted, range **-5 dB to 15 dB** (confirmed in manifest `snr_db` values).
- Split: **speaker-disjoint**, 80/10/10 by speaker — train 32 speakers/2162 utterances, val 4/242,
  test 4/299. Verified zero speaker-ID overlap across all three splits.
- Augmentation: reverb (p=0.3) + clipping (p=0.15), post-mixing.
- Noise-source resampling: MUSAN/LibriSpeech are natively 16kHz; UrbanSound8K and Kabealo
  gunshots are not (44.1-96kHz). Initial `build_dataset.py` run crashed on this (strict
  sample-rate check, by design, meant to catch real bugs — but this was expected raw-source
  variation, not a bug). Fixed by adding `_load_noise_wav()` which resamples noise clips
  on-the-fly via `torchaudio.functional.resample` (pure tensor op, no I/O backend) during
  mixing; clean-speech loading (`_load_wav()`) stays strict since LibriSpeech is always 16kHz
  natively, so a mismatch there would mean an actual problem.
- Output verified: `data/processed/{train,val,test}/` contain clean+noisy wav pairs (16kHz mono
  PCM_16, confirmed via `soundfile.info()`), `manifests/{train,val,test}.json` contain one entry
  per pair with pair_id/paths/speaker_id/split/snr_db/noise_category/noise_source/source_utterance.

## Baseline — PHASE 2 COMPLETE, verified with real output
- Hand-built **spectral subtraction** (not `noisereduce` or any third-party library) — `src/baseline/spectral_subtraction.py`. STFT-domain, noise estimated from the first N frames of the noisy signal (Boll 1979 convention), Berouti et al. 1979 oversubtraction + spectral floor. Reads/writes via `soundfile` only.
- Tunables (`baseline.*` in `configs/finetune.yaml`): `stft.{n_fft=512, hop_length=128, win_length=512}`, `noise_estimate_frames=6`, `oversubtraction_factor=1.5`, `spectral_floor=0.02`.
- Known documented limitation: noise-estimate window assumes leading frames are noise-only, which isn't guaranteed by Phase 1's mixing (noise overlays the full utterance) — left as-is since exposing this classical failure mode is the baseline's purpose.
- Ran successfully on full test split (299 pairs) → `results/baseline/*_enhanced.wav` (299 files, verified count).

## Loss (planned)
- **L1 + multi-resolution STFT loss**.

## Evaluation — PHASE 2 COMPLETE, verified with real output
- `src/eval/metrics.py`: `snr()`, `stoi_score()` (via `pystoi`), `pesq_score()` (via `pesq`, wideband mode — `eval.pesq_mode` in config).
- `src/eval/evaluate.py`: scores `noisy` (raw floor) and `baseline` columns vs. clean test set, per-noise-category (gunshot/stationary/general) + overall breakdown. `finetuned` column wired but empty (n=0) until Phase 3. Writes `results/baseline_results.{json,csv}` (both verified present).

### Real results (test split, n=299)
| column   | category   | n   | SNR(dB) | STOI  | PESQ  |
|----------|-----------|-----|---------|-------|-------|
| noisy    | gunshot   | 97  | -1.09   | 0.566 | 1.296 |
| noisy    | stationary| 98  | 2.01    | 0.631 | 1.368 |
| noisy    | general   | 104 | 0.47    | 0.596 | 1.260 |
| noisy    | overall   | 299 | 0.47    | 0.598 | 1.307 |
| baseline | gunshot   | 97  | -0.12   | 0.561 | 1.297 |
| baseline | stationary| 98  | 3.22    | 0.626 | 1.309 |
| baseline | general   | 104 | 2.36    | 0.588 | 1.271 |
| baseline | overall   | 299 | 1.84    | 0.592 | 1.292 |

**Reading:** baseline lifts overall SNR (+1.37 dB) but STOI and PESQ both drop slightly vs. doing nothing — this is exactly the classical spectral-subtraction failure mode the baseline was built to expose (trades raw energy-ratio for perceptual/intelligibility quality via musical noise). Gunshot category is hardest and stays SNR-negative (transient noise breaks the "stationary noise fingerprint" assumption worst). This is the number fine-tuning (Phase 3) needs to beat, especially on STOI/PESQ, not just SNR.

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
**Phase 1 complete — Data pipeline.** All datasets downloaded, `build_dataset.py` ran
successfully end-to-end, outputs verified (wav counts, manifest schema, speaker-disjointness,
audio format).

**Phase 2 complete — Baseline + eval.** Spectral subtraction baseline ran on full test split
(299/299 pairs), evaluate.py produced real SNR/STOI/PESQ numbers with per-category breakdown
(see Evaluation section above), both results files verified present. Ready to move to
model-loading/fine-tuning (Phase 3) — no such code exists yet.
