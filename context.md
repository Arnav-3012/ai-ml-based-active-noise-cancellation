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
**→ Latest status: v5 is the FINAL fine-tuned model (locked 2026-09-11, no further training
planned). ONNX export (Phase 5a) is COMPLETE and VERIFIED CLEAN (zero export drift, isolated
from a real crop confound) — Core ML/TFLite conversion has not started. Task 1 (focused demo
comparison) is written but not yet run/reported. See "Phase 3c/4 iteration — v2, v4, v5",
"Phase 5a — ONNX export", and "Task 1" sections below. Phases 1-4 below are the historical
build-up to v1.**

**Phase 1 complete — Data pipeline.** All datasets downloaded, `build_dataset.py` ran
successfully end-to-end, outputs verified (wav counts, manifest schema, speaker-disjointness,
audio format).

**Phase 2 complete — Baseline + eval.** Spectral subtraction baseline ran on full test split
(299/299 pairs), evaluate.py produced real SNR/STOI/PESQ numbers with per-category breakdown
(see Evaluation section above), both results files verified present.

**Phase 3a complete — dns48 model loading, verified.** `src/model/load.py` loads the pretrained
dns48 checkpoint via `denoiser.pretrained.dns48()` (denoiser's own mechanism, pip package as-is —
confirmed not to import either patched vendor file from Phase 0). Architecture verified against a
locked spec in `configs/finetune.yaml['model']['expected']`: **18,867,937 params** (exact figure,
not the rounded "~18.9M"), 5 encoder + 5 decoder layers, kernel_size=8, stride=4, 2-layer LSTM
hidden 768 — all matched exactly, zero mismatches. MPS backend used (no CPU fallback needed).
`src/model/verify_load.py` ran real inference on one real Phase 1 test-split file
(`data/processed/test/test_000000_noisy.wav`, 4.8s @ 16kHz): output shape exactly matched input
shape `(1, 76800)`, single-run inference time 615.12 ms on MPS (later confirmed to be cold-start
per-shape MPS kernel compilation, not a real regression — warm timing is ~7-9ms/sec of audio).

**Phase 3b complete — data loading + training-loop smoke test, verified with real values.**
`src/model/dataset.py` (`NoisyCleanDataset`/`make_dataloader`) loads noisy/clean pairs from Phase 1
manifests via `soundfile`, fixed-length random-crop (`training.segment_seconds=4.0`) to collate
variable-duration clips into batches (`training.batch_size=2` for the smoke test).
`src/model/finetune.py` wires forward -> L1+multi-resolution-STFT loss -> backward ->
`optimizer.step()` (Adam, `training.learning_rate=3e-5` — 1/10th of a typical from-scratch rate,
standard fine-tuning heuristic), with optional encoder-layer freezing implemented as a mechanism
but defaulted off (`training.freeze_encoder_layers: 0`, a 3c decision). Ran one real smoke-test
batch: **L1 loss 0.022368, STFT loss 0.175597, total 0.197965** — all non-zero/non-NaN/non-Inf,
plausible for a pretrained-but-not-fine-tuned model on this project's real noisy/clean pairs.
Gradient spot-check on 5 real params after `backward()` — all non-zero, confirming the loss is
actually connected to the model. **This was also the first real test of the Phase 0 STFT loss
patch (`return_complex` fix) against real batched audio via the soundfile I/O path** (not just
Phase 0's synthetic sine tensors) — confirmed working, patched-module import path independently
re-verified via `__file__` inspection (resolves to `src/vendor/denoiser_patched/stft_loss.py`).

**Phase 3c COMPLETE — real training run finished, plateau-stopped, best checkpoint saved and
verified.** Real train/val manifest counts confirmed: **2162 train pairs, 242 val pairs**.
`configs/finetune.yaml['training_run']`: `batch_size: 8`, `max_epochs: 40` (hard ceiling,
confirmed by user — deadline 2026-09-12, 2-2.5hr budget), `early_stopping: {min_delta: 0.001,
patience_epochs: 5}`, `checkpoint_dir: "checkpoints"`, `results_csv:
"results/finetune_training_log.csv"`. Full math/justification for each value in logs.md's Phase
3c ceiling entry.

**Real run result (user ran `python -m src.model.finetune --run` themselves):**
- **Stopped early on plateau at epoch 12** of the 40-epoch ceiling — 5 consecutive epochs (8-12)
  without val-loss improvement > `min_delta=0.001`.
- **Best checkpoint: epoch 7**, `checkpoints/dns48_finetuned_best.pt` (226,473,933 bytes),
  `val_total_loss=0.143079` (`val_l1=0.012954`, `val_stft=0.130126`) — verified by loading the
  checkpoint directly and confirming its internal epoch/loss fields match the console log exactly.
- **Wall-clock: ~26.5 min total** (~132s/epoch avg, faster than the ~3.64min/epoch pre-run
  estimate — the 3.5x training-vs-inference multiplier assumption was conservative).
- **Trend:** train loss decreased monotonically (0.1587→0.1413 over 12 epochs); val loss
  plateaued/oscillated in a tight band from epoch 7 onward; train/val gap widened (epoch 1: ~0.011
  → epoch 11: ~0.019) — an early-overfitting signature for this dataset size (2162 pairs), flagged
  honestly, not smoothed over. This is the genuine plateau case (NOT ceiling-reached-without-
  plateau) — must be reported as "plateaued at epoch 12, best val loss at epoch 7," never as
  "ceiling reached" (it wasn't) or silently upgraded to "fully converged" (unresolved until
  evaluated against real metrics).

**Decision:** No hyperparameter tweaks made preemptively, per user instruction. Next step: Phase 4
evaluation (below).

**Phase 4 COMPLETE — real evaluation run, all three columns populated (v1).** `src/model/inference.py`
loaded the fine-tuned checkpoint (weight-diff verified non-trivial: `torch.equal()` → False vs.
pretrained, max abs diff 0.005874 on `encoder.0.0.weight`) and ran inference over all 299 test
pairs → `results/finetuned/*_enhanced.wav`. `src/eval/evaluate.py` scored all three columns in one
run, wrote `results/baseline_results.{json,csv}`.

**PS-target assessment (SNR>15dB, STOI>0.85, PESQ>2.5) — NEVER cleared, by any version through v5**
(see Phase 5 section below). v1's numbers superseded by v5 as the final model; full v1 numbers and
v1-vs-baseline analysis are in `logs.md`'s Phase 4 entry, not repeated here.

## Phase 3c/4 iteration — v2, v4, v5 fine-tuning runs (v5 is FINAL, locked 2026-09-11)

Full rationale/config diffs for each version are in `configs/finetune.yaml`
(`training_run`/`training_run_v4`/`training_run_v5` sections, heavily commented) and in
`logs.md`'s "v2 through v5 fine-tuning iterations + FINAL model decision" entry. Summary:

- **v1 → v2 diagnosis:** `diagnose_failures.py` labeled 18/18 of v1's worst test pairs
  "likely over-suppression" (model erasing signal, not failing to clean it) — this diagnosis
  drove every subsequent version's levers, not blind hyperparameter search.
- **v2** (`checkpoints/dns48_finetuned_v2_best.pt`, epoch 7 best-val): LR warmup+cosine
  (replacing v1's flat 3e-5), STFT loss weight 1.0→2.0, 3x hard-mixture (low-SNR) oversampling.
  Result: near-identical to v1 on all metrics. `diagnose_data_gaps.py` follow-up ruled OUT
  gunshot-specific data scarcity — the gap is general low-SNR capability, not category-specific.
- **v4** (`checkpoints/dns48_finetuned_v4_best.pt`, epoch 11 best-val): reverted v2's unproven
  STFT reweight and static oversampling; added epoch-aware **curriculum learning** (ramps
  low-SNR and gunshot sampling weight over first 40% of epochs, multiplicative when both apply)
  and a **silence-collapse penalty** (targets over-suppression directly, keyed off the noisy
  input's own energy envelope). Result: again near-identical to v1/v2.
- **v5 — FINAL MODEL** (`checkpoints/dns48_finetuned_v5_best.pt`, epoch 27 best-val,
  `val_total_loss=0.13689`, the best of all four): **dynamic mixing** replaces the static
  2162-pair manifest entirely (fresh mixtures drawn per step, uniform 1/3-per-category weights
  to match v1-v4's natural balance, `steps_per_epoch=270` kept identical to v1-v4 for
  comparability); **reactive LR** (`ReduceLROnPlateau`, factor 0.5, patience 5, min_lr 1e-5,
  replacing fixed cosine decay). Silence-penalty term computed/logged every epoch but
  `silence_penalty_enabled: false` — verified numerically this session (`train_total ≈
  train_l1 + train_stft` to 8 decimals in `finetune_training_log_v5.csv`), i.e. measured but
  NOT applied to gradients this run. Ran 42 epochs total.

### Real results, all four versions, overall (test split, n=299; source: `results/baseline_results.csv`)
| version | n   | SNR(dB) | STOI   | PESQ   |
|---------|-----|---------|--------|--------|
| v1      | 299 | 8.671   | 0.6620 | 1.7882 |
| v2      | 299 | 8.564   | 0.6656 | 1.7753 |
| v4      | 299 | 8.586   | 0.6631 | 1.7718 |
| **v5**  | 299 | **8.750** | **0.6744** | **1.7863** |

v5 is best or near-best on every metric (narrowly below v1 on PESQ specifically). Full
per-category and per-SNR-bucket tables for all four versions are in `logs.md`, not repeated here.

**FINAL DECISION (user, 2026-09-11): v5 is locked — no further training iterations planned.**
PS numeric targets (SNR>15dB, STOI>0.85, PESQ>2.5) remain unreached by any version. This is
reported as-is, not smoothed over: the project's real, consistent contribution is beating the
classical spectral-subtraction baseline by a wide margin on every metric/category across four
methodologically-motivated iterations — not clearing the PS's absolute numeric bar.

## Phase 5a — ONNX export: COMPLETE, VERIFIED CLEAN

Scope: ONNX export of the v5 checkpoint, then Core ML/TFLite conversion (only if ONNX
verification passes cleanly — do not proceed on a broken export). **ONNX step is done and
passed; Core ML/TFLite conversion has not started.**

- `src/export/to_onnx.py`: exports `checkpoints/dns48_finetuned_v5_best.pt` to ONNX
  (opset 17). Known risk (dns48's 2-layer LSTM, documented ONNX dynamic-sequence-length export
  fragility) was flagged explicitly in-code before running — **did not materialize**: both the
  fixed-length export (`checkpoints/onnx/dns48_finetuned_v5.onnx`, no `dynamic_axes`) AND the
  dynamic-length export (`checkpoints/onnx/dynamic_dns48_finetuned_v5.onnx`) succeeded on the
  real run, no silent fallback needed.
- `src/export/verify_onnx.py`: ran the fixed-length ONNX model via `onnxruntime` over the full
  299-pair test split. Real result: overall SNR 8.4508dB, STOI 0.6700, PESQ 1.7505 — delta vs.
  v5's PyTorch numbers (SNR 8.75, STOI 0.674, PESQ 1.786): **SNR -0.2992dB, STOI -0.0040,
  PESQ -0.0355**. This run's own logged warning flagged a real confound: test clips are
  center-cropped/zero-padded to the 4.0s fixed export length before scoring, and **184/299 test
  clips (61.5%) are longer than 4s** (confirmed by direct manifest inspection: min 1.445s, max
  32.485s, mean 6.47s) — so this delta could not, by itself, be attributed to export drift vs.
  crop-discarded content.
- `src/export/verify_onnx_isolated.py` (new, written specifically to resolve that ambiguity):
  feeds the IDENTICAL cropped waveform to both the ONNX graph and the PyTorch model (crop
  variable held constant, only the runtime differs) and separately reports the crop effect
  (pytorch-on-cropped-input vs. v5's original full-length PyTorch numbers).

### Real isolated result (test split, n=299)
| comparison | SNR(dB) | STOI | PESQ |
|---|---|---|---|
| **export drift** (onnx vs. pytorch, identical cropped input) | **-0.0000** | **-0.0000** | **-0.0000** |
| **crop effect** (pytorch-on-cropped-input vs. v5's full-length PyTorch eval) | -0.2992 | -0.0040 | -0.0355 |

**Conclusion: export drift is exactly zero (to displayed precision).** The entire gap
`verify_onnx.py` first reported is fully explained by the fixed-length crop discarding real
audio on the 61.5% of clips longer than 4s — not by any loss from the ONNX export itself. This
satisfies the task's own stopping condition ("don't proceed to Core ML/TFLite on a broken
export") — the export is clean, proceeding is supported.

- New `export:` config block in `configs/finetune.yaml` (`checkpoint_path`, `onnx_dir`,
  `onnx_filename`, `fixed_length_seconds: 4.0` — matches `training.segment_seconds`,
  `opset_version: 17`) — per rule 1, no magic numbers inline in export code.
- `onnx`/`onnxruntime` now installed (user ran the real export/verify commands successfully) —
  not yet added to `requirements.txt` as a formal pin (should be done before this is considered
  fully closed out).

**Open decision, not made here:** whether the fixed-length-only (4.0s window) constraint is
acceptable for the eventual on-device deployment shape, or whether the dynamic-length export
(which also succeeded) should be carried forward instead to avoid chunking/padding longer audio
at inference time — for the user to decide before Core ML/TFLite conversion begins.

## Task 1 — focused demo comparison (`evaluate_final.py`)

Code written (`src/eval/evaluate_final.py`, `results/final_comparison.csv` output target) but
**real output not yet reported by the user** — status unresolved, not run/verified as of this
snapshot.

## Open decisions
- Whether v5's silence-penalty term should be re-enabled (it was computed but not applied to
  gradients this run) — was pending user confirmation per the config's own comment; not resolved
  in this session, not assumed either way.
- Whether/how to close the remaining gap to the PS's absolute numeric targets is explicitly
  NOT part of current scope — v5 is locked, no further training planned. Any future work on this
  gap (architecture changes, more data, etc.) would be a new, separately-scoped effort.
- Whether Core ML/TFLite conversion proceeds from the fixed-length ONNX export or the
  dynamic-length one — both succeeded and are verified equally available; fixed-length avoids
  any residual dynamic-axis runtime risk, dynamic-length avoids a chunk/pad step at inference
  time. Not decided here.
- `onnx`/`onnxruntime` are installed and working but not yet formally pinned in
  `requirements.txt` — should be added before Phase 5a is considered fully closed out.
