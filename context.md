# context.md — Current State Snapshot

> This file is overwritten fully each session. It reflects *current* state only — history lives in `logs.md`.

## Project
SIH26052 — fine-tuning Facebook Research's Denoiser (dns48 checkpoint) for defence-noise speech enhancement (e.g. gunshots, battlefield/vehicle noise) on top of clean/noisy speech pairs, deployed as a real-time-feeling on-device iOS demo.

## Model
- Base: **dns48** (Facebook Research `denoiser`), **18,867,937 params** (exact, not the rounded "~18.9M").
- Fine-tuned via transfer learning only — no training from scratch (CLAUDE.md rule 6).
- **v5 is the FINAL, locked fine-tuned model** (locked 2026-09-11, no further training planned). Checkpoint: `checkpoints/dns48_finetuned_v5_best.pt`, best-val epoch 27 of 42 run, `val_total_loss=0.13689` (best of all four versions: v1/v2/v4/v5).
- Verified warm inference latency: **~7-9ms/sec of audio** on MPS (M4 Pro, PyTorch). **On-device (iPhone, Core ML INT8): ~274ms cold (first call), ~53-55ms warm** — same cold/warm-per-unique-shape pattern as the Mac, now also confirmed on Core ML.
- Shipped for on-device deployment: `dns48_finetuned_v5_int8.mlpackage` (18MB, weights-only INT8 quantization, verified rounding-level accuracy delta vs. fp32). Shipping precision is **locked at FLOAT32 for the reference/verification artifact** on defence-context safety-margin grounds (fp16 rejected despite acceptable isolated drift, due to ~111x real-time headroom making the speed/size tradeoff unnecessary); INT8 is a separately verified, available option for app-bundle-size reasons specifically, not a change to that safety-margin call.

## Datasets — ALL DOWNLOADED
- **LibriSpeech dev-clean** — 2703 utterances, 40 speakers (clean speech source). Native 16kHz.
- **MUSAN** — 930 noise clips (noise/all subset: 87 "stationary" from sound-bible subfolder, 843 "general"). Native 16kHz.
- **UrbanSound8K** — gun_shot category (classID 6). Replaces ESC-50 (verified to have NO gun_shot class). Native rate mixed 44.1/48/96kHz.
- **Kabealo Zenodo gunshot dataset** — Zenodo record 7004819, full 2,148-clip set (not sampled) in `data/raw/gunshots/`. Native 44.1kHz.
- Combined gunshot noise pool (static, v1-v4 era): 2522 clips (2148 Kabealo + ~374 UrbanSound8K).
- **v5's expanded dynamic speech pool:** 134 speakers / 44.71 hours (vs. the original static split's 32 speakers/2,162 utterances) — see "v5 fine-tuning" section below.

## Data pipeline — PHASE 1 COMPLETE (static manifests), v5 supersedes with dynamic mixing
- Static manifests (used by v1/v2/v4): SNR-targeted mixing, range **-5 dB to 15 dB**, speaker-disjoint 80/10/10 split — train 32 speakers/2162 utterances, val 4/242, test 4/299 (zero speaker-ID overlap, verified). Augmentation: reverb (p=0.3) + clipping (p=0.15).
- **v5 replaced the static train manifest with dynamic on-the-fly mixing** (`src/data/dynamic_dataset.py`, `src/data/build_dynamic_pool.py`, `manifests/dynamic_train_pool.json`) — fresh mixtures drawn per step from a 134-speaker/44.71hr speech pool, uniform 1/3-per-category noise weights (matches v1-v4's natural balance), `steps_per_epoch=270` kept identical to v1-v4 for comparability. Val/test splits are unchanged (same fixed 242/299 pairs used across all versions, for fair comparison).
- Noise-source resampling: `_load_noise_wav()` resamples non-16kHz noise (UrbanSound8K/Kabealo) on-the-fly via `torchaudio.functional.resample`; clean speech loading stays strict (LibriSpeech always native 16kHz).

## Baseline — PHASE 2 COMPLETE, verified with real output
- Hand-built **spectral subtraction** — `src/baseline/spectral_subtraction.py`. STFT-domain, Boll 1979 noise estimate + Berouti et al. 1979 oversubtraction/floor.
- Tunables (`baseline.*` in `configs/finetune.yaml`): `stft.{n_fft=512, hop_length=128, win_length=512}`, `noise_estimate_frames=6`, `oversubtraction_factor=1.5`, `spectral_floor=0.02`.
- Real results (test split, n=299, overall): SNR 1.84dB, STOI 0.592, PESQ 1.292 — improves raw-noisy SNR (+1.37dB) but *regresses* STOI/PESQ slightly vs. doing nothing (classical musical-noise failure mode, the exact thing this baseline exists to expose).

## Loss
- **L1 + multi-resolution STFT loss** (`src/model/finetune.py::compute_loss`, `src/vendor/denoiser_patched/stft_loss.py`). 3 resolutions: `fft_sizes=[1024,2048,512]`, `hop_sizes=[120,240,50]`, `win_lengths=[600,1200,240]`. L1:STFT weight 1:1 for v1/v4/v5 (v2 tried 2.0 STFT weight, reverted — no measured benefit).

## Evaluation — PHASE 2/4 COMPLETE, verified with real output
- `src/eval/metrics.py`: `snr()`, `stoi_score()` (pystoi), `pesq_score()` (pesq, wideband).
- `src/eval/evaluate.py`: scores noisy/baseline/finetuned columns, per-category + overall. `src/eval/evaluate_final.py`: focused v5-vs-baseline-vs-noisy demo comparison (Task 1) — **written, real output status not reconfirmed this session, treat as unresolved unless re-checked.**
- `src/eval/diagnose_failures.py` / `diagnose_data_gaps.py`: the real diagnostic tooling behind the v1→v5 iteration story (see Phase 3 section below and `docs/phase3_training_deep_dive.md`).

## v1 → v5 fine-tuning iteration summary (full diagnostic story in `logs.md` and `docs/phase3_training_deep_dive.md`)
- **v1** (epoch 7 best-val): flat LR 3e-5. Overall: SNR 8.671dB, STOI 0.6620, PESQ 1.7882.
- **v1→v2 diagnosis:** `diagnose_failures.py` labeled 18/18 of v1's worst pairs "likely over-suppression" — drove every subsequent lever.
- **v2** (epoch 7 best-val): warmup+cosine LR (peak 1.5e-4), STFT weight 2.0, 3x hard-mixture oversampling. Near-identical to v1 (SNR 8.564, STOI 0.6656, PESQ 1.7753) — ruled out optimizer/loss config as the bottleneck. `diagnose_data_gaps.py` ruled out gunshot-specific data scarcity (general low-SNR gap, not category-specific).
- **v4** (epoch 11 best-val, ran to epoch 31): reverted v2's unproven levers; added curriculum learning (low-SNR/gunshot sampling ramp, first 40% of epochs) + silence-collapse penalty (computed, weight 0.05). Again near-identical (SNR 8.586, STOI 0.6631, PESQ 1.7718) — confirmed the bottleneck wasn't the training recipe at all.
- **v5 — FINAL** (epoch 27 best-val, ran 42 epochs): root cause pinpointed as **static-dataset exhaustion** (all 3 prior versions plateaued identically despite different recipes because all trained on the same fixed 2,162-pair manifest). Fix: dynamic mixing (134 speakers/44.71hr, fresh mixtures per step) + `ReduceLROnPlateau` (factor 0.5, patience 5, min_lr 1e-5) replacing fixed cosine decay. **Best of all four versions on SNR/STOI, narrowly below v1 on PESQ**: SNR 8.750dB, STOI 0.6744, PESQ 1.7863.
- **PS-target assessment (SNR>15dB, STOI>0.85, PESQ>2.5) — NEVER cleared, by any version.** Reported honestly: the real, consistent contribution is beating the classical baseline by a wide margin (SNR +6.91dB, STOI +0.082, PESQ +0.494, overall) across four methodologically-motivated iterations, not clearing the PS's absolute numeric bar. Δ-SNR (not absolute output SNR) is the honest reporting frame given the test set's -5 to 15dB input range — see `docs/phase3_training_deep_dive.md` Section 6 for the full mathematical justification.

**FINAL DECISION (user, 2026-09-11): v5 is locked. No further training iterations planned.**

## Phase 5a — ONNX export: COMPLETE, VERIFIED CLEAN
- `src/export/to_onnx.py`: both fixed-length (`checkpoints/onnx/dns48_finetuned_v5.onnx`, opset 17) and dynamic-length exports succeeded — the flagged LSTM/dynamic-sequence-length export risk did not materialize.
- Verification (isolated from a real crop confound via `verify_onnx_isolated.py`): **export drift is exactly 0.0000 (SNR/STOI/PESQ) to displayed precision.** The raw first-pass delta (SNR -0.2992dB etc.) was fully attributable to the fixed-length crop discarding real content on 61.5% of test clips (184/299 longer than 4s), not to the export itself.

## Phase 5b — Core ML conversion (iOS target): COMPLETE, all three precisions verified
Real path included two genuine dead-ends, root-caused and fixed (full detail in `logs.md` and `docs/export_quantization_deep_dive.md`):
1. `coremltools` 9.0 has no ONNX converter — switched to tracing the PyTorch checkpoint directly (`torch.jit.trace`).
2. `mlprogram` (required for iOS15+ deployment target) defaults to fp16 compute precision — forced `compute_precision=FLOAT32` for the reference conversion so precision drift wasn't conflated with format drift.
3. **Real bug, root-caused and fixed:** `Demucs`'s internal Python-int length arithmetic (`valid_length`, `downsample2`/`upsample2` parity checks) got traced as tensor ops by `torch.jit.trace`, crashing `coremltools`' MIL frontend (`TypeError: only 0-dimensional arrays...`). Fixed via `src/export/traceable_demucs.py::TraceableDemucs` — hardcodes all length-dependent arithmetic as precomputed Python constants, valid ONLY because this export targets one fixed input length (64,000 samples / 4.0s). Verified exact 0.0 max-abs-diff, both eager-wrapper-vs-original and traced-vs-eager, before trusting the fix.

**All three precision artifacts verified against the same fixed-length-crop methodology (n=299), each isolated as its own variable:**

| artifact | size | SNR(dB) delta vs fp32 | STOI delta | PESQ delta | verdict |
|---|---|---|---|---|---|
| `dns48_finetuned_v5.mlpackage` (fp32) | 72MB | — (reference) | — | — | verified clean vs. ONNX (0.0000 delta) |
| `dns48_finetuned_v5_fp16.mlpackage` | 54MB (~25% smaller) | -0.2472 | -0.0007 | -0.0624 | **fails** script's own small-delta threshold on SNR/PESQ (~5x/~3x over) — real, non-trivial cost |
| `dns48_finetuned_v5_int8.mlpackage` (weights-only) | 18MB (exactly 4.00x smaller) | -0.0065 | -0.0002 | -0.0024 | **passes** threshold comfortably — rounding-level cost |

**Shipping decision:** FLOAT32 locked as the precision of record (defence-context safety margin — ~111x real-time headroom on M4 Pro makes fp16/int8's speed benefit unnecessary, so there's no reason to accept fp16's measured non-trivial cost). INT8 is verified clean and available as an option specifically for app-bundle-size reasons, independent of the safety-margin call — **this is the artifact actually shipped in the iOS app** (`ANCDemo/ANCDemo/Resources/dns48_finetuned_v5_int8.mlpackage`), since on-device bundle size is a real constraint that reasoning doesn't waive.

## iOS demo app — `ANCDemo/` — COMPLETE, all four stages verified on a physical device
Native SwiftUI app (bundle ID `com.arnav.anc.ANCDemo`, automatic signing, team `MGWJ67MC23`), shipping `dns48_finetuned_v5_int8.mlpackage`. Four stages, each verified before the next started:
1. **Model load & verify** — confirmed the shipped model loads/runs correctly on-device; this is where the real cold/warm latency numbers were established.
2. **Record → infer → play** — `AVAudioRecorder` (not `AVAudioEngine` — no real-time streaming need), reuses the exact model instance and fit-to-64000 logic from stage 1, Raw/Enhanced A/B playback.
3. **Clean & Share / AirDrop** — native `UIActivityViewController` share sheet, surfaces AirDrop automatically.
4. **UI polish** — status states, recording pulse animation, inference-time readout, playback-overlap fix.

**Real on-device numbers:** cold (first call) ~274ms, warm (every call after) ~53-55ms — matches the Mac's per-shape MPS cold/warm pattern. **Design decision:** silent launch warmup (`ModelRunner.warmup()`, fire-and-forget background task at app launch) absorbs the cold cost before the user ever taps record, so the first real recording's displayed time reflects the warm number.

Model input contract (locked, verified against export code, not assumed): 16kHz mono Float32, exactly 64,000 samples, tensor names `noisy_waveform`/`enhanced_waveform`.

## Teaching documentation — `docs/` — COMPLETE
Three long-form, beginner-to-master documents, grounded entirely in this repo's real numbers/config/code (no invented figures):
- `docs/phase3_training_deep_dive.md` (~7,325 words) — fine-tuning fundamentals, training loop mechanics, loss function derivation, LR scheduling derivation, the full v1→v5 diagnostic journey (centerpiece), Δ-SNR reporting rationale, final results.
- `docs/export_quantization_deep_dive.md` (~5,220 words) — ONNX fundamentals, the real `TraceableDemucs` tracing bug and fix, the full verification-chain isolation methodology, fp32/fp16/int8 from first principles, the real precision shipping decisions.
- `docs/ios_app_brief.md` (~1,286 words, intentionally brief) — Core ML basics, Xcode/device pairing chain, four build stages, real latency numbers.

## MVP scope
- End-to-end: mix defence-noise-corrupted speech → fine-tune dns48 → beat spectral-subtraction baseline on SNR/STOI/PESQ on a held-out speaker-disjoint test set → export/quantize for on-device → working iOS demo. **All stages complete.**

## Explicitly out of scope
- No dashboard/UI (Streamlit/Gradio) in `src/` — decided against (a standalone judge-facing Streamlit dashboard outside `src/`, reading only `results/`, is in scope per CLAUDE.md rule 5's 2026-09-12 exception, but has not been built this session).
- No training from scratch.
- No real-time streaming inference (batch/offline record-then-process only).
- TFLite conversion was never pursued once iOS/Core ML was confirmed as the actual target platform.

## Environment / dependencies
- **torch==2.14.0**, **torchaudio==2.11.0** — exact-pinned, confirmed correct current pairing (torchaudio's cadence lags torch's).
- **denoiser==0.1.5** — two runtime breakages against this torch/torchaudio pairing, patched locally in `src/vendor/denoiser_patched/` (`audio.py`, `stft_loss.py`). Do not import `denoiser.audio`/`denoiser.stft_loss` directly.
- **All audio file I/O goes through `soundfile`**, not `torchaudio.load`/`save` (torchcodec's native FFmpeg linking proved too fragile to depend on).
- MPS backend confirmed working throughout.
- **Still not formally pinned in `requirements.txt`:** `onnx`, `onnxruntime`, `coremltools` — all installed and working, used successfully across Phase 5a/5b, but the pin was never added. Open item, carried forward from earlier sessions.

## Repo hygiene — RESOLVED this session
A prior commit (`a54e699`, "prototype over") accidentally committed and pushed ~300MB of model binaries to `origin/main` — three Core ML `.mlpackage` bundles, two ONNX `.onnx.data` sidecars, and a duplicate `.mlpackage` bundled into the iOS app's Resources — because `.gitignore` covered `*.onnx`/`*.pt` but not `*.onnx.data` or `*.mlpackage` (a directory-based format those patterns never matched). Fixed via `git filter-repo` (installed this session), stripping all three paths from every commit in history (backed up first via `git clone --mirror`), followed by a `.gitignore` fix (`*.onnx.data`, `*.mlpackage/`, `checkpoints/coreml/`, `checkpoints/onnx/`, the app's Resources path) and a force-push of the rewritten history to `origin/main`. Verified clean on both local and remote after the fact — `.git` dropped from 300MB+ to 656KB, zero references to the stripped paths anywhere in history. **Flagged: any other clone/fork of this repo is now diverged and needs a fresh clone, not a pull.** Model files themselves remain present and usable on local disk — only untracked from git going forward.

## Open decisions
- Whether v5's silence-penalty term should be re-enabled (computed but not applied to gradients in the v5 run) — still not resolved, not assumed either way.
- Whether/how to close the remaining gap to the PS's absolute numeric targets (SNR>15dB/STOI>0.85/PESQ>2.5) is explicitly NOT current scope — v5 is locked; any future work here would be a new, separately-scoped effort.
- `onnx`/`onnxruntime`/`coremltools` pinning in `requirements.txt` — still open.
- Task 1 (`evaluate_final.py` real output) — status not reconfirmed this session; treat as unresolved unless re-checked against real console output.
- Whether a standalone judge-facing Streamlit dashboard (in-scope per CLAUDE.md's 2026-09-12 exception) gets built — not started, not requested yet this session.
