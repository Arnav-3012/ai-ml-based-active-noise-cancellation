# logs.md — Append-Only Change Log

> Never edit or delete past entries. Always append new entries at the bottom.

---

## 2026-09-10 — Phase 0: Repo scaffold + venv setup

**What was scaffolded:**
- Local git repo initialized (`git init`) in `/Users/arnav/Active_noise_cancel_dl` — GitHub remote does not exist yet, was never cloned (empty local dir, no URL supplied). Repo creation + push deferred to user via provided commands.
- Full folder tree created per `sih26052_framework.md`: `data/{raw,processed}/*`, `src/{data,baseline,model,eval,export}/`, `configs/`, `manifests/`, `results/`, `notebooks/`.
- `.gitkeep` placeholders added to `data/raw/*`, `data/processed/*`, `results/`, `manifests/`, `notebooks/` (these dirs are otherwise empty and would not exist in git without them).
- `.gitignore` written: `data/raw/`, `data/processed/`, `*.wav`, `*.pt`, `*.onnx`, `__pycache__/`, `.venv/`, `*.pyc`, `.DS_Store`, `results/*.wav`.
  - Note: since `data/raw/` and `data/processed/` are ignored wholesale, their `.gitkeep` files require `git add -f` to be tracked — flagged for the user in the final report, not done silently.
- `CLAUDE.md`, `context.md`, `plan.md`, `info.md` written per spec.

**venv status: SUCCESS**
- Created with `python3 -m venv .venv` using system Python **3.13.9**.
- pip upgraded to 26.2.1.

**Install results (all succeeded):**
| Package | Installed version |
|---|---|
| torch | 2.14.0 |
| torchaudio | 2.11.0 |
| denoiser | 0.1.5 |
| pystoi | 0.4.1 |
| pesq | 0.0.4 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| soundfile | 0.14.0 |
| PyYAML | 6.0.3 |
| tqdm | 4.70.0 |

**pesq install — flagged risk, actual result: BUILT SUCCESSFULLY.**
- `pip install pesq` compiled its C extension (`cypesq`) against Python 3.13 on macOS arm64 (clang, `pesq/pesqmod.c`, `pesq/dsp.c`, `pesq/pesqdsp.c`) with only benign "unused variable" warnings — no errors.
- Verified with a functional smoke test (not just import): `pesq.pesq(16000, ref, deg, 'wb')` on synthetic signals returned a valid MOS score (~4.58 for near-identical signals), confirming the compiled extension actually runs, not just imports cleanly.
- No action needed — no fallback or skip required.

**Other risk noted (not a failure, but flagged per "don't guess" rule):** `torch==2.14.0` resolved alongside `torchaudio==2.11.0`. These are not from the same release cycle (torchaudio normally version-locks to torch, e.g. 2.14.x/2.14.x). Basic import and `torch.backends.mps.is_available()` checks pass, but this mismatch should be re-verified before any real audio I/O or resampling work in Phase 1 — if `torchaudio` ops break against `torch` 2.14 internals, may need to pin both explicitly.

**MPS check:** `torch.backends.mps.is_available()` → `True`, `torch.backends.mps.is_built()` → `True`, confirming target M4 Pro MPS backend is usable from this venv.

**Structure verification:** `find .` output confirms all directories from `sih26052_framework.md` exist with `.gitkeep` in the gitignored/empty ones. No data-mixing, model-loading, or training code was written — Phase 0 scope only, per instructions.

**Not done (out of scope for this session, needs user input):** GitHub repo creation and initial push — no repo URL was available; user asked for the commands to run this themselves rather than have it done automatically.

---

## 2026-09-10 — Version-mismatch investigation: torch 2.14.0 / torchaudio 2.11.0 / denoiser 0.1.5

**Trigger:** Phase 0 log flagged `torch==2.14.0` resolved alongside `torchaudio==2.11.0` as a possible cross-release mismatch, to be verified before any Phase 1 audio I/O work.

**Finding 1 — the assumed "correct pairing" doesn't exist:** Checked PyPI's JSON API directly (`https://pypi.org/pypi/torch/json`, `.../torchaudio/json`), not just local pip cache. torchaudio's latest published release is **2.11.0** (uploaded 2026-03-23); there is no 2.14.x torchaudio. torch 2.14.0 was uploaded 2026-09-02. torchaudio's release cadence has fallen behind torch's — `torch==2.14.0` + `torchaudio==2.11.0` **is** the current real-world pairing, not a mismatch to fix by "finding the right torchaudio version."

**Finding 2 — denoiser 0.1.5 has no upper version bound but two real runtime breakages against this pairing (verified by actually running the code, not just reading pins):**
- `denoiser/audio.py` `Audioset.__getitem__` calls `torchaudio.get_audio_backend()`, which has been removed from torchaudio 2.11 (`hasattr(torchaudio, 'get_audio_backend')` → `False`). Would raise `AttributeError` on any dataset load through denoiser's own dataset class.
- `denoiser/stft_loss.py`'s `stft()` calls `torch.stft(...)` without `return_complex`, which now raises `RuntimeError: stft requires the return_complex parameter be given for real inputs` under torch 2.14 — confirmed by direct reproduction. This is the exact multi-resolution STFT loss function planned for fine-tuning (see `context.md`), so this would have broken Phase 2 training, not just an edge case.

Reported both to the user before making any changes, per "ambiguity → ask, don't guess."

**Decision (user-directed):** Keep torch 2.14.0 + torchaudio 2.11.0 (do not downgrade torch to chase an older matched pair — would lose MPS improvements for the M4 Pro target). Vendor and patch only the two broken denoiser modules into `src/vendor/denoiser_patched/`, rather than editing the installed pip package in place (fragile — lost on any venv rebuild/`pip install --upgrade`, untracked by git) or depending on denoiser's own future fix (0.1.5 is unmaintained/pinned by the project, per `CLAUDE.md` rule 6 — no training-from-scratch fork, transfer-learning only, so we stay on the released package plus a minimal local patch rather than forking the whole repo).

**Patches applied** (full before/after diffs and rationale in `src/vendor/denoiser_patched/README.md`):
1. `stft_loss.py`: `torch.stft(..., return_complex=True)`, read `.real`/`.imag` off the resulting complex tensor instead of indexing `x_stft[..., 0]`/`[..., 1]` (the old real-tensor-with-trailing-pair-dim output no longer exists). Numerically equivalent.
2. `audio.py`: removed the dead `get_audio_backend()` branch.

**Finding 3 — surfaced only by actually running the smoke test, not present in the original ask:** torchaudio 2.11's `load()`/`save()` route through a `torchcodec` I/O backend. `torchcodec` is not installed by default and was not a declared denoiser/torchaudio dependency. Installed it (`pip install torchcodec`, latest 0.16.0) — import succeeded but native library loading failed: `libtorchcodec` couldn't find `libavutil.60.dylib`. Homebrew ffmpeg 8.0 **is** installed and does have the library at `/opt/homebrew/lib/libavutil.60.dylib`, but it isn't on the dynamic linker search path, and the loader's own error trace showed it also probing stale `/opt/anaconda3/...` paths despite running from `.venv` — a native/system linking issue, not a Python dependency issue. Reported to the user rather than silently mutating `DYLD_FALLBACK_LIBRARY_PATH` or installing system packages unprompted.

**Decision (user-directed):** Drop `torchcodec` entirely (`pip uninstall torchcodec` — never added to `requirements.txt`, so no removal needed there). Route all file I/O in the patched `audio.py` through `soundfile` (already an installed, pure-C `libsndfile`-backed dependency with no FFmpeg/codec runtime requirement) instead of `torchaudio.load`/`torchaudio.save`. Rationale given: this environment may later move to Colab/Kaggle if MPS training hits limits, and `torchcodec`'s native FFmpeg linking is fragile across environments (broke on the first environment tried). `torchaudio` itself remains installed and pinned — still used for non-I/O ops elsewhere — just not for reading/writing files.

**`audio.py` I/O patch details:** `soundfile.read(..., dtype='float32', always_2d=True)` returns `[samples, channels]` float64-by-default numpy; converted explicitly to `[channels, samples]` float32 torch tensors via `torch.from_numpy(data).transpose(0, 1)` to match torchaudio's convention that the rest of denoiser's code expects. `get_info()` rewritten against `soundfile.info()`'s `.frames`/`.samplerate`/`.channels` attributes. Confirmed via `grep -rln "torchaudio\.\(load\|save\)" src/` (excluding `.venv/`) that no references to `torchaudio.load`/`torchaudio.save` remain anywhere in our own code.

**Smoke tests (all executed and passed, exact output shown):**
1. Patched `MultiResolutionSTFTLoss` on synthetic clean (440Hz sine) vs. noisy (440Hz + 0.05×445Hz) 1s/16kHz tensors → `sc_loss: 0.003540559...`, `mag_loss: 0.001778998...`, both finite scalars. PASS.
2. `soundfile` save/load round-trip: synthetic sine → `data/processed/_smoke_test.wav` → read back. Original shape `(1, 16000)` dtype `float32` sr `16000`; loaded shape `(1, 16000)` dtype `float32` sr `16000`; `torch.allclose(loaded, original, atol=1e-3)` → True. PASS. Test file deleted after (`os.remove`).
3. Patched `Audioset.__getitem__` (soundfile-backed) on the same test file: item shape `(1, 16000)`, dtype `float32`, matches original within `atol=1e-3`. PASS.
4. Re-ran STFT loss smoke test after the `audio.py` I/O swap (independent module, but re-verified per "verify before checking off") — identical PASS result.

**MPS check (re-run after all install/uninstall changes):** `torch.backends.mps.is_available()` → `True`, `torch.backends.mps.is_built()` → `True`. Confirms the reinstall/uninstall sequence did not break MPS.

**`requirements.txt` updated:** `torch==2.14.0` and `torchaudio==2.11.0` now pinned exactly (previously unpinned). `denoiser` left unpinned intentionally — we vendor/patch the two broken modules locally rather than depending on denoiser's internals staying fixed upstream. `torchcodec` never added (installed transiently, then uninstalled — not a project dependency).

**New files:** `src/vendor/denoiser_patched/{__init__.py, audio.py, stft_loss.py, README.md}` — the README documents both files' exact before/after diffs, the upstream denoiser version forked from (0.1.5, PyPI), and why each change was made, for future audit.

**Not done:** No Phase 1 data-pipeline code touched, per instructions to stop after the version fix was verified.

---

## 2026-09-10 — Phase 1: Data pipeline code written (nothing executed)

**Scope:** Wrote all Phase 1 data-pipeline code per the locked spec. Per explicit instruction, did NOT run any script — no download, no mixing, no dataset build. Only syntax-checked (`py_compile`) each new file; no data/model logic was executed.

**New files:**
- `configs/finetune.yaml` — new file (didn't exist before). Holds every tunable for this phase: dataset URLs/targets, SNR range (-5 to 15 dB), noise categories (gunshot/stationary/general), reverb/clipping augmentation probabilities+ranges, speaker-disjoint split ratios (80/10/10), output format. No magic numbers left in any `.py` file — all read from this config.
- `src/data/download.py` — downloads/verifies LibriSpeech dev-clean, MUSAN noise, ESC-50 gun_shot class; checks `data/raw/{librispeech,musan,esc50,gunshots}/` first and skips anything already present in sufficient quantity. Kabealo/Zenodo gunshot dataset has no stable programmatic API (browser-gated) — script detects the missing files and prints manual download instructions instead of failing or guessing a URL. On a successful run, fills in the dataset-URL placeholders in `info.md` itself (not done by me directly this session, since nothing was executed).
- `src/data/mixing.py` — `mix_pair()`: SNR-targeted mixing (random target SNR per pair, drawn from config range), fits noise length to clean length (loop or random crop), tags every pair with its noise category (gunshot/stationary/general) up front per the locked Tier-1 breakdown requirement.
- `src/data/augment.py` — `augment()`: reverb (synthetic exponentially-decaying IR convolution, no external SoX dependency) and clipping (percentile-based hard limiting), each applied per a configured probability.
- `src/data/build_dataset.py` — orchestrator: indexes LibriSpeech by speaker (parses speaker ID from path structure), splits speakers (not utterances) into train/val/test to avoid identity leakage, builds a 3-way noise pool (ESC-50 gun_shot + Kabealo clips → gunshot; MUSAN noise split into stationary/general by subfolder convention, documented in-code), runs mixing+augmentation per utterance, writes noisy/clean wav pairs to `data/processed/{train,val,test}/` and one manifest JSON per split to `manifests/` with fields: pair_id, clean_path, noisy_path, speaker_id, split, snr_db, noise_category, noise_source, source_utterance.
- `src/__init__.py`, `src/data/__init__.py` — added so `src.data.*` imports resolve (needed by `build_dataset.py`'s imports of `mixing`/`augment`).

**Design notes not previously locked, decided now (flagged for visibility, not asked about since they're implementation details within the existing spec, not new decisions):**
- MUSAN noise has no built-in stationary/general label; split by subfolder convention (`sound-bible` → stationary, `free-sound` → general) with the rationale documented directly in `build_dataset.py`'s docstring.
- Reverb implemented as a synthetic randomized exponential-decay impulse response convolved via `torch.nn.functional.conv1d`, rather than pulling in SoX/pyroomacoustics, to avoid adding a new heavy dependency for Phase 1.

**Verification performed:** `python3 -m py_compile` on all four new `src/data/*.py` files — syntax-only check, confirms no import-order or syntax errors. No script was executed end-to-end; no network calls, no audio I/O, no dataset build ran this session.

**Not done:** Nothing executed per explicit instruction — download, manual Zenodo step, and build_dataset run are all pending the user's own terminal session (commands provided separately in chat, not run here).

---

## 2026-09-10 — Resolved Kabealo/Zenodo gunshot dataset record ID

**Trigger:** User ran `download.py` themselves; it correctly printed the manual-download instructions for the gunshot dataset, but wrote the literal placeholder URL (`.../records/PLACEHOLDER_ZENODO_RECORD_ID`) into `info.md`'s "downloaded" line — confusing, since nothing was actually downloaded. User opened that URL in-browser and hit a real Zenodo 404, then flagged it.

**Resolution:** Used web search to identify the actual record: Kabealo, Wyatt et al., "A multi-firearm, multi-orientation audio dataset of gunshots" (*Data in Brief*, 2023, DOI 10.1016/j.dib.2023.109091), Zenodo record **7004819** (https://zenodo.org/records/7004819) — 2,148 gunshot samples across 4 firearms, 27 mics/9 phones/Raspberry Pi mics at an outdoor firing range.

**Files updated:**
- `configs/finetune.yaml` — `dataset.gunshots.zenodo_record_id`/`zenodo_url` now point to the real record (was the literal placeholder string).
- `info.md` — corrected the gunshot dataset line to the real Zenodo URL + citation, and removed the false "(downloaded ...)" claim since the manual step has not actually been completed yet.
- `dataset.md` — updated the Kabealo/gunshots section's config snippet and explanation to reference the real record instead of the placeholder, and reworded "why is this still a placeholder" to "why is this still a manual step" since the ID is now known.

**Not done:** Still have not run the manual Zenodo download itself — that remains the user's step (visit the record, download, extract 800–1200 sampled `.wav` clips into `data/raw/gunshots/`).

---

## 2026-09-11 — Kabealo gunshots imported (full set, not sampled); UrbanSound8K downloaded; sample-rate mismatch bug fixed

**User's manual Kabealo download:** User already had the Zenodo record 7004819 downloaded locally at `/Users/arnav/edge-collected-gunshot-audio/` (2,148 `.wav` files across 4 firearm subfolders: `remington_870_12_gauge` (379), `glock_17_9mm_caliber` (669), `ruger_ar_556_dot223_caliber` (597), `38s&ws_dot38_caliber` (503) — matches the Kabealo et al. dataset exactly). Asked user whether to sample 800-1200 per the original config or use the full set; **decision: use the full set** (more real diversity, no reason to discard data). Copied all 2,148 `.wav` files flat into `data/raw/gunshots/` (`cp -n`, no filename collisions — source files are UUID-named). Updated `configs/finetune.yaml`: `gunshots.sample_min`/`sample_max` changed from `800`/`1200` to `2000`/`2148` to reflect the full-set decision (this is the threshold `download.py`'s `check_gunshots()` uses to confirm the manual step succeeded — verified independently that `2148 >= 2000` passes).

**UrbanSound8K:** User ran `python -m src.data.download` — LibriSpeech/MUSAN skipped (already present), gunshots skipped (now satisfied), UrbanSound8K downloaded and extracted successfully (confirmed indirectly: `build_dataset.py`'s noise pool later reported 2522 gunshot-category clips = 2148 Kabealo + ~374 UrbanSound8K, consistent with the ~370-390 expected range from `dataset.md`).

**Bug found on first `build_dataset.py` run (user-run, real output):** Crashed on the very first noise clip processed — `RuntimeError: data/raw/gunshots/....wav has sample rate 44100, expected 16000`. Investigated directly: checked native sample rates across all four raw sources with `soundfile.info()` —
- `data/raw/gunshots` (Kabealo): all 44100 Hz.
- `data/raw/urbansound8k`: mixed 44100/48000/96000 Hz (varies by clip).
- `data/raw/musan`, `data/raw/librispeech`: already 16000 Hz natively.

This is expected raw-source variation, not a bug in the download step — MUSAN and LibriSpeech happen to already ship at the project's target rate, but UrbanSound8K and Kabealo don't, and there was no resampling step anywhere in the pipeline to reconcile this before `_load_wav()`'s strict equality check (intentionally strict, to catch *real* mismatches — see Phase 1 write-up) rejected them.

**Fix (user-directed, asked before implementing):** Resample noise clips on-the-fly during mixing, in `build_dataset.py`, rather than as a separate one-time preprocessing pass over `data/raw/`. Split the old single `_load_wav()` into two functions:
- `_load_wav()` — unchanged, stays strict, used only for clean speech (LibriSpeech). A mismatch here would mean something is actually wrong (corrupted download, wrong file), since LibriSpeech is natively 16kHz, so it still raises rather than silently resampling.
- `_load_noise_wav()` — new. Loads via the same `soundfile` path, but resamples with `torchaudio.functional.resample()` (a pure tensor op, not an I/O backend — confirmed via smoke test that it doesn't touch `torchaudio.load`/`torchcodec`, consistent with this project's existing soundfile-only-for-I/O rule from Phase 0) whenever the clip's native rate doesn't match `sample_rate` from config.
- `build_split()` updated to call `_load_noise_wav()` instead of `_load_wav()` for the noise path specifically (clean speech path unchanged).

**Verification performed:** `python3 -m py_compile` on `build_dataset.py` — passes. Isolated smoke test: `torchaudio.functional.resample(torch.randn(1, 44100), orig_freq=44100, new_freq=16000)` → output shape `(1, 16000)`, correct dtype `float32`. Checked for partial output from the crashed run: only 2 `.wav` files existed under `data/processed/` and no `manifests/*.json` were written, so no cleanup was needed — a rerun overwrites the same sequential pair IDs harmlessly.

**Not done:** `build_dataset.py` has not been rerun with the fix yet — that's the user's next step.

---

## 2026-09-11 — Phase 1 COMPLETE: build_dataset.py ran successfully, output verified

**User re-ran `python -m src.data.build_dataset`** with the resample fix in place. Full real output, no errors:
- 40 speakers, 2703 utterances indexed from LibriSpeech.
- Speaker-disjoint split: train 32 speakers/2162 utterances, val 4/242, test 4/299.
- Noise pool: gunshot 2522 (2148 Kabealo + ~374 UrbanSound8K), stationary 87, general 843.
- All 3 manifests written: `manifests/train.json` (2162 entries), `manifests/val.json` (242), `manifests/test.json` (299).

**Independent verification performed (not just trusting the script's own success message):**
- `find data/processed/{train,val,test} -iname '*.wav' | wc -l` → 4324/484/598 files respectively — exactly 2× each split's manifest entry count (one clean + one noisy file per pair), confirming no partial/missing writes.
- Loaded all 3 manifests and inspected a sample entry per split — schema matches spec (pair_id, clean_path, noisy_path, speaker_id, split, snr_db, noise_category, noise_source, source_utterance), `snr_db` values fall within the configured -5 to 15 dB range, `noise_category` correctly varies (stationary/gunshot seen across samples), `noise_source` paths correctly point into the right raw dataset per category.
- **Speaker-disjointness verified programmatically**, not assumed: computed `set(speaker_id)` per split and checked pairwise intersections — train∩val, train∩test, val∩test all empty sets. Confirms the leakage-prevention design actually holds on real output, not just in the split logic's intent.
- Spot-checked one output file (`data/processed/train/train_000000_noisy.wav`) via `soundfile.info()`: 16000 Hz, 1 channel, PCM_16 — matches `configs/finetune.yaml`'s `sample_rate`/`output` settings exactly.

**Phase 1 status: DONE.** `context.md` fully rewritten to reflect: all 4 datasets downloaded and characterized (native sample rates, clip counts), full pipeline run completed with verified output, current phase updated from "Phase 0 — Scaffold" to "Phase 1 complete — Data pipeline," ready to move to baseline or model-loading work next.

**Not done:** No baseline (spectral subtraction) or model-loading code written yet — out of scope for Phase 1 per the original task framing ("this phase produces labeled noisy/clean pairs, nothing else. No baseline, no model loading yet").

---

## 2026-09-11 — Phase 2: Baseline + eval code written (nothing executed)

**Scope:** Wrote all Phase 2 baseline + evaluation code per the locked spec. Per explicit instruction, did NOT run anything — only `py_compile` syntax checks and a YAML parse check were performed; no audio I/O, no metric computation ran this session.

**New files:**
- `src/baseline/spectral_subtraction.py` — hand-implemented STFT spectral subtraction (Boll 1979 noise estimate over the first N frames of the noisy signal itself + Berouti et al. 1979 oversubtraction/spectral-floor). Uses `torch.stft`/`torch.istft` for the transform (not a magic reimplementation, but not `noisereduce` either — locked decision honored), `soundfile` for all file I/O (no `torchaudio.load`/`save`, consistent with the Phase 0 fix). Documents in-comment why the noise-estimate-frames approach has a real limitation on this dataset specifically (Phase 1's `mixing.py` overlays noise under the full utterance, so leading frames aren't guaranteed silence) — left as a known classical-baseline weakness, not silently fixed, since demonstrating that weakness is the point of this baseline per the task's own framing.
- `src/eval/metrics.py` — `snr()`, `stoi_score()`, `pesq_score()`, each a thin wrapper (SNR hand-computed, STOI via `pystoi`, PESQ via `pesq` — the package verified to build correctly in Phase 0) with a plain-English one-line docstring each, written judge-Q&A-ready for later reuse in `info.md`'s metric-definitions section.
- `src/eval/evaluate.py` — scores `noisy` (raw, unprocessed — reference floor) and `baseline` (spectral subtraction output) columns against clean test audio across all three metrics, with a `finetuned` column already wired up but empty (`n=0`, `null` scores) until Phase 3 supplies `results/finetuned/` output. Per-noise-category breakdown (gunshot/stationary/general) plus an `overall` row, computed from the `noise_category` field already present in the Phase 1 manifest. Writes both `results/baseline_results.json` (nested, full precision) and `results/baseline_results.csv` (flat table, one row per column×category) — the CSV is what becomes the first columns of the baseline-vs-fine-tuned comparison table.

**Config decision:** Added a `baseline:` and `eval:` section to the existing `configs/finetune.yaml` rather than creating a separate `configs/baseline.yaml` — kept one file since it's already the project's single source of truth per CLAUDE.md rule 1, and the baseline/eval tunables are few (STFT n_fft/hop/win, oversubtraction alpha, spectral floor beta, PESQ mode) and don't warrant a second config file to track. New keys: `baseline.stft.{n_fft,hop_length,win_length,window}`, `baseline.noise_estimate_frames`, `baseline.oversubtraction_factor`, `baseline.spectral_floor`, `eval.pesq_mode`.

**Verification performed:** `python3 -m py_compile` on all 5 new/touched `.py` files (including new `src/baseline/__init__.py`, `src/eval/__init__.py`) — all pass. `yaml.safe_load` on the updated `configs/finetune.yaml` — parses cleanly. No script was executed end-to-end; no audio was read, transformed, or scored this session.

**Not done:** Nothing executed per explicit instruction — running `spectral_subtraction.py` then `evaluate.py` and confirming real SNR/STOI/PESQ numbers on the test split are the user's next manual steps (exact commands given separately in chat).

---

## 2026-09-11 — Phase 2 COMPLETE: baseline + eval ran successfully, output verified

**User ran both scripts.** `spectral_subtraction.py`: 299/299 test pairs processed, no errors. `evaluate.py`: 299/299 pairs scored, no errors, printed full results table.

**Independent verification performed:**
- `ls results/baseline | wc -l` → 299, matches test manifest size exactly (one enhanced file per pair).
- `results/baseline_results.json` and `results/baseline_results.csv` both confirmed present.

**Real results (test split, n=299):**
| column | category | n | SNR(dB) | STOI | PESQ |
|---|---|---|---|---|---|
| noisy | overall | 299 | 0.47 | 0.598 | 1.307 |
| baseline | overall | 299 | 1.84 | 0.592 | 1.292 |
(full per-category table in context.md)

**Finding:** baseline improves overall SNR by +1.37 dB over the unprocessed floor, but STOI and PESQ both *decrease* slightly relative to raw noisy audio. This is the expected/intended classical-baseline failure mode (musical noise from spectral subtraction trades energy-ratio gain for perceptual-quality loss) — not a bug, this is the exact result the baseline was built to produce per the task's own framing. Gunshot category (transient, non-stationary noise) is the hardest across all three metrics and stays SNR-negative even after processing, consistent with the noise-estimate assumption (constant fingerprint from leading frames) being least valid for transient noise.

**Phase 2 status: DONE.** `context.md` updated with the real results table and current-phase marker moved to "Phase 2 complete," ready for Phase 3 (model-loading/fine-tuning).

**Not done:** No model-loading or fine-tuning code written yet — out of scope for Phase 2.

---

## 2026-09-11 — Phase 3a COMPLETE: dns48 model loading + inference verified on real audio

**Scope:** First checkpointable sub-step of Phase 3 (fine-tuning), split 3a/3b/3c per user instruction. 3a covers model loading and architecture verification only — no training loop, optimizer, or loss code (that's 3b).

**New files:**
- `src/model/__init__.py`.
- `src/model/load.py` — `load_dns48()` loads the pretrained checkpoint via `denoiser.pretrained.dns48()` (denoiser's own loading mechanism, used as-is from the pip package). Verified directly (not assumed) that `denoiser.pretrained` imports only `.demucs`/`.utils` — it does NOT import `denoiser.audio` or `denoiser.stft_loss`, so neither of the two patched files from `src/vendor/denoiser_patched/` (Phase 0) is touched by model loading; those patches only matter later for training/loss and dataset loading. `_check_architecture()` compares the loaded model against a locked spec in `configs/finetune.yaml['model']['expected']` and raises + logs `ERROR` on ANY mismatch rather than assuming correctness. MPS check (`get_device()`) follows CLAUDE.md rule 8: uses MPS if available, else falls back to CPU with a visible `logger.warning`, never silent.
- `src/model/verify_load.py` — loads the model via `load.py`, picks the FIRST entry from `manifests/test.json` (not a hardcoded path) and loads its real `noisy_path` wav via `soundfile` (project convention, no `torchaudio.load`), runs one `torch.no_grad()` inference pass, confirms output shape equals input shape, times the single call as a sanity number only.

**Config change:** Added `model:` section to `configs/finetune.yaml` with `name: dns48` and an `expected:` block (total_params, encoder/decoder layer counts, kernel_size, stride, chin/chout, lstm_layers, lstm_hidden) used only for the mismatch check in `load.py` — not passed into the model constructor (denoiser's `dns48()` builds the architecture itself to match its own checkpoint).

**Param count corrected before locking the spec, not just copied from the task's "~18.9M" framing:** Instantiated `denoiser.demucs.Demucs` locally with dns48's exact hyperparameters (chin=1, chout=1, hidden=48, depth=5, kernel_size=8, stride=4, causal=True, resample=4) and summed `p.numel()` directly → **18,867,937**, not a rounded figure. This is the number locked into config and checked against.

**Verification performed (mine, before handing off):**
- Confirmed via `inspect`/`grep` that `denoiser.pretrained.dns48()` → `_demucs(pretrained, DNS_48_URL, hidden=48)`, and that `pretrained.py`'s only imports are `.demucs`/`.utils` — no path to the patched files.
- Confirmed `Demucs` exposes `.kernel_size`, `.stride`, `.encoder`/`.decoder` (`ModuleList`, len 5 each), `.lstm.lstm` (`torch.nn.LSTM`, `num_layers=2`, `hidden_size=768`, `input_size=768`) — all attributes used in `_check_architecture()` are real, not guessed.
- Read `Demucs.forward()` source directly: handles 2D or 3D input (auto-unsqueezes channel dim), internally pads to `valid_length()` before the encoder/LSTM/decoder stack and crops back to the original input length before returning (`x[..., :length]`) — confirms the output-shape-matches-input-shape check in `verify_load.py` is architecturally guaranteed, not a lucky assumption.
- `python3 -m py_compile` on both new files — pass. `yaml.safe_load` on the updated config — pass. No checkpoint download, no full inference, no training run by me — left entirely to the user's own run per explicit instruction.

**User ran `python -m src.model.verify_load` themselves. Real output, no errors:**
- Checkpoint loaded via `denoiser.pretrained.dns48()`.
- **Param count: 18,867,937 — exact match to locked spec, zero mismatches.** Encoder layers: 5, decoder layers: 5, kernel_size: 8, stride: 4, LSTM: num_layers=2, hidden_size=768, input_size=768 — all match.
- **MPS backend available — used device 'mps'** (no CPU fallback triggered).
- Real test audio loaded: `data/processed/test/test_000000_noisy.wav` (`pair_id=test_000000`, from `manifests/test.json`, not synthetic).
- Input shape `(1, 76800)` (4.8s @ 16kHz) → output shape `(1, 76800)` — **exact match, PASS**.
- Single-run inference timing (sanity number, not a rigorous benchmark): **615.12 ms** for 4.8s of audio on MPS.

**Phase 3a status: DONE.** dns48 loads correctly via denoiser's own mechanism, confirmed not to depend on the two patched vendor files, architecture matches the locked spec exactly (real count 18,867,937, not the rounded ~18.9M figure), runs real inference on real Phase 1 processed audio on MPS with correct output shape.

**Not done:** No training loop, optimizer, or loss code — that's Phase 3b, out of scope for this sub-step per explicit instruction.

---

## 2026-09-11 — Phase 3a follow-up: two findings verified and resolved before 3b

**Trigger:** User asked for two things to be concretely re-verified, not assumed, before proceeding to 3b.

**Finding 1 — patched-code import path: CONFIRMED CORRECT, nothing needed fixing.** Checked directly via `sys.modules` inspection after actually calling `denoiser.pretrained.dns48()` (not just reading source): only `denoiser`, `denoiser.pretrained`, `denoiser.demucs`, `denoiser.resample`, `denoiser.utils` load, all resolving to the raw pip package under `.venv/lib/python3.13/site-packages/denoiser/`. `'denoiser.audio' in sys.modules` → `False`, `'denoiser.stft_loss' in sys.modules` → `False`. Confirmed transitively too: `pretrained.py` imports only `.demucs`/`.utils`; `demucs.py` imports only `.resample`/`.utils`; `denoiser/__init__.py` is empty (no eager re-exports). Model loading architecturally never touches `audio.py`/`stft_loss.py`, patched or unpatched — those two files are only imported later, in Phase 3b, by training code that explicitly imports `MultiResolutionSTFTLoss`/`Audioset` from `src/vendor/denoiser_patched/`, which `load.py`/`verify_load.py` correctly never do.

**Finding 2 — 615ms timing: CONFIRMED cold-start/per-shape MPS kernel compilation, not a regression.** Isolated the variable properly (same file repeated 6x, then a controlled fixed-length synthetic tensor repeated 5x, both with `torch.mps.synchronize()` around the timed region for accurate measurement): run 1 on any never-before-seen input length is consistently slow (36–67ms/sec of audio), while every subsequent run on that same length drops to a tight **~6.9–7.1ms/sec**, matching the earlier-verified ~9ms/sec warm benchmark. Further isolated that the cost is **per unique input length**, not a single global one-time warmup: touching a new file with a new duration after the model was already warmed up on a different length still pays a first-touch cost (e.g. 16.42ms/s → 6.49ms/s on second touch of that same new length). Root cause: `Demucs.forward()` pads input to `valid_length(length)` before running through the encoder/LSTM/decoder stack, so each distinct padded shape triggers its own MPS kernel dispatch/compilation the first time it's seen. The original 615ms/4.8s figure was simply the very first inference call in a freshly-loaded model in a fresh process — expected, not a bug.

**Noted for Phase 3b (advance notice only, no code started):** 3b will be the FIRST real test of the Phase 0 STFT loss patch (`torch.stft(..., return_complex=True)` fix in `src/vendor/denoiser_patched/stft_loss.py`) under live conditions. The Phase 0 smoke test used synthetic sine-wave tensors only, not real batched multi-channel audio flowing through the soundfile-based I/O path (`src/vendor/denoiser_patched/audio.py`) at training time. Per this project's standing verify-before-trusting standard, 3b must not assume the patch works because it worked on synthetic data before — it needs to be re-verified concretely against real data (real noisy/clean pairs from `manifests/train.json`, loaded via the patched `Audioset`/`soundfile` path, run through `MultiResolutionSTFTLoss`) as part of that phase's own verification, not carried over as an assumption from Phase 0.

**Per-shape timing behavior also noted for 3b/inference benchmarking:** since cost is per unique padded length (not a one-time global warmup), any later inference benchmarking work should consider fixed-length bucketing or an explicit warmup pass before timing real batches, rather than timing cold first-touch calls.

**Phase 3a follow-up status: RESOLVED, both findings reviewed and approved by user.** No code changes were required for either finding — both were verification-only. `context.md` not touched this entry (no state change beyond what's already recorded there for Phase 3a).

**Not done:** No Phase 3b code (training loop, optimizer, loss) written or started this session, per explicit instruction to stop after this log entry.

---

## 2026-09-11 — Phase 3b COMPLETE: data loading + training-loop smoke test, verified with real values

**Scope:** Data loading + training loop wiring, smoke-test only (1 batch, 1 gradient step) — per explicit instruction, no real training, no checkpointing, no validation loop (that's 3c).

**New files:**
- `src/model/dataset.py` — `NoisyCleanDataset` reads pairs from `manifests/{train,val,test}.json` (Phase 1), loads noisy+clean wavs via `soundfile` (matches Phase 0/3a I/O convention, no `torchaudio.load`). Manifest clips have variable duration (verified in the 3a follow-up: 3.8s-32.3s range) so each `__getitem__` random-crops (same offset for both noisy/clean, keeping them time-aligned) or zero-pads to a fixed `training.segment_seconds` window so `DataLoader`'s default collate can batch them. Returns 1D `[samples]` mono tensors (confirmed real Phase 1 output is mono via `soundfile.info()` on an actual train file before assuming it). `make_dataloader()` wraps it with `training.batch_size`/`num_workers` from config.
- `src/model/finetune.py` — `verify_patched_loss_import()` confirms `MultiResolutionSTFTLoss` resolves to `src/vendor/denoiser_patched/stft_loss.py` via the module's `__file__` (same concrete method as the 3a follow-up's `sys.modules` check), raises if it doesn't match the expected path suffix. `freeze_encoder_layers()` implements optional early-layer freezing as a mechanism, defaulted OFF (`training.freeze_encoder_layers: 0`) — whether to actually freeze is left as a 3c decision, per instruction. `build_optimizer()` sets up Adam at `training.learning_rate`. `compute_loss()` returns L1 and STFT loss as separate scalars (not just combined) plus the weighted total. `smoke_test_train()` runs exactly one batch through forward -> loss -> backward -> `optimizer.step()` -> (no `zero_grad()` needed after, single step only), with explicit NaN/Inf/exactly-zero checks on each loss component and a gradient spot-check (first 5 params with non-None grad) after `backward()`.

**Config additions:** New `training:` section in `configs/finetune.yaml` — `segment_seconds: 4.0` (fixed-length crop, reasoned in-comment: needed for batch collation given variable clip lengths), `batch_size: 2` (deliberately tiny, smoke-test scope), `num_workers: 0`, `learning_rate: 3.0e-5` (1/10th of a typical from-scratch Adam LR ~3e-4 for speech enhancement — standard fine-tuning heuristic to avoid catastrophically forgetting dns48's pretrained weights, reasoning documented in-comment), `loss_weights: {l1: 1.0, stft: 1.0}` (equal weighting for the smoke test; rebalancing left to 3c), `freeze_encoder_layers: 0` (mechanism implemented, defaulted off).

**Verification performed (mine, before handing off):**
- Confirmed `MultiResolutionSTFTLoss.forward()` expects `(B, T)` 2D tensors (read its docstring/source directly) — `Demucs.forward()` outputs `[batch, channels=1, samples]`, so `finetune.py` explicitly squeezes the channel dim before the loss call; confirmed this shape flow on a synthetic dry run (untrained architecture-only `Demucs` instance + real patched `MultiResolutionSTFTLoss`, batch=2) end-to-end: forward -> squeeze -> L1+STFT loss -> backward -> non-zero gradient, all before running the real checkpoint-based smoke test myself (explicitly out of scope for me this session).
- Confirmed via `grep` that `load_dns48()` (3a) calls `model.eval()`, so `finetune.py` explicitly calls `model.train()` before the forward pass, not assumed.
- Confirmed `MultiResolutionSTFTLoss`/`STFTLoss` register a `window` buffer (`register_buffer`), so `.to(device)` is required and correctly applied — not just Python-side attributes that would silently no-op on `.to()`.
- Confirmed real Phase 1 output is mono (`soundfile.info()` on an actual `data/processed/train/*.wav` file → `channels: 1`) before writing the `.squeeze(0)` in `dataset.py`'s `_load_wav()`.
- `python3 -m py_compile` on both new files — pass. `yaml.safe_load` on the updated config — pass. No checkpoint download or real training run performed by me — left to the user's own run per explicit instruction.

**User ran `python -m src.model.finetune` themselves. Real output, no errors:**
- **Patched-code path confirmed concretely (not assumed):** `MultiResolutionSTFTLoss resolved from: /Users/arnav/Active_noise_cancel_dl/src/vendor/denoiser_patched/stft_loss.py` — matches the vendored patch, not the raw pip package.
- Model loaded via `denoiser.pretrained.dns48()`, architecture re-verified against locked spec (18,867,937 params, 5+5 layers, kernel=8/stride=4, LSTM 2×768) — zero mismatches, consistent with 3a.
- MPS backend used, no CPU fallback.
- **Input batch:** noisy `(2, 64000)` float32, clean `(2, 64000)` float32 — matches `training.batch_size=2` × `training.segment_seconds=4.0` × `sample_rate=16000` exactly.
- **Model output (pre-squeeze):** `(2, 1, 64000)` float32 — confirms `Demucs` adds a channel dim as expected. **Post-squeeze (loss input):** `(2, 64000)` float32 — matches clean batch shape, correct alignment into the loss.
- **Loss values, printed separately (this is the actual verification, not just "no crash"):**
  - **L1 loss: 0.022368** — small but non-zero, plausible for a pretrained model's raw output vs. real clean speech (dns48 was never trained on this project's specific noise distribution, so a small-but-present residual error is expected, not suspiciously large or suspiciously perfect).
  - **STFT loss: 0.175597** — non-zero, in a sane range given the patched loss's `factor_sc=0.1`/`factor_mag=0.1` internal scaling.
  - **Total (weighted, 1.0/1.0): 0.197965.**
  - None of the three were NaN, Inf, or exactly zero — all three explicit checks in `smoke_test_train()` passed silently (no `RuntimeError` raised).
- **Gradient spot-check (first 5 real params with non-None grad after `backward()`), all non-zero:** `encoder.0.0.weight` (0.00453883), `encoder.0.0.bias` (0.00641049), `encoder.0.2.weight` (0.00068696), `encoder.0.2.bias` (0.00381913), `encoder.1.0.weight` (0.00037252) — concrete proof the loss is connected to real model parameters, not silently detached from the graph.
- `optimizer.step()` ran, no checkpoint saved, no further steps — exactly one smoke-test iteration, as scoped.

**Phase 3b status: DONE.** The Phase 0 STFT loss patch (`return_complex` fix) is now verified against real batched audio through the soundfile-based I/O path, not just Phase 0's synthetic sine tensors — loss values are sensible (non-zero, non-exploding, in a plausible range for a pretrained-but-not-fine-tuned model), and gradients concretely reach real model parameters. Data loading (`dataset.py`) and training-loop wiring (`finetune.py`) both confirmed working end-to-end on real data.

**Not done:** No multi-batch or multi-epoch training, no checkpointing, no validation loop, no plateau-stopping — all explicitly deferred to Phase 3c per task scope.

---

## 2026-09-11 — Phase 3c epoch ceiling: confirmed before writing training code

**Trigger:** Per explicit instruction, epoch ceiling must be proposed with real justification and confirmed by user BEFORE any Phase 3c code (validation loop, early stopping, checkpointing) is written.

**Real inputs used (verified, not assumed):**
- Train pairs: **2162** (exact count from `manifests/train.json`, re-confirmed this session via direct read — not the rounded ~2160 figure floated in the task prompt).
- Val pairs: **242** (exact count from `manifests/val.json`).
- Segment length: **4.0s fixed** (`training.segment_seconds`, 3b) — every training sample is cropped/padded to this window regardless of source clip duration. Sampled 200 real train-manifest clips directly (`soundfile.info()`) to confirm source durations aren't mostly sub-4s (would've meant excessive padding): mean 6.93s, median 5.57s, range 1.88s–26.6s. Confirms segment cropping is sane, not silently padding-dominated.
- Warm inference timing: **~7ms/sec of audio** on MPS, per 3a/3a-follow-up's isolated warm benchmark (used the lower end of the previously-confirmed 6.9–9ms/sec warm range, per task instruction).
- Batch size: **8** — NOT the 3b value of 2 (which was explicitly a smoke-test placeholder, deferred to 3c for a real decision per 3b's own log entry). User asked for a recommendation balancing fitting-the-project's-SNR/STOI/PESQ-targets against over/underfitting risk; recommended and user accepted **batch_size=8**: dns48 is small (~18.9M params, ~225MB fixed weights+Adam-state memory), so 24GB unified memory is not the binding constraint at this model scale; batch=8 gives meaningfully more stable gradients than batch=2 for a low-LR (3e-5) fine-tune without batch=16's higher first-run OOM/swap risk on MPS's less mature memory manager.

**Calculation (shown, not just stated):**
- `steps_per_epoch = 2162 // 8 = 270` (floor, drops final partial batch).
- Forward-only per-sample cost: `7ms/s × 4.0s segment = 28ms`.
- **Training step ≠ inference step** — assumption used: training step (forward+backward+optimizer) ≈ **3.5× forward-only time** (backward pass ~2x forward FLOPs per standard autodiff rule of thumb, plus Adam optimizer overhead and the LSTM's known-slower MPS backward for recurrent backprop-through-time, folded into the extra 1.5x margin). This is an explicit assumption, not a measured multiplier — flagged as such to the user.
- Per-sample training-step time: `28ms × 3.5 = 98ms` → per-batch (×8): `784ms ≈ 0.784s/step`.
- Train time/epoch: `270 × 0.784s ≈ 211.7s`.
- Val time/epoch (forward-only, batch=8, 242 pairs → 31 steps ceil): `31 × (28ms×8=224ms) ≈ 6.9s`.
- **Total ≈ 218.6s ≈ 3.64 min/epoch.**

**Ceiling decision:** Deadline confirmed by user as **2026-09-12** (tomorrow), with a **2–2.5 hr budget, extendable if it clearly improves the model**. At ~3.64 min/epoch, proposed and user-confirmed ceiling: **40 epochs (~2.4 hr wall-clock)**. Validation-loss plateau is the stopping rule that can end training EARLIER than 40; the ceiling is a hard stop regardless of whether a plateau was reached — if the ceiling is hit without a clear plateau, this is a locked project-wide rule: the report must say "ceiling reached, still improving," not be silently framed as convergence.

**Phase 3c ceiling status: CONFIRMED by user.** Proceeding to extend `src/model/finetune.py` with validation loop, plateau early stopping, and checkpointing per Task 2.

---

## 2026-09-11 — Phase 3c code extension COMPLETE (not yet run — real run is the user's, per explicit instruction)

**Scope:** Extended `src/model/finetune.py` and `configs/finetune.yaml` with the real multi-epoch training run (validation loop, plateau early stopping, best-val-loss checkpointing, per-epoch CSV logging). Did NOT run the real training loop myself — per explicit instruction, this is the user's run to start and watch.

**Config additions (`configs/finetune.yaml['training_run']`):**
- `batch_size: 8` — real-run batch size (separate from 3b's `training.batch_size: 2` smoke-test value, which is untouched). Justification: see Phase 3c ceiling entry above.
- `max_epochs: 40` — the confirmed hard ceiling from the ceiling-approval entry above. Hard stop regardless of plateau status.
- `early_stopping.min_delta: 0.001` — chosen relative to 3b's real smoke-tested total-loss scale (~0.198): ~0.5% of that value, tight enough not to stop on noise, loose enough to catch a genuine plateau.
- `early_stopping.patience_epochs: 5` — ~18 min at this run's ~3.64 min/epoch estimate; enough epochs to distinguish a real plateau from a temporary flat stretch without eating too much of the 40-epoch/2.4hr ceiling.
- `checkpoint_dir: "checkpoints"` — new top-level dir (not `results/`, which is reserved for eval-artifact outputs per Phase 2 convention). Added `checkpoints/.gitkeep` (empty placeholder, consistent with the project's existing `.gitkeep` convention for gitignored/generated dirs). Note: `.gitignore` already excludes `*.pt` globally, so checkpoint files themselves are never a commit risk regardless of directory.
- `results_csv: "results/finetune_training_log.csv"` — per-epoch loss CSV, written under the existing `results/` convention.

**`src/model/finetune.py` changes:**
- `_run_validation()` — full pass over the val split, `model.eval()` + `torch.no_grad()`, restores `model.train()` after. Returns mean L1/STFT/total across all val batches (same `compute_loss()` used by training, so components are directly comparable).
- `train()` — the real training loop: per epoch, iterates all train batches (forward -> loss -> backward -> `optimizer.step()` -> `zero_grad()`), then runs one full validation pass, logs train+val L1/STFT/total plus epoch wall-time to both stdout and `results/finetune_training_log.csv` (flushed every epoch so an interrupted run doesn't lose logged progress).
  - Checkpointing: saves `{epoch, model_state_dict, optimizer_state_dict, val_total_loss, val_l1_loss, val_stft_loss}` to `checkpoints/dns48_finetuned_best.pt` ONLY when val total loss improves by more than `min_delta` — overwritten in place, so only the single best-so-far checkpoint is ever kept (not every epoch, not just the last), per task instruction.
  - Plateau early stopping: tracks `epochs_since_improvement`; stops the loop when it reaches `patience_epochs`, before hitting `max_epochs` if a plateau is detected earlier.
  - **Honest end-of-run framing (locked project-wide rule, not optional):** two distinct log messages depending on exit path — plateau-triggered stop is logged as "STOPPED EARLY on plateau"; ceiling-triggered stop (loop completes all `max_epochs` with no plateau) is logged as "EPOCH CEILING REACHED, no plateau detected" with an explicit "do NOT report this as convergence" note. These are never conflated into one generic "training complete" message.
- Entry point: `python -m src.model.finetune` (no args) still runs the unchanged 3b `smoke_test_train()` — did not touch that behavior. `python -m src.model.finetune --run` runs the new real Phase 3c `train()`.

**Verification performed (mine, before handing off):**
- `python3 -m py_compile` on both changed files (`finetune.py`, `dataset.py`) — pass.
- `yaml.safe_load` on the updated config, confirmed `training_run` section parses to the expected dict — pass.
- Confirmed `make_dataloader()` (in `dataset.py`) now accepts an explicit `batch_size` override parameter (used by `train()` to pass `training_run.batch_size=8`, distinct from 3b's `training.batch_size=2` still used by `smoke_test_train()`) — added without changing default behavior when the override isn't passed.
- Confirmed `results/` and `checkpoints/` dirs exist (created `checkpoints/` + its `.gitkeep` this session; `results/` already existed from Phase 2) so `train()`'s `mkdir(parents=True, exist_ok=True)` calls are belt-and-suspenders, not covering up a missing-dir assumption.
- Did NOT run `train()` myself, and did NOT run any real training — per explicit instruction, this is the user's run.

**Exact command for the user to start the real run themselves:**
```
python -m src.model.finetune --run
```

**Phase 3c code-extension status: DONE, code written and statically verified, NOT yet executed.** Awaiting the user's real training run result before the next `logs.md` entry (per explicit instruction: two separate append points, this one and a later one after the actual run completes — not combined).

---

## 2026-09-11 — Phase 3c REAL TRAINING RUN COMPLETE — plateau-stopped at epoch 12, best checkpoint epoch 7

**Trigger:** User ran `python -m src.model.finetune --run` themselves (per explicit instruction — not run by the agent). Reported real console output; separately mid-run (11 epochs in) asked whether to tweak anything given 3-4/5 stale epochs — decision was to let plateau-stopping run its course rather than kill/restart, then evaluate against real target metrics (SNR>15dB, STOI>0.85, PESQ>2.5) before tweaking. Run finished naturally before that follow-up was needed.

**Real result (verified against actual console output + checkpoint file inspection, not assumed from the log alone):**
- **Stopped early on plateau at epoch 12** (of the confirmed 40-epoch ceiling) — 5 consecutive epochs (8-12) without a val-loss improvement greater than `min_delta=0.001`, matching the exact early-stopping rule in `configs/finetune.yaml['training_run']['early_stopping']`.
- **Best checkpoint: epoch 7**, `val_total_loss=0.143079` (`val_l1_loss=0.012954`, `val_stft_loss=0.130126`). Verified by loading `checkpoints/dns48_finetuned_best.pt` directly (`torch.load`) — `epoch`/loss fields inside the checkpoint match the epoch-7 console line exactly, confirming the checkpointing logic saved the right epoch's state, not a later/stale one. File size 226,473,933 bytes (~216MB), sane for ~18.9M fp32 params × 2 (model weights + Adam's two optimizer moment buffers).
- **Total wall-clock: ~26.5 min** (12 epochs × 126.8-139.5s each, avg ~132s/epoch) — well under the confirmed 2-2.5hr budget and faster than the ~3.64min/epoch pre-run estimate (actual ~2.2min/epoch; the 3.5x forward-to-training-step multiplier assumption in the Phase 3c ceiling entry was conservative, real MPS training throughput came in faster than that estimate).
- **Trend observed:** train loss decreased monotonically epoch 1→12 (0.158689→0.141294 total), while val loss plateaued/oscillated in a tight band from epoch 7 onward (0.143079→0.144212, essentially flat noise). Train/val gap widened over the run (epoch 1: ~0.011, epoch 11: ~0.019) — classic early-overfitting signature for a fine-tune on this dataset size (2162 train pairs), not an underfitting signal. Flagged to user as an honest read, not silently smoothed over.

**Per-locked-rule honest framing:** this is the genuine plateau case, NOT the ceiling-reached-without-plateau case — the run's own final log line ("STOPPED EARLY on plateau at epoch 12... Best val loss 0.143079 at epoch 7") is accurate and should be quoted as-is in any later report; it must NOT be described as "ceiling reached" (it wasn't — training stopped at epoch 12 of a 40-epoch ceiling) and must NOT be silently upgraded to "fully converged" (val loss plateaued, but whether this is a true global optimum for the loss vs. a fixable overfitting artifact is unresolved until evaluated against real SNR/STOI/PESQ targets).

**Decision, per user instruction:** do NOT tweak hyperparameters preemptively. Next step is evaluating `checkpoints/dns48_finetuned_best.pt` (epoch 7) against the real target metrics (SNR>15dB, STOI>0.85, PESQ>2.5) and the Phase 2 baseline numbers already in `context.md` — only if it falls short of those targets will LR/regularization/architecture tweaks be considered, and that would be a new, explicitly-scoped task, not assumed here.

**Phase 3c status: DONE — real training run complete, best checkpoint saved and verified.** Not done: evaluation of the fine-tuned checkpoint against test-split SNR/STOI/PESQ (next phase, not started).

---

## 2026-09-11 — Phase 4 code COMPLETE (not yet run — real run is the user's, per explicit instruction)

**Scope:** New `src/model/inference.py` (runs the Phase 3c fine-tuned checkpoint over the full 299-pair test split, writes enhanced wavs) + extended `src/eval/evaluate.py` (fills in the previously-empty `finetuned` column using those wavs). Did NOT run either myself — per explicit instruction, this is the user's run.

**Verification performed BEFORE writing the eval extension (per task's explicit ask — confirm the checkpoint load actually changes weights, don't assume success from "no error"):**
- Loaded the pretrained dns48 via `load_dns48()`, captured `encoder.0.0.weight` as a real tensor.
- Loaded `checkpoints/dns48_finetuned_best.pt`, called `model.load_state_dict(ckpt['model_state_dict'])`, re-read the same tensor.
- **Result: `torch.equal()` → `False`. Max abs diff = 0.005874408408999443.** Concretely confirms the fine-tuned checkpoint's weights are genuinely different from the raw pretrained checkpoint, not a silent no-op load. This exact check is now also baked into `inference.py` itself as `verify_finetuned_weights_differ()` (runs automatically every time `python -m src.model.inference` is run, raises `RuntimeError` if the weights ever come back identical — not a one-off manual check that could silently stop being true later).

**New file `src/model/inference.py`:**
- `load_finetuned_model()` — loads pretrained dns48 via `load.py`'s existing `load_dns48()` mechanism (same architecture-spec verification as Phase 3a/3c, unchanged), then overwrites weights with `checkpoints/dns48_finetuned_best.pt`'s `model_state_dict`. Raises `FileNotFoundError` with a clear message if the checkpoint doesn't exist (rather than a confusing downstream error) — checked path is exactly the one Phase 3c produced.
- `verify_finetuned_weights_differ()` — the concrete tensor-diff check described above, run automatically as part of `run()`, not left as a one-time manual verification.
- `run()` — iterates all 299 test-manifest pairs, loads each noisy wav via `soundfile` (same I/O convention as the rest of the project), runs inference (`torch.no_grad()`), writes `results/finetuned/{pair_id}_enhanced.wav` via `soundfile` at `output.bit_depth` (PCM_16) — same write convention as `src/baseline/spectral_subtraction.py`'s existing output, so `evaluate.py` can score it identically.

**`src/eval/evaluate.py` changes (extension, not duplication):**
- Added `FINETUNED_DIR = Path("results/finetuned")`.
- Added one data-gathering block in `run()` for the `finetuned` column, scoring `results/finetuned/{pair_id}_enhanced.wav` against clean via the SAME `_score_pair()` used for noisy/baseline (unchanged) -- mirrors the existing `baseline` block exactly, no new scoring logic.
- Replaced the old "finetuned column stays empty" hardcoded-placeholder loop with a generic empty-list normalization that applies uniformly to all three columns -- functionally identical behavior for noisy/baseline (which always had real data) and correctly falls back to `n=0` for finetuned if `results/finetuned/` doesn't exist yet, same placeholder behavior Phase 2 relied on, just no longer hardcoded to always be empty.
- `_summarize()`, `_write_json()`, `_write_csv()`, `_print_table()` are completely UNCHANGED -- confirmed by re-reading the full file after editing; noisy/baseline rows are not regenerated or altered, only the finetuned rows go from `n=0` placeholders to real scored values once the wavs exist.
- Output files unchanged: `results/baseline_results.json` / `results/baseline_results.csv` (filenames deliberately not renamed, to avoid breaking any existing reference to these paths -- they simply now contain all three populated columns instead of two).

**Verification performed (mine, before handing off):**
- `python3 -m py_compile` on both `inference.py` and `evaluate.py` -- pass.
- Re-read the full edited `evaluate.py` after editing to confirm the noisy/baseline code paths are byte-for-byte the same logic as Phase 2, only the finetuned block and the placeholder-normalization loop changed.
- Did NOT run `python -m src.model.inference` or `python -m src.eval.evaluate` myself -- per explicit instruction, this is the user's run.

**Exact commands for the user to run themselves, in order:**
```
python -m src.model.inference
python -m src.eval.evaluate
```

**Phase 4 code status: DONE, code written and statically verified (including a concrete non-assumption check that the checkpoint load changes real weights), NOT yet executed.** Awaiting the user's real evaluation results before the next `logs.md` entry (per explicit instruction: append the real finetuned numbers + honest PS-target assessment only after the user reports back -- not combined with this entry).

---

## 2026-09-11 — Phase 4 REAL RESULTS: finetuned column populated, test split n=299

**Trigger:** User ran `python -m src.model.inference` then `python -m src.eval.evaluate` themselves and reported real console output. `verify_finetuned_weights_differ()` (the runtime guard added this session) passed silently (no `RuntimeError`), confirming the checkpoint load during this real run genuinely used the fine-tuned weights, not a no-op.

**Real results (test split, n=299, all three columns from one evaluate.py run):**

| column    | category   | n   | SNR(dB) | STOI  | PESQ  |
|-----------|-----------|-----|---------|-------|-------|
| noisy     | gunshot   | 97  | -1.09   | 0.566 | 1.296 |
| noisy     | stationary| 98  | 2.01    | 0.631 | 1.368 |
| noisy     | general   | 104 | 0.47    | 0.596 | 1.260 |
| noisy     | overall   | 299 | 0.47    | 0.598 | 1.307 |
| baseline  | gunshot   | 97  | -0.12   | 0.561 | 1.297 |
| baseline  | stationary| 98  | 3.22    | 0.626 | 1.309 |
| baseline  | general   | 104 | 2.36    | 0.588 | 1.271 |
| baseline  | overall   | 299 | 1.84    | 0.592 | 1.292 |
| finetuned | gunshot   | 97  | 8.81    | 0.639 | 1.859 |
| finetuned | stationary| 98  | 9.41    | 0.689 | 1.852 |
| finetuned | general   | 104 | 7.85    | 0.658 | 1.662 |
| finetuned | overall   | 299 | 8.67    | 0.662 | 1.788 |

**PS-target assessment (SNR>15dB, STOI>0.85, PESQ>2.5), overall (n=299):**
- SNR: 8.67 dB vs. target >15 dB — **NOT cleared**.
- STOI: 0.662 vs. target >0.85 — **NOT cleared**.
- PESQ: 1.788 vs. target >2.5 — **NOT cleared**.

**Finetuned vs. baseline, overall:**
- SNR: 1.84 → 8.67 dB — **improved** (+6.83 dB).
- STOI: 0.592 → 0.662 — **improved** (+0.070) — this is one of the two axes baseline regressed on vs. doing nothing; finetuned reverses that regression and improves past the no-processing floor too (0.598 → 0.662).
- PESQ: 1.292 → 1.788 — **improved** (+0.496) — the other axis baseline regressed on; finetuned also clears the no-processing floor (1.307 → 1.788).

**Per-category (all three categories, finetuned vs. baseline):** SNR/STOI/PESQ all improved in every category (gunshot, stationary, general) — no category regressed on any metric.

**Phase 4 status: DONE — real evaluation complete, all three columns populated in `results/baseline_results.{json,csv}`.** Not done: none of the three PS numeric targets are cleared yet at this checkpoint (epoch 7); whether/how to close the remaining gap (further tuning, architecture changes, more data, etc.) is an open decision for the user, not decided or assumed here.

## 2026-09-11 — v2 through v5 fine-tuning iterations + FINAL model decision (v5 locked, no further training)

**Note on this entry:** back-filled from real artifacts already on disk (`results/finetune_training_log_v{2,4,5}.csv`, `results/baseline_results.csv`, `results/snr_bucket_results.csv`, `results/failure_diagnosis*.csv`, `results/v1_vs_v2_worst_pairs_comparison.csv`, checkpoint metadata read directly via `torch.load`, and the fully-documented `training_run_v2`/`training_run_v4`/`training_run_v5` sections of `configs/finetune.yaml`) — this session's own runs were Task 1 (`evaluate_final.py`) and Task 2 (ONNX export scripts), not these training iterations. All numbers below were read from existing files, not re-derived or approximated.

**v1 → v2: diagnosis before any new lever.** `src/eval/diagnose_failures.py` labeled v1's 18 worst-scoring test pairs (by PESQ/STOI) using an STFT-based residual-noise/speech-distortion heuristic: **18/18 labeled "likely over-suppression"** (not under-suppression / leftover noise) — the model was erasing signal, not failing to clean it. This diagnosis, not a blind hyperparameter sweep, is what motivated every subsequent version's specific levers.

**v2 (`training_run` in config, checkpoint `dns48_finetuned_v2_best.pt`, epoch 7 best-val, `val_total_loss=0.27122`)** — three concurrent levers, motivated by the over-suppression diagnosis and v1's still-improving-not-plateaued stop point:
- LR schedule: v1's flat `3e-5` → warmup (5%) + cosine decay, peak `1.5e-4`, floor `3e-5`.
- Loss reweighting: STFT term weight 1.0 → 2.0 (L1 stays 1.0) — targets STOI/PESQ more directly than raw L1.
- Train-only oversampling of hard (low-SNR, <5dB) mixtures, 3x via `WeightedRandomSampler` (val/test kept uniform).
- Early-stopping patience 5 → 10 epochs (v1's 5-epoch patience stopped on a still-fluctuating, not clearly plateaued, val curve).
- Real result (n=299 overall): SNR 8.56dB, STOI 0.666, PESQ 1.775 — **near-identical to v1** (SNR 8.67, STOI 0.662, PESQ 1.788) despite three simultaneous changes.
- `diagnose_data_gaps.py` + `v1_vs_v2_worst_pairs_comparison.csv` followed up: confirmed gunshot's poor performance is NOT gunshot-specific data scarcity — cross-tabulation showed all categories were similarly bad at low-SNR, i.e. a general low-SNR capability gap, not a category-specific one. This ruled out "just add more gunshot data" as the next lever.

**v4 (`training_run_v4`, checkpoint `dns48_finetuned_v4_best.pt`, epoch 11 best-val, `val_total_loss=0.14202`)** — since v2's three simultaneous levers changed nothing measurable, v4 reverted the two unproven ones and added two new, diagnosis-targeted ones instead:
- **Reverted** (evidence didn't support them): STFT reweight back to 1:1 L1:STFT; static 3x oversampling replaced (not just removed) by curriculum below.
- **Kept** (not implicated by any diagnosis): v2's LR warmup+cosine schedule, unchanged.
- **New — curriculum learning:** epoch-aware sampling weight, linearly ramped over the first 40% of epochs then held: low-SNR pairs (<5dB) ramp to 3x weight, gunshot-category pairs ramp to 2x weight (multiplicative when both apply — the exact worst cell from diagnosis). Ends training at the true, unskewed distribution rather than a permanently harder one.
- **New — silence-collapse penalty** (`silence_penalty`, weight 0.05): penalizes the enhanced output going near-silent in frames where the *noisy input* had substantial energy (RMS > 0.01) and output/input RMS ratio drops below 0.1 — targets the diagnosed over-suppression failure mode directly, keyed off the model's own input rather than a blanket minimum-energy floor against clean (which would also penalize correctly-suppressed silence).
- Epoch ceiling raised to 100 (patience 20) — curriculum means the back half of training faces a genuinely different, harder distribution than the front half, so a real second learning phase was expected post-ramp, not just noise around an already-found optimum. Run plateau-stopped at **epoch 31**; epoch 11 was the last new-best checkpoint.
- `diagnose_failures_v4.py` re-ran the same over-suppression heuristic on v4's worst pairs — result available in `results/failure_diagnosis_v4.csv` (same 18-pair-worst-case structure as v1's, direct comparison point; this entry does not re-summarize the aggregate count here since it wasn't independently re-verified against console output this session).
- Real result (n=299 overall): SNR 8.59dB, STOI 0.663, PESQ 1.772 — again **near-identical to v1/v2** across all three metrics.

**v5 (`training_run_v5`, checkpoint `dns48_finetuned_v5_best.pt`, epoch 27 best-val, `val_total_loss=0.13689` — the best val loss of all four versions) — FINAL MODEL, no further training iterations planned.**
- **Dynamic mixing, replacing the static 2162-pair train manifest entirely:** noise mixtures are drawn fresh (not from a fixed pre-built pool) with `noise_category_weights` explicitly uniform (gunshot=stationary=general=1.0) to preserve the same ~1/3-per-category exposure v1-v4's `rng.choice()` produced naturally. `steps_per_epoch=270` deliberately kept identical to v1/v2/v4's static-manifest epoch size (`2162 // 8`) so wall-clock and epoch-count comparisons across all four versions stay apples-to-apples. Seeded once at dataset construction (reproducible run-to-run) but *not* re-seeded per epoch, so successive epochs within one run see genuinely different mixtures — see `src/data/dynamic_dataset.py` / `src/data/build_dynamic_pool.py` (both untracked/new this phase) and `manifests/dynamic_train_pool.json`.
- **Reactive LR, replacing fixed cosine decay:** same warmup shape as v2 (5%, peak `1.5e-4`), then `ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)` on val loss instead of a schedule pre-shaped around a committed `max_epochs` — chosen because v5's `max_epochs=120` is a loose ceiling, not a tight budget a cosine curve needs to reach zero by.
- **Loss:** L1:STFT kept 1:1 (v4's reverted config, reused). **Silence-collapse penalty term was computed/logged every epoch but `silence_penalty_enabled: false`** — confirmed numerically this session (`train_total ≈ train_l1 + train_stft` to 8 decimal places in `finetune_training_log_v5.csv`, NOT `l1+stft+0.05*silence_penalty`) — i.e. the penalty was measured, not applied to gradients, for this run. Per the config's own comment this was pending explicit user confirmation to re-enable once v4's eval showed it working; that confirmation/decision is not recorded in this entry — flagging as open, not assumed either way.
- Early-stopping patience 15 (between v2's 10 and v4's 20 — v5 has no curriculum ramp, so doesn't need v4's extra margin for ramp-induced fluctuation). Ran **42 epochs total**, best-val at epoch 27.
- Real result (n=299 overall, from `results/baseline_results.csv`): **SNR 8.750dB, STOI 0.6744, PESQ 1.7863** — the best of all four versions on every metric, though narrowly (v1: 8.671/0.6620/1.7882 — note v5 is actually slightly *below* v1 on PESQ specifically).

**Real per-category results, all four versions (test split, n=299; source: `results/baseline_results.csv`):**

| version | category   | n   | SNR(dB) | STOI  | PESQ  |
|---------|-----------|-----|---------|-------|-------|
| v1      | gunshot   | 97  | 8.807   | 0.639 | 1.859 |
| v1      | stationary| 98  | 9.408   | 0.689 | 1.852 |
| v1      | general   | 104 | 7.849   | 0.658 | 1.662 |
| v1      | overall   | 299 | 8.671   | 0.662 | 1.788 |
| v2      | gunshot   | 97  | 8.736   | 0.643 | 1.864 |
| v2      | stationary| 98  | 9.275   | 0.694 | 1.824 |
| v2      | general   | 104 | 7.733   | 0.660 | 1.647 |
| v2      | overall   | 299 | 8.564   | 0.666 | 1.775 |
| v4      | gunshot   | 97  | 8.768   | 0.642 | 1.874 |
| v4      | stationary| 98  | 9.339   | 0.691 | 1.818 |
| v4      | general   | 104 | 7.705   | 0.656 | 1.633 |
| v4      | overall   | 299 | 8.586   | 0.663 | 1.772 |
| v5      | gunshot   | 97  | 8.901   | 0.654 | 1.870 |
| v5      | stationary| 98  | 9.430   | 0.702 | 1.833 |
| v5      | general   | 104 | 7.969   | 0.667 | 1.664 |
| v5      | overall   | 299 | 8.750   | 0.674 | 1.786 |

**Real per-SNR-bucket results, v5 vs. v1 (test split; source: `results/snr_bucket_results.csv`):**

| version | bucket      | n  | SNR(dB) | STOI  | PESQ  |
|---------|-------------|----|---------|-------|-------|
| v1      | -5 to 0 dB  | 83 | 5.929   | 0.591 | 1.478 |
| v5      | -5 to 0 dB  | 83 | 6.147   | 0.606 | 1.495 |
| v1      | 0 to 5 dB   | 78 | 7.767   | 0.663 | 1.720 |
| v5      | 0 to 5 dB   | 78 | 7.823   | 0.673 | 1.745 |
| v1      | 5 to 10 dB  | 60 | 10.560  | 0.714 | 1.978 |
| v5      | 5 to 10 dB  | 60 | 10.571  | 0.725 | 1.940 |
| v1      | 10 to 15 dB | 78 | 11.040  | 0.697 | 2.040 |
| v5      | 10 to 15 dB | 78 | 11.047  | 0.710 | 2.019 |

The hardest bucket (-5 to 0dB, the diagnosed weak spot) improved the most in relative terms from v1→v5 (STOI +0.015, largest of any bucket), consistent with v4/v5's low-SNR-targeted curriculum/dynamic-mixing levers, though PESQ in the two highest-SNR buckets is marginally below v1.

**PS-target assessment, v5 overall (n=299) — still NONE cleared:** SNR 8.75dB vs. target >15dB; STOI 0.674 vs. target >0.85; PESQ 1.786 vs. target >2.5. Same conclusion as v1: real, consistent improvement over the classical baseline on every metric/category, but none of the three numeric PS targets reached by any of the four fine-tuning iterations.

**FINAL DECISION (user, 2026-09-11): v5 is locked as the final model for this prototype. No further training iterations planned.** Four fine-tuning iterations (v1/v2/v4/v5) were run; each subsequent version was motivated by a concrete diagnosis of the previous version's failure mode (v1→v2: over-suppression diagnosis; v2→v4: ruled out data-scarcity explanation, added curriculum + silence-penalty; v4→v5: dynamic mixing + reactive LR) rather than blind hyperparameter search, and v5 is the best-performing version on nearly every metric, but the PS's numeric targets remain unreached. Next work shifts to Phase 5 (export), not further fine-tuning.

## 2026-09-11 — Phase 5a code written (not yet run — real run is the user's, per explicit instruction)

**Trigger:** user requested (1) a demo-focused v5-vs-baseline-vs-noisy comparison output for the PPT, and (2) Phase 5a ONNX export + on-Mac verification, both as exact commands to run themselves — no code executed by the assistant this session.

**Task 1 — `src/eval/evaluate_final.py` (new file).** Reuses `_score_pair`/`_snr_bucket`/`_summarize`/`load_config` from the existing `src/eval/evaluate.py` (no duplicated metric-computation logic) restricted to three columns (`noisy`/`baseline`/`finetuned_v5`). Writes `results/final_comparison.csv`; prints overall/per-category/per-SNR-bucket tables plus an explicit `HEADLINE` section for the gunshot category row and the `-5 to 0 dB` bucket row (the two evidence points named for the PS's own framing). Does not modify or duplicate `evaluate.py`'s existing full six-column output.

**Task 2 — `src/export/to_onnx.py` + `src/export/verify_onnx.py` (new files, new `src/export/` package), plus a new `export:` block added to `configs/finetune.yaml`** (`checkpoint_path`, `onnx_dir`, `onnx_filename`, `fixed_length_seconds: 4.0` — matches `training.segment_seconds`, `opset_version: 17` — per CLAUDE.md rule 1, no magic numbers inline).
- `to_onnx.py`: exports `checkpoints/dns48_finetuned_v5_best.pt` to ONNX. Explicitly flags the known LSTM/ONNX dynamic-sequence-length export risk in code comments. Attempts fixed-length export first (no `dynamic_axes` — the known-safer path); only then attempts a dynamic-length export, which is reported explicitly on failure (logged error, non-silent) rather than silently falling back to fixed-length-only.
- `verify_onnx.py`: runs the fixed-length ONNX export via `onnxruntime` over the same 299-pair test split used for v5's PyTorch evaluation, computes SNR/STOI/PESQ (reusing `src/eval/metrics.py`, no duplicated logic), and reports the exact delta against v5's known PyTorch numbers (hardcoded from the already-reported real run: SNR 8.75dB, STOI 0.674, PESQ 1.786 overall). Explicitly flags that test clips are center-cropped/zero-padded to the fixed export length before scoring, which is a real confound in the delta (not identical input to v5's original variable-length PyTorch eval) — logged loudly, not hidden, so a nonzero delta isn't misattributed to export drift alone if the crop/pad effect turns out to be the actual cause.
- Neither `onnx` nor `onnxruntime` is installed in this environment (checked, both `ModuleNotFoundError`) — not added to `requirements.txt` by the assistant; install command given to the user to run explicitly, per instruction not to run/install anything this session.

**Status: code written, nothing executed.** Real results for both tasks are pending the user's own run — per explicit instruction, this entry will not be updated with pass/fail or numeric ONNX-delta outcomes until the user reports back.

## 2026-09-11 — Phase 5a REAL RESULTS: ONNX export verified clean, export drift isolated from a real crop confound

**Trigger:** user ran `python -m src.export.to_onnx` then `python -m src.export.verify_onnx` themselves and reported real console output.

**Export step:** both the fixed-length export (`checkpoints/onnx/dns48_finetuned_v5.onnx`, no `dynamic_axes`) AND the dynamic-length export (`checkpoints/onnx/dynamic_dns48_finetuned_v5.onnx`, time axis dynamic) **succeeded**. The LSTM/dynamic-sequence-length ONNX risk flagged explicitly in `to_onnx.py`'s comments (per task instruction) did not materialize as a failure — dynamic export worked on the first attempt, no silent fallback needed.

**First verification pass (`verify_onnx.py`, fixed-length export, n=299):** ONNX overall SNR 8.4508dB, STOI 0.6700, PESQ 1.7505 vs. v5's known PyTorch numbers (SNR 8.75dB, STOI 0.674, PESQ 1.786) — delta SNR -0.2992dB, STOI -0.0040, PESQ -0.0355. As flagged in the script's own logged warning, this comparison was confounded: test clips were center-cropped/zero-padded to the 4.0s fixed export length before scoring, and **184/299 test clips (61.5%, confirmed by direct manifest inspection this session — min 1.445s, max 32.485s, mean 6.47s) are longer than 4s**, so most of the test set had real audio content discarded by the crop, independent of export quality. This delta could not be attributed to export drift vs. crop effect from this pass alone.

**Isolation pass (`src/export/verify_onnx_isolated.py`, new script written this session specifically to resolve the ambiguity — feeds the IDENTICAL cropped waveform to both the ONNX graph and the PyTorch model, so any difference between them is pure export drift with the crop variable held constant):**

| comparison | SNR(dB) | STOI | PESQ |
|---|---|---|---|
| **export drift** (onnx vs. pytorch, same cropped input) | **-0.0000** | **-0.0000** | **-0.0000** |
| **crop effect** (pytorch_cropped vs. v5's original full-length PyTorch eval) | -0.2992 | -0.0040 | -0.0355 |

**Conclusion: export drift is exactly zero (to displayed precision) — the ONNX graph computes identically to the PyTorch model on identical input.** The entire gap `verify_onnx.py` first reported is fully and exclusively attributable to the fixed-length center-crop discarding real audio content on the 61.5% of test clips longer than 4s, not to any loss introduced by the ONNX export itself.

**Phase 5a status: ONNX export VERIFIED CLEAN.** Per the task's own stopping condition ("if there's a meaningful gap, stop and report it, don't proceed to Core ML/TFLite conversion on a broken export") — there is no meaningful gap once isolated; proceeding to Core ML/TFLite conversion is supported by this verification. Open, not decided here: whether the fixed-length-only constraint (4.0s window) is acceptable for the eventual on-device deployment shape, or whether the dynamic-length export (which also succeeded) should be carried forward instead to avoid needing to chunk/pad longer audio at inference time — that tradeoff is for the user to decide before Core ML/TFLite conversion begins.

**Task 1 (focused demo comparison, `evaluate_final.py`) — not yet reported.** This entry covers Task 2 only; Task 1's real output has not been provided by the user as of this entry.

## 2026-09-11 — Phase 5b: Core ML conversion (iOS target), fp32, REAL RESULTS verified clean

**Trigger:** user confirmed platform (iOS/Core ML) and conversion source (fixed-length 4.0s ONNX export, not dynamic-length) via explicit questions before any platform-specific code was written, per CLAUDE.md rule 10.

**Path taken, with two real failures diagnosed and fixed along the way (not glossed over):**

1. **First real failure — ONNX has no converter path.** Initial `to_coreml.py` called `ct.convert()` directly on the `.onnx` file, per the original plan. Real run failed: `ValueError: Unable to determine the type of the model... source... ["tensorflow", "pytorch", "milinternal"]`. Root cause: the installed coremltools version (9.0) has no built-in ONNX converter — confirmed, not assumed. **Fix:** rewrote `to_coreml.py` to trace the v5 PyTorch checkpoint directly (`torch.jit.trace`) and convert that (`source="pytorch"` implied), bypassing ONNX entirely for the Core ML leg. Phase 5a's ONNX export/verification remain valid and untouched for their own purposes.

2. **Second real failure — backend/deployment-target conflict.** `convert_to="neuralnetwork"` (chosen on the now-invalidated assumption that it has more mature LSTM op coverage) failed: `ValueError: If minimum deployment target is iOS15... or higher, then 'convert_to' cannot be neuralnetwork. It must be 'mlprogram'`. **Fix (user-confirmed):** `convert_to` changed to `"mlprogram"`, `minimum_deployment_target` kept at `iOS15`. `configs/finetune.yaml`'s `coreml:` block comments updated to state the real constraint, not the superseded LSTM-coverage rationale.

3. **Third real failure — LSTM/shape-tracing crash inside coremltools' MIL frontend**, hit twice, both root-caused via direct investigation (not patched blindly, per explicit instruction) before any fix was proposed:
   - **Task 1 finding (verified, not assumed):** `Demucs.valid_length(length)` and `downsample2`'s `x.shape[-1] % 2 != 0` parity check are pure functions of the model's static hyperparameters (`resample=4, depth=5, kernel_size=8, stride=4`) and the one fixed input length (64000 samples / 4.0s @ 16kHz) — not audio-content-dependent. Confirmed by direct arithmetic: `valid_length(64000) = 64085` (pad amount 85), and both `downsample2` calls in the `resample=4` path see even-length input (256340, then 128170) — the odd-pad branch is never taken for this shape.
   - **First fix attempt:** new `src/export/traceable_demucs.py::TraceableDemucs` — wraps the trained `Demucs` instance as-is (same weights/layers, `denoiser/demucs.py`/`resample.py` untouched), replacing the length-dependent glue arithmetic with the precomputed constants above. Conversion got further but crashed again at a *different* op (`ops.py::_int`/`_cast`, same `TypeError: only 0-dimensional arrays can be converted to Python scalars`), traced via `/debug-sensei` to a second, self-inflicted instance of the identical bug class: the wrapper's own `_upsample2_traceable`/`_downsample2_traceable` helpers (copied near-verbatim from `denoiser/resample.py`) still contained `*other, time = x.shape` / `*other, time = xodd.shape` — reading a traced tensor's shape into a Python value, the exact hazard the wrapper was built to eliminate, reintroduced by copying the original code too literally. Also found and removed a second self-inflicted instance: an added runtime guard (`if mix.shape[-1] != FIXED_INPUT_LENGTH`) inside `forward()` itself traced into a graph op — moved that check to `to_coreml.py`, executed once before tracing, not inside the traced function.
   - **Second fix:** precomputed all remaining length-dependent values as plain Python ints — upsample2's two input `time`s (64085, 128170), downsample2's two `odd_time`s (128170, 64085), and the decoder's 5 skip-connection slice lengths (`_DECODER_LENGTHS = [249, 1000, 4004, 16020, 64084, 256340]`, replacing `skip[..., :x.shape[-1]]`).
   - **Verified behaviorally correct before calling it fixed** (per explicit instruction — tracing/converting without error is not sufficient): (a) eager `TraceableDemucs` output vs. unmodified `Demucs.forward()`, same random input at length 64000 — **exact 0.0 max abs diff**; (b) `torch.jit.trace`d wrapper output vs. eager wrapper output — **exact 0.0 max abs diff**, no shape-cast `TracerWarning` remaining. Both checks run before attempting Core ML conversion again.

4. **Fourth real issue, non-blocking — fp16 default precision.** First successful `mlprogram` conversion's log showed `cast_fp16_to_fp32` ops throughout — `mlprogram`'s default `compute_precision` is `FLOAT16` on Apple targets, which would have conflated fp16 rounding drift with pure conversion-format drift (the task's explicit separation-of-concerns goal, same discipline as Phase 5a's crop-vs-export isolation). **Fix (user-confirmed):** added `compute_precision=ct.precision.FLOAT32` to `ct.convert()`; reconverted.

**Real conversion result:** `checkpoints/coreml/dns48_finetuned_v5.mlpackage` written successfully (fp32, `mlprogram`, `iOS15` min target). **File size: 72MB** (`weight.bin`) — the reference point for quantization's later size-reduction claim.

**Real verification result (`src/export/verify_coreml.py`, coremltools' `predict()` API, on-Mac, no phone required — same fixed-length-crop methodology as `verify_onnx_isolated.py`, test split n=299):**

| comparison | SNR(dB) | STOI | PESQ |
|---|---|---|---|
| Core ML (fp32 mlprogram) | 8.4508 | 0.6700 | 1.7505 |
| ONNX isolated baseline | 8.4508 | 0.6700 | 1.7505 |
| **delta (coreml − onnx)** | **+0.0000** | **-0.0000** | **+0.0000** |

Per-category (n=299): gunshot 8.67dB/0.652/1.828, stationary 8.97dB/0.695/1.767, general 7.76dB/0.664/1.663 — all consistent with the ONNX/PyTorch numbers already reported for this fixed-length-crop methodology.

**Conclusion: Core ML conversion introduces zero measurable drift (to displayed precision) vs. the already-verified-clean ONNX export**, at full fp32 precision — behavioral match, not just a clean conversion, is the bar this was held to. **Phase 5b status: Core ML conversion (iOS, fp32) VERIFIED CLEAN.**

**New files this session:** `src/export/traceable_demucs.py` (export-only wrapper, does not modify `denoiser/demucs.py` or `denoiser/resample.py`), `src/export/to_coreml.py`, `src/export/verify_coreml.py`. New `coreml:` config block in `configs/finetune.yaml` (`mlpackage_dir`, `mlpackage_filename`, `convert_to: "mlprogram"`, `minimum_deployment_target: "iOS15"`).

**Open, not decided here:** `coremltools` is not yet pinned in `requirements.txt` (same open item as `onnx`/`onnxruntime` from Phase 5a). Quantization (fp16 or int8) is explicitly out of scope for this entry — this was the fp32-only conversion step, deliberately isolated from precision-reduction so a later quantization step's size/quality tradeoff can be measured cleanly against this 72MB fp32 baseline, not conflated with conversion-format drift.

## 2026-09-11 — Phase 5b follow-up: fp16 compute-precision isolation, REAL RESULTS (non-trivial cost measured)

**Trigger:** user wanted fp16 compute precision's cost measured as its own isolated quantity, separate from conversion-format drift (already proven zero) and from quantization (not run), so a ship fp32-vs-fp16 decision can be made from real numbers rather than assumption.

**Method:** `src/export/to_coreml.py` parameterized with `run(precision=...)` / `--fp16` CLI flag — same `TraceableDemucs` wrapper, same fixed 4.0s input, same `mlprogram`/`iOS15` target, ONLY `compute_precision` changed (`ct.precision.FLOAT16` vs. the already-verified `FLOAT32`). Written to a separate `.mlpackage` (`dns48_finetuned_v5_fp16.mlpackage`, config: `coreml.mlpackage_filename_fp16`) — the fp32 artifact was not overwritten, both kept on disk. `src/export/verify_coreml.py` parameterized with `run(use_fp16=...)` / `--fp16` to compare the fp16 `.mlpackage` against the fp32 Core ML baseline (not the ONNX baseline), isolating pure precision effect.

**Real result (`results/coreml_fp16_verification.csv`, test split n=299, same fixed-length-crop methodology as every prior verification):**

| comparison | SNR(dB) | STOI | PESQ |
|---|---|---|---|
| Core ML fp16 | 8.2036 | 0.6693 | 1.6881 |
| Core ML fp32 (baseline) | 8.4508 | 0.6700 | 1.7505 |
| **delta (fp16 − fp32)** | **-0.2472** | **-0.0007** | **-0.0624** |

Per-category (n=299): gunshot 8.41dB/0.650/1.739, stationary 8.73dB/0.695/1.713, general 7.52dB/0.663/1.617 — all three categories moved in the same direction vs. their fp32 counterparts (no category anomaly), consistent with a real, consistent precision-noise effect rather than a one-off artifact.

**Verdict, per the verification script's own threshold (real numbers, not "looks fine"): delta is NOT small** (script's own threshold: SNR<0.05, STOI<0.005, PESQ<0.02 — SNR and PESQ both exceed it, by ~5x and ~3x respectively). This is a real, reportable fp16 compute-precision cost, most pronounced in PESQ (-0.0624, the largest proportional hit of the three metrics) — plausible given fp16 accumulation error compounding across this model's many sequential layers (5 encoder + 2-layer LSTM + 5 decoder + 2 downsample stages).

**File size:** fp32 `.mlpackage` 72MB, fp16 `.mlpackage` 54MB — a ~25% reduction, notably less than the ~50% a naive halving-of-weight-storage assumption would predict. Reported as-is, not smoothed over: `compute_precision=FLOAT16` changes compute/intermediate-activation precision, not necessarily every stored weight/buffer's width, so the size reduction is partial.

**Status: fp16 compute-precision cost is now a measured quantity, not an assumption.** Ship decision (fp32 vs. fp16) is an open decision for the user, informed by this real tradeoff: fp16 saves ~18MB (~25%) at a real, non-negligible quality cost (SNR -0.25dB, PESQ -0.06) — not free, and not devastating, but not rounding-level either. Quantization (a further, separate step, still not started) was explicitly not conflated with this measurement.

## 2026-09-11 — Phase 5b follow-up: fp32 locked as shipping precision; INT8 weights-only quantization isolation, REAL RESULTS

**Decision (user, prior to this step):** FLOAT32 is locked as the shipping precision. fp16 was verified acceptable in numeric terms (see prior entry) but rejected for defence-context safety margin — the model's ~111x real-time inference headroom on the M4 Pro target makes fp16's speed/size benefit unnecessary, so there is no reason to accept its non-trivial quality cost (SNR -0.25dB, PESQ -0.06). INT8 quantization is now tested as its own isolated measurement, on top of the verified fp32 artifact — same discipline as every precision step so far (real numbers, not assumption).

**Trigger:** user requested the exact command only, no execution by the agent — command specified, task not run, `logs.md` not touched until user reported real output (per explicit instruction). User then ran the commands themselves and reported console output.

**Method:** new `src/export/quantize_coreml.py`, applying `coremltools.optimize.coreml.linear_quantize_weights` (mode `"linear_symmetric"`, dtype `int8`, per-channel granularity — coremltools 9.0's documented default for this call) directly to the **already-verified fp32 `.mlpackage`** (`checkpoints/coreml/dns48_finetuned_v5.mlpackage`) — not requantized from ONNX, not re-traced from the PyTorch checkpoint. Saved as a separate artifact (`dns48_finetuned_v5_int8.mlpackage`); fp32 and fp16 artifacts untouched. **Weights-only, confirmed explicitly against the coremltools 9.0 API and its own docstring:** `linear_quantize_weights` quantizes only the stored `const` weight tensors (via `constexpr_affine_dequantize`/`constexpr_blockwise_shift_scale` ops); `coremltools.optimize.coreml.linear_quantize_activations` is a separate, uncalled function in this coremltools version — activations remain float at runtime for this artifact, not int8. This distinction is real, not a formality: the fp32-vs-int8 delta measured below reflects weight quantization only.

**Verification:** `verify_coreml.py --int8`, same isolated fixed-length-crop methodology as every prior step (n=299), fp32 Core ML as baseline (SNR 8.4508, STOI 0.6700, PESQ 1.7505) — isolating pure weight-quantization effect, separate from conversion-format drift (already proven zero) and fp16 compute-precision effect (already measured separately).

**File size (`.mlpackage` total, `du -sh`):**

| precision | size | vs. fp32 |
|---|---|---|
| fp32 | 72MB | 1.00x |
| fp16 | 54MB | 0.75x (~25% reduction) |
| int8 (weights-only) | 18MB | 0.25x (**exactly 4.00x reduction**) |

The expected ~4x reduction from fp32 weight storage is confirmed exactly (72MB / 18MB = 4.00), a cleaner result than fp16's partial ~25% reduction — consistent with weights-only int8 quantizing the dominant contributor to `.mlpackage` size (stored weights) directly, vs. fp16's `compute_precision` flag which does not uniformly halve every stored tensor's width.

**Real result (`results/coreml_int8_verification.csv`, n=299):**

| comparison | SNR(dB) | STOI | PESQ |
|---|---|---|---|
| Core ML int8 (weights-only) | 8.4443 | 0.6698 | 1.7481 |
| Core ML fp32 (baseline) | 8.4508 | 0.6700 | 1.7505 |
| **delta (int8 − fp32)** | **-0.0065** | **-0.0002** | **-0.0024** |

Per-category (n=299): gunshot 8.66dB/0.651/1.820, stationary 8.97dB/0.695/1.766, general 7.75dB/0.663/1.664 — all three within ~0.01–0.02 of their fp32 counterparts, no category standing out.

**Verdict, per the verification script's own threshold (SNR<0.05, STOI<0.005, PESQ<0.02): delta IS small/rounding-level** — int8 weights-only quantization introduces no meaningful drift at this step. This is a materially different outcome from fp16 (which failed the same threshold on SNR and PESQ by ~5x and ~3x) despite int8 being the more aggressive bit-width reduction — consistent with int8 here only touching stored weights (dequantized to float before each op, per coremltools' own documented behavior) while fp16 changed the actual runtime compute/accumulation precision throughout the graph.

**LSTM quantization-error hypothesis — checked, not confirmed:** per-SNR-bucket breakdown (new addition to `verify_coreml.py`, reused `evaluate.py`'s existing bucket edges) for int8:

| snr_bucket | n | SNR(dB) | STOI | PESQ |
|---|---|---|---|---|
| -5 to 0 dB | 83 | 5.70 | 0.599 | 1.483 |
| 0 to 5 dB | 78 | 7.73 | 0.671 | 1.716 |
| 5 to 10 dB | 60 | 10.18 | 0.718 | 1.855 |
| 10 to 15 dB | 78 | 10.75 | 0.706 | 1.980 |

These are int8's own absolute scores per bucket, not a per-bucket delta against fp32 (fp32's per-bucket numbers were never separately recorded in an earlier step, so no bucket-level baseline exists to diff against here) — the absolute pattern (monotonically increasing SNR/STOI/PESQ with higher input SNR) is the expected general trend for any precision, not evidence specific to quantization error. **The overall delta (-0.0065 SNR, -0.0024 PESQ) is close enough to zero that meaningfully attributing any of it to a specific SNR bucket — let alone the LSTM specifically — is not supportable from this data.** A real per-bucket fp32-vs-int8 delta comparison (which would require re-running fp32 through the same new bucket-reporting code, not yet done) is the next step if this hypothesis is worth pursuing further; given the overall delta is already an order of magnitude inside the "small" threshold, that follow-up was not run as it is unlikely to change the ship decision.

**Status: INT8 weights-only quantization is VERIFIED — expected ~4x size reduction confirmed exactly, with rounding-level quality delta (not the fp16 tradeoff).** Combined with the fp32-locked shipping decision above: fp32 remains the precision decision for defence-context safety margin regardless of int8's clean numbers here, since the ~111x real-time headroom argument against fp16 applies equally against adopting int8 purely for size/speed — this entry records int8 as a measured, available option, not a change to the locked shipping precision. That call (whether int8's 4x size reduction is independently worth taking despite headroom being a non-issue, e.g. for app bundle size) is open for the user to decide, not decided here.
