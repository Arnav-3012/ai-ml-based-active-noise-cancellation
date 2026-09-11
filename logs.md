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
