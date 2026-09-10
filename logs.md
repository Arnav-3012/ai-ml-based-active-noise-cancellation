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
