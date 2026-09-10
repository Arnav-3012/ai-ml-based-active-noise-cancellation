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
