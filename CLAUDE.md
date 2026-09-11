# CLAUDE.md — SIH26052 Agent Rules

Non-negotiable constraints for any coding agent (including Claude Code) working in this repo.

1. **No magic numbers.** Every hyperparameter (learning rate, batch size, SNR range, STFT window/hop, loss weights, epochs, etc.) lives in `configs/finetune.yaml`. Code reads from config — never hardcodes a tunable value inline.

2. **Never commit audio.** `data/raw/` and `data/processed/` are gitignored. Do not write code that stages, commits, or force-adds files under these paths (the `.gitkeep` placeholders are the sole exception, added once at scaffold time).

3. **`logs.md` is append-only.** Always append a new entry at the bottom. Never edit, reorder, or delete past entries — it is the audit trail of what was done and why.

4. **`context.md` is a snapshot, not a log.** Overwrite it fully each session to reflect current state (model, data, phase, decisions). Do not accumulate history there — that's what `logs.md` is for.

5. **No dashboard/UI code as part of the pipeline.** `src/` stays free of Streamlit/Gradio/web-UI code — the pipeline itself is not a dashboarded product. Exception (decided 2026-09-12): a standalone Streamlit results dashboard for judge presentation is in scope, kept fully outside `src/` (e.g. `dashboard/`), reading only already-computed files under `results/` — it does not run training/inference/eval itself.

6. **No training from scratch.** The dns48 checkpoint (Facebook Research Denoiser) is fine-tuned via transfer learning only. Do not initialize or train a model from random weights.

7. **`notebooks/` is scratch only.** Nothing in `src/` may import from or depend on anything in `notebooks/`. Notebooks are for exploration; `src/` is the source of truth.

8. **Target device: M4 Pro MacBook, PyTorch MPS backend.** All model/tensor code must check `torch.backends.mps.is_available()` and fall back to CPU with an explicit, visible warning — never fail silently and never assume CUDA.

9. **Verify before checking off.** Before marking any task done in `plan.md`, actually run the code and show real output. Do not check a box based on code that "should work."

10. **Ambiguity → ask, don't guess.** If a task is underspecified or a design choice isn't already decided in `context.md`/`info.md`, stop and ask the user rather than silently picking a default.
