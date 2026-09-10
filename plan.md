# plan.md — Phased Task List

High-level phase tracker. Full spec lives in `context.md`; details of each phase are decided as we enter it, not re-derived here.

- [x] **Phase 0 — Scaffold**: repo structure, .gitignore, venv, dependency install, CLAUDE.md/context.md/info.md/plan.md/logs.md.
- [ ] **Phase 1 — Data**: download + verify corpora, SNR-targeted mixing, speaker-disjoint splits, manifests.
- [ ] **Phase 2 — Baseline**: hand-built spectral subtraction implementation + baseline metrics on test set.
- [ ] **Phase 3 — Fine-tuning**: load dns48 on MPS, fine-tune with L1 + multi-res STFT loss per `configs/finetune.yaml`.
- [ ] **Phase 4 — Evaluation**: SNR/STOI/PESQ for model vs. baseline, results table + samples.
- [ ] **Phase 5 — Export**: ONNX export → CoreML/TFLite.
