"""Phase 4 v2 pre-flight: concretely verifies the v2 checkpoint before any
evaluation runs against it.

Two checks, both required before Phase 4 v2 evaluation proceeds:

1. Integrity -- loads checkpoints/dns48_finetuned_v2_best.pt cleanly and
   confirms its saved metadata reads epoch=7 (per the interrupted-run
   post-mortem: epoch 7 was v2's last new-best before the run was Ctrl+C'd
   mid-epoch-14, and checkpoint saves only happen on new-best at completed-
   epoch boundaries, so the file should be untouched since epoch 7).

2. Distinctness -- confirms the v2 checkpoint's weights are a genuinely
   distinct fine-tune, not an accidental copy of the pretrained checkpoint
   OR the v1 checkpoint. Compares encoder.0.0.weight (same tensor
   inference.py's verify_finetuned_weights_differ() already uses for the
   pretrained-vs-finetuned check) pairwise across all three: pretrained,
   v1, v2.

Run standalone: `python -m src.model.verify_v2_checkpoint` from repo root.
Exits non-zero and prints a clear failure reason if either check fails --
never silently proceeds on a bad checkpoint.
"""

import logging

import torch

from src.model.load import load_config, load_dns48

V1_CHECKPOINT_PATH = "checkpoints/dns48_finetuned_best.pt"
V2_CHECKPOINT_PATH = "checkpoints/dns48_finetuned_v2_best.pt"
EXPECTED_EPOCH = 7
EXPECTED_VAL_TOTAL_LOSS = 0.271223
VAL_TOTAL_LOSS_ATOL = 1e-5

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def verify_integrity(checkpoint_path: str = V2_CHECKPOINT_PATH) -> dict:
    """Loads the checkpoint and confirms its metadata matches the expected
    epoch/val_total_loss. Raises RuntimeError on any mismatch or load failure
    -- never returns a "looks fine" result on a bad checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(
            f"FAILED to load {checkpoint_path} -- file is corrupted or unreadable: {e}"
        ) from e

    for key in ("model_state_dict", "epoch", "val_total_loss"):
        if key not in checkpoint:
            raise RuntimeError(
                f"{checkpoint_path} loaded but is missing expected key '{key}' -- "
                f"this looks like a partial/corrupted checkpoint, not epoch {EXPECTED_EPOCH}'s save."
            )

    epoch = checkpoint["epoch"]
    val_total_loss = checkpoint["val_total_loss"]

    if epoch != EXPECTED_EPOCH:
        raise RuntimeError(
            f"{checkpoint_path} epoch={epoch}, expected {EXPECTED_EPOCH} -- "
            f"this is NOT the epoch-7 checkpoint. Do not proceed to evaluation."
        )

    if abs(val_total_loss - EXPECTED_VAL_TOTAL_LOSS) > VAL_TOTAL_LOSS_ATOL:
        raise RuntimeError(
            f"{checkpoint_path} val_total_loss={val_total_loss}, expected "
            f"~{EXPECTED_VAL_TOTAL_LOSS} (atol={VAL_TOTAL_LOSS_ATOL}) -- "
            f"this does NOT match epoch 7's recorded value. Do not proceed to evaluation."
        )

    logger.info(
        f"Integrity PASS: {checkpoint_path} loads cleanly, epoch={epoch}, "
        f"val_total_loss={val_total_loss} (matches expected epoch {EXPECTED_EPOCH} / "
        f"~{EXPECTED_VAL_TOTAL_LOSS})."
    )
    return checkpoint


def verify_distinctness(v2_checkpoint: dict) -> None:
    """Confirms v2's encoder.0.0.weight differs from BOTH the raw pretrained
    weights AND the v1 checkpoint's weights -- rules out v2 accidentally
    being a no-op copy of either."""
    cfg = load_config()
    pretrained_model, device = load_dns48(cfg)
    pretrained_w = pretrained_model.encoder[0][0].weight.detach().to("cpu")

    v1_checkpoint = torch.load(V1_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    v1_w = v1_checkpoint["model_state_dict"]["encoder.0.0.weight"].detach()

    v2_w = v2_checkpoint["model_state_dict"]["encoder.0.0.weight"].detach()

    if torch.equal(v2_w, pretrained_w):
        raise RuntimeError(
            "v2 checkpoint's encoder.0.0.weight is IDENTICAL to the raw pretrained "
            "checkpoint -- v2 did not actually fine-tune. Do not proceed to evaluation."
        )
    if torch.equal(v2_w, v1_w):
        raise RuntimeError(
            "v2 checkpoint's encoder.0.0.weight is IDENTICAL to v1's checkpoint -- "
            "this looks like v1's checkpoint accidentally saved under the v2 filename, "
            "not a distinct fine-tune. Do not proceed to evaluation."
        )

    diff_vs_pretrained = (v2_w - pretrained_w).abs().max().item()
    diff_vs_v1 = (v2_w - v1_w).abs().max().item()
    logger.info(
        f"Distinctness PASS: v2 encoder.0.0.weight differs from pretrained "
        f"(max abs diff={diff_vs_pretrained:.6f}) and from v1 "
        f"(max abs diff={diff_vs_v1:.6f}). Genuinely a distinct fine-tune."
    )


def main() -> None:
    v2_checkpoint = verify_integrity()
    verify_distinctness(v2_checkpoint)
    logger.info("All v2 checkpoint pre-flight checks PASSED. Safe to proceed to Phase 4 v2 evaluation.")


if __name__ == "__main__":
    main()
