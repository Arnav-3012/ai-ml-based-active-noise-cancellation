"""Phase 4: runs a fine-tuned dns48 checkpoint over the full test split.

Loads the pretrained dns48 architecture via `load.py`'s existing mechanism
(same as Phase 3a/3c), then overwrites its weights with the fine-tuned
state dict from a checkpoint (default: `checkpoints/dns48_finetuned_best.pt`,
Phase 3c's v1 best-val checkpoint, epoch 7). This is a real weight swap, not
the pretrained model run twice under a different name -- verified once,
concretely, in `verify_finetuned_weights_differ()` below (direct tensor
comparison, same method the Phase 3c ceiling planning used when
sanity-checking this before writing the eval extension).

Writes enhanced wavs to `results/finetuned/{pair_id}_enhanced.wav` by default,
mirroring `src/baseline/spectral_subtraction.py`'s output convention exactly
(same `soundfile` write, same `output.bit_depth` subtype from config) so
`src/eval/evaluate.py` can score this column the same way it already scores
the baseline column. --checkpoint/--output-dir let a second run (e.g. v2's
checkpoint) write to a separate output dir without overwriting v1's.

Run standalone: `python -m src.model.inference` from repo root, or with
`--checkpoint checkpoints/dns48_finetuned_v2_best.pt --output-dir results/finetuned_v2`
for a second checkpoint.
"""

import argparse
import json
import logging
from pathlib import Path

import soundfile as sf
import torch

from src.model.load import load_config, load_dns48

CONFIG_PATH = "configs/finetune.yaml"
TEST_MANIFEST = Path("manifests/test.json")
OUTPUT_DIR = Path("results/finetuned")
CHECKPOINT_PATH = Path("checkpoints/dns48_finetuned_best.pt")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_finetuned_model(
    cfg: dict | None = None, checkpoint_path: Path = CHECKPOINT_PATH
) -> tuple[torch.nn.Module, torch.device, dict]:
    """Loads the pretrained dns48 architecture, then restores the fine-tuned
    weights from the given checkpoint on top of it.

    Returns: (model, device, checkpoint_metadata) -- metadata is the
    checkpoint dict minus the two state_dicts (epoch/val losses), useful
    for logging which checkpoint is actually in use.
    """
    if cfg is None:
        cfg = load_config()

    model, device = load_dns48(cfg)  # loads pretrained weights first, per load.py's own mechanism

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} not found -- training must be run first "
            f"before evaluation."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metadata = {k: v for k, v in checkpoint.items() if k not in ("model_state_dict", "optimizer_state_dict")}
    logger.info(f"Loaded fine-tuned checkpoint {checkpoint_path}: {metadata}")

    return model, device, metadata


def verify_finetuned_weights_differ(model: torch.nn.Module, cfg: dict, device: torch.device) -> None:
    """Concretely confirms the loaded state dict actually changed the
    weights vs. the raw pretrained checkpoint -- does NOT just assume
    `load_state_dict()` succeeding (no error) means the right weights are
    in place. Compares one real weight tensor (encoder.0.0.weight) between
    a freshly-loaded pretrained model and the current (post fine-tuned-load)
    model; raises if they are identical, which would mean the checkpoint
    load silently did nothing.
    """
    pretrained_model, _ = load_dns48(cfg)
    pretrained_w = pretrained_model.encoder[0][0].weight.detach().to(device)
    finetuned_w = model.encoder[0][0].weight.detach()

    if torch.equal(pretrained_w, finetuned_w):
        raise RuntimeError(
            "Fine-tuned model's encoder.0.0.weight is IDENTICAL to the raw pretrained "
            "checkpoint -- the fine-tuned state dict did not actually change the weights. "
            "This would mean Phase 4 is silently evaluating the pretrained model, not the "
            "fine-tuned one."
        )

    max_abs_diff = (pretrained_w - finetuned_w).abs().max().item()
    logger.info(
        f"Verified: fine-tuned weights differ from pretrained "
        f"(encoder.0.0.weight max abs diff = {max_abs_diff:.6f}). Not evaluating a no-op checkpoint."
    )


def _load_wav(path: Path) -> tuple[torch.Tensor, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data[:, 0]).contiguous(), sr


def run(
    manifest_path: Path = TEST_MANIFEST,
    output_dir: Path = OUTPUT_DIR,
    checkpoint_path: Path = CHECKPOINT_PATH,
) -> None:
    cfg = load_config()
    sample_rate = cfg["sample_rate"]
    subtype = cfg["output"]["bit_depth"]

    model, device, metadata = load_finetuned_model(cfg, checkpoint_path)
    verify_finetuned_weights_differ(model, cfg, device)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, entry in enumerate(manifest):
            noisy, sr = _load_wav(Path(entry["noisy_path"]))
            if sr != sample_rate:
                raise RuntimeError(
                    f"{entry['noisy_path']} has sample rate {sr}, expected {sample_rate}"
                )

            noisy_batch = noisy.unsqueeze(0).to(device)  # [1, samples]
            enhanced = model(noisy_batch).squeeze(0).squeeze(0).cpu().numpy()  # [samples]

            out_path = output_dir / f"{entry['pair_id']}_enhanced.wav"
            sf.write(str(out_path), enhanced, sample_rate, subtype=subtype)

            if (i + 1) % 50 == 0 or (i + 1) == len(manifest):
                logger.info(f"[inference] {i + 1}/{len(manifest)} pairs processed")

    logger.info(f"Done. Wrote {len(manifest)} enhanced wavs to {output_dir} using checkpoint {metadata}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help=f"Path to the fine-tuned checkpoint (default: {CHECKPOINT_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory to write enhanced wavs to (default: {OUTPUT_DIR}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(output_dir=args.output_dir, checkpoint_path=args.checkpoint)
