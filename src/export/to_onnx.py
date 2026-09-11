"""Phase 5a: exports the v5 fine-tuned dns48 checkpoint to ONNX.

KNOWN RISK (flagged explicitly, not glossed over): dns48
(denoiser.demucs.Demucs) contains a 2-layer LSTM. LSTM modules have a
well-documented history of ONNX export fragility -- especially around
dynamic sequence-length handling (the exporter has to trace/unroll the
recurrent loop, and dynamic-axis tracing for RNNs has had correctness and
opset-compatibility issues across torch/onnx versions).

Strategy, in order:
  1. Export with a FIXED input length (config: export.fixed_length_seconds,
     default matches training.segment_seconds) via torch.onnx.export with NO
     dynamic_axes. This is the safer, more likely-to-work path since the
     traced graph has concrete shapes throughout, including through the LSTM.
  2. Only after (1) succeeds, attempt a dynamic-length export (dynamic_axes
     on the time dimension). If this fails, the failure is reported
     explicitly (exception message printed, exit non-zero) -- this script
     does NOT silently fall back to fixed-length-only and claim success.
     A fixed-length-only export is a real deployment constraint (the on-Mac
     app would need to chunk/pad audio to a fixed window) and must be
     visible to the user, not hidden.

This script does not run inference or compute metrics -- that is Phase 5a's
second step, done separately against the exported model on the same test
set used for v5's PyTorch evaluation (see project instructions for the
verification script/commands).

Run standalone: `python -m src.export.to_onnx` from repo root. Requires
`onnx` and `onnxruntime` installed (not yet in requirements.txt -- add
explicitly, this script does not install anything itself).
"""

import logging
from pathlib import Path

import torch
import yaml

from src.model.load import load_config
from src.model.inference import load_finetuned_model, verify_finetuned_weights_differ

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "configs/finetune.yaml"


def _fixed_length_samples(cfg: dict) -> int:
    sample_rate = cfg["sample_rate"]
    seconds = cfg["export"]["fixed_length_seconds"]
    return int(round(sample_rate * seconds))


def export_fixed_length(model: torch.nn.Module, cfg: dict, device: torch.device, output_path: Path) -> None:
    """Exports with NO dynamic_axes -- every shape in the traced graph is
    concrete. This is the known-working path for LSTM-containing models."""
    length = _fixed_length_samples(cfg)
    dummy = torch.randn(1, 1, length, device=device)

    logger.info(f"Attempting FIXED-length ONNX export (length={length} samples, {cfg['export']['fixed_length_seconds']}s)...")
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["noisy_waveform"],
        output_names=["enhanced_waveform"],
        opset_version=cfg["export"]["opset_version"],
        dynamic_axes=None,
    )
    logger.info(f"Fixed-length export succeeded -- wrote {output_path}")


def export_dynamic_length(model: torch.nn.Module, cfg: dict, device: torch.device, output_path: Path) -> bool:
    """Attempts dynamic-length export (time axis marked dynamic). Returns
    True on success, False on failure -- caller must report failure
    explicitly, never silently treat False as equivalent to the fixed-length
    result."""
    length = _fixed_length_samples(cfg)
    dummy = torch.randn(1, 1, length, device=device)

    logger.info("Attempting DYNAMIC-length ONNX export (time axis dynamic)...")
    try:
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            input_names=["noisy_waveform"],
            output_names=["enhanced_waveform"],
            opset_version=cfg["export"]["opset_version"],
            dynamic_axes={
                "noisy_waveform": {2: "num_samples"},
                "enhanced_waveform": {2: "num_samples"},
            },
        )
        logger.info(f"Dynamic-length export succeeded -- wrote {output_path}")
        return True
    except Exception as e:
        logger.error(
            "DYNAMIC-length ONNX export FAILED. This is being reported explicitly, "
            "not silently swallowed -- the fixed-length export above (if it succeeded) "
            "remains the only usable artifact. Deployment must chunk/pad audio to the "
            f"fixed length ({length} samples) until this is resolved.\n"
            f"Underlying error: {type(e).__name__}: {e}"
        )
        return False


def run(cfg: dict | None = None) -> Path:
    if cfg is None:
        cfg = load_config()

    export_cfg = cfg["export"]
    checkpoint_path = Path(export_cfg["checkpoint_path"])
    onnx_dir = Path(export_cfg["onnx_dir"])
    onnx_dir.mkdir(parents=True, exist_ok=True)
    fixed_output_path = onnx_dir / export_cfg["onnx_filename"]
    dynamic_output_path = onnx_dir / f"dynamic_{export_cfg['onnx_filename']}"

    model, device, metadata = load_finetuned_model(cfg, checkpoint_path)
    verify_finetuned_weights_differ(model, cfg, device)
    logger.info(f"Loaded checkpoint {checkpoint_path} for export: {metadata}")

    if device.type != "cpu":
        logger.info(f"Moving model to CPU for export (traced from device={device.type}; ONNX export is run on CPU for portability).")
        model = model.to("cpu")
        device = torch.device("cpu")

    export_fixed_length(model, cfg, device, fixed_output_path)
    dynamic_ok = export_dynamic_length(model, cfg, device, dynamic_output_path)

    logger.info("=== Export summary ===")
    logger.info(f"Fixed-length export: {fixed_output_path} (OK)")
    logger.info(f"Dynamic-length export: {'OK -> ' + str(dynamic_output_path) if dynamic_ok else 'FAILED -- see error above, fixed-length is the only usable artifact'}")

    return fixed_output_path


if __name__ == "__main__":
    run()
