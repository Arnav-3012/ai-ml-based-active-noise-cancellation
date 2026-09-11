"""Phase 5b: converts the v5 fine-tuned dns48 checkpoint to Core ML
(.mlpackage) for the iOS target.

REVISED PATH (was ONNX -> Core ML, changed after a real run failed): the
installed coremltools version no longer has a built-in ONNX converter --
`ct.convert()` only auto-detects "pytorch"/"tensorflow"/"milinternal"
sources, not .onnx files (confirmed via a real `ValueError` on the actual
run, not assumed). coremltools' actively-maintained, best-supported path
is now direct-from-PyTorch: torch.jit.trace the model, then
`ct.convert(traced, ...)`. This bypasses ONNX entirely for the Core ML
leg -- Phase 5a's ONNX export/verification stays valid and unchanged for
its own purposes, it's just no longer this script's input.

SECOND REAL ISSUE, ALSO RESOLVED: tracing the raw Demucs model directly
also failed inside coremltools' MIL frontend (`ops.py::_int`/`_cast`,
`TypeError: only 0-dimensional arrays can be converted to Python scalars`).
Root cause: Demucs.forward() computes `self.valid_length(length)` and a
resample-internal odd/even shape check as Python-level arithmetic on
`mix.shape[-1]`, which torch.jit.trace can capture as traced tensor ops
instead of Python ints/bools (this is what the earlier TracerWarnings at
demucs.py:146 and resample.py:67 were flagging). This is now worked around
via `src/export/traceable_demucs.py::TraceableDemucs`, a wrapper that
reuses the same trained model instance/weights but replaces that
length-dependent glue arithmetic with precomputed constants -- valid ONLY
for this export's fixed 64000-sample (4.0s @ 16kHz) input length. See that
module's docstring for the full derivation/justification. denoiser/demucs.py
and denoiser/resample.py are NOT modified -- training/eval/ONNX-export
continue to use the original, unmodified forward path.

Uses the SAME fixed input length as Phase 5a's ONNX export
(config: export.fixed_length_seconds) so the traced shape matches an
input the model has actually been evaluated on, and so Phase 5b's
verification script can crop test audio identically for an apples-to-
apples comparison against the ONNX baseline numbers.

This is the fp32/full-precision conversion step only -- no quantization.
Quantization is a separate, later step so its size-reduction claim can be
isolated from conversion-format drift, same separation-of-concerns
discipline as Phase 5a's crop-vs-export isolation.

FP16 COMPUTE-PRECISION ISOLATION (Phase 5b follow-up): `run(precision="FLOAT16")`
produces a SEPARATE .mlpackage (config: coreml.mlpackage_filename_fp16),
identical in every other respect to the fp32 conversion (same
TraceableDemucs wrapper, same fixed 4.0s input, same convert_to/target) --
the ONLY change is compute_precision. This isolates pure fp16 precision
cost from conversion-format drift (already verified zero in the fp32 run)
and from quantization (a separate, later step, not this one). Both
.mlpackage files are kept on disk for comparison/potential shipping either
way -- this call does not overwrite the fp32 artifact.

Run standalone: `python -m src.export.to_coreml` from repo root (fp32).
For the fp16 isolation artifact: `python -m src.export.to_coreml --fp16`.
Requires `coremltools` installed (not yet in requirements.txt) and the v5
PyTorch checkpoint (checkpoints/dns48_finetuned_v5_best.pt, already
required elsewhere in this project).
"""

import argparse
import logging
from pathlib import Path

import coremltools as ct
import torch

from src.model.load import load_config
from src.model.inference import load_finetuned_model, verify_finetuned_weights_differ
from src.export.traceable_demucs import TraceableDemucs, FIXED_INPUT_LENGTH

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _fixed_length_samples(cfg: dict) -> int:
    sample_rate = cfg["sample_rate"]
    seconds = cfg["export"]["fixed_length_seconds"]
    return int(round(sample_rate * seconds))


def run(cfg: dict | None = None, precision: str = "FLOAT32") -> Path:
    if cfg is None:
        cfg = load_config()

    coreml_cfg = cfg["coreml"]
    checkpoint_path = Path(cfg["export"]["checkpoint_path"])

    model, device, metadata = load_finetuned_model(cfg, checkpoint_path)
    verify_finetuned_weights_differ(model, cfg, device)
    logger.info(f"Loaded checkpoint {checkpoint_path} for Core ML conversion: {metadata}")

    if device.type != "cpu":
        logger.info(f"Moving model to CPU for tracing (traced from device={device.type}; coremltools conversion is run on CPU for portability).")
        model = model.to("cpu")
        device = torch.device("cpu")
    model.eval()

    length = _fixed_length_samples(cfg)
    if length != FIXED_INPUT_LENGTH:
        raise ValueError(
            f"export.fixed_length_seconds resolves to {length} samples, but "
            f"TraceableDemucs's precomputed constants are only valid for "
            f"{FIXED_INPUT_LENGTH} samples -- see traceable_demucs.py's docstring. "
            "Recompute and update that module before changing this length."
        )

    traceable_model = TraceableDemucs(model).to(device)
    traceable_model.eval()
    dummy = torch.randn(1, 1, length, device=device)

    logger.info(f"Tracing TraceableDemucs wrapper (fixed length={length} samples, {cfg['export']['fixed_length_seconds']}s)...")
    with torch.no_grad():
        traced = torch.jit.trace(traceable_model, dummy)

    precision = precision.upper()
    if precision not in ("FLOAT32", "FLOAT16"):
        raise ValueError(f"precision must be 'FLOAT32' or 'FLOAT16', got {precision!r}")
    filename_key = "mlpackage_filename" if precision == "FLOAT32" else "mlpackage_filename_fp16"

    output_dir = Path(coreml_cfg["mlpackage_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / coreml_cfg[filename_key]

    convert_to = coreml_cfg["convert_to"]
    min_target = getattr(ct.target, coreml_cfg["minimum_deployment_target"])
    compute_precision = getattr(ct.precision, precision)

    logger.info(f"Converting traced model -> Core ML ({convert_to}, min target {coreml_cfg['minimum_deployment_target']}, compute_precision={precision})...")
    mlmodel = ct.convert(
        traced,
        convert_to=convert_to,
        inputs=[ct.TensorType(name="noisy_waveform", shape=dummy.shape)],
        outputs=[ct.TensorType(name="enhanced_waveform")],
        minimum_deployment_target=min_target,
        # mlprogram defaults to FLOAT16 compute precision (confirmed via a real
        # conversion log showing fp16 casts throughout) -- explicit FLOAT32 is
        # this function's default so ordinary calls stay isolated from fp16
        # precision loss. precision="FLOAT16" is a deliberate, separate
        # measurement path (see module docstring), not this function's default.
        compute_precision=compute_precision,
    )

    mlmodel.save(str(output_path))
    logger.info(f"Core ML conversion succeeded ({precision}) -- wrote {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", action="store_true", help="Convert with compute_precision=FLOAT16 instead of the default FLOAT32 (writes a separate .mlpackage, does not overwrite the fp32 one).")
    args = parser.parse_args()
    run(precision="FLOAT16" if args.fp16 else "FLOAT32")
