"""Phase 5b follow-up: INT8 weights-only post-training quantization of the
ALREADY-VERIFIED fp32 Core ML model, as its own isolated measurement --
same discipline as the fp16 compute-precision step (to_coreml.py --fp16):
one variable changed, everything else held fixed.

Starts from the trusted, already-converted fp32 .mlpackage
(coreml.mlpackage_filename) -- does NOT requantize from ONNX and does NOT
re-trace TraceableDemucs. Uses coremltools.optimize.coreml.linear_quantize_weights,
which quantizes stored WEIGHTS only (per-channel linear int8, dequantized
back to float at runtime before each op). This is distinct from activation
quantization: coremltools also exposes linear_quantize_activations (calibration-
data-driven, quantizes intermediate tensors too) -- that function is NOT
called here, so this artifact's activations remain float at inference time.
Report this distinction explicitly when reporting results; do not conflate
"quantized" with "weights AND activations quantized."

Writes a SEPARATE .mlpackage (config: coreml.mlpackage_filename_int8) --
the fp32 and fp16 artifacts are untouched.

Run standalone: `python -m src.export.quantize_coreml` from repo root.
Requires coremltools and the fp32 .mlpackage already converted
(`python -m src.export.to_coreml` first).
"""

import logging
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto

from src.model.load import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run(cfg: dict | None = None) -> Path:
    if cfg is None:
        cfg = load_config()

    coreml_cfg = cfg["coreml"]
    mlpackage_dir = Path(coreml_cfg["mlpackage_dir"])
    fp32_path = mlpackage_dir / coreml_cfg["mlpackage_filename"]
    int8_path = mlpackage_dir / coreml_cfg["mlpackage_filename_int8"]

    if not fp32_path.exists():
        raise FileNotFoundError(f"{fp32_path} not found -- run `python -m src.export.to_coreml` first.")

    logger.info(f"Loading verified fp32 Core ML model {fp32_path}...")
    fp32_model = ct.models.MLModel(str(fp32_path))

    dtype = coreml_cfg["int8_quantize_dtype"]
    logger.info(f"Applying weights-only linear quantization (dtype={dtype})...")
    quant_config = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype=dtype)
    )
    int8_model = cto.linear_quantize_weights(fp32_model, config=quant_config)

    mlpackage_dir.mkdir(parents=True, exist_ok=True)
    int8_model.save(str(int8_path))
    logger.info(f"INT8 weights-only quantization complete -- wrote {int8_path}")

    fp32_size = _mlpackage_weight_size(fp32_path)
    int8_size = _mlpackage_weight_size(int8_path)
    logger.info(
        f"weight.bin size: fp32={fp32_size / 1e6:.1f}MB, int8={int8_size / 1e6:.1f}MB "
        f"(ratio={fp32_size / int8_size:.2f}x)"
    )

    return int8_path


def _mlpackage_weight_size(mlpackage_path: Path) -> int:
    weight_file = mlpackage_path / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"
    if not weight_file.exists():
        raise FileNotFoundError(f"expected weight file not found at {weight_file}")
    return weight_file.stat().st_size


if __name__ == "__main__":
    run()
