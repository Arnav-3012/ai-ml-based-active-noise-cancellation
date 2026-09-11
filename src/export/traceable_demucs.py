"""Export-only wrapper around dns48's Demucs model, working around a real
torch.jit.trace / coremltools MIL conversion failure.

BACKGROUND: `Demucs.forward()` (denoiser/demucs.py) calls
`self.valid_length(length)` where `length = mix.shape[-1]`, then
`F.pad(x, (0, self.valid_length(length) - length))`. Under eager execution
`length` is a plain Python int, but under `torch.jit.trace`, `mix.shape[-1]`
can be captured as a traced 0-d tensor rather than a Python int (confirmed
by a real TracerWarning at demucs.py:146, `math.ceil(length * self.resample)`,
and a real downstream coremltools MIL crash: `ops.py::_int`/`_cast` raising
`TypeError: only 0-dimensional arrays can be converted to Python scalars`
when the converter tries to cast that traced value to a plain int).
`downsample2`'s `x.shape[-1] % 2 != 0` check (denoiser/resample.py:67) has
the same class of problem.

WHY THIS IS SAFE TO HARDCODE (verified by direct computation, not assumed):
for a FIXED input length, `valid_length(length)` and the parity checks
inside `downsample2` are pure functions of that one integer and the model's
static hyperparameters (resample=4, depth=5, kernel_size=8, stride=4) --
they do not depend on audio content, only on the input length, which this
project's export path has already fixed at 64000 samples (4.0s @ 16kHz,
config: export.fixed_length_seconds). Computed once via direct arithmetic
for length=64000:
  - valid_length(64000) == 64085
  - F.pad padding amount == 85 (i.e. F.pad(x, (0, 85)))
  - Both downsample2 calls in the resample=4 path see EVEN-length input
    (256340, then 128170) -- the odd-length pad-by-one branch is never
    taken for this specific length/architecture combination.

THIS WRAPPER IS VALID ONLY FOR length=64000 (4.0s @ 16kHz, resample=4,
depth=5, kernel_size=8, stride=4 -- dns48's exact architecture). Using it
with any OTHER input length will silently apply the WRONG padding amount
and produce incorrect output -- there is no runtime check against this,
by design (a Python-level branch would reintroduce the same tracing
problem this wrapper exists to avoid). Do not reuse this module outside
the fixed-length export path.

Does NOT modify denoiser/demucs.py or denoiser/resample.py. Wraps the
existing trained model instance as-is (same weights, same layers, same
encoder/decoder/lstm modules) -- only the length-dependent glue arithmetic
around F.pad and the resample calls is replaced with pre-computed
constants for tracing purposes. Training, PyTorch-side inference, and
ONNX export are entirely unaffected (they still use the original,
unmodified forward path).
"""

import torch as th
from torch.nn import functional as F

from denoiser.resample import kernel_downsample2, kernel_upsample2

# Valid ONLY for length=64000 (4.0s @ 16kHz) with dns48's architecture
# (resample=4, depth=5, kernel_size=8, stride=4) -- see module docstring.
FIXED_INPUT_LENGTH = 64000
FIXED_VALID_LENGTH = 64085  # Demucs.valid_length(64000), precomputed
FIXED_PAD_AMOUNT = FIXED_VALID_LENGTH - FIXED_INPUT_LENGTH  # 85

# Precomputed time-dimension lengths through the resample=4 pipeline, for
# length=64000 (-> 64085 after F.pad) -- see module docstring for the full
# derivation. Each is a plain Python int, used in place of a traced
# `x.shape[-1]` read inside _upsample2_traceable/_downsample2_traceable.
_UPSAMPLE1_IN_TIME = FIXED_VALID_LENGTH        # 64085 (input to 1st upsample2)
_UPSAMPLE2_IN_TIME = _UPSAMPLE1_IN_TIME * 2    # 128170 (input to 2nd upsample2)
_ENCODER_INPUT_LENGTH = _UPSAMPLE2_IN_TIME * 2  # 256340 (input to encoder)

_DOWNSAMPLE1_ODD_TIME = _ENCODER_INPUT_LENGTH // 2   # 128170 (xodd len, 1st downsample2)
_DOWNSAMPLE2_ODD_TIME = (_ENCODER_INPUT_LENGTH // 2) // 2  # 64085 (xodd len, 2nd downsample2)

# Precomputed encoder/decoder time lengths (depth=5, kernel_size=8, stride=4),
# starting from _ENCODER_INPUT_LENGTH (256340). Used in place of a traced
# `x.shape[-1]` read at each decoder skip-connection slice
# (`skip[..., :x.shape[-1]]` in the original Demucs.forward()).
_ENCODER_LENGTHS = [_ENCODER_INPUT_LENGTH]
for _ in range(5):
    _ENCODER_LENGTHS.append((_ENCODER_LENGTHS[-1] - 8) // 4 + 1)
# _ENCODER_LENGTHS = [256340, 64084, 16020, 4004, 1000, 249]

_DECODER_LENGTHS = [_ENCODER_LENGTHS[-1]]
for _ in range(5):
    _DECODER_LENGTHS.append((_DECODER_LENGTHS[-1] - 1) * 4 + 8)
# _DECODER_LENGTHS = [249, 1000, 4004, 16020, 64084, 256340]
# skip length needed at decoder step i is _DECODER_LENGTHS[i] (the x length
# BEFORE that decode() call, i.e. before it grows to _DECODER_LENGTHS[i+1]).


def _upsample2_traceable(x, kernel, time: int):
    """Same math as denoiser.resample.upsample2, but with the kernel
    precomputed/passed in AND the `time` dimension passed in as a plain
    Python int instead of read off `x.shape` inside this function.

    WHY: the original `*other, time = x.shape` line (copied verbatim from
    denoiser.resample.upsample2) unpacks a traced tensor's shape into a
    Python value used in `.reshape(...)` -- under torch.jit.trace this
    itself becomes a traced `size`/`numtotensor`/`int` op sequence (same
    class of bug as the `valid_length`/F.pad issue this module already
    works around; confirmed by a second real coremltools MIL crash on the
    exact same 'int'/'_cast' failure, traced to this line). Since the
    input shape is fully fixed for this export (see module docstring),
    `time` is precomputed by the caller and passed in as a constant,
    exactly like FIXED_PAD_AMOUNT."""
    out = F.conv1d(x.reshape(-1, 1, time), kernel, padding=56)[..., 1:].reshape(1, 1, time)
    y = th.stack([x, out], dim=-1)
    return y.reshape(1, 1, -1)


def _downsample2_traceable(x, kernel, input_is_even: bool, odd_time: int):
    """Same math as denoiser.resample.downsample2, but the odd/even check
    is a precomputed Python bool (input_is_even) and the xodd time
    dimension (odd_time) is a precomputed Python int, instead of both
    being read off a traced tensor's shape (`x.shape[-1] % 2 != 0` and
    `*other, time = xodd.shape`) -- same tracing hazard and same fix
    rationale as _upsample2_traceable above. Verified for this export's
    fixed shape: both downsample2 call sites see even-length input
    (256340, then 128170), so `input_is_even=True` is correct for both --
    see module docstring for the computation."""
    if not input_is_even:
        x = F.pad(x, (0, 1))
    xeven = x[..., ::2]
    xodd = x[..., 1::2]
    out = xeven + F.conv1d(xodd.reshape(-1, 1, odd_time), kernel, padding=56)[..., :-1].reshape(
        1, 1, odd_time)
    return out.reshape(1, 1, -1).mul(0.5)


class TraceableDemucs(th.nn.Module):
    """Wraps a trained Demucs instance for tracing/Core ML export at the
    fixed input length FIXED_INPUT_LENGTH (64000 samples, 4.0s @ 16kHz)
    ONLY. See module docstring for why this is safe and why it must not
    be reused at any other length."""

    def __init__(self, demucs: th.nn.Module):
        super().__init__()
        if demucs.resample != 4:
            raise ValueError(f"TraceableDemucs assumes resample=4, got {demucs.resample}")
        if demucs.depth != 5 or demucs.kernel_size != 8 or demucs.stride != 4:
            raise ValueError(
                "TraceableDemucs's precomputed constants assume depth=5, kernel_size=8, "
                f"stride=4 (dns48's architecture) -- got depth={demucs.depth}, "
                f"kernel_size={demucs.kernel_size}, stride={demucs.stride}."
            )
        self.demucs = demucs
        self.register_buffer("_up_kernel", kernel_upsample2())
        self.register_buffer("_down_kernel", kernel_downsample2())

    def forward(self, mix):
        # NOTE: no runtime check of mix.shape[-1] here, deliberately -- a Python-level
        # comparison against a traced tensor's shape (`mix.shape[-1] != FIXED_INPUT_LENGTH`)
        # itself gets traced as a graph op (confirmed by a real run: TracerWarning +
        # downstream coremltools MIL crash on the exact same 'int'/'_cast' failure this
        # wrapper exists to avoid). The length is validated ONCE, before tracing, by
        # to_coreml.py (which checks it against FIXED_INPUT_LENGTH from config). This
        # wrapper's docstring states the single-length constraint; there is no in-graph
        # enforcement of it, by necessity.
        demucs = self.demucs
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        if demucs.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (demucs.floor + std)
        else:
            std = 1

        x = mix
        # Precomputed pad amount (85) replaces demucs.valid_length(length) - length.
        x = F.pad(x, (0, FIXED_PAD_AMOUNT))

        # resample == 4: two upsample2 calls, each fed its precomputed input `time` length.
        x = _upsample2_traceable(x, self._up_kernel, time=_UPSAMPLE1_IN_TIME)
        x = _upsample2_traceable(x, self._up_kernel, time=_UPSAMPLE2_IN_TIME)

        skips = []
        for encode in demucs.encoder:
            x = encode(x)
            skips.append(x)
        x = x.permute(2, 0, 1)
        x, _ = demucs.lstm(x)
        x = x.permute(1, 2, 0)
        for decoder_idx, decode in enumerate(demucs.decoder):
            skip = skips.pop(-1)
            # Precomputed length (was `x.shape[-1]`) -- see _DECODER_LENGTHS derivation above.
            x = x + skip[..., :_DECODER_LENGTHS[decoder_idx]]
            x = decode(x)

        # Precomputed parity + odd_time (both calls see even-length input for this fixed shape).
        x = _downsample2_traceable(x, self._down_kernel, input_is_even=True, odd_time=_DOWNSAMPLE1_ODD_TIME)
        x = _downsample2_traceable(x, self._down_kernel, input_is_even=True, odd_time=_DOWNSAMPLE2_ODD_TIME)

        x = x[..., :FIXED_INPUT_LENGTH]
        return std * x
