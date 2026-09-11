"""Reverb + clipping augmentation, applied post-mixing per the locked spec.

Reverb is synthesized as a random exponentially-decaying impulse response
(no external SoX/pyroomacoustics dependency) and applied via convolution.
Clipping hard-limits waveform peaks at a random percentile of the signal's
own amplitude distribution. All probabilities/ranges come from
configs/finetune.yaml — no magic numbers here.
"""

import random

import torch
import torch.nn.functional as F
import yaml

CONFIG_PATH = "configs/finetune.yaml"


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _synth_impulse_response(
    num_samples: int, room_scale: float, reverberance: float, sample_rate: int
) -> torch.Tensor:
    """Exponentially decaying noise IR — a lightweight stand-in for a room IR.

    `room_scale` controls IR length (bigger room -> longer tail).
    `reverberance` (0-100-ish) controls decay rate (higher -> slower decay).
    """
    ir_len = max(1, int(room_scale * sample_rate))
    decay = torch.exp(-torch.arange(ir_len, dtype=torch.float32) / (reverberance * sample_rate / 100 + 1))
    noise = torch.randn(ir_len)
    ir = noise * decay
    ir = ir / (ir.abs().max() + 1e-8)
    ir[0] = 1.0  # preserve direct path
    return ir.unsqueeze(0).unsqueeze(0)  # [1, 1, ir_len] for conv1d


def apply_reverb(waveform: torch.Tensor, cfg: dict, rng: random.Random, sample_rate: int) -> torch.Tensor:
    """Convolve `waveform` [channels, samples] with a random synthetic IR."""
    reverb_cfg = cfg["augment"]["reverb"]
    room_scale = rng.uniform(reverb_cfg["room_scale_min"], reverb_cfg["room_scale_max"])
    reverberance = rng.uniform(reverb_cfg["reverberance_min"], reverb_cfg["reverberance_max"])

    ir = _synth_impulse_response(waveform.shape[-1], room_scale, reverberance, sample_rate)
    ir_len = ir.shape[-1]

    x = waveform.unsqueeze(1)  # [channels, 1, samples]
    x_padded = F.pad(x, (ir_len - 1, 0))
    wet = F.conv1d(x_padded, ir)
    wet = wet.squeeze(1)

    peak = waveform.abs().max()
    wet_peak = wet.abs().max()
    if wet_peak > 1e-8:
        wet = wet * (peak / wet_peak)
    return wet


def apply_clipping(waveform: torch.Tensor, cfg: dict, rng: random.Random) -> torch.Tensor:
    """Hard-clip waveform peaks at a random percentile of its own amplitude."""
    clip_cfg = cfg["augment"]["clipping"]
    percentile = rng.uniform(clip_cfg["clip_percentile_min"], clip_cfg["clip_percentile_max"])

    abs_vals = waveform.abs().flatten()
    threshold = torch.quantile(abs_vals, percentile)
    return torch.clamp(waveform, min=-threshold, max=threshold)


def augment(waveform: torch.Tensor, cfg: dict, rng: random.Random, sample_rate: int) -> torch.Tensor:
    """Apply reverb and/or clipping to `waveform` per configured probabilities."""
    out = waveform
    if rng.random() < cfg["augment"]["reverb"]["probability"]:
        out = apply_reverb(out, cfg, rng, sample_rate)
    if rng.random() < cfg["augment"]["clipping"]["probability"]:
        out = apply_clipping(out, cfg, rng)
    return out
