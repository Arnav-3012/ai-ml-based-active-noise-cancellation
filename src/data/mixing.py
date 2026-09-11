"""SNR-targeted mixing of clean speech with noise.

Given a clean speech waveform and a noise waveform, scales the noise to hit
a target SNR (sampled uniformly from the range in configs/finetune.yaml) and
sums them. Every mixed pair is tagged with its noise category (gunshot /
stationary / general) so later Tier-1 evaluation can break results down by
noise type without retrofitting labels.
"""

import random

import numpy as np
import torch
import yaml

CONFIG_PATH = "configs/finetune.yaml"


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x ** 2))


def sample_target_snr(cfg: dict, rng: random.Random) -> float:
    """Draw a target SNR (dB) uniformly from the locked spec's range."""
    snr_min = cfg["mixing"]["snr_db_min"]
    snr_max = cfg["mixing"]["snr_db_max"]
    return rng.uniform(snr_min, snr_max)


def fit_noise_to_length(noise: torch.Tensor, num_samples: int, rng: random.Random) -> torch.Tensor:
    """Loop or randomly crop a noise clip to match the clean clip's length."""
    noise_len = noise.shape[-1]
    if noise_len == num_samples:
        return noise
    if noise_len > num_samples:
        start = rng.randint(0, noise_len - num_samples)
        return noise[..., start:start + num_samples]
    reps = int(np.ceil(num_samples / noise_len))
    tiled = noise.repeat(1, reps)
    return tiled[..., :num_samples]


def mix_at_snr(clean: torch.Tensor, noise: torch.Tensor, target_snr_db: float, eps: float) -> torch.Tensor:
    """Scale `noise` so the mix hits `target_snr_db` relative to `clean`, then sum.

    clean, noise: [channels, samples] tensors of equal length.
    """
    clean_rms = _rms(clean)
    noise_rms = _rms(noise)
    noise_rms = torch.clamp(noise_rms, min=eps)

    target_noise_rms = clean_rms / (10 ** (target_snr_db / 20))
    scale = target_noise_rms / noise_rms
    scaled_noise = noise * scale

    return clean + scaled_noise


def mix_pair(
    clean: torch.Tensor,
    noise: torch.Tensor,
    noise_category: str,
    cfg: dict,
    rng: random.Random,
) -> dict:
    """Build one noisy/clean training pair tagged with its noise category.

    Returns a dict with the mixed waveform, the untouched clean waveform,
    the SNR used, and the noise category — everything build_dataset.py needs
    to write both the audio and the manifest entry.
    """
    if noise_category not in cfg["mixing"]["noise_categories"]:
        raise ValueError(
            f"Unknown noise_category '{noise_category}', expected one of "
            f"{cfg['mixing']['noise_categories']}"
        )

    num_samples = clean.shape[-1]
    noise_fit = fit_noise_to_length(noise, num_samples, rng)

    target_snr_db = sample_target_snr(cfg, rng)
    noisy = mix_at_snr(clean, noise_fit, target_snr_db, cfg["mixing"]["eps"])

    return {
        "noisy": noisy,
        "clean": clean,
        "snr_db": target_snr_db,
        "noise_category": noise_category,
    }
