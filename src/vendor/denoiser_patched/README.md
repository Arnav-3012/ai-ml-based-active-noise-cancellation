# denoiser_patched

Vendored, minimally patched copies of two modules from `facebookresearch/denoiser`
(PyPI package `denoiser==0.1.5`), forked to fix breakage against
`torch==2.14.0` + `torchaudio==2.11.0` (the environment pinned in
`requirements.txt` — see `logs.md` for the version-mismatch investigation
this came out of).

Forked from: `denoiser` 0.1.5 as published on PyPI (installed at
`.venv/lib/python3.13/site-packages/denoiser/`).

Everything else in `denoiser` (model definition, pretrained checkpoint
loading, resample, etc.) is used as-is from the pip package. Only these two
files are patched, because only these two files broke.

## Why patched instead of downgrading torch

torch 2.14.0 has no matching torchaudio release (torchaudio's latest on
PyPI is 2.11.0 — its release cadence has fallen behind torch's). Downgrading
torch to find an older matched pair was considered and rejected in favor of
staying on current torch/torchaudio and patching the two broken denoiser
call sites, to keep MPS backend improvements on the M4 Pro target.

## Patch 1 — `audio.py`: file I/O switched from `torchaudio` to `soundfile`

Two rounds of changes here, both about audio I/O:

**Round 1:** `torchaudio.get_audio_backend()` was removed from current
torchaudio. The original code branched on its return value to decide which
`torchaudio.load()` keyword signature to use.

**Round 2 (superseded round 1 — final state):** torchaudio 2.11's
`load()`/`save()` route through a `torchcodec` I/O backend, which requires a
native `libtorchcodec` build linked against a specific FFmpeg version.
Installing `torchcodec` on this machine failed to load its dylib
(`libavutil.60.dylib` not found on the linker search path, even with
Homebrew FFmpeg 8.0 installed — the loader was also picking up stale
`/opt/anaconda3` rpath assumptions). Rather than fight native linking that
could easily break again in a different environment (e.g. Colab/Kaggle, if
training ever moves off this M4 Pro), all file I/O was moved to `soundfile`
(already an installed dependency, pure C extension via libsndfile, no
FFmpeg/codec dependency for WAV).

`torchaudio` itself is still installed and still used for algorithmic ops
elsewhere (denoiser's `convert_audio`/resampling does not touch its I/O
backend), just not for reading/writing files.

Behavioral differences handled in the swap:
- `soundfile.read(..., dtype='float32', always_2d=True)` returns a numpy
  array shaped `[samples, channels]`; torchaudio's convention is
  `[channels, samples]`. Converted via `torch.from_numpy(data).transpose(0, 1)`.
- `soundfile.read` defaults to `float64`; forced `dtype='float32'` explicitly
  to match the float32 tensors the rest of the pipeline expects.
- `soundfile.info()` exposes `.frames`/`.samplerate`/`.channels` (vs.
  torchaudio's `.num_frames`/`.sample_rate`/`.num_channels`) — `get_info()`
  updated accordingly.

Before (original pip package):
```python
def get_info(path):
    info = torchaudio.info(path)
    if hasattr(info, 'num_frames'):
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels)
...
if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
    out, sr = torchaudio.load(str(file),
                              frame_offset=offset,
                              num_frames=num_frames or -1)
else:
    out, sr = torchaudio.load(str(file), offset=offset, num_frames=num_frames)
```

After (final, soundfile-based):
```python
def get_info(path):
    info = sf.info(str(path))
    return Info(info.frames, info.samplerate, info.channels)
...
data, sr = sf.read(str(file), start=offset,
                   frames=(num_frames or -1) if num_frames else -1,
                   dtype='float32', always_2d=True)
out = torch.from_numpy(data).transpose(0, 1).contiguous()
```

## Patch 2 — `stft_loss.py`: `stft()`

`torch.stft()` now raises `RuntimeError: stft requires the return_complex
parameter be given for real inputs` when called without `return_complex`.
The original code relied on the old behavior of returning a real-valued
tensor with a trailing size-2 (real, imag) dimension, indexed via
`x_stft[..., 0]` / `x_stft[..., 1]`. With `return_complex=True`, `torch.stft`
returns a genuine complex-dtype tensor instead, so real/imag are now read
via `.real` / `.imag`.

Before:
```python
x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
real = x_stft[..., 0]
imag = x_stft[..., 1]
```

After:
```python
x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
real = x_stft.real
imag = x_stft.imag
```

Numerically equivalent — magnitude computed from real/imag components is
unchanged, only how those components are extracted from `torch.stft`'s
output changed.

## Usage

Import the STFT loss and the `Audioset` dataset class from here instead of
from `denoiser` directly:

```python
from src.vendor.denoiser_patched.stft_loss import MultiResolutionSTFTLoss
from src.vendor.denoiser_patched.audio import Audioset
```

Everything else (model architecture, pretrained checkpoint loading, DSP
utilities) should still be imported from the pip-installed `denoiser`
package as normal.
