"""Backend capability probing and portable FFT dispatch.

Only composed pure-torch operations are used. When native complex FFT
works on a device we use ``torch.fft``; otherwise we fall back to a
cached real-valued DFT expressed as matmuls (cos/sin basis), which runs
on any backend that supports real matmul (CUDA, XLA/TPU, HPU/Gaudi, CPU).
"""
from __future__ import annotations

import math
from functools import lru_cache

import torch

_DFT_CACHE: dict = {}


def _device_key(device) -> str:
    return str(torch.device(device))


@lru_cache(maxsize=None)
def complex_fft_supported(device_str: str) -> bool:
    """Cached probe: True if a tiny rfft succeeds on ``device``."""
    try:
        device = torch.device(device_str)
        x = torch.zeros(8, dtype=torch.float32, device=device)
        y = torch.fft.rfft(x)
        _ = torch.fft.irfft(y, n=8)
        # Force materialisation on lazy backends (XLA/HPU).
        _ = float(torch.abs(y).sum().cpu()) if y.numel() else 0.0
        return True
    except Exception:
        return False


# Native FFT existing is not the same as native FFT being the fast
# path: on Gaudi the hpu-complex plugin provides a correct but dense
# FFT that measured ~50 ms/call at EEG-training shapes where the
# real-valued DFT matmul takes ~0.25 ms. Default to the fallback
# there; users can override per device type.
_FFT_PREFERENCE: dict = {"hpu": "fallback"}
_FFT_MODES = ("auto", "native", "fallback")


def set_fft_preference(device_type: str, mode: str) -> None:
    """Set FFT path preference for a device type.

    ``mode`` is ``"auto"`` (use native FFT when the probe passes),
    ``"native"`` (same, explicitly) or ``"fallback"`` (always use the
    real-valued DFT matmul, even when native complex FFT works).
    """
    if mode not in _FFT_MODES:
        raise ValueError(f"mode must be one of {_FFT_MODES}, got {mode!r}")
    _FFT_PREFERENCE[str(device_type)] = mode


def use_native_fft(device) -> bool:
    """Decision used by every op in this package: native FFT or DFT?"""
    device = torch.device(device)
    if _FFT_PREFERENCE.get(device.type, "auto") == "fallback":
        return False
    return complex_fft_supported(_device_key(device))


def get_backend_name(device) -> str:
    """Human-readable backend name for ``device``."""
    dtype = torch.device(device).type
    names = {
        "cuda": "cuda",
        "xla": "tpu-xla",
        "hpu": "gaudi-hpu",
        "cpu": "cpu",
        "mps": "mps",
        "privateuseone": "privateuseone",
    }
    base = names.get(dtype, dtype)
    mode = "native-fft" if use_native_fft(device) else "dft-matmul"
    return f"{base}-{mode}"


# --- Cached real DFT bases -------------------------------------------------

def _dft_matrices(n: int, device, dtype):
    """Full DFT cos/sin matrices of shape (n, n), cached per (n, device, dtype).

    Phase indices are reduced modulo ``n`` in int64 before scaling:
    forming ``k * nn`` in float32 loses the fractional phase at long
    lengths (measured 5.2e-4 max rel. err at n=6000, vs ~3e-6 with
    the integer reduction, matching Braindecode's real-DFT basis).
    """
    key = (n, _device_key(device), dtype)
    if key not in _DFT_CACHE:
        k = torch.arange(n, dtype=torch.int64, device=device).view(n, 1)
        nn = torch.arange(n, dtype=torch.int64, device=device).view(1, n)
        ang = 2.0 * math.pi * ((k * nn) % n).to(dtype) / float(n)
        _DFT_CACHE[key] = (torch.cos(ang), torch.sin(ang))
    return _DFT_CACHE[key]


def _real_dtype(dtype):
    return torch.float32 if dtype == torch.complex64 else torch.float64 if dtype == torch.complex128 else dtype


def _move_last(x, dim):
    dim = dim % x.ndim
    return torch.movedim(x, dim, -1), dim


# --- Pair (real, imag) FFT helpers -----------------------------------------

def fft_pair(x: torch.Tensor, dim: int = -1):
    """FFT of a real tensor, returned as (real, imag) real tensors."""
    if use_native_fft(x.device):
        f = torch.fft.fft(x, dim=dim)
        return f.real, f.imag
    xm, dim0 = _move_last(x, dim)
    n = xm.shape[-1]
    cos_m, sin_m = _dft_matrices(n, xm.device, xm.dtype)
    re = torch.matmul(xm, cos_m.t())
    im = -torch.matmul(xm, sin_m.t())
    return torch.movedim(re, -1, dim0), torch.movedim(im, -1, dim0)


def ifft_pair(re: torch.Tensor, im: torch.Tensor, dim: int = -1):
    """Inverse FFT from (real, imag) parts, returned as (real, imag)."""
    if use_native_fft(re.device):
        f = torch.complex(re, im)
        out = torch.fft.ifft(f, dim=dim)
        return out.real, out.imag
    rem, dim0 = _move_last(re, dim)
    imm, _ = _move_last(im, dim)
    n = rem.shape[-1]
    cos_m, sin_m = _dft_matrices(n, rem.device, rem.dtype)
    out_re = (torch.matmul(rem, cos_m) - torch.matmul(imm, sin_m)) / float(n)
    out_im = (torch.matmul(rem, sin_m) + torch.matmul(imm, cos_m)) / float(n)
    return torch.movedim(out_re, -1, dim0), torch.movedim(out_im, -1, dim0)


def fft(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Portable full FFT of a real tensor -> complex tensor."""
    re, im = fft_pair(x, dim=dim)
    return torch.complex(re, im)


def ifft(f: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Portable inverse FFT of a complex tensor -> complex tensor."""
    re, im = ifft_pair(f.real, f.imag, dim=dim)
    return torch.complex(re, im)


def rfft_pair(x: torch.Tensor, dim: int = -1):
    """One-sided FFT of a real tensor as (real, imag) real tensors.

    Never builds a complex tensor on the fallback path: on Gaudi
    without a complex plugin ``torch.complex`` silently returns a CPU
    tensor, so magnitude/power consumers must stay in pair form.
    """
    if use_native_fft(x.device):
        f = torch.fft.rfft(x, dim=dim)
        return f.real, f.imag
    xm, dim0 = _move_last(x, dim)
    n = xm.shape[-1]
    cos_m, sin_m = _dft_matrices(n, xm.device, xm.dtype)
    m = n // 2 + 1
    re = torch.matmul(xm, cos_m[:m].t())
    im = -torch.matmul(xm, sin_m[:m].t())
    return torch.movedim(re, -1, dim0), torch.movedim(im, -1, dim0)


def rfft(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Portable real FFT -> complex tensor with n//2+1 bins."""
    re, im = rfft_pair(x, dim=dim)
    return torch.complex(re, im)
