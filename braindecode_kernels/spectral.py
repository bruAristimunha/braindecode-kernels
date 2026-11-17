"""Portable spectral operations built from composed torch ops."""
from __future__ import annotations

import torch

from . import backend


def rfft_abs(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Magnitude of the real FFT along ``dim``.

    Computed as ``sqrt(re^2 + im^2)`` from the pair form, never via a
    complex tensor: on devices without native complex support (Gaudi
    without a plugin) building the complex tensor would silently
    round-trip through the CPU.
    """
    re, im = backend.rfft_pair(x, dim=dim)
    return torch.sqrt(re * re + im * im)


def analytic_signal(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Analytic signal (Hilbert transform) along ``dim``.

    Mirrors braindecode's ``_analytic_transform`` / scipy's ``hilbert``:
    FFT, zero negative frequencies, double positive frequencies
    (DC and Nyquist kept at unit gain for even lengths), inverse FFT.
    Uses the real-DFT matmul fallback when native FFT is unavailable.
    """
    re, im = _analytic_pair(x, dim=dim)
    return torch.complex(re, im)


def _analytic_pair(x: torch.Tensor, dim: int = -1):
    """Analytic signal as (real, imag) tensors, no complex built."""
    if torch.is_complex(x):
        raise ValueError("x must be real.")
    n = x.shape[dim]
    re, im = backend.fft_pair(x, dim=dim)
    # Build the Hilbert multiplier along `dim`.
    h_shape = [1] * x.ndim
    h_shape[dim] = n
    h = torch.zeros(h_shape, dtype=re.dtype, device=re.device)
    idx = [slice(None)] * x.ndim

    def _set(pos, val):
        idx[dim] = pos
        h[tuple(idx)] = val

    if n % 2 == 0:
        _set(0, 1.0)
        idx[dim] = n // 2
        h[tuple(idx)] = 1.0
        idx[dim] = slice(1, n // 2)
        h[tuple(idx)] = 2.0
    else:
        _set(0, 1.0)
        idx[dim] = slice(1, (n + 1) // 2)
        h[tuple(idx)] = 2.0
    out_re, out_im = backend.ifft_pair(re * h, im * h, dim=dim)
    return out_re, out_im


def plv(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Phase-locking value.

    Computes the analytic signal along the last axis, converts each
    sample to a unit phasor, averages phasors along ``dim`` and returns
    the magnitude (in [0, 1]). Pair form throughout, so it stays
    on-device where complex tensors are unavailable.
    """
    re, im = _analytic_pair(x, dim=-1)
    mag = torch.sqrt(re * re + im * im)
    mag = torch.where(mag == 0, torch.ones_like(mag), mag)
    mean_re = (re / mag).mean(dim=dim)
    mean_im = (im / mag).mean(dim=dim)
    return torch.sqrt(mean_re * mean_re + mean_im * mean_im)


def stft_log_power(
    x: torch.Tensor,
    n_fft: int,
    hop: int,
    window: torch.Tensor | None = None,
) -> torch.Tensor:
    """Log-power spectrogram along the last axis.

    Frames the signal (no centering/padding, i.e. torchaudio
    ``Spectrogram(center=False, power=2)`` framing), applies ``window``
    (Hann by default), takes the portable rFFT, and returns
    ``log10(|X|^2 + 1e-10)`` with shape ``(..., n_frames, n_fft//2+1)``.
    """
    if window is None:
        window = torch.hann_window(n_fft, dtype=x.dtype, device=x.device)
    else:
        window = window.to(dtype=x.dtype, device=x.device)
    n = x.shape[-1]
    if n < n_fft:
        x = torch.nn.functional.pad(x, (0, n_fft - n))
        n = n_fft
    # Framing via unfold on the last axis -> (..., n_frames, n_fft).
    frames = x.unfold(-1, n_fft, hop)
    frames = frames * window
    re, im = backend.rfft_pair(frames, dim=-1)
    power = re * re + im * im
    return torch.log10(power + 1e-10)


def spectrogram(x, n_fft, hop_length=None, window=None):
    """Complex STFT spectrogram with the window on ``x``'s device.

    torchaudio's transform keeps its window on CPU on some stacks,
    which fails on accelerator tensors; this is the portable spelling
    of the same computation via ``torch.stft``.
    """
    if window is None:
        window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    return torch.stft(
        x, n_fft, hop_length=hop_length, window=window, return_complex=True)
