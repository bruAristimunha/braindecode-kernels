"""Portable spectral operations built from composed torch ops."""
from __future__ import annotations

import torch

from . import backend


def rfft_abs(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Magnitude of the real FFT along ``dim``."""
    f = backend.rfft(x, dim=dim)
    return torch.abs(f)


def analytic_signal(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Analytic signal (Hilbert transform) along ``dim``.

    Mirrors braindecode's ``_analytic_transform`` / scipy's ``hilbert``:
    FFT, zero negative frequencies, double positive frequencies
    (DC and Nyquist kept at unit gain for even lengths), inverse FFT.
    Uses the real-DFT matmul fallback when native FFT is unavailable.
    """
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
    return torch.complex(out_re, out_im)


def plv(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Phase-locking value.

    Computes the analytic signal along the last axis, converts each
    sample to a unit phasor, averages phasors along ``dim`` and returns
    the magnitude (in [0, 1]).
    """
    analytic = analytic_signal(x, dim=-1)
    mag = torch.abs(analytic)
    mag = torch.where(mag == 0, torch.ones_like(mag), mag)
    unit_re = analytic.real / mag
    unit_im = analytic.imag / mag
    mean_re = unit_re.mean(dim=dim)
    mean_im = unit_im.mean(dim=dim)
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
    spec = backend.rfft(frames, dim=-1)
    power = spec.real * spec.real + spec.imag * spec.imag
    return torch.log10(power + 1e-10)
