"""Portable augmentations mirroring braindecode functional semantics.

Reference: braindecode/augmentation/functional.py
``ft_surrogate``, ``_analytic_transform`` and ``_frequency_shift``.
All complex arithmetic is done on (real, imag) pairs with real ops so
the DFT-matmul fallback path needs no native complex FFT.
"""
from __future__ import annotations

import math

import torch

from . import backend
from .spectral import analytic_signal


def _rotate(re: torch.Tensor, im: torch.Tensor, angle: torch.Tensor):
    """Multiply complex (re, im) by exp(1j * angle) using real ops."""
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    return re * cos_a - im * sin_a, re * sin_a + im * cos_a


def ft_surrogate(
    x: torch.Tensor,
    phase: torch.Tensor,
    phase_noise_magnitude: float,
) -> torch.Tensor:
    """Fourier-transform surrogate with a caller-supplied phase perturbation.

    Mirrors braindecode's ``ft_surrogate``: FFT the (real) signal,
    multiply by ``exp(1j * phase_noise_magnitude * phase)`` and return
    the real part of the inverse FFT. ``phase`` is a real tensor of
    phase perturbations (radians) broadcastable to ``x``'s shape; with
    the Hermitian-symmetric phases braindecode samples, the result is
    real up to numerical error, exactly as in the original (which also
    discards the imaginary part). Magnitude 0 returns ``x`` unchanged.
    """
    if torch.is_complex(x):
        raise ValueError("x must be real.")
    if not (0 <= float(phase_noise_magnitude) <= 1):
        raise ValueError("phase_noise_magnitude must be between 0 and 1.")
    phase = phase.to(dtype=x.dtype, device=x.device)
    re, im = backend.fft_pair(x, dim=-1)
    angle = phase * float(phase_noise_magnitude)
    # Angle must broadcast against the full-FFT shape (same as x for real input).
    angle = torch.broadcast_to(angle, x.shape)
    re2, im2 = _rotate(re, im, angle)
    out_re, _ = backend.ifft_pair(re2, im2, dim=-1)
    return out_re.to(dtype=x.dtype)


def _nextpow2(n: int) -> int:
    return int(math.ceil(math.log2(abs(n)))) if n > 1 else 0


def frequency_shift(x: torch.Tensor, shift, sfreq: float) -> torch.Tensor:
    """Shift a signal in frequency, mirroring braindecode ``_frequency_shift``.

    Pads the last axis to the next power of two, forms the analytic
    signal, multiplies by ``exp(2j*pi*shift*t)`` and returns the real
    part truncated to the original length. ``shift`` may be a float or
    a scalar/per-leading-batch tensor broadcastable over channels/time.
    """
    n_orig = x.shape[-1]
    n_padded = 2 ** _nextpow2(n_orig)
    padded = torch.nn.functional.pad(x, (0, n_padded - n_orig))
    analytic = analytic_signal(padded, dim=-1)
    t = torch.arange(n_padded, dtype=x.dtype, device=x.device) / float(sfreq)
    shift_t = torch.as_tensor(shift, dtype=x.dtype, device=x.device) if not isinstance(shift, torch.Tensor) else shift.to(dtype=x.dtype, device=x.device)
    # Build angle of shape broadcastable to padded: (..., n_padded).
    if shift_t.ndim == 0:
        angle = 2.0 * math.pi * shift_t * t
    else:
        # Per-batch shift, e.g. shape (batch,) for x (batch, channels, time).
        # Align shift dims to leading dims: reshape shift to (..., 1, 1...) with time last.
        lead = shift_t.shape
        new_shape = list(lead) + [1] * (x.ndim - len(lead) - 1) + [1]
        angle = 2.0 * math.pi * shift_t.reshape(new_shape) * t
    re2, _im2 = _rotate(analytic.real, analytic.imag, angle)
    return re2[..., :n_orig].to(dtype=x.dtype)
