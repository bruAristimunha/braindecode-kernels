"""Portable trial/loader ops composed from basic torch operations."""
from __future__ import annotations

import torch


def aggregate_trial_predictions(
    window_preds: torch.Tensor,
    i_window_in_trial: torch.Tensor,
    n_trials: int,
) -> torch.Tensor:
    """Mean of window predictions per trial via ``index_add``.

    ``window_preds`` has shape ``(n_windows, ...)`` and
    ``i_window_in_trial`` is a long tensor of shape ``(n_windows,)``
    with values in ``[0, n_trials)``. Returns shape ``(n_trials, ...)``.
    Trials with no windows yield zeros.
    """
    idx = i_window_in_trial.to(device=window_preds.device, dtype=torch.long)
    out_shape = (n_trials,) + tuple(window_preds.shape[1:])
    out = torch.zeros(out_shape, dtype=window_preds.dtype, device=window_preds.device)
    out.index_add_(0, idx, window_preds)
    counts = torch.zeros(n_trials, dtype=window_preds.dtype, device=window_preds.device)
    ones = torch.ones(idx.shape[0], dtype=window_preds.dtype, device=window_preds.device)
    counts.index_add_(0, idx, ones)
    view = (n_trials,) + (1,) * (window_preds.ndim - 1)
    counts = torch.where(counts == 0, torch.ones_like(counts), counts)
    return out / counts.view(view)


def renorm(x: torch.Tensor, maxnorm: float, dim: int = 0) -> torch.Tensor:
    """Composed equivalent of ``Tensor.renorm(2, dim, maxnorm)``.

    Scales each sub-tensor indexed along ``dim`` (norm taken over all
    other axes, as in ``torch.renorm``) so its L2 norm does not exceed
    ``maxnorm``; slices already within the limit are unchanged.
    """
    reduce_dims = tuple(d for d in range(x.ndim) if d != dim % x.ndim)
    norm = torch.sqrt(torch.sum(x * x, dim=reduce_dims, keepdim=True))
    denom = torch.clamp(norm, min=float(maxnorm))
    # Factor is 1 when norm <= maxnorm; guard the maxnorm == norm == 0 case.
    factor = torch.where(denom == 0, torch.ones_like(denom), float(maxnorm) / denom.clamp_min(1e-12))
    factor = torch.where(denom == 0, torch.ones_like(factor), factor)
    return x * factor


def loader_kwargs() -> dict:
    """Portable DataLoader keyword defaults.

    Conservative cross-backend defaults: single-process loading (safe on
    CPU, CUDA, XLA and HPU workers) and pinned memory only on CUDA.
    """
    return {
        "num_workers": 0,
        "pin_memory": bool(torch.cuda.is_available()),
        "persistent_workers": False,
    }
