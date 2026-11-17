"""Braindecode-oriented portable torch kernels (CUDA / TPU / Gaudi / CPU)."""
from .backend import complex_fft_supported, get_backend_name
from .spectral import analytic_signal, plv, rfft_abs, spectrogram, stft_log_power
from .augment import frequency_shift, ft_surrogate
from .ops import aggregate_trial_predictions, loader_kwargs, renorm
from .integrate import (
    BRAINDECODE_GATES,
    GateSpec,
    patch_braindecode,
    patch_gates,
    unpatch_braindecode,
    unpatch_gates,
    verify_patched,
)

__version__ = "0.2.0"
__all__ = [
    "complex_fft_supported",
    "get_backend_name",
    "rfft_abs",
    "stft_log_power",
    "analytic_signal",
    "plv",
    "ft_surrogate",
    "frequency_shift",
    "aggregate_trial_predictions",
    "renorm",
    "loader_kwargs",
    "patch_braindecode",
    "unpatch_braindecode",
    "spectrogram",
    "GateSpec",
    "BRAINDECODE_GATES",
    "patch_gates",
    "unpatch_gates",
    "verify_patched",
]
