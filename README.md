# braindecode-kernels

Backend-portable, Braindecode-oriented PyTorch operations that run on NVIDIA CUDA, Google TPU, Intel Gaudi and CPU — using only composed pure-torch operations, with no backend-specific kernels.

## What / why

Braindecode models and augmentations rely on a small set of signal operations (FFT magnitudes, Hilbert/analytic signals, phase-locking values, Fourier surrogates, frequency shifts, trial aggregation, weight renorm). Native complex FFT is not available or not reliable on every accelerator — notably Gaudi HPUs via the `hpu-complex` plugin and some XLA paths. This package implements those ops once, in portable torch, and transparently falls back to a real-valued DFT expressed as cached matmuls when native complex FFT does not work on the target device.

## Backends

| Backend | Device type | FFT path | Status |
|---|---|---|---|
| NVIDIA CUDA | `cuda` | native `torch.fft` when probe passes | supported |
| Google TPU | `xla` (torch_xla) | native XLA FFT or DFT matmul fallback | supported |
| Intel Gaudi | `hpu` (hpu-complex plugin) | DFT matmul fallback when complex FFT fails | supported |
| CPU | `cpu` | native `torch.fft` | supported |

## Install

```bash
pip install -e .
# with test dependencies:
pip install -e ".[test]"
```

`torch` is intentionally **not** a hard dependency — install the build matched to your accelerator (CUDA, torch_xla, Gaudi SynapseAI, or CPU) first.

## Usage

```python
import torch
from braindecode_kernels import (
    analytic_signal, rfft_abs, stft_log_power, plv,
    ft_surrogate, frequency_shift,
    aggregate_trial_predictions, renorm, loader_kwargs,
    get_backend_name,
)

x = torch.randn(8, 22, 1000)          # (batch, channels, time)
print(get_backend_name(x.device))     # e.g. "cpu-native-fft"

mag = rfft_abs(x)                      # |rFFT| along time
analytic = analytic_signal(x)          # Hilbert analytic signal
phase_lock = plv(x, dim=0)             # PLV across trials
logp = stft_log_power(x, n_fft=256, hop=64)

phase = torch.zeros_like(x)            # Hermitian-symmetric phase noise in practice
surrogate = ft_surrogate(x, phase, phase_noise_magnitude=0.5)
shifted = frequency_shift(x, shift=2.0, sfreq=250.0)

trial_preds = aggregate_trial_predictions(window_preds, i_window_in_trial, n_trials)
w = renorm(weight, maxnorm=1.0, dim=0)
dl_kwargs = loader_kwargs()
```

## How backend selection works

`complex_fft_supported(device)` runs a tiny cached probe — an 8-point `rfft`/`irfft` on the device inside `try/except` (materialised so lazy XLA/HPU backends actually execute it). If it succeeds, `torch.fft` is used. If it raises, every FFT in the package is computed from a cached real DFT basis: cos/sin matrices cached per `(n, device, dtype)`, applied with real matmuls on `(real, imag)` pairs — operations every backend supports. `get_backend_name(device)` reports e.g. `cuda-native-fft`, `gaudi-hpu-dft-matmul`. Moving a tensor to a new device needs no code change.

## Limits

- The DFT fallback is O(n²) in time and memory per transform length; it is a correctness fallback, not a performance path, and the cached basis costs `2·n²` elements per length/device/dtype.
- `stft_log_power` uses no centering/padding (`center=False` framing) and returns `log10(power + 1e-10)`; it does not replicate torchaudio's reflect padding or dB scaling.
- `ft_surrogate` takes a caller-supplied phase tensor (braindecode samples its own internally); callers should supply Hermitian-symmetric phases for real outputs, as braindecode does.
- Ops are tested on CPU against scipy/torch/braindecode oracles; CUDA/TPU/Gaudi execution follows from the composed-ops design but is not exercised in CI here.

## License

Apache-2.0. See `LICENSE`.
