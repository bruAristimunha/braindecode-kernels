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

## Library integration (agnostic gate patching)

Backend plugins should not embed library-specific code, so the
patching machinery here is generic. A library's device gates are
declared as data — `GateSpec(module, attribute, kind)` with kind
`"input"` (tensor preparation that may copy to CPU) or `"predicate"`
(per-tensor "use the fallback" decision) — and one engine does the
work for any library:

```python
from braindecode_kernels import GateSpec, patch_gates, verify_patched

gates = [GateSpec("mylib.functional", "prepare_input", "input"),
         GateSpec("mylib.fallback", "needs_fallback", "predicate")]
report = patch_gates(gates)          # report["rebound"]: what took effect
verify_patched("mylib.models.some_model", "prepare_input")
```

When the package's FFT probe passes on a tensor's device, input gates
keep the tensor on-device (at least float32) and predicate gates
answer False; otherwise the library's original functions run
unchanged, so no fallback path is ever disabled globally. A raising
probe also falls back. Because models often bind gate functions by
name at import, patching rebinds every already-imported module
attribute still pointing at the original; modules imported later get
the patched function from the defining module. Re-calling with a
different configuration raises — `unpatch_gates()` first, so A/B runs
never silently keep the first setup.

Braindecode is the bundled preset (`BRAINDECODE_GATES`):

```python
import braindecode_kernels
braindecode_kernels.patch_braindecode()   # after hpu_complex.enable() on Gaudi
```

Call it before creating models. The longer-term fix is upstream: if
Braindecode replaces its `device.type == "hpu"` literals with a
registerable FFT-capability probe, this preset becomes a thin
registration shim and the patching engine is only needed for
libraries without such a hook.

## How backend selection works

`complex_fft_supported(device)` runs a tiny cached probe — an 8-point `rfft`/`irfft` on the device inside `try/except` (materialised so lazy XLA/HPU backends actually execute it). If it succeeds, `torch.fft` is used. If it raises, every FFT in the package is computed from a cached real DFT basis: cos/sin matrices cached per `(n, device, dtype)`, applied with real matmuls on `(real, imag)` pairs — operations every backend supports. `get_backend_name(device)` reports e.g. `cuda-native-fft`, `gaudi-hpu-dft-matmul`. Moving a tensor to a new device needs no code change.

Native FFT existing is not always native FFT being fast: on Gaudi, a plugin FFT can be correct but far slower than the DFT matmul at EEG shapes, so `use_native_fft(device)` — the decision every op and gate actually uses — defaults to the fallback for `hpu` even when the probe passes. Override with `set_fft_preference("hpu", "native")`. Magnitude, power and PLV are computed from `(real, imag)` pairs (`sqrt(re² + im²)`), never by building a complex tensor, so they stay on-device where `torch.complex` would silently round-trip through the CPU. The DFT basis reduces phase indices modulo `n` in integers, keeping the fallback at ~1e-6 relative error even at n=6000.

## Limits

- The DFT fallback is O(n²) in time and memory per transform length; it is a correctness fallback, not a performance path, and the cached basis costs `2·n²` elements per length/device/dtype.
- `stft_log_power` uses no centering/padding (`center=False` framing) and returns `log10(power + 1e-10)`; it does not replicate torchaudio's reflect padding or dB scaling.
- `ft_surrogate` takes a caller-supplied phase tensor (braindecode samples its own internally); callers should supply Hermitian-symmetric phases for real outputs, as braindecode does.
- Ops are tested on CPU against scipy/torch/braindecode oracles; CUDA/TPU/Gaudi execution follows from the composed-ops design but is not exercised in CI here.

## License

Apache-2.0. See `LICENSE`.
