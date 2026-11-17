import sys

import numpy as np
import pytest
import torch
from scipy.signal import hilbert as scipy_hilbert

sys.path.insert(0, "/Users/bruaristimunha/Projects/braindecode/braindecode")

from braindecode_kernels import (
    aggregate_trial_predictions,
    analytic_signal,
    complex_fft_supported,
    frequency_shift,
    ft_surrogate,
    get_backend_name,
    loader_kwargs,
    plv,
    renorm,
    rfft_abs,
    stft_log_power,
)
from braindecode_kernels import backend


def test_backend_cpu():
    assert complex_fft_supported(torch.device("cpu")) is True
    assert "cpu" in get_backend_name(torch.device("cpu"))


def test_rfft_abs_vs_torch_fft():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64)
    expected = torch.abs(torch.fft.rfft(x, dim=-1))
    torch.testing.assert_close(rfft_abs(x), expected, rtol=1e-5, atol=1e-5)


def test_rfft_abs_dim():
    torch.manual_seed(0)
    x = torch.randn(16, 4)
    expected = torch.abs(torch.fft.rfft(x, dim=0))
    torch.testing.assert_close(rfft_abs(x, dim=0), expected, rtol=1e-5, atol=1e-5)


def test_dft_fallback_matches_native(monkeypatch):
    torch.manual_seed(0)
    x = torch.randn(2, 33)
    monkeypatch.setattr(backend, "complex_fft_supported", lambda d: False)
    expected = torch.abs(torch.fft.rfft(x, dim=-1))
    torch.testing.assert_close(rfft_abs(x), expected, rtol=1e-4, atol=1e-4)
    f_expected = torch.fft.fft(x, dim=-1)
    torch.testing.assert_close(backend.fft(x), f_expected, rtol=1e-4, atol=1e-4)


def test_analytic_signal_vs_scipy():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64)
    expected = torch.tensor(scipy_hilbert(x.numpy(), axis=-1))
    torch.testing.assert_close(analytic_signal(x), expected, rtol=1e-4, atol=1e-4)


def test_analytic_signal_odd_length_vs_scipy():
    torch.manual_seed(1)
    x = torch.randn(4, 65)
    expected = torch.tensor(scipy_hilbert(x.numpy(), axis=-1))
    torch.testing.assert_close(analytic_signal(x), expected, rtol=1e-4, atol=1e-4)


def test_analytic_signal_dft_fallback_vs_scipy(monkeypatch):
    torch.manual_seed(0)
    x = torch.randn(2, 32)
    monkeypatch.setattr(backend, "complex_fft_supported", lambda d: False)
    expected = torch.tensor(scipy_hilbert(x.numpy(), axis=-1))
    torch.testing.assert_close(analytic_signal(x), expected, rtol=1e-3, atol=1e-3)


def test_plv_vs_scipy_oracle():
    torch.manual_seed(0)
    x = torch.randn(5, 8, 64)
    analytic_np = scipy_hilbert(x.numpy(), axis=-1)
    unit = analytic_np / np.abs(analytic_np)
    expected = torch.tensor(np.abs(unit.mean(axis=0)))
    torch.testing.assert_close(plv(x, dim=0), expected, rtol=1e-4, atol=1e-4)
    assert torch.all((plv(x, dim=0) >= 0) & (plv(x, dim=0) <= 1.0 + 1e-6))


def test_plv_identical_signals_is_one():
    t = torch.arange(128, dtype=torch.float32)
    sig = torch.sin(2 * np.pi * 5 * t / 128.0)
    x = sig.repeat(4, 2, 1)
    torch.testing.assert_close(plv(x, dim=0), torch.ones(2, 128), rtol=1e-4, atol=1e-4)


def test_stft_log_power_vs_torch_fft():
    torch.manual_seed(0)
    x = torch.randn(2, 256)
    n_fft, hop = 64, 32
    window = torch.hann_window(n_fft)
    out = stft_log_power(x, n_fft=n_fft, hop=hop, window=window)
    frames = x.unfold(-1, n_fft, hop) * window
    spec = torch.fft.rfft(frames, dim=-1)
    expected = torch.log10(spec.real**2 + spec.imag**2 + 1e-10)
    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_stft_default_window_shape():
    x = torch.randn(3, 100)
    out = stft_log_power(x, n_fft=32, hop=16)
    assert out.shape[-1] == 17
    assert torch.isfinite(out).all()


def test_aggregate_vs_naive_loop():
    torch.manual_seed(0)
    preds = torch.randn(7, 4)
    idx = torch.tensor([0, 0, 1, 2, 2, 2, 1])
    out = aggregate_trial_predictions(preds, idx, n_trials=3)
    rows = []
    for trial in range(3):
        rows.append(preds[idx == trial].mean(dim=0))
    torch.testing.assert_close(out, torch.stack(rows))


def test_aggregate_empty_trial_zero():
    preds = torch.ones(2, 2)
    idx = torch.tensor([0, 0])
    out = aggregate_trial_predictions(preds, idx, n_trials=3)
    torch.testing.assert_close(out[1:], torch.zeros(2, 2))


def test_renorm_vs_torch():
    torch.manual_seed(0)
    x = torch.randn(6, 5)
    for dim in (0, 1):
        expected = x.renorm(2, dim, 1.0)
        torch.testing.assert_close(renorm(x, 1.0, dim=dim), expected, rtol=1e-5, atol=1e-5)
    # Slices already within maxnorm unchanged.
    small = torch.full((3, 3), 0.1)
    torch.testing.assert_close(renorm(small, 10.0, dim=0), small)


def test_loader_kwargs():
    kw = loader_kwargs()
    assert isinstance(kw, dict) and "num_workers" in kw


def _hermitian_phase(shape, seed=0):
    rng = np.random.RandomState(seed)
    n = shape[-1]
    phase = np.zeros(shape)
    if n % 2 == 0:
        half = rng.uniform(0, 2 * np.pi, size=shape[:-1] + (n // 2 - 1,))
        phase[..., 1 : n // 2] = half
        phase[..., n // 2 + 1 :] = -np.flip(half, axis=-1)
    else:
        half = rng.uniform(0, 2 * np.pi, size=shape[:-1] + ((n - 1) // 2,))
        phase[..., 1 : (n + 1) // 2] = half
        phase[..., (n + 1) // 2 :] = -np.flip(half, axis=-1)
    return torch.tensor(phase, dtype=torch.float32)


def test_ft_surrogate_zero_magnitude_identity():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64)
    phase = _hermitian_phase(x.shape)
    torch.testing.assert_close(ft_surrogate(x, phase, 0.0), x, rtol=1e-4, atol=1e-4)


def test_ft_surrogate_preserves_amplitude_spectrum():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64)
    phase = _hermitian_phase(x.shape, seed=3)
    out = ft_surrogate(x, phase, 1.0)
    torch.testing.assert_close(
        torch.abs(torch.fft.rfft(out, dim=-1)),
        torch.abs(torch.fft.rfft(x, dim=-1)),
        rtol=1e-3,
        atol=1e-3,
    )


def test_ft_surrogate_vs_braindecode_zero_magnitude():
    bd = pytest.importorskip("braindecode.augmentation.functional")
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32)
    y = torch.zeros(2, dtype=torch.long)
    expected, _ = bd.ft_surrogate(x.clone(), y, 0.0, False, random_state=0)
    phase = _hermitian_phase(x.shape)
    torch.testing.assert_close(ft_surrogate(x, phase, 0.0), expected.float(), rtol=1e-4, atol=1e-4)


def test_frequency_shift_pure_tone():
    sfreq = 256.0
    t = torch.arange(256, dtype=torch.float32) / sfreq
    x = torch.sin(2 * np.pi * 10.0 * t).view(1, 1, -1).repeat(2, 2, 1)
    out = frequency_shift(x, 5.0, sfreq)
    assert out.shape == x.shape
    # Shifted tone peaks near 15 Hz in the rFFT magnitude.
    mag = torch.abs(torch.fft.rfft(out[0, 0].double()))
    peak_hz = float(torch.argmax(mag)) * sfreq / 256.0
    assert abs(peak_hz - 15.0) <= 1.0


def test_frequency_shift_vs_braindecode():
    bd = pytest.importorskip("braindecode.augmentation.functional")
    torch.manual_seed(0)
    x = torch.randn(2, 2, 64)
    y = torch.zeros(2, dtype=torch.long)
    expected, _ = bd.frequency_shift(x.clone(), y, 3.0, 100.0)
    out = frequency_shift(x, 3.0, 100.0)
    torch.testing.assert_close(out, expected.float(), rtol=1e-3, atol=1e-3)


def test_fallback_magnitude_paths_never_build_complex(monkeypatch):
    """On Gaudi-without-plugin torch.complex silently lands on CPU, so
    magnitude/PLV/STFT paths must work from (re, im) pairs only."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 200)
    expected_abs = torch.abs(torch.fft.rfft(x, dim=-1))
    expected_plv = plv(x, dim=0)
    expected_stft = stft_log_power(x[:, 0], n_fft=64, hop=32)
    monkeypatch.setattr(backend, "complex_fft_supported", lambda d: False)

    def _no_complex(*a, **k):
        raise AssertionError("torch.complex called on the fallback path")

    monkeypatch.setattr(torch, "complex", _no_complex)
    torch.testing.assert_close(rfft_abs(x), expected_abs, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(plv(x, dim=0), expected_plv, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        stft_log_power(x[:, 0], n_fft=64, hop=32), expected_stft,
        rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n", [200, 1920, 6000])
def test_dft_fallback_accuracy_long_lengths(monkeypatch, n):
    """Phase indices reduced modulo n in integers (Braindecode's basis):
    float32 k*n phases gave 5.2e-4 at n=6000; this must stay ~1e-6."""
    torch.manual_seed(0)
    x = torch.randn(4, n)
    monkeypatch.setattr(backend, "complex_fft_supported", lambda d: False)
    got = rfft_abs(x)
    ref = torch.abs(torch.fft.rfft(x.double(), dim=-1)).float()
    rel = ((got - ref).abs().max() / ref.abs().max()).item()
    assert rel < 1e-5, f"n={n} rel err {rel}"


def test_fft_preference_forces_fallback(monkeypatch):
    torch.manual_seed(0)
    x = torch.randn(2, 64)
    expected = torch.abs(torch.fft.rfft(x, dim=-1))
    backend.set_fft_preference("cpu", "fallback")
    try:
        assert backend.use_native_fft(x.device) is False
        assert "dft-matmul" in backend.get_backend_name(x.device)
        torch.testing.assert_close(rfft_abs(x), expected, rtol=1e-4, atol=1e-4)
    finally:
        backend.set_fft_preference("cpu", "auto")
    assert backend.use_native_fft(x.device) is True


def test_fft_preference_rejects_bad_mode():
    with pytest.raises(ValueError):
        backend.set_fft_preference("cpu", "sometimes")
