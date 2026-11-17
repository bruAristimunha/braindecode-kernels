"""Tests for the generic gate patcher against stub packages.

The Braindecode stub reproduces the two real facts that matter: models
hold ``spectral_input`` by name (module global bound at import), and
``needs_real_dft`` is a plain module-level function.
"""
import importlib
import sys
import types

import pytest
import torch

from braindecode_kernels import integrate
from braindecode_kernels.integrate import GateSpec


@pytest.fixture(autouse=True)
def _restore():
    yield
    integrate.unpatch_gates()
    integrate.unpatch_braindecode()


def _make_stub(monkeypatch, prefix="braindecode"):
    bd = types.ModuleType(prefix)
    functional = types.ModuleType(f"{prefix}.functional")
    functions = types.ModuleType(f"{prefix}.functional.functions")
    real_dft = types.ModuleType(f"{prefix}.functional._real_dft")
    models = types.ModuleType(f"{prefix}.models")
    fake_model = types.ModuleType(f"{prefix}.models.fake")

    def orig_spectral_input(x):
        return x.to(torch.promote_types(x.dtype, torch.float32)) + 1000.0

    def orig_needs_real_dft(x):
        return True  # as on HPU

    functions.spectral_input = orig_spectral_input
    functional.spectral_input = orig_spectral_input
    functional.functions = functions
    functional._real_dft = real_dft
    real_dft.needs_real_dft = orig_needs_real_dft
    fake_model.spectral_input = orig_spectral_input  # by-name import
    bd.functional = functional

    for name, mod in [
        (prefix, bd), (f"{prefix}.functional", functional),
        (f"{prefix}.functional.functions", functions),
        (f"{prefix}.functional._real_dft", real_dft),
        (f"{prefix}.models", models),
        (f"{prefix}.models.fake", fake_model),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    return functions, real_dft, fake_model, orig_spectral_input, orig_needs_real_dft


def test_patch_rebinds_by_name_imports_and_scopes_gate(monkeypatch):
    functions, real_dft, fake_model, orig_si, orig_nd = _make_stub(monkeypatch)
    probe = lambda dev: True  # noqa: E731
    report = integrate.patch_braindecode(fft_probe=probe)
    assert report["patched"] is True
    assert "braindecode.models.fake" in report["rebound"]
    assert fake_model.spectral_input is not orig_si
    assert functions.spectral_input is fake_model.spectral_input
    assert integrate.verify_patched("braindecode.models.fake") is True
    x = torch.randn(2, 8)
    out = fake_model.spectral_input(x)
    assert (out - x).abs().max() < 1e-6
    assert real_dft.needs_real_dft(x) is False
    # Idempotent for the same probe; a different probe is not ignored.
    assert integrate.patch_braindecode(fft_probe=probe) is report
    with pytest.raises(RuntimeError, match="unpatch"):
        integrate.patch_braindecode(fft_probe=lambda dev: True)
    assert integrate.unpatch_braindecode() is True
    assert integrate.verify_patched("braindecode.models.fake") is False
    assert fake_model.spectral_input is orig_si
    assert functions.spectral_input is orig_si
    assert real_dft.needs_real_dft is orig_nd


def test_patch_defers_to_original_when_fft_missing(monkeypatch):
    functions, real_dft, fake_model, orig_si, orig_nd = _make_stub(monkeypatch)
    integrate.patch_braindecode(fft_probe=lambda dev: False)
    x = torch.randn(2, 8)
    out = functions.spectral_input(x)
    assert (out - (x + 1000.0)).abs().max() < 1e-5  # original ran
    assert real_dft.needs_real_dft(x) is True  # real-DFT preserved


def test_raising_probe_falls_back_to_original(monkeypatch):
    functions, real_dft, *_ = _make_stub(monkeypatch)

    def bad_probe(dev):
        raise RuntimeError("probe exploded")

    integrate.patch_braindecode(fft_probe=bad_probe)
    x = torch.randn(2, 8)
    out = functions.spectral_input(x)
    assert (out - (x + 1000.0)).abs().max() < 1e-5
    assert real_dft.needs_real_dft(x) is True


def test_scan_does_not_trigger_lazy_module_getattr(monkeypatch):
    _make_stub(monkeypatch)
    calls = []

    class Lazy(types.ModuleType):
        def __getattr__(self, name):
            calls.append(name)
            raise AttributeError(name)

    monkeypatch.setitem(sys.modules, "lazy_fake_mod", Lazy("lazy_fake_mod"))
    integrate.patch_braindecode(fft_probe=lambda dev: True)
    assert "spectral_input" not in calls


def test_patch_noop_without_braindecode(monkeypatch):
    monkeypatch.setitem(sys.modules, "braindecode", None)
    # ``braindecode.functional`` is resolved from sys.modules too.
    monkeypatch.setitem(sys.modules, "braindecode.functional", None)
    assert integrate.patch_braindecode() is None


def test_generic_patch_gates_for_another_library(monkeypatch):
    _make_stub(monkeypatch, prefix="acme")
    gates = [
        GateSpec("acme.functional.functions", "spectral_input", "input"),
        GateSpec("acme.functional._real_dft", "needs_real_dft", "predicate"),
    ]
    report = integrate.patch_gates(gates, fft_probe=lambda dev: False)
    assert report["patched"] is True
    assert set(report["rebound"]) >= {
        "acme.functional.functions", "acme.models.fake"}
    assert integrate.verify_patched("acme.models.fake") is True
    assert integrate.unpatch_gates() is True
    assert integrate.verify_patched("acme.models.fake") is False


def test_generic_patch_gates_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        integrate.patch_gates(
            [GateSpec("torch", "fft", "bogus")], fft_probe=lambda dev: True)
    assert integrate.unpatch_gates() is False  # nothing was applied


def test_verify_patched_sees_late_importer(monkeypatch):
    functions, *_ = _make_stub(monkeypatch)
    integrate.patch_braindecode(fft_probe=lambda dev: True)
    late = types.ModuleType("braindecode.models.late")
    # What ``from braindecode.functional import spectral_input`` binds
    # when the model module is first imported after patching.
    late.spectral_input = sys.modules["braindecode.functional"].spectral_input
    monkeypatch.setitem(sys.modules, "braindecode.models.late", late)
    assert integrate.verify_patched("braindecode.models.late") is True


def test_spectrogram_matches_torch_stft():
    from braindecode_kernels.spectral import spectrogram

    torch.manual_seed(0)
    x = torch.randn(2, 512)
    got = spectrogram(x, 128, hop_length=64)
    ref = torch.stft(x, 128, hop_length=64,
                     window=torch.hann_window(128), return_complex=True)
    torch.testing.assert_close(got, ref)


def test_real_braindecode_import_order():
    try:
        cbramod = importlib.import_module("braindecode.models.cbramod")
        functions = importlib.import_module("braindecode.functional.functions")
        importlib.import_module("braindecode.functional._real_dft")
    except Exception as exc:  # installed Braindecode absent/different
        pytest.skip(f"real braindecode unavailable: {exc}")
    orig = cbramod.spectral_input
    if orig is not functions.spectral_input:
        pytest.skip("installed braindecode does not share spectral_input")
    integrate.patch_braindecode(fft_probe=lambda dev: True)
    try:
        assert cbramod.spectral_input is not orig
        assert integrate.verify_patched("braindecode.models.cbramod") is True
    finally:
        integrate.unpatch_braindecode()
    assert cbramod.spectral_input is orig
