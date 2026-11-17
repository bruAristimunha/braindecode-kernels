"""Generic device-gate patching for libraries with CPU fallbacks.

Some libraries route spectral work through two kinds of module-level
gates: an *input* gate that prepares a tensor for an FFT (and may copy
it to the CPU), and a *predicate* gate that decides per tensor whether
a real-valued fallback must be used. When a native complex FFT actually
works on the tensor's device, both gates can be relaxed safely:

- input gate: keep the tensor on-device, in at least float32;
- predicate gate: answer False (native FFT path), scoped per device.

A gate is declared as data (:class:`GateSpec`), so supporting a new
library needs no new patching logic — only a new gate list. Models
often bind gate functions by name (``from lib import spectral_input``),
so patching rebinds every already-imported module attribute that still
points at the original function object (found via ``vars(module)``,
which never triggers lazy module ``__getattr__``), and
:func:`verify_patched` reports whether a given module took the patch.

``BRAINDECODE_GATES`` is the preset for Braindecode's
``spectral_input`` / ``needs_real_dft`` pair; ``patch_braindecode()``
is the compatibility wrapper. Nothing is imported from the target
library at module import time.
"""
from __future__ import annotations

import functools
import importlib
import sys
from dataclasses import dataclass

import torch

from . import backend
from .spectral import spectrogram  # re-export; defined with spectral ops

__all__ = [
    "GateSpec",
    "BRAINDECODE_GATES",
    "patch_gates",
    "unpatch_gates",
    "verify_patched",
    "patch_braindecode",
    "unpatch_braindecode",
    "spectrogram",
]

_KINDS = ("input", "predicate")


@dataclass(frozen=True)
class GateSpec:
    """One patchable gate: ``attribute`` of ``module``, of ``kind``.

    ``kind`` is ``"input"`` (tensor -> prepared tensor) or
    ``"predicate"`` (tensor -> bool, True meaning "use the fallback").
    """

    module: str
    attribute: str
    kind: str


BRAINDECODE_GATES = (
    GateSpec("braindecode.functional.functions", "spectral_input", "input"),
    GateSpec("braindecode.functional._real_dft", "needs_real_dft", "predicate"),
)

_STATE: dict | None = None


def _default_probe(device) -> bool:
    """FFT decision for gates: the backend's preference-aware choice.

    This is deliberately not just "native FFT exists": where the
    native FFT is a slow plugin path (Gaudi + hpu-complex), the
    backend prefers the real-valued fallback, so gates keep the
    library's fast real-DFT path there. See
    ``backend.set_fft_preference``.
    """
    return backend.use_native_fft(device)


def _probe_ok(probe, device) -> bool:
    """A raising probe means "not verified": fall back, never crash."""
    try:
        return bool(probe(device))
    except Exception:
        return False


def _wrap_input(original, probe):
    @functools.wraps(original)
    def wrapped(x):
        if _probe_ok(probe, x.device):
            return x.to(torch.promote_types(x.dtype, torch.float32))
        return original(x)

    return wrapped


def _wrap_predicate(original, probe):
    @functools.wraps(original)
    def wrapped(x):
        if _probe_ok(probe, x.device):
            return False
        return original(x)

    return wrapped


def _rebind(attribute, original, replacement):
    """Replace every module attribute that is ``original``.

    Reads ``vars(module)`` only, so lazy module-level ``__getattr__``
    hooks are never triggered. Returns the rebound module names.
    """
    rebound = []
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        try:
            attrs = vars(mod)
        except Exception:
            continue
        if attrs.get(attribute) is original:
            setattr(mod, attribute, replacement)
            rebound.append(name)
    return rebound


def _resolve(spec):
    module = importlib.import_module(spec.module)
    try:
        original = vars(module).get(spec.attribute)
    except Exception:
        original = None
    if not callable(original):
        raise AttributeError(f"{spec.module}.{spec.attribute} is not callable")
    return module, original


def patch_gates(gates, fft_probe=None):
    """Patch ``gates`` (an iterable of :class:`GateSpec`). Idempotent.

    Returns a report dict whose ``rebound`` lists every module whose
    attribute was replaced. Calling again with the same gates and probe
    returns the same report; calling with different gates or a
    different probe raises ``RuntimeError`` — unpatch first, so an A/B
    run never silently keeps the first configuration.
    """
    global _STATE
    gates = tuple(gates)
    for spec in gates:
        if spec.kind not in _KINDS:
            raise ValueError(
                f"unknown gate kind {spec.kind!r}; expected one of {_KINDS}")
    probe = _default_probe if fft_probe is None else fft_probe
    if _STATE is not None:
        if gates == _STATE["gates"] and probe is _STATE["probe"]:
            return _STATE["report"]
        raise RuntimeError(
            "gates already patched with a different configuration; "
            "call unpatch_gates() first")

    # Resolve everything before mutating anything.
    records = []
    for spec in gates:
        module, original = _resolve(spec)
        wrap = _wrap_input if spec.kind == "input" else _wrap_predicate
        records.append({
            "spec": spec, "module": module, "original": original,
            "replacement": wrap(original, probe),
        })

    gate_reports = []
    union = set()
    try:
        for rec in records:
            spec = rec["spec"]
            rebound = _rebind(spec.attribute, rec["original"],
                              rec["replacement"])
            # Defining module is covered by the scan when imported, but
            # set it explicitly: it is the source for late importers.
            setattr(rec["module"], spec.attribute, rec["replacement"])
            if spec.module not in rebound:
                rebound.append(spec.module)
            union.update(rebound)
            gate_reports.append({
                "module": spec.module, "attribute": spec.attribute,
                "kind": spec.kind, "rebound": sorted(rebound),
            })
    except Exception:
        _restore(records)
        raise

    report = {"patched": True, "gates": gate_reports,
              "rebound": sorted(union)}
    _STATE = {"report": report, "gates": gates, "probe": probe,
              "records": records}
    return report


def _restore(records):
    for rec in records:
        spec = rec["spec"]
        _rebind(spec.attribute, rec["replacement"], rec["original"])
        try:
            setattr(rec["module"], spec.attribute, rec["original"])
        except Exception:
            pass


def unpatch_gates():
    """Undo :func:`patch_gates`. Returns True when something was undone."""
    global _STATE
    if _STATE is None:
        return False
    _restore(_STATE["records"])
    _STATE = None
    return True


def verify_patched(module_name: str, attribute: str = "spectral_input") -> bool:
    """True when ``module_name``'s ``attribute`` currently is a patch.

    Answers the "did my model take it?" question for modules imported
    before or after patching; False for unknown/unpatched modules.
    """
    if _STATE is None:
        return False
    mod = sys.modules.get(module_name)
    if mod is None:
        return False
    try:
        current = vars(mod).get(attribute)
    except Exception:
        return False
    return any(
        current is rec["replacement"]
        for rec in _STATE["records"]
        if rec["spec"].attribute == attribute)


def patch_braindecode(fft_probe=None):
    """Patch Braindecode's spectral gates. Idempotent.

    Returns the :func:`patch_gates` report, or ``None`` when
    Braindecode is not importable (nothing is changed then).
    """
    try:
        return patch_gates(BRAINDECODE_GATES, fft_probe=fft_probe)
    except (ImportError, AttributeError, TypeError):
        return None


def unpatch_braindecode():
    """Undo :func:`patch_braindecode`. Returns True if it was applied."""
    if _STATE is None or _STATE["gates"] != BRAINDECODE_GATES:
        return False
    return unpatch_gates()
