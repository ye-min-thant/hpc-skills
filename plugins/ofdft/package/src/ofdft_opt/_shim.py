"""DFTpy line-search compatibility shim -- safe with any DFTpy (2.1.x, 2.2.x) and any SciPy.

(ofdft_opt._shim is scipy_dcsrch_shim.py; ``import ofdft_opt`` imports it before anything
else, so ``ofdft_opt._shim.PATCHED`` tells whether it patched.)

DFTpy <= 2.1.x runs its density-optimization line search through SciPy's old Fortran
routine ``minpack2.dcsrch`` (math_utils.py: ``from scipy.optimize import _minpack2 as
minpack2``, falling back to ``scipy.optimize.minpack2``). SciPy >= 1.15 no longer
provides it (and SciPy 2.0 also drops the ``scipy.optimize.minpack2`` stub, which would
make ``import dftpy.math_utils`` itself fail). DFTpy >= 2.2.0 uses
``scipy.optimize._linesearch.line_search_wolfe1`` instead and needs nothing.

Import this module BEFORE anything that imports dftpy. It patches only when the
installed DFTpy still uses minpack2 AND SciPy lacks dcsrch; otherwise it does nothing.
It never raises. ``scipy_dcsrch_shim.PATCHED`` tells whether it patched.
"""
import os
import sys
import types
import warnings

PATCHED = False


def _scipy_has_dcsrch():
    """True if SciPy still ships the Fortran dcsrch (old SciPy): nothing to do."""
    for name in ("scipy.optimize._minpack2", "scipy.optimize.minpack2"):
        try:
            mod = __import__(name, fromlist=["dcsrch"])
            if getattr(mod, "dcsrch", None) is not None:
                return True
        except Exception:
            pass
    return False


def _dftpy_uses_minpack2():
    """True for DFTpy <= 2.1.x, whose math_utils references minpack2.
    Decided from the source text so dftpy.math_utils is not imported (it would fail on
    SciPy 2.0 before we can patch). Unknown layout -> True (patching is harmless)."""
    mu = sys.modules.get("dftpy.math_utils")
    if mu is not None:                      # dftpy imported before this shim
        return hasattr(mu, "minpack2")
    try:
        import importlib.util
        spec = importlib.util.find_spec("dftpy")   # top-level: does not execute dftpy
        if spec is None:
            return False                    # DFTpy not installed in this python
        for d in spec.submodule_search_locations or []:
            path = os.path.join(d, "math_utils.py")
            if os.path.isfile(path):
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    return "minpack2" in fh.read()
    except Exception:
        pass
    return True


def _make_dcsrch():
    """Old Fortran call signature, implemented on SciPy's pure-Python DCSRCH port."""
    from scipy.optimize._dcsrch import DCSRCH
    states = {}

    def dcsrch(stp, f, g, ftol, gtol, xtol, task, stpmin, stpmax, isave, dsave):
        key = id(isave)                     # DFTpy allocates one isave per line search
        if task[:5] == b"START":
            states[key] = DCSRCH(None, None, ftol, gtol, xtol, stpmin, stpmax)
        stp, f, g, task = states[key]._iterate(stp, f, g, task)
        if task[:2] != b"FG":               # CONVERGENCE / WARNING / ERROR: search done
            states.pop(key, None)
        return stp, f, g, task

    return dcsrch


def _install():
    global PATCHED
    if _scipy_has_dcsrch() or not _dftpy_uses_minpack2():
        return
    dcsrch = _make_dcsrch()
    import scipy.optimize as so
    try:
        import scipy.optimize._minpack2 as target   # exists but without dcsrch
    except Exception:
        target = types.ModuleType("scipy.optimize._minpack2")
        sys.modules["scipy.optimize._minpack2"] = target
        so._minpack2 = target
    target.dcsrch = dcsrch                  # DFTpy 2.1.x imports this first
    mu = sys.modules.get("dftpy.math_utils")
    if mu is not None and hasattr(mu, "minpack2"):
        mu.minpack2 = target                # dftpy was imported before this shim
    PATCHED = True


try:
    _install()
except Exception as exc:  # never break the caller
    warnings.warn("scipy_dcsrch_shim: dcsrch fallback not installed (%r); DFTpy <= 2.1.x "
                  "line search will fail on this SciPy" % (exc,), RuntimeWarning)
