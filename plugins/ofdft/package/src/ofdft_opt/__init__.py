"""ofdft-opt: orbital-free DFT (DFTpy) structure relaxation and stress-strain curves.

Command line: ``ofdft <command>`` or ``python -m ofdft_opt <command>`` (relax, sscurve, jobscript,
doctor, pp, plot, version); under MPI ``mpirun -np N python -m mpi4py -m ofdft_opt <command>``.

Importing this package has two side effects that must happen before DFTpy is imported (every
module of the package imports it first, automatically):
  * ``ofdft_opt._shim`` patches SciPy's removed ``dcsrch`` line search for DFTpy <= 2.1.x
    (no-op on DFTpy >= 2.2);
  * if no ``pylibxc`` is importable, ``<package>/_vendor`` is APPENDED to sys.path so DFTpy finds
    the vendored NumPy stand-in (LDA/PZ81, PBE). A real pylibxc (LibXC bindings) always wins.
``XC_BACKEND`` says which one is used: "libxc" or "ofdft-standin" (decided by the content of the
pylibxc that is found, not by its location: a copy of the stand-in elsewhere on sys.path, e.g. a
``pylibxc/`` folder in the current directory, is the stand-in too; ``XC_STANDIN_COPY`` then names it,
with "modified" when it differs from the vendored files). ``xc_description()`` is the text for logs.
"""
import hashlib
import importlib.util
import os
import sys

__version__ = "0.1.0"

PKG_DIR = os.path.dirname(os.path.abspath(__file__))     # .../ofdft_opt
LIB_DIR = os.path.dirname(PKG_DIR)                       # what must be on PYTHONPATH

from . import _shim  # noqa: E402,F401  (before anything that imports dftpy)

_VENDOR_DIR = os.path.join(PKG_DIR, "_vendor")


def _find_pylibxc():
    """Directory of the importable pylibxc package (None if there is none). Does not import it."""
    try:
        spec = importlib.util.find_spec("pylibxc")
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    locs = list(spec.submodule_search_locations or [])
    if locs:
        return os.path.abspath(locs[0])
    return os.path.dirname(os.path.abspath(spec.origin)) if spec.origin else None


_STANDIN_MARK = b"Minimal NumPy stand-in for the `pylibxc` bindings"   # first docstring line of the stand-in


def _sha_lf(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
    except OSError:
        return None


def _standin_copy(where):
    """None if `where` (a pylibxc package dir outside _vendor) is not the stand-in, else "identical" or
    "modified" (compared with the vendored files, CRLF-normalised)."""
    try:
        with open(os.path.join(where, "__init__.py"), "rb") as f:
            head = f.read(4096)
    except OSError:
        return None
    if _STANDIN_MARK not in head:
        return None
    vend = os.path.join(_VENDOR_DIR, "pylibxc")
    for name in ("__init__.py", "functional.py"):
        mine = _sha_lf(os.path.join(where, name))
        if mine is None or mine != _sha_lf(os.path.join(vend, name)):
            return "modified"
    return "identical"


def _setup_xc():
    where = _find_pylibxc()
    if where is None:
        if _VENDOR_DIR not in sys.path:
            sys.path.append(_VENDOR_DIR)                 # append: a real pylibxc always wins
        importlib.invalidate_caches()
        where = _find_pylibxc()
    if where is None:                                    # broken install (no _vendor): PBE unavailable
        return "none", None, None
    vendor = os.path.normcase(os.path.realpath(_VENDOR_DIR))
    if os.path.normcase(os.path.realpath(where)).startswith(vendor + os.sep):
        return "ofdft-standin", where, None
    copy = _standin_copy(where)                          # e.g. pylibxc/ of an old run folder in the cwd
    if copy:
        return "ofdft-standin", where, copy
    return "libxc", where, None


# XC_ORIGIN: directory of the pylibxc in use (or None); XC_STANDIN_COPY: None, "identical" or "modified"
# when that pylibxc is a copy of the stand-in outside the package (it shadows the vendored one)
XC_BACKEND, XC_ORIGIN, XC_STANDIN_COPY = _setup_xc()


def xc_description():
    """XC backend for log headers and `ofdft doctor`, e.g. "ofdft-standin (copy at /x/pylibxc)"."""
    if XC_STANDIN_COPY == "identical":
        return f"{XC_BACKEND} (copy at {XC_ORIGIN}, identical to the vendored one)"
    if XC_STANDIN_COPY == "modified":
        return f"{XC_BACKEND} (MODIFIED copy at {XC_ORIGIN}, used instead of the vendored one)"
    return XC_BACKEND
