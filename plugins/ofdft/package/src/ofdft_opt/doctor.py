"""`ofdft doctor` -- check the Python environment for ofdft-opt (seconds; safe on a login node).

  ofdft doctor            full report
  ofdft doctor --brief    only the WARN/FAIL lines plus one summary line (used by generated job scripts)
  mpirun -np N python -m ofdft_opt doctor --mpi
                          MPI wire-up test: rank 0 prints exactly one line "<size> | <MPI library>",
                          the other ranks print nothing

Exit code 0 when nothing FAILed, 1 otherwise. Without --mpi, MPI is never initialized
(mpi4py.rc.initialize = False is set before any mpi4py import).
"""
import argparse
import importlib
import importlib.util
import os
import re
import sys

from . import PKG_DIR, XC_BACKEND, __version__

DFTPY_OK_RANGE = ((2, 1, 2), (2, 3, 0))          # tested: 2.1.2 (serial) and 2.2.0 (MPI, cluster)
ASE_MIN = (3, 23)
PY_MIN = (3, 10)
MPI_HINT = "conda install -c conda-forge openmpi mpi4py mpi4py-fft"


def _vtuple(ver, n=3):
    m = re.match(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", str(ver))
    if not m:
        return None
    return tuple(int(g or 0) for g in m.groups())[:n]


def _real(path):
    return os.path.normcase(os.path.realpath(path))


def _under(path, prefix):
    p, d = _real(path), _real(prefix)
    return p == d or p.startswith(d.rstrip(os.sep) + os.sep)


def _dist_version(*names):
    try:
        from importlib.metadata import version
    except ImportError:
        return None
    for n in names:
        try:
            return version(n)
        except Exception:
            pass
    return None


class Report:
    def __init__(self):
        self.items = []

    def add(self, level, name, text):
        self.items.append((level, name, text))

    def count(self, level):
        return sum(1 for it in self.items if it[0] == level)


def _check_module(rep, name, required=True, min_version=None, hint=""):
    """Import `name`, report version + location. Returns the module or None."""
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        rep.add("FAIL" if required else "WARN", name, f"not importable ({exc.__class__.__name__}: {exc})"
                + (f"; {hint}" if hint else ""))
        return None
    ver = getattr(mod, "__version__", "?")
    path = os.path.dirname(getattr(mod, "__file__", "") or "") or "?"
    level, note = "OK", ""
    if min_version and (_vtuple(ver) or (0,)) < min_version:
        level, note = "WARN", f"  (older than the required {'.'.join(map(str, min_version))})"
    elif path != "?" and not _under(path, sys.prefix):
        level, note = "WARN", f"  (not from the env prefix {sys.prefix}: ~/.local or PYTHONPATH shadowing?)"
    rep.add(level, name, f"{ver}  {path}{note}")
    return mod


def _check_dftpy(rep):
    try:
        import dftpy
    except Exception as exc:
        rep.add("FAIL", "dftpy", f"not importable ({exc.__class__.__name__}: {exc}); install the PyPI "
                "release: python -m pip install dftpy==2.2.0")
        return None
    ver = str(getattr(dftpy, "__version__", "?"))
    pkg = os.path.dirname(os.path.abspath(dftpy.__file__))
    where = f"{ver}  {pkg}"
    git = os.path.isdir(os.path.join(os.path.dirname(pkg), ".git"))
    if "dev" in ver.lower() or "+" in ver or git:
        rep.add("FAIL", "dftpy", f"{where}  -- a git/development build of DFTpy: the dev branch computes a wrong "
                "XC stress under MPI; install the PyPI release (python -m pip install dftpy==2.2.0)")
        return dftpy
    v = _vtuple(ver)
    lo, hi = DFTPY_OK_RANGE
    if v is None or not (lo <= v < hi):
        rep.add("WARN", "dftpy", f"{where}  -- outside the tested range 2.1.2-2.2.x")
    elif not _under(pkg, sys.prefix):
        rep.add("WARN", "dftpy", f"{where}  -- not from the env prefix {sys.prefix} (~/.local or PYTHONPATH?)")
    else:
        rep.add("OK", "dftpy", where)
    return dftpy


def _check_mpi4py(rep):
    try:
        import mpi4py
    except Exception:
        rep.add("WARN", "mpi4py", f"not installed: needed for MPI runs ({MPI_HINT})")
        return
    lib = ""
    try:
        from mpi4py import MPI              # not initialized (rc.initialize = False)
        first = MPI.Get_library_version().strip().splitlines()
        lib = f"  [{first[0].strip()}]" if first else ""
    except Exception as exc:
        lib = f"  [MPI library not loadable: {exc}]"
    rep.add("OK", "mpi4py", f"{getattr(mpi4py, '__version__', '?')}  "
            f"{os.path.dirname(mpi4py.__file__)}{lib}")


def _check_mpi4py_fft(rep):
    # presence only: importing mpi4py_fft loads MPI machinery; the real import is tested by --mpi
    try:
        spec = importlib.util.find_spec("mpi4py_fft")
    except Exception:
        spec = None
    if spec is None:
        rep.add("WARN", "mpi4py_fft", f"not installed: needed for MPI runs ({MPI_HINT})")
        return
    ver = _dist_version("mpi4py-fft", "mpi4py_fft") or "?"
    loc = (list(spec.submodule_search_locations or []) or [spec.origin or "?"])[0]
    rep.add("OK", "mpi4py_fft", f"{ver}  {loc}")


def _check_shim(rep):
    try:
        from . import _shim
    except Exception as exc:
        rep.add("FAIL", "scipy shim", f"cannot import ofdft_opt._shim: {exc}")
        return
    if getattr(_shim, "PATCHED", False):
        rep.add("OK", "scipy shim", "active (DFTpy <= 2.1 line search on SciPy without dcsrch)")
        return
    needed = False
    try:
        needed = _shim._dftpy_uses_minpack2() and not _shim._scipy_has_dcsrch()
    except Exception:
        pass
    if needed:
        rep.add("FAIL", "scipy shim", "needed (DFTpy <= 2.1 on a SciPy without dcsrch) but not installed: "
                "upgrade DFTpy to 2.2.0")
    else:
        rep.add("OK", "scipy shim", "not needed")


def _check_xc(rep):
    import ofdft_opt
    origin = getattr(ofdft_opt, "XC_ORIGIN", None)
    copy = getattr(ofdft_opt, "XC_STANDIN_COPY", None)
    if XC_BACKEND == "libxc":
        if origin and not _under(origin, sys.prefix):
            rep.add("WARN", "XC backend", f"libxc from {origin}: not from the env prefix {sys.prefix} (a pylibxc "
                    "folder in the current directory or on PYTHONPATH shadows the environment's)")
        else:
            rep.add("OK", "XC backend", f"libxc (real pylibxc: {origin or '?'})")
    elif XC_BACKEND == "ofdft-standin" and copy:
        rep.add("WARN", "XC backend", f"{ofdft_opt.xc_description()}: a pylibxc folder outside the package (e.g. "
                "in the current directory) shadows the vendored stand-in; remove or rename it"
                + ("" if copy == "identical" else " -- results may differ from the vendored stand-in"))
    elif XC_BACKEND == "ofdft-standin":
        rep.add("OK", "XC backend", "ofdft-standin (vendored NumPy LDA/PBE; LibXC not installed)")
    else:
        rep.add("FAIL", "XC backend", f"{XC_BACKEND}: no pylibxc and the vendored stand-in "
                f"{os.path.join(PKG_DIR, '_vendor', 'pylibxc')} is missing (broken install: copy the package again)")


def _check_pp(rep):
    try:
        from . import pp
    except Exception as exc:
        rep.add("FAIL", "pseudopot.", f"cannot import ofdft_opt.pp: {exc}")
        return
    try:
        path_list = pp.search_path()
    except Exception as exc:
        path_list = [f"? ({exc})"]
    found = 0
    for name in sorted(pp.REGISTRY):
        try:
            path = pp.find(name)
        except Exception as exc:
            rep.add("WARN", "pseudopot.", f"{name}: search failed ({exc})")
            continue
        if not path:
            rep.add("INFO", "pseudopot.", f"{name}: not found")
            continue
        found += 1
        try:
            _, ok = pp.verify(path)
        except Exception as exc:
            rep.add("FAIL", "pseudopot.", f"{name}: {path} not readable ({exc})")
            continue
        if ok:
            rep.add("OK", "pseudopot.", f"{name}: {path} (sha256 OK)")
        else:
            rep.add("FAIL", "pseudopot.", f"{name}: {path} has the wrong sha256 (corrupted or a different file): "
                    "delete it and run `ofdft pp fetch`")
    if not found:
        rep.add("WARN", "pseudopot.", "no registry pseudopotential found in " + os.pathsep.join(path_list)
                + ": run `ofdft pp fetch`")
    else:
        rep.add("INFO", "pp search", os.pathsep.join(path_list))


def run_checks():
    rep = Report()
    rep.add("OK", "ofdft-opt", f"{__version__}  {PKG_DIR}")
    pyver = sys.version.split()[0]
    if sys.version_info[:2] < PY_MIN:
        rep.add("FAIL", "python", f"{pyver}  {sys.executable}  (needs >= {'.'.join(map(str, PY_MIN))})")
    else:
        rep.add("OK", "python", f"{pyver}  {sys.executable}")
    conda = os.path.isdir(os.path.join(sys.prefix, "conda-meta"))
    env = f"{sys.prefix}" + ("  (conda env)" if conda else "")
    cp = os.environ.get("CONDA_PREFIX")
    if cp and _real(cp) != _real(sys.prefix):
        env += f"  [CONDA_PREFIX={cp} is a different env; fine when python is called by absolute path]"
    rep.add("INFO", "env", env)
    rep.add("INFO", "user site", "ignored (PYTHONNOUSERSITE)" if os.environ.get("PYTHONNOUSERSITE")
            else "enabled (packages in ~/.local can shadow the env; job scripts set PYTHONNOUSERSITE=1)")
    _check_module(rep, "numpy")
    _check_module(rep, "scipy")
    _check_module(rep, "ase", min_version=ASE_MIN, hint="conda install -c conda-forge 'ase>=3.23'")
    dftpy = _check_dftpy(rep)
    _check_mpi4py(rep)
    _check_mpi4py_fft(rep)
    _check_shim(rep)
    _check_xc(rep)
    _check_pp(rep)
    rep.dftpy_version = getattr(dftpy, "__version__", None) if dftpy else None
    return rep


def mpi_check():
    """Rank 0 prints exactly one line '<size> | <MPI library version>' (run under mpirun)."""
    try:
        import mpi4py_fft  # noqa: F401  (DFTpy needs it for distributed FFTs)
        from mpi4py import MPI
    except Exception as exc:
        print(f"ofdft doctor --mpi: mpi4py/mpi4py_fft not importable: {exc!r}", file=sys.stderr, flush=True)
        return 1
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        lines = MPI.Get_library_version().strip().splitlines()
        print(f"{comm.size} | {lines[0].strip() if lines else '?'}", flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="ofdft doctor", description=__doc__.split("\n\n")[0])
    p.add_argument("--brief", action="store_true", help="only WARN/FAIL lines and a summary line")
    p.add_argument("--mpi", action="store_true",
                   help="MPI wire-up test under mpirun: rank 0 prints '<size> | <MPI library>'")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    if args.mpi:
        return mpi_check()
    if "mpi4py.MPI" not in sys.modules:            # never initialize MPI on a login node
        try:
            import mpi4py
            mpi4py.rc.initialize = False
        except Exception:
            pass

    rep = run_checks()
    nfail, nwarn = rep.count("FAIL"), rep.count("WARN")
    if args.brief:
        for level, name, text in rep.items:
            if level in ("FAIL", "WARN"):
                print(f"{level}  {name}: {text}")
    else:
        print(f"ofdft doctor -- ofdft-opt {__version__}")
        for level, name, text in rep.items:
            print(f"  {level:4s}  {name:11s} {text}")
    if nfail:
        print(f"doctor FAILED: {nfail} FAIL, {nwarn} WARN" + ("" if not args.brief else
              " (run `ofdft doctor` for the full report)"))
        return 1
    print(f"doctor OK (ofdft-opt {__version__}, python {sys.version.split()[0]}, "
          f"dftpy {rep.dftpy_version}, XC {XC_BACKEND}"
          + (f"; {nwarn} warning{'s' if nwarn != 1 else ''})" if nwarn else ")"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
