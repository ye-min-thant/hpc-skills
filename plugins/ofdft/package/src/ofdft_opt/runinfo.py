"""Shared run helpers: file checksums, atomic JSON params/status files, STOP files, MPI-safe exits.

MPI notes: everything here is rank-local. Callers decide refusals on rank 0 and broadcast the
decision (see `gate`), so that every rank exits identically and no rank waits in a collective.
"""
import datetime
import hashlib
import json
import os
import signal
import sys
import time
import traceback


# ---------------------------------------------------------------------------------------- checksums
def sha256_bytes(b):
    """sha256 hex digest of bytes."""
    return hashlib.sha256(b).hexdigest()


def sha256_text(path):
    """sha256 of a text file with CRLF -> LF normalisation (a Windows-copied file gives the same sum)."""
    with open(path, "rb") as f:
        return sha256_bytes(f.read().replace(b"\r\n", b"\n"))


# ---------------------------------------------------------------------------------------- JSON files
def write_json_atomic(path, obj):
    """Write JSON to a temporary file next to `path`, then rename: readers never see half a file."""
    d = os.path.dirname(os.path.abspath(path))
    tmp = os.path.join(d, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, default=_json_default)
        f.write("\n")
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:                    # Windows: target briefly open in another process
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def _json_default(o):
    try:                                           # numpy scalars / arrays
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
    except ImportError:
        pass
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def read_json(path, default=None):
    """Parsed JSON of `path`; `default` if the file is missing or unreadable."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def now_iso():
    """Local time, ISO 8601 with UTC offset, seconds resolution."""
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------- STOP files
def stop_requested(prefix):
    """'<prefix>.STOP' or 'STOP' if that file exists in the current directory, else None.
    Rank-local: callers decide on rank 0 and broadcast."""
    for name in (f"{prefix}.STOP", "STOP"):
        if os.path.exists(name):
            return name
    return None


# ---------------------------------------------------------------------------------------- MPI helpers
def _mpi_module():
    """mpi4py.MPI if it is already imported and MPI is running (never initializes MPI)."""
    MPI = sys.modules.get("mpi4py.MPI")
    if MPI is None:
        return None
    try:
        if MPI.Is_initialized() and not MPI.Is_finalized():
            return MPI
    except Exception:
        pass
    return None


def world_rank():
    """MPI rank of this process (0 when serial); never initializes MPI."""
    MPI = _mpi_module()
    if MPI is not None:
        try:
            return MPI.COMM_WORLD.Get_rank()
        except Exception:
            pass
    for key in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK"):
        v = os.environ.get(key, "")
        if v.isdigit():
            return int(v)
    return 0


def install_mpi_abort_hook():
    """sys.excepthook wrapper: print the traceback, then MPI.COMM_WORLD.Abort(1) if MPI is
    initialized with more than one rank, so an exception on one rank ends the whole job
    instead of leaving the others waiting in a collective until the time limit."""
    if getattr(sys.excepthook, "_ofdft_abort_hook", False):
        return
    previous = sys.excepthook

    def hook(etype, value, tb):
        try:
            previous(etype, value, tb)
        except Exception:
            traceback.print_exception(etype, value, tb)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        MPI = _mpi_module()
        if MPI is not None:
            try:
                if MPI.COMM_WORLD.Get_size() > 1:
                    MPI.COMM_WORLD.Abort(1)
            except Exception:
                pass

    hook._ofdft_abort_hook = True
    sys.excepthook = hook


class Terminated(Exception):
    """Raised on SIGTERM (Slurm time limit / scancel) so the status file can say so."""


def install_term_handler():
    """Best effort: turn SIGTERM into a `Terminated` exception (main thread only), so the status
    file can record it. Returns a function that restores the previous handler."""
    def handler(signum, frame):
        raise Terminated(f"terminated by signal {signum} (Slurm time limit, scancel or node problem); "
                         "resubmit the same job script to continue")
    try:
        previous = signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError, AttributeError):
        return lambda: None

    def restore():
        try:
            signal.signal(signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)
        except (ValueError, OSError, TypeError):
            pass
    return restore


# ---------------------------------------------------------------------------------------- messages, exits
def die(msg, code=2):
    """Print 'ERROR: msg' (rank 0 only) and exit with `code` (2 = refused / bad input)."""
    if world_rank() == 0:
        print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def warn(msg):
    """Print 'WARNING: msg' to stderr on rank 0 only."""
    if world_rank() == 0:
        print(f"WARNING: {msg}", file=sys.stderr, flush=True)


def gate(comm, fn):
    """Run fn() on rank 0 only and give its result to every rank. If fn exits (SystemExit, e.g.
    from `die`), every rank exits with the same code; an unexpected exception is re-raised on
    rank 0 (abort hook) and turned into exit code 1 on the others. `comm` may be None (serial)."""
    size = getattr(comm, "size", 1) if comm is not None else 1
    rank = getattr(comm, "rank", 0) if comm is not None else 0
    outcome, exc = None, None
    if rank == 0:
        try:
            outcome = ("ok", fn())
        except SystemExit as e:
            outcome = ("exit", e.code)
        except BaseException as e:             # noqa: BLE001  (re-raised below)
            outcome, exc = ("error", repr(e)), e
    if size > 1:
        outcome = comm.bcast(outcome, root=0)
    kind, value = outcome
    if kind == "ok":
        return value
    if kind == "exit":
        raise SystemExit(value)
    if exc is not None:
        raise exc
    raise SystemExit(1)
