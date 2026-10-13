"""`ofdft <command> [args]` -- command dispatcher (also `python -m ofdft_opt <command>`)."""
import importlib
import sys

from . import PKG_DIR, __version__

COMMANDS = {
    "relax":     ("ofdft_opt.relax",     "relax atomic positions and/or the cell (L-BFGS, DFTpy OF-DFT)"),
    "sscurve":   ("ofdft_opt.sscurve",   "quasi-static stress-strain curve from a relaxed structure"),
    "jobscript": ("ofdft_opt.jobscript", "write a Slurm job script for relax/sscurve (does not submit)"),
    "doctor":    ("ofdft_opt.doctor",    "check the Python environment (dftpy, mpi4py, PP files, ...)"),
    "pp":        ("ofdft_opt.pp",        "pseudopotentials: `pp list`, `pp fetch [--dest DIR]`"),
    "plot":      ("ofdft_opt.plot",      "plot stress-strain csv file(s) to PNG"),
    "version":   (None,                  "print the ofdft-opt version and location"),
}


def usage():
    lines = [f"usage: ofdft <command> [options]      (ofdft-opt {__version__})", "", "commands:"]
    lines += [f"  {name:10s} {text}" for name, (_, text) in COMMANDS.items()]
    lines += ["", "`ofdft <command> -h` shows the options of a command.",
              "MPI: mpirun -np N python -m mpi4py -m ofdft_opt <command> ..."]
    return "\n".join(lines)


def version_string():
    return f"ofdft-opt {__version__} ({PKG_DIR})"


def main(argv=None):
    """Dispatch to <command>.main(remaining argv); returns its exit code (0/None = ok)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(usage(), file=sys.stderr)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd in ("-h", "--help", "help"):
        print(usage())
        return 0
    if cmd in ("version", "--version", "-V"):
        print(version_string())
        return 0
    if cmd not in COMMANDS:
        print(f"ofdft: unknown command '{cmd}'\n\n{usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(COMMANDS[cmd][0])
    return module.main(rest)


if __name__ == "__main__":
    sys.exit(main())
