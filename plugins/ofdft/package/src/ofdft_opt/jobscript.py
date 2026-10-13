"""`ofdft jobscript` -- write a Slurm job script (job.sh) + run.json for one run folder.

The script is only written, never submitted: review it, then `sbatch job.sh` from the run folder.

  ofdft jobscript --task relax   --poscar Al.vasp        -- --relax full --fmax 0.01
  ofdft jobscript --task sscurve --poscar Al_CONTCAR     -- --axis z --strain-step 0.005 --strain-max 0.05
  ofdft jobscript --task relax   --poscar al_vacancy_255.vasp --ntasks 4 --mem 8G --time 00:20:00 -- --steps 2

Everything after `--` is passed to `ofdft <task>` verbatim. The generated script (modelled on a
production script used for 13,140-atom runs) activates the conda env of --python (or uses absolute
paths when that python is not in a conda env), pins PYTHONPATH to --libdir, runs `ofdft doctor --brief`,
checks that mpi4py sees exactly one MPI job of N ranks, launches `mpirun -np N python -m mpi4py -m
ofdft_opt <task> ...` (an exception on one rank aborts the job instead of hanging) and continues
automatically when resubmitted (relax: --restart when <prefix>_CONTCAR exists; sscurve: from its csv).
"""
import argparse
import os
import re
import shlex
import shutil
import sys

from . import LIB_DIR, __version__, runinfo

TASKS = ("relax", "sscurve")
DEFAULT_TIME = {"relax": "3-00:00:00", "sscurve": "10-00:00:00"}
MEM_MIN_GB = 8
MEM_GB_PER_10K_ATOMS = 65      # 0.0065 GB/atom: measured 38-43 GB for 13,140 atoms at 64 ranks (~2x headroom)
NODE_CORES_MAX = 128           # largest node of the cluster this was tested on (warn above)
LOCK_FILE = ".ofdft_job.lock"
NAME_BAD = re.compile(r"[^A-Za-z0-9._-]+")
TIME_RE = re.compile(r"^(\d+|\d+-\d{1,2}(:\d{2}){0,2}|\d{1,2}(:\d{2}){1,2}|UNLIMITED|INFINITE)$", re.I)
MEM_RE = re.compile(r"^(\d+)\s*([GgMm])?[Bb]?$")
PARTITION_RE = re.compile(r"^[A-Za-z0-9._,-]+$")


def auto_ntasks(natoms):
    """MPI ranks by system size (one node)."""
    if natoms < 500:
        return 8
    if natoms < 2000:
        return 16
    if natoms < 6000:
        return 32
    return 64


def auto_mem_gb(natoms):
    """max(8, ceil(0.0065 GB x atoms)), integer arithmetic (no float rounding surprises)."""
    return max(MEM_MIN_GB, -(-MEM_GB_PER_10K_ATOMS * natoms // 10000))


def sanitize_name(s):
    return NAME_BAD.sub("_", s).strip("_") or "ofdft"


def split_passthrough(argv):
    """argv -> (own args, task args after the first '--')."""
    argv = list(argv)
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def _is_abs(path):
    return os.path.isabs(path) or path.startswith("/")   # POSIX path given on Windows (tests)


def task_prefix(task, poscar, task_args):
    """Output prefix the task will use (mirrors the defaults of `ofdft relax` / `ofdft sscurve`)."""
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--prefix", default=None)
    p.add_argument("--axis", default="z")
    p.add_argument("--mode", default="tension")
    p.add_argument("--strain-step", type=float, default=0.005)
    p.add_argument("--strains", type=float, nargs="+", default=None)
    ns, _ = p.parse_known_args(task_args)
    if ns.prefix:
        return ns.prefix
    stem = os.path.splitext(os.path.basename(poscar))[0]
    if task == "relax":
        return stem
    comp = (min(ns.strains) < 0) if ns.strains else (ns.strain_step < 0)   # same rule as `ofdft sscurve`
    return f"{stem}_ss_{ns.axis}_{ns.mode}" + ("_comp" if comp else "")


def mixed_sign_strains(task_args):
    """True if a --strains list in the task options mixes tension and compression (sscurve refuses it)."""
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--strains", type=float, nargs="+", default=None)
    try:
        ns, _ = p.parse_known_args(task_args)
    except SystemExit:
        return False
    return bool(ns.strains) and min(ns.strains) < 0 < max(ns.strains)


def has_control_chars(s):
    """Newlines, tabs and other control characters (they would break out of the job script's comments)."""
    return any(ord(c) < 32 or ord(c) == 127 for c in str(s))


def find_conda_sh(python):
    """(env_prefix, conda_sh) for a python inside a conda env, else (env_prefix, None)."""
    prefix = os.path.dirname(os.path.dirname(python))          # <prefix>/bin/python
    cands = []
    parent = os.path.dirname(prefix)
    if os.path.basename(parent) == "envs":                     # <base>/envs/<name>
        cands.append(os.path.dirname(parent))
    cands.append(prefix)                                       # the base env itself
    if os.environ.get("CONDA_EXE") and os.path.isdir(os.path.join(prefix, "conda-meta")):
        cands.append(os.path.dirname(os.path.dirname(os.environ["CONDA_EXE"])))   # env made with -p elsewhere
    for base in cands:
        sh = os.path.join(base, "etc", "profile.d", "conda.sh")
        if os.path.isfile(sh):
            if "\\" not in python:                              # keep the separator style of --python
                sh = sh.replace("\\", "/")
            return prefix, sh
    return prefix, None


def parse_args(argv=None):
    own, task_args = split_passthrough(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        prog="ofdft jobscript",
        usage="ofdft jobscript --task {relax,sscurve} --poscar FILE [options] [-- TASK OPTIONS ...]",
        description="Write a Slurm job script for `ofdft relax` or `ofdft sscurve` (does not submit it).",
        epilog="Everything after `--` is passed to `ofdft <task>` verbatim, e.g. "
               "`-- --relax full --fmax 0.01`.")
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--poscar", required=True, metavar="FILE",
                   help="structure file in the run folder (sscurve: the RELAXED reference CONTCAR)")
    p.add_argument("--name", default=None, help="Slurm job name [<task>_<poscar stem>]")
    p.add_argument("--ntasks", default="auto", metavar="N|auto",
                   help="MPI ranks on one node [auto: <500 atoms 8, <2000 16, <6000 32, else 64]")
    p.add_argument("--mem", default="auto", metavar="auto|<GB>G",
                   help="memory per job [auto: max(8, ceil(0.0065 x atoms)) G]")
    p.add_argument("--time", default=None, metavar="D-HH:MM:SS",
                   help="wall time [relax 3-00:00:00, sscurve 10-00:00:00]")
    p.add_argument("--partition", default=None, help="Slurm partition [cluster default]")
    p.add_argument("--python", default=sys.executable,
                   help="python of the environment with dftpy [this python: %(default)s]")
    p.add_argument("--libdir", default=None,
                   help=f"directory put on PYTHONPATH, containing ofdft_opt/ [{LIB_DIR}]")
    p.add_argument("-o", "--output", default="job.sh", help="job script to write [job.sh]")
    bad = [a for a in own + task_args if has_control_chars(a)]
    if bad:
        p.error(f"control characters (newline, tab, ...) are not allowed in arguments: {bad[0]!r}")
    args, extra = p.parse_known_args(own)
    if extra:
        p.error(f"unrecognized arguments: {' '.join(extra)} (options for `ofdft <task>` go after `--`, "
                "e.g. ofdft jobscript --task relax --poscar X.vasp -- --steps 2)")
    args.task_args = task_args
    args.argv = list(sys.argv[1:] if argv is None else argv)

    if args.ntasks != "auto":
        try:
            args.ntasks = int(args.ntasks)
        except ValueError:
            p.error(f"--ntasks must be a positive integer or 'auto', not {args.ntasks!r}")
        if args.ntasks < 1:
            p.error("--ntasks must be >= 1")
    if args.mem != "auto":
        m = MEM_RE.match(args.mem.strip())
        if not m or int(m.group(1)) < 1:
            p.error(f"--mem must be 'auto' or e.g. 40G (GB), not {args.mem!r}")
        args.mem = m.group(1) + (m.group(2) or "G").upper()
    args.time = args.time or DEFAULT_TIME[args.task]
    if not TIME_RE.match(args.time):
        p.error(f"--time must be a Slurm time like 3-00:00:00 or 00:20:00, not {args.time!r}")
    if args.partition is not None and not PARTITION_RE.match(args.partition):
        p.error(f"--partition {args.partition!r}: only letters, digits and . _ , - are allowed")
    if args.name is not None:
        args.name = sanitize_name(args.name)
    return args, p


def _warn(msg):
    print(f"WARNING: {msg}", file=sys.stderr, flush=True)


def _compare_previous(run_json, args, poscar_name, prefix, ntasks, mem):
    """Regenerating job.sh in a used run folder: warn when the new script would not continue the run that
    the previous run.json describes (other prefix = a NEW run on resubmit), and list changed settings."""
    old = runinfo.read_json(run_json)
    if not isinstance(old, dict):
        return
    if old.get("task") != args.task or old.get("prefix") != prefix:
        _warn(f"run.json in this folder describes `ofdft {old.get('task')} {old.get('poscar')}` with prefix "
              f"{old.get('prefix')}; the new job.sh runs `ofdft {args.task} {poscar_name}` with prefix {prefix}: "
              f"submitting it starts a NEW run instead of continuing that one. To continue it, repeat ALL its task "
              f"options after `--`: {' '.join(old.get('args') or []) or '(none)'}")
    changes = []
    if old.get("args") is not None and list(old.get("args")) != list(args.task_args):
        changes.append(f"task options {' '.join(old['args']) or '(none)'} -> {' '.join(args.task_args) or '(none)'}")
    for key, new in (("ntasks", ntasks), ("mem", mem), ("time", args.time), ("partition", args.partition)):
        if key in old and old[key] != new:
            changes.append(f"{key} {old[key]} -> {new}")
    if changes:
        print("note: settings changed from the previous run.json: " + "; ".join(changes), file=sys.stderr,
              flush=True)


def build_script(r):
    """Return the job script text for the resolved settings dict r."""
    q = shlex.quote
    mpi = r["ntasks"] > 1
    task_args = " ".join(q(a) for a in [r["task"], r["poscar"]] + r["task_args"])
    L = ["#!/bin/bash",
         f"#SBATCH --job-name={r['name']}",
         "#SBATCH --nodes=1",
         f"#SBATCH --ntasks-per-node={r['ntasks']}",
         "#SBATCH --cpus-per-task=1",
         f"#SBATCH --mem={r['mem']}",
         f"#SBATCH --time={r['time']}"]
    if r["partition"]:
        L.append(f"#SBATCH --partition={r['partition']}")
    L += ["#SBATCH --output=%x_%j.out",
          "#",
          f"# Generated by ofdft-opt {__version__} on {r['created']} with:",
          f"#   {r['command_line']}",
          f"# Task: ofdft {r['task']} {r['poscar']} ({r['natoms']} atoms), {r['ntasks']} MPI rank(s), "
          f"output prefix {r['prefix']}",
          "# Submit, and resubmit after a time limit / node failure / clean stop, from the run folder",
          f"# (the folder containing {r['poscar']}):",
          f"#   sbatch {r['script_rel']}",
          ("#   relax continues the SAME structure: --restart is added when "
           f"{r['prefix']}_CONTCAR exists" if r["task"] == "relax" else
           "#   sscurve skips the strain points already in its csv and continues an unfinished point"),
          "# Stop cleanly (the current step finishes and all files are written):",
          f"#   touch {r['prefix']}.STOP        (or: touch STOP)",
          "#   delete the STOP file before resubmitting (this script refuses to start while one exists).",
          f"# Progress: {r['prefix']}.log; result/convergence: {r['prefix']}_status.json; "
          f"job output: {r['name']}_<jobid>.out",
          f"# Do not edit on Windows (CRLF breaks bash; fix with: sed -i 's/\\r$//' {r['script_name']});",
          "# better regenerate with `ofdft jobscript` (settings are in run.json).",
          "",
          f"PY={q(r['python'])}",
          f"POSCAR={q(r['poscar'])}",
          f"PREFIX={q(r['prefix'])}",
          f"OFDFT_VERSION={q(__version__)}",
          "",
          "# ---------------- environment ----------------",
          'if [ -n "$SLURM_SUBMIT_DIR" ]; then cd "$SLURM_SUBMIT_DIR" || exit 1; fi']
    if r["conda_sh"]:
        L += [f"# conda env {r['env_prefix']}",
              f"CONDA_SH={q(r['conda_sh'])}",
              f"CONDA_ENV={q(r['env_prefix'])}",
              'if [ -f "$CONDA_SH" ]; then',
              '    source "$CONDA_SH"                # makes \'conda activate\' work in a batch shell',
              "else",
              '    echo "ERROR: $CONDA_SH not found (conda moved? regenerate job.sh)" >&2; exit 1',
              "fi",
              'conda activate "$CONDA_ENV" || { echo "ERROR: conda activate $CONDA_ENV failed" >&2; exit 1; }',
              'if [ "$(command -v python)" != "$CONDA_PREFIX/bin/python" ]; then',
              '    echo "ERROR: python is $(command -v python), expected $CONDA_PREFIX/bin/python" >&2; exit 1',
              "fi"]
    else:
        L += [f"# {r['python']} is not inside a conda env with an etc/profile.d/conda.sh above it:",
              "# no activation; the python and its directory are used by absolute path only.",
              f"export PATH={q(r['bindir'])}:\"$PATH\""]
    L += ["unset PYTHONHOME",
          "export PYTHONNOUSERSITE=1             # ignore ~/.local packages that could shadow the env's dftpy",
          f"export PYTHONPATH={q(r['libdir'])}   # exactly this ofdft-opt version (old versions stay installed)",
          "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # 1 thread per MPI rank",
          "",
          "# ---------------- sanity checks (seconds) ----------------",
          'echo "== job ${SLURM_JOB_ID:-none} host $(hostname) ntasks ${SLURM_NTASKS:-1} dir $PWD"',
          'echo "== python $PY"',
          '[ -x "$PY" ] || { echo "ERROR: $PY not found or not executable" >&2; exit 1; }',
          'if [ ! -f "$POSCAR" ]; then',
          '    echo "ERROR: $POSCAR not found in $PWD: submit from the run folder (cd <run folder>; sbatch job.sh)" >&2',
          "    exit 1",
          "fi",
          'for f in "$PREFIX.STOP" STOP; do',
          '    if [ -e "$f" ]; then',
          '        echo "ERROR: stop file $PWD/$f exists (left from an earlier stop): delete it, then resubmit" >&2',
          "        exit 1",
          "    fi",
          "done",
          "# one job at a time in this folder (two jobs would overwrite each other's files)",
          "if command -v flock >/dev/null 2>&1; then",
          f"    exec 9<>{q(LOCK_FILE)}",
          "    flock -n 9; LRC=$?",
          '    if [ "$LRC" -eq 1 ]; then',
          f'        echo "ERROR: another job is already running in $PWD (lock {LOCK_FILE})" >&2; exit 1',
          '    elif [ "$LRC" -ne 0 ]; then',
          '        echo "WARNING: could not lock (flock rc $LRC); continuing without the folder lock" >&2',
          "    fi",
          "fi",
          "\"$PY\" -c 'import sys, ofdft_opt as o; print(\"== ofdft-opt\", o.__version__, o.PKG_DIR); "
          "sys.exit(o.__version__ != sys.argv[1])' \"$OFDFT_VERSION\" \\",
          '    || { echo "ERROR: ofdft-opt $OFDFT_VERSION is not importable from $PYTHONPATH (install it or regenerate job.sh)" >&2; exit 1; }',
          '"$PY" -m ofdft_opt doctor --brief || { echo "ERROR: ofdft doctor reported a FAIL (see above)" >&2; exit 1; }',
          ""]
    if mpi:
        L += ["# ---------------- MPI wire-up test (seconds) ----------------",
              f': "${{SLURM_NTASKS:={r["ntasks"]}}}"          # set by Slurm; default when run outside a job',
              f"MPIRUN={q(r['mpirun'])}       # the env's own mpirun (mpi4py was built against this MPI)",
              'if [ ! -x "$MPIRUN" ]; then',
              '    MPIRUN="$(command -v mpirun)" || { echo "ERROR: no mpirun next to $PY or on PATH" >&2; exit 1; }',
              "fi",
              'echo "== mpirun $MPIRUN"',
              'OUT=$("$MPIRUN" -np "$SLURM_NTASKS" "$PY" -m ofdft_opt doctor --mpi)',
              'echo "== MPI test (expect exactly one line starting with $SLURM_NTASKS):"; echo "$OUT"',
              'if [ "$(echo "$OUT" | awk \'NF{print $1}\' | sort -u)" != "$SLURM_NTASKS" ]; then',
              '    echo "ERROR: mpi4py does not see one MPI job of $SLURM_NTASKS ranks under $MPIRUN" >&2',
              "    echo \"       many '1 | ...' lines = launcher/MPI mismatch; empty = mpi4py/mpi4py_fft missing\" >&2",
              "    exit 1",
              "fi",
              "",
              "# ---------------- run ----------------",
              "# -m mpi4py: an exception on one rank aborts the whole job instead of hanging until --time",
              f'CMD=("$MPIRUN" -np "$SLURM_NTASKS" "$PY" -m mpi4py -m ofdft_opt {task_args})']
    else:
        L += ["# single process (generated with --ntasks 1): no MPI launcher",
              'if [ "${SLURM_NTASKS:-1}" -gt 1 ]; then',
              '    echo "WARNING: $SLURM_NTASKS tasks allocated but this script runs 1 process (regenerate with --ntasks)" >&2',
              "fi",
              "",
              "# ---------------- run ----------------",
              f'CMD=("$PY" -m ofdft_opt {task_args})']
    if r["task"] == "relax":
        L += ['if [ -s "${PREFIX}_CONTCAR" ]; then CMD+=(--restart); fi    # continue the SAME structure']
    else:
        L += ["# (no restart flag: sscurve continues automatically from its csv)"]
    L += ['echo "== $(date) ${CMD[*]}"',
          '"${CMD[@]}"',
          'RC=$?; echo "== $(date) exit code $RC"; exit $RC',
          ""]
    return "\n".join(L)


def _write_text_atomic(path, text, mode=None):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    if mode is not None:
        try:
            os.chmod(tmp, mode)
        except OSError:
            pass
    os.replace(tmp, path)


def main(argv=None):
    args, p = parse_args(argv)
    out = os.path.abspath(os.path.expanduser(args.output))
    run_dir = os.path.dirname(out)
    if os.path.isdir(out):
        p.error(f"-o {args.output} is a directory; give a file name such as {args.output}/job.sh")
    if not os.path.isdir(run_dir):
        p.error(f"output folder {run_dir} does not exist")

    poscar = os.path.abspath(os.path.expanduser(args.poscar))
    if not os.path.isfile(poscar):
        p.error(f"--poscar {args.poscar}: file not found")
    # the run folder = where the job is submitted from (the job cd's to $SLURM_SUBMIT_DIR and opens the
    # POSCAR by its bare name): the POSCAR must be in the current folder or next to the job script
    submit_dir = os.path.dirname(poscar)
    if not (os.path.samefile(submit_dir, os.getcwd()) or os.path.samefile(submit_dir, run_dir)):
        p.error(f"--poscar {args.poscar} must be in the run folder (the current folder or the folder of the "
                f"job script): copy it there first")
    try:
        from .core import read_poscar
        natoms = len(read_poscar(poscar)["symbols"])
    except SystemExit:
        raise
    except Exception as exc:
        p.error(f"cannot read {args.poscar} as a VASP5 POSCAR: {exc}")
    if natoms < 1:
        p.error(f"{args.poscar}: no atoms")

    python = os.path.expanduser(args.python)
    if not _is_abs(python):
        found = shutil.which(python)
        if not found:
            p.error(f"--python {args.python}: give the absolute path of the environment's python")
        python = os.path.abspath(found)
    if not os.path.exists(python):
        _warn(f"python {python} does not exist on this machine (the job will fail unless it exists on the nodes)")
    libdir = os.path.expanduser(args.libdir) if args.libdir else LIB_DIR
    if not _is_abs(libdir):
        libdir = os.path.abspath(libdir)
    if not os.path.isfile(os.path.join(libdir, "ofdft_opt", "__init__.py")):
        _warn(f"{libdir} does not contain ofdft_opt/ (the job's version check will fail)")

    ntasks_auto = args.ntasks == "auto"
    ntasks = auto_ntasks(natoms) if ntasks_auto else args.ntasks
    if ntasks > NODE_CORES_MAX:
        _warn(f"{ntasks} ranks on ONE node: more than the {NODE_CORES_MAX} cores of the largest node tested")
    mem_auto = args.mem == "auto"
    mem = f"{auto_mem_gb(natoms)}G" if mem_auto else args.mem
    if re.search(r"(^| )--engine[= ]emt( |$)", " ".join(args.task_args)):
        _warn("--engine emt in the task options: EMT is for testing only, not OF-DFT results")
    if "--overwrite" in args.task_args and args.task == "relax":
        _warn("--overwrite in the task options: a resubmitted job would start over instead of continuing")
    if args.task == "sscurve" and mixed_sign_strains(args.task_args):
        p.error("--strains mixes tension (> 0) and compression (< 0) strains: run them as two curves "
                "(two run folders)")

    poscar_name = os.path.basename(poscar)
    if args.task_args and args.task_args[0] in (args.poscar, poscar_name):
        p.error("do not repeat the structure after `--`: --poscar is passed to the task automatically")
    stem = os.path.splitext(poscar_name)[0]
    name = args.name or sanitize_name(f"{args.task}_{stem}")
    prefix = task_prefix(args.task, poscar_name, args.task_args)
    _compare_previous(os.path.join(run_dir, "run.json"), args, poscar_name, prefix, ntasks, mem)
    env_prefix, conda_sh = find_conda_sh(python)
    bindir = os.path.dirname(python)
    sep = "\\" if "\\" in python else "/"
    created = runinfo.now_iso()
    command_line = "ofdft jobscript " + " ".join(shlex.quote(a) for a in args.argv)
    try:
        script_rel = os.path.relpath(out, submit_dir).replace(os.sep, "/")
    except ValueError:                                         # other drive (Windows)
        script_rel = out
    r = dict(task=args.task, poscar=poscar_name, natoms=natoms, name=name, ntasks=ntasks, mem=mem,
             time=args.time, partition=args.partition, python=python, libdir=libdir, prefix=prefix,
             task_args=list(args.task_args), env_prefix=env_prefix, conda_sh=conda_sh, bindir=bindir,
             mpirun=bindir.rstrip("/\\") + sep + "mpirun", created=created,
             command_line=command_line.replace("\n", " "), script_name=os.path.basename(out),
             script_rel=script_rel)
    _write_text_atomic(out, build_script(r), mode=0o755)

    run_json = os.path.join(run_dir, "run.json")
    runinfo.write_json_atomic(run_json, {
        "version": __version__, "task": args.task, "poscar": poscar_name,
        "poscar_sha256": runinfo.sha256_text(poscar), "atoms": natoms, "prefix": prefix,
        "name": name, "ntasks": ntasks, "mem": mem, "time": args.time, "partition": args.partition,
        "args": list(args.task_args), "python": python, "libdir": libdir,
        "conda_env": env_prefix if conda_sh else None, "jobscript": os.path.basename(out),
        "command": command_line, "created": created})

    launch = (f"mpirun -np {ntasks} {python} -m mpi4py" if ntasks > 1 else python)
    restart = (f"  [+ --restart when {prefix}_CONTCAR exists]" if args.task == "relax" else
               "  [continues from the csv when resubmitted]")
    print(f"wrote {os.path.basename(out)} and run.json in {run_dir}")
    print(f"  structure  {poscar_name}: {natoms} atoms")
    print(f"  resources  1 node, {ntasks} MPI rank(s){' (auto)' if ntasks_auto else ''}, mem {mem}"
          f"{' (auto)' if mem_auto else ''}, time {args.time}, partition {args.partition or '(cluster default)'}")
    print(f"  env        {'conda ' + env_prefix if conda_sh else 'absolute paths (no conda activation)'}; "
          f"PYTHONPATH={libdir}")
    print(f"  command    {launch} -m ofdft_opt {' '.join(shlex.quote(a) for a in [args.task, poscar_name] + args.task_args)}"
          f"{restart}")
    print(f"  outputs    prefix {prefix} (status: {prefix}_status.json); stop: touch {prefix}.STOP")
    print(f"  submit     cd {submit_dir} && sbatch {script_rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
