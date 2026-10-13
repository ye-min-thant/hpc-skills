"""ofdft relax -- simultaneous relaxation of atomic positions and cell with DFTpy (OF-DFT),
own preconditioned L-BFGS on [reference positions, strain] (no ASE optimizer), fixed FFT grid
(re-chosen and continued when the cell drifted > --regrid-tol after convergence).
Converged when max|F| < --fmax (eV/A) AND max|sigma + p| < --smax (GPa) ('positions': forces only).

Usage:
  ofdft relax POSCAR --relax full                       (PP: registry default, e.g. Al -> al.gga.recpot)
  mpirun -np N python -m mpi4py -m ofdft_opt relax POSCAR --relax full        (MPI)
  ofdft relax POSCAR --restart          continue from <prefix>_CONTCAR (physics options must match)
  touch <prefix>.STOP   (or STOP)       finish the current step, write files, exit
Outputs (prefix = POSCAR file name without extension): <prefix>.log (one line per step),
<prefix>_CONTCAR (every step), <prefix>.xyz (every --traj-every steps), <prefix>_scf.log (DFTpy SCF),
<prefix>_params.json (options, PP/POSCAR checksums), <prefix>_status.json (authoritative state:
running / converged / not_converged / error).
Exit code 0 also when not converged (the status file tells); 2 = refused / bad input; 1 = error.
"""
import argparse
import json
import os
import socket
import sys
import time

import numpy as np

from . import PKG_DIR, XC_ORIGIN, __version__, _shim, xc_description
from . import pp as ppmod
from .core import (VOIGT, EV_A3_TO_GPA, DFTpyEngine, EMTEngine, ExpPrecon, Relaxer, append_xyz,
                   init_mpi, read_poscar, rss_gb, write_poscar)
from .runinfo import (die, gate, install_term_handler, now_iso, read_json, sha256_text, stop_requested,
                      write_json_atomic)

# options that change the physics: a --restart with different values is refused
PHYSICS_KEYS = ["engine", "xc", "kedf", "ecut", "nr", "relax", "pressure", "pp_sha256"]


def parse_args(argv=None):
    # allow_abbrev=False: `ofdft jobscript` must see the same options as the task (e.g. --prefix)
    p = argparse.ArgumentParser(prog="ofdft relax", description=__doc__, allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("poscar", help="initial structure (VASP5 POSCAR/CONTCAR)")
    p.add_argument("--pp", action="append", default=[], metavar="EL=FILE",
                   help="local pseudopotential per element (repeatable). FILE: a path, or a bare name searched in "
                        "--pppath, $OFDFT_PPPATH, ~/.ofdft/pp, the current directory [registry default for --xc: "
                        "Al -> al.gga.recpot (PBE) / al.lda.recpot (LDA)]")
    p.add_argument("--pppath", default=None, metavar="DIR", help="directory searched first for PP files")
    p.add_argument("--xc", choices=["pbe", "lda"], default="pbe", help="[pbe]")
    p.add_argument("--kedf", default="WT", help="[WT]")
    p.add_argument("--ecut", type=float, default=600.0, help="grid cutoff, eV [600]")
    p.add_argument("--nr", type=int, nargs=3, default=None, help="fix FFT grid explicitly (else from ecut)")
    p.add_argument("--econv", type=float, default=None,
                   help="density convergence, Ha/atom [1e-7; 1e-8 if --fmax < 0.01]")
    p.add_argument("--econv-loose", type=float, default=0.0,
                   help="looser density convergence while max|F| > econv-switch * fmax, Ha/atom "
                        "[0 = off: always use --econv]")
    p.add_argument("--econv-switch", type=float, default=5.0, help="see --econv-loose [5]")
    p.add_argument("--precon", choices=["exp", "none"], default="none",
                   help="Exp preconditioner for the atomic positions [none; exp did not pay off for "
                        "large OF-DFT polycrystals]")
    p.add_argument("--relax", choices=["positions", "iso", "full", "cell"], default="full",
                   help="positions only | positions + volume | positions + full cell ('cell' = 'full') [full]")
    p.add_argument("--fmax", type=float, default=0.01, help="force criterion, eV/A [0.01]")
    p.add_argument("--smax", type=float, default=0.05, help="stress criterion, GPa [0.05]")
    p.add_argument("--pressure", type=float, default=0.0, help="target hydrostatic pressure, GPa [0]")
    p.add_argument("--steps", type=int, default=1000, help="max optimizer steps [1000]")
    p.add_argument("--maxstep", type=float, default=0.2, help="max atomic step, A [0.2]")
    p.add_argument("--maxstrain", type=float, default=0.01, help="max strain change per step [0.01]")
    p.add_argument("--memory", type=int, default=20, help="L-BFGS memory [20]")
    p.add_argument("--alpha", type=float, default=10.0, help="initial Hessian guess, eV/A^2 [10]")
    p.add_argument("--stiffness", type=float, default=100.0,
                   help="elastic stiffness used to scale strain, GPa [100; ~C11 of the material]")
    p.add_argument("--regrid-tol", type=float, default=0.02,
                   help="after convergence, re-choose the grid if spacing drifted by more than this "
                        "fraction and continue [0.02; 0 = never]")
    p.add_argument("--prefix", default=None,
                   help="output prefix [POSCAR file name without extension, so different structures never "
                        "restart from each other's files]")
    p.add_argument("--traj-every", type=int, default=10, help="write trajectory frame every N steps [10]")
    p.add_argument("--engine", choices=["dftpy", "emt"], default="dftpy",
                   help="dftpy = OF-DFT [default]; emt = ASE EMT classical potential, TESTING / DRY RUNS ONLY")
    p.add_argument("--restart", action="store_true",
                   help="continue from <prefix>_CONTCAR if present (physics options must match <prefix>_params.json)")
    p.add_argument("--overwrite", action="store_true",
                   help="start from POSCAR even if <prefix>.log / <prefix>_CONTCAR exist (they are replaced)")
    p.add_argument("--allow-xc-mismatch", action="store_true",
                   help="allow a PP whose name says LDA with --xc pbe (or GGA with --xc lda)")
    p.add_argument("--verbose", action="store_true", help="show DFTpy output")
    return p.parse_args(argv)


def _jsonable(obj):
    return json.loads(json.dumps(obj))


def _prepare(args, contcar, logpath, parpath):
    """Rank 0: every refusal decision and every input read, before anything is written."""
    if args.fmax <= 0 or args.smax <= 0 or args.steps < 0 or args.traj_every < 1:
        die("--fmax and --smax must be > 0, --steps >= 0, --traj-every >= 1")
    if args.econv <= 0:
        die("--econv must be > 0")
    exists = [f for f in (logpath, contcar) if os.path.exists(f)]
    if exists and not args.restart and not args.overwrite:
        die(f"{' and '.join(exists)} already exist: a run with prefix '{args.prefix}' was started here before. "
            f"Use --restart to continue it, --overwrite to start again from {args.poscar} (its files are "
            f"replaced), or another --prefix. Nothing was changed.")
    src = contcar if (args.restart and os.path.isfile(contcar)) else args.poscar
    if not os.path.isfile(src):
        die(f"structure file not found: {src}")
    try:
        s = read_poscar(src)
    except (OSError, ValueError, IndexError) as exc:
        die(f"cannot read {src} as a VASP5 POSCAR: {exc}")
    if args.engine == "dftpy":
        pp_files = ppmod.resolve(args.pp, s["symbols"], args.xc, args.pppath,
                                 allow_xc_mismatch=args.allow_xc_mismatch)
    else:
        pp_files = {}
    params = _jsonable({
        "version": __version__, "engine": args.engine, "xc": args.xc, "kedf": args.kedf, "ecut": args.ecut,
        "nr": None if args.nr is None else [int(v) for v in args.nr], "relax": args.relax,
        "pressure": args.pressure, "pp": {el: os.path.basename(p) for el, p in pp_files.items()},
        "pp_sha256": {el: sha256_text(p) for el, p in pp_files.items()},
        "poscar": os.path.basename(args.poscar),
        "poscar_sha256": sha256_text(args.poscar) if os.path.isfile(args.poscar) else None,
        "fmax": args.fmax, "smax": args.smax, "econv": args.econv})
    notes = []
    if args.restart and os.path.exists(parpath):
        old = read_json(parpath)
        if not isinstance(old, dict):
            die(f"{parpath} is unreadable; cannot check that --restart uses the same physics "
                f"(use --overwrite or another --prefix)")
        diff = {k: (old.get(k), params[k]) for k in PHYSICS_KEYS if old.get(k) != params[k]}
        if diff:
            lines = "\n".join(f"    {k}: was {a!r}, now {b!r}" for k, (a, b) in diff.items())
            die(f"--restart with options that change the physics of run '{args.prefix}' "
                f"({parpath}):\n{lines}\n  Use the original options, a new --prefix, or --overwrite to start again.")
        if old.get("poscar_sha256") and params["poscar_sha256"] and old["poscar_sha256"] != params["poscar_sha256"]:
            if src == contcar:                     # --restart would silently continue the OLD structure
                die(f"{args.poscar} is not the structure that run '{args.prefix}' started from (its content "
                    f"differs from {parpath}): --restart would continue the OLD structure from {contcar}. "
                    f"To relax the new structure use a new run folder or --prefix (or --overwrite, which replaces "
                    f"the old outputs); to continue the old run, restore its original {args.poscar}.")
            notes.append(f"{args.poscar} differs from the POSCAR of the first run (no {contcar}: starting from it)")
        for k in ("fmax", "smax", "econv"):
            if old.get(k) != params[k]:
                notes.append(f"{k} changed on restart: {old.get(k)} -> {params[k]}")
    return dict(s=s, src=src, pp_files=pp_files, params=params, notes=notes)


def _dftpy_info():
    try:
        import dftpy
        return f"dftpy {getattr(dftpy, '__version__', '?')} ({os.path.dirname(dftpy.__file__)})"
    except Exception:
        return "dftpy not importable"


def main(argv=None):
    args = parse_args(argv)
    if args.relax == "cell":
        args.relax = "full"
    if args.prefix is None:
        args.prefix = os.path.splitext(os.path.basename(args.poscar))[0]
    econv_note = None
    if args.econv is None:
        args.econv = 1e-8 if args.fmax < 0.01 else 1e-7
        if args.fmax < 0.01:
            econv_note = f"econv auto-tightened to 1e-8 Ha/atom because fmax {args.fmax} < 0.01 eV/A"
    mp = init_mpi()
    root = mp.comm.rank == 0
    if args.engine == "emt" and mp.comm.size > 1:
        die("--engine emt is serial only (testing / dry runs): run it without mpirun")
    contcar = f"{args.prefix}_CONTCAR"
    logpath, parpath, statpath = f"{args.prefix}.log", f"{args.prefix}_params.json", f"{args.prefix}_status.json"
    setup = gate(mp.comm, lambda: _prepare(args, contcar, logpath, parpath))
    params = setup["params"]

    # ---- all checks passed (identically on every rank): from here on files are written ----
    out = sys.stdout
    logf = open(logpath, "a" if args.restart else "w") if root else None

    def log(msg=""):
        if root:
            print(msg, file=out, flush=True)
            print(msg, file=logf, flush=True)

    smax_used = args.smax if args.relax != "positions" else None
    status_base = {"version": __version__, "engine": args.engine, "prefix": args.prefix, "relax": args.relax,
                   "fmax": args.fmax, "smax": smax_used, "pp_sha256": params["pp_sha256"], "contcar": contcar,
                   "restart": bool(args.restart), "started": now_iso(), "mpi_ranks": mp.comm.size,
                   "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}

    def status(**kw):                              # rank 0, atomic; "state" first for readability
        if root:
            write_json_atomic(statpath, {"state": kw.pop("state"), **status_base, **kw})

    if root:
        write_json_atomic(parpath, params)
        status(state="running", steps=0, steps_max=args.steps, updated=now_iso())
    restore_term = install_term_handler() if root else (lambda: None)
    t0 = time.time()
    state = {"E": None, "step": 0, "stop": None, "engine": None}
    try:
        _run(args, mp, root, setup, log, status, state, econv_note, statpath, logpath)
    except BaseException as exc:
        if root:
            if isinstance(exc, SystemExit):
                reason = f"refused or aborted (exit code {exc.code}); see the error message in the job output"
            else:
                reason = f"{type(exc).__name__}: {exc}"
            try:
                status(state="error", reason=reason, steps=state["step"],
                       dftpy_calls=getattr(state.get("engine"), "ncalls", 0),
                       wall_s=round(time.time() - t0, 1), finished=now_iso())
                log(f" ERROR: {reason}")
                log(f" status: {statpath}")
            except Exception:
                pass
        raise
    finally:
        restore_term()
        if root:
            eng = state.get("engine")
            if eng is not None and eng.scf_log is not None:
                eng.scf_log.close()
            logf.close()
    return 0


def _run(args, mp, root, setup, log, status, state, econv_note, statpath, logpath):
    """The relaxation itself (as in dftpy_relax.py); `state` is shared with main for the status file."""
    s, src, params = setup["s"], setup["src"], setup["params"]
    symbols, cell, pos, fixed = s["symbols"], s["cell"], s["positions"], s["fixed"]
    n = len(symbols)
    contcar = f"{args.prefix}_CONTCAR"
    traj = f"{args.prefix}.xyz"
    if root and not args.restart and os.path.exists(traj):
        os.remove(traj)
    if args.engine == "emt":
        engine = EMTEngine(symbols, econv=args.econv, mp=mp)
    else:
        engine = DFTpyEngine(args, symbols, mp, verbose=args.verbose, pp_files=setup["pp_files"])
    state["engine"] = engine
    if root and args.engine == "dftpy":            # live DFTpy SCF output: tail -f <prefix>_scf.log
        engine.scf_log = open(f"{args.prefix}_scf.log", "a" if args.restart else "w", buffering=1)
    smax_used = args.smax if args.relax != "positions" else None

    import scipy
    log("=" * 96)
    log(f" ofdft-opt {__version__}  XC backend {xc_description()}  python {sys.version.split()[0]} ({sys.executable})  "
        f"{_dftpy_info()}  numpy {np.__version__}  scipy {scipy.__version__}  "
        f"dcsrch shim {'active' if _shim.PATCHED else 'not needed'}")
    log(f" ofdft-opt package {PKG_DIR}   pylibxc {XC_ORIGIN}")
    if args.engine == "emt":
        log(" " + "!" * 95)
        log(" !!! ENGINE = EMT (ASE classical potential): TESTING / DRY RUN ONLY -- NOT OF-DFT, NOT PHYSICAL !!!")
        log(" " + "!" * 95)
    log(f" ofdft relax: {src}  ({n} atoms: " +
        ", ".join(f"{el} {symbols.count(el)}" for el in sorted(set(symbols))) + f")   MPI ranks: {mp.comm.size}")
    if args.engine == "emt":
        pp_txt = "none (EMT)"
    else:
        pp_txt = ", ".join(f"{el}: {engine.pp[el]} (sha256 {engine.pp_sha256[el][:12]})" for el in sorted(engine.pp))
    log(f" XC={args.xc.upper()}  KEDF={args.kedf}  ecut={args.ecut} eV  econv={args.econv} Ha/atom"
        + (f" (loose {args.econv_loose} while max|F| > {args.econv_switch * args.fmax:.3g})" if args.econv_loose else "")
        + f"  precon={args.precon}  PP={pp_txt}")
    if econv_note:
        log(f" note: {econv_note}")
    log(f" relax={args.relax}  fmax={args.fmax} eV/A  smax={args.smax} GPa  pressure={args.pressure} GPa  "
        f"fixed components={0 if fixed is None else int(fixed.sum())}  max steps={args.steps}")
    for note in setup["notes"]:
        log(f" restart: {note}")
    if root:
        stale = stop_requested(args.prefix)
        if stale:
            log(f" WARNING: STOP file '{stale}' exists: the run will stop after the first evaluation "
                "(delete it to run normally)")
    log(f" cell lengths {np.round(np.linalg.norm(cell, axis=1), 4)} A   V/atom {abs(np.linalg.det(cell))/n:.4f} A^3")
    log("=" * 96)
    log(f"{'step':>5} {'energy (eV)':>18} {'dE (eV)':>11} {'max|F|':>9} {'max|sig|':>9} "
        f"{'P (GPa)':>8} {'V/atom':>8} {'step(A)':>8} {'ev':>3} {'time (s)':>9} {'mem(GB)':>8}")

    t0 = time.time()

    def max_stress(st):                            # max |sigma + p| over all components, GPa
        if args.relax == "positions":
            return None
        return float(np.abs(st["stress"] * EV_A3_TO_GPA + args.pressure * np.eye(3)).max())

    def on_step(it, st, d, nls):
        P = -np.trace(st["stress"]) / 3.0 * EV_A3_TO_GPA
        dE = 0.0 if state["E"] is None else st["E"] - state["E"]
        dmax = 0.0 if d is None else np.linalg.norm(d[:3 * n].reshape(n, 3), axis=1).max()
        state["E"] = st["E"]
        state["step"] += 0 if it == 0 else 1
        mem = rss_gb()                             # total resident memory over all MPI ranks
        mem = mp.comm.allreduce(mem) if mp.is_mpi else mem
        log(f"{state['step']:5d} {st['E']:18.6f} {dE:11.3e} {st['fmax']:9.5f} {st['smax']:9.5f} "
            f"{P:8.4f} {st['V']/n:8.4f} {dmax:8.4f} {nls:3d} {time.time()-t0:9.1f} {mem:8.2f}")
        if root:
            write_poscar(contcar, symbols, st["cell"], st["R"], fixed,
                         comment=f"dftpy_relax step {state['step']} E={st['E']:.6f} eV")
            if state["step"] % args.traj_every == 0:
                append_xyz(traj, symbols, st["cell"], st["R"], st["forces"], st["E"], st["stress"])
            try:                                   # progress only: never let it stop the run
                status(state="running", steps=state["step"], steps_max=args.steps, energy_eV=st["E"],
                       max_force_eVA=float(st["fmax"]), max_stress_GPa=max_stress(st),
                       dftpy_calls=engine.ncalls, wall_s=round(time.time() - t0, 1), updated=now_iso())
            except OSError:
                pass

    def should_stop():                             # decided on rank 0 so all ranks stop together
        flag = stop_requested(args.prefix) if root else None
        flag = mp.comm.bcast(flag, root=0) if mp.is_mpi else flag
        if flag:
            state["stop"] = flag
        return flag

    remaining = args.steps
    last_fmax = None                               # loose econv at the start, tight after a regrid
    while True:
        rel = Relaxer(engine, cell, pos, symbols, fixed, args.relax, args.fmax,
                      args.smax if args.relax != "positions" else None, args.pressure,
                      alpha=args.alpha, stiffness_gpa=args.stiffness, memory=args.memory,
                      maxstep=args.maxstep, maxstrain=args.maxstrain, log=log,
                      precon=ExpPrecon() if args.precon == "exp" else None,
                      comm=mp.comm if mp.is_mpi else None, econv_tight=args.econv,
                      econv_loose=args.econv_loose or None, econv_switch=args.econv_switch)
        rel.last_fmax = last_fmax
        steps_before = state["step"]
        st, conv = rel.run(remaining, on_step=on_step, should_stop=should_stop)
        last_fmax = st["fmax"]
        remaining -= state["step"] - steps_before
        cell, pos = st["cell"], st["R"]
        drift = engine.spacing_drift(cell)          # relative to the cell the grid was chosen for
        if conv and args.relax != "positions" and args.regrid_tol > 0 and drift > args.regrid_tol and remaining > 0:
            log(f" grid spacing drifted {drift*100:.1f}% (> {args.regrid_tol*100:.1f}%): re-choosing grid and continuing")
            engine.reset_grid()
            s["cell"] = cell.copy()
            state["E"] = None
            continue
        break

    if root:
        write_poscar(contcar, symbols, cell, pos, fixed, comment=f"dftpy_relax final E={st['E']:.6f} eV")
        append_xyz(traj, symbols, cell, pos, st["forces"], st["E"], st["stress"])
    sig = st["stress"] * EV_A3_TO_GPA
    wall = time.time() - t0
    log("=" * 96)
    log(f" {'CONVERGED' if conv else 'NOT converged'} after {state['step']} steps, "
        f"{engine.ncalls} DFTpy calls, wall {wall:.1f} s (DFTpy {engine.time:.1f} s)"
        + (f", {engine.nwarn} density restarts" if engine.nwarn else ""))
    log(f" E = {st['E']:.6f} eV   E/atom = {st['E']/n:.6f} eV   max|F| = {st['fmax']:.5f} eV/A")
    log(f" stress (GPa) xx yy zz yz xz xy = " + " ".join("%.5f" % sig[i, j] for i, j in VOIGT))
    log(f" cell lengths {np.round(np.linalg.norm(cell, axis=1), 5)} A  "
        f"angles {np.round([np.degrees(np.arccos(np.dot(cell[j], cell[k]) / np.linalg.norm(cell[j]) / np.linalg.norm(cell[k]))) for j, k in ((1, 2), (0, 2), (0, 1))], 4)}  "
        f"V/atom {st['V']/n:.5f} A^3   grid {list(map(int, engine.nr))}")
    log(f" structure: {contcar}   trajectory: {traj}   log: {logpath}   SCF output: {args.prefix}_scf.log")
    log(f" status: {statpath}")
    if not conv:
        log(" not converged: rerun with --restart (and/or more --steps)")
    if state["stop"]:                              # last line: which STOP file stopped the run
        log(f" stopped by STOP file '{state['stop']}': delete it before restarting")
    elif root and stop_requested(args.prefix):
        log(f" STOP file '{stop_requested(args.prefix)}' exists: delete it before restarting")
    status(state="converged" if conv else "not_converged",
           reason="converged" if conv else ("stop" if state["stop"] else "steps"),
           steps=state["step"], dftpy_calls=engine.ncalls, energy_eV=st["E"], energy_per_atom_eV=st["E"] / n,
           max_force_eVA=float(st["fmax"]), max_stress_GPa=max_stress(st), fmax=args.fmax, smax=smax_used,
           relax=args.relax, version=__version__, pp_sha256=params["pp_sha256"], contcar=contcar,
           wall_s=round(wall, 1), finished=now_iso())


if __name__ == "__main__":
    sys.exit(main())
