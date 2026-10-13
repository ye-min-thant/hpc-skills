"""ofdft sscurve -- quasi-static stress-strain curve with DFTpy (OF-DFT), using the `ofdft relax`
L-BFGS relaxer (no ASE optimizer).

Start from a RELAXED structure (zero stress, e.g. <name>_CONTCAR of `ofdft relax --relax full`). The
engineering strain along one axis is increased step by step; at every strain point the atoms are relaxed
and, for --mode tension (uniaxial stress), also the other five cell components (their stresses -> 0) with
'rigid grips' along the loading axis (as LAMMPS fix deform): the box length along the axis is exactly
(1 + strain) x the reference, the loading-axis shears are tilts of that axis.
--mode strain (uniaxial strain) keeps the other cell components at the reference.

Reference check: if <ref stem without _CONTCAR>_status.json exists it must say "converged"; at strain 0
the first evaluation must have max|F| <= 2 fmax and all six |sigma| <= 2 smax (a reference relaxed with
looser thresholds than the curve's fails this: relax it to at least the curve's --fmax/--smax). Otherwise
the curve is refused (exit 2) unless --allow-unrelaxed-reference. The force/stress check needs strain 0
in the list, computed from the reference (the log says when it is not applied).

Usage:
  ofdft sscurve relaxed_CONTCAR --axis z --mode tension --strain-step 0.005 --strain-max 0.05
  mpirun -np N python -m mpi4py -m ofdft_opt sscurve ...          (MPI)
Compression: negative --strain-step and --strain-max, or a --strains list with negative values (files
get the suffix _comp); a --strains list must not mix tension and compression.
Output: <prefix>.csv   one line per converged strain point (strain, stresses in GPa (positive =
                       tensile), energy, max|F|, steps, DFTpy calls, time)
        <prefix>_e+0.0100_CONTCAR   relaxed structure at each strain
        <prefix>_e+0.0100_partial_CONTCAR   latest structure of an unfinished point (for restart)
        <prefix>.log, <prefix>_scf.log, <prefix>_params.json,
        <prefix>_status.json   running / complete / stopped / point_not_converged / error / reference_not_relaxed
Restart: run the same command again; converged strains are skipped and an unfinished point continues
from its partial CONTCAR. Changing physics/tolerance options with the same prefix is refused. A curve
started by the old ss_curve.py script continues too (its params file is checked by name and rewritten).
A new curve refuses to replace <prefix>.log/_params.json/_status.json of another run (--overwrite).
Stop: touch <prefix>.STOP (or STOP). Exit code 0 (the status file tells), 2 = refused, 1 = error.
"""
import argparse
import csv
import json
import os
import re
import socket
import sys
import time

import numpy as np

from . import PKG_DIR, XC_ORIGIN, __version__, _shim, xc_description
from . import pp as ppmod
from .core import (DFTpyEngine, EMTEngine, ExpPrecon, Relaxer, VOIGT, EV_A3_TO_GPA, init_mpi,
                   read_poscar, write_poscar, rss_gb)
from .runinfo import (die, gate, install_term_handler, now_iso, read_json, sha256_text, stop_requested,
                      write_json_atomic)

AXIS = {"x": 0, "y": 1, "z": 2}
CSV_COLS = ["strain", "sxx_GPa", "syy_GPa", "szz_GPa", "syz_GPa", "sxz_GPa", "sxy_GPa",
            "energy_eV", "energy_per_atom_eV", "max_force_eVA", "max_free_stress_GPa",
            "steps", "dftpy_calls", "time_s", "converged"]
# options that change the physics or the convergence of the curve (must match on restart)
PARAM_KEYS = ["poscar_sha256", "pp_sha256", "engine", "xc", "kedf", "ecut", "nr", "axis", "mode", "branch",
              "strain_step", "fmax", "smax", "econv", "econv_loose", "econv_switch"]
# keys of a params file written by the original ss_curve.py (no "version" key; "poscar" = absolute
# path, "pp" = list of "EL=FILE") that mean the same in both formats
LEGACY_SHARED_KEYS = ["xc", "kedf", "ecut", "nr", "axis", "mode", "strain_step", "fmax", "smax", "econv",
                      "econv_loose", "econv_switch"]


class ReferenceNotRelaxed(RuntimeError):
    """The reference structure is not relaxed (forces or stresses too large at strain 0)."""


def unwrap_like(Q, Qref, cell0):
    """Shift each atom of Q by lattice vectors of cell0 so it is closest to Qref (undo wrapping)."""
    d = (Q - Qref) @ np.linalg.inv(cell0)
    d -= np.round(d)
    return Qref + d @ cell0


def state_from_structure(cell, R, cell0, Qref):
    """(Q, eps) of a structure, with cell = cell0 (I+eps)^T and R = Q (I+eps)^T (eps not symmetrized,
    so rigid-grips states round-trip exactly)."""
    D = np.linalg.solve(cell0, cell).T - np.eye(3)
    Q = R @ np.linalg.inv(np.eye(3) + D).T
    return unwrap_like(Q, Qref, cell0), D


def strain_list(step, smax):
    if step == 0 or step * smax < 0:
        die("--strain-step and --strain-max must be nonzero and have the same sign")
    n = int(np.floor(smax / step + 1e-9))
    return [round(k * step, 10) + 0.0 for k in range(n + 1)]


def _check_reference(st, tol, comm):
    """Raise ReferenceNotRelaxed (on every rank) if the first evaluation at strain 0 exceeds the limits."""
    fmax_tol, smax_tol = tol
    sig_all = float(np.abs(st["stress"]).max() * EV_A3_TO_GPA)
    bad = None
    over = []                                      # `not <=` also catches NaN
    if not st["fmax"] <= fmax_tol:
        over.append(f"max|F| = {st['fmax']:.4f} eV/A exceeds the limit {fmax_tol:.4g} (2 x fmax)")
    if not sig_all <= smax_tol:
        over.append(f"max|sigma| over all six components = {sig_all:.4f} GPa exceeds the limit {smax_tol:.4g} "
                    f"(2 x smax)")
    if over:
        bad = (f"reference not relaxed: at strain 0 {' and '.join(over)} (max|F| {st['fmax']:.4f} eV/A, "
               f"max|sigma| {sig_all:.4f} GPa). Relax it with `ofdft relax <structure> --relax full --fmax "
               f"{fmax_tol / 2:.4g} --smax {smax_tol / 2:.4g}` or tighter (at least this curve's thresholds; "
               f"e.g. a --restart of that relaxation with these thresholds), or pass --allow-unrelaxed-reference")
    if comm is not None:
        bad = comm.bcast(bad, root=0)
    if bad:
        raise ReferenceNotRelaxed(bad)


def run_curve(engine, cell0, R0, symbols, fixed, opt, log=print, comm=None, done=None, resume=None,
              on_point=None, on_partial=None, should_stop=lambda: False):
    """Strain loop. opt: dict with axis, mode, strains, fmax, smax, steps, precon, predictor,
    keep_memory, regrid_tol, alpha, stiffness, memory, maxstep, maxstrain, econv_tight,
    econv_loose, econv_switch, and optionally ref_check = (fmax_tol eV/A, smax_tol GPa) or None:
    at strain 0 started from the reference, the first evaluation must have max|F| <= fmax_tol and
    all six |sigma| <= smax_tol, else ReferenceNotRelaxed is raised. done: list of (strain, cell, R)
    already converged (restart); resume: {strain: (cell, R)} partially relaxed structures to continue from.
    on_point(e, st, row) after each point (row['converged'] tells the outcome); on_partial(e, st)
    every 10 steps. Returns a list of result rows (dicts)."""
    a = AXIS[opt["axis"]]
    free = [k for k in range(6) if k != a] if opt["mode"] == "tension" else []
    rel = Relaxer(engine, cell0, R0, symbols, fixed, "custom", opt["fmax"],
                  opt["smax"] if free else None, 0.0, free=free, axis=a if free else None,
                  alpha=opt["alpha"], stiffness_gpa=opt["stiffness"], memory=opt["memory"],
                  maxstep=opt["maxstep"], maxstrain=opt["maxstrain"], log=log,
                  precon=ExpPrecon() if opt["precon"] else None, comm=comm,
                  econv_tight=opt.get("econv_tight"), econv_loose=opt.get("econv_loose"),
                  econv_switch=opt.get("econv_switch", 5.0))
    n = len(R0)
    hist = {}                                      # strain -> (strain, Q, eps) of converged points
    for e, cell, R in (done or []):
        Q, eps = state_from_structure(cell, R, cell0, R0)
        hist[round(e, 10)] = (e, Q, eps)
    resume = resume or {}
    rows = []
    for e in opt["strains"]:
        if round(e, 10) in hist:
            continue
        if should_stop():
            break
        # ---- predictor: previous converged points on this loading branch (same sign, smaller |e|) ----
        path = sorted((h for h in hist.values() if h[0] * e >= 0 and abs(h[0]) < abs(e) - 1e-12),
                      key=lambda h: abs(h[0]))
        how = "reference"
        if not path:
            Qp, epsp = np.array(R0, float), np.zeros((3, 3))
        else:
            Qp, epsp = path[-1][1].copy(), path[-1][2].copy()
            how = "previous point"
            if opt["predictor"] == "linear" and len(path) >= 2:
                (e0, Q0, eps0), (e1, Q1, eps1) = path[-2], path[-1]
                dQ = Q1 - unwrap_like(Q0, Q1, cell0)
                if np.linalg.norm(dQ, axis=1).max() < 0.1:   # no slip in the last increment
                    t = (e - e1) / (e1 - e0)
                    Qp, epsp = Q1 + t * dQ, eps1 + t * (eps1 - eps0)
                    how = "linear extrapolation"
        if round(e, 10) in resume:                 # continue an unfinished point
            cell_r, R_r = resume[round(e, 10)]
            Qp, epsp = state_from_structure(cell_r, R_r, cell0, Qp)
            how = "saved CONTCAR (--start or partial)"
        if not free:                               # uniaxial strain: other components stay at 0
            epsp = np.zeros((3, 3))
        epsp[a, a] = e                             # imposed strain along the loading axis
        if free:                                   # rigid grips: row a of F is (1+e) e_a
            epsp[a, [k for k in range(3) if k != a]] = 0.0
        rel.eps_fixed = epsp.copy()
        rel.Q, rel.eps = Qp, epsp.copy()
        rel.last_fmax = 0.0 if not path and e == 0 else None   # new strain: start with loose econv
        if not opt["keep_memory"]:
            rel.S, rel.Y = [], []
        cell_new = cell0 @ (np.eye(3) + epsp).T
        if hasattr(engine, "spacing_drift") and opt["regrid_tol"] > 0 and \
                engine.spacing_drift(cell_new) > opt["regrid_tol"]:
            log(f" strain {e:+.4f}: FFT grid re-chosen (spacing drift > {opt['regrid_tol']*100:.1f}%)")
            engine.reset_grid()
        log(f" strain {e:+.4f}: start from {how}")
        ref_check = opt.get("ref_check") if (round(e, 10) == 0 and how == "reference") else None
        # ---- relax at this strain ----
        calls0, t0 = getattr(engine, "ncalls", 0), time.time()
        info = {"steps": 0}

        def step_cb(it, st, d, nls):
            info["steps"] = it
            log(f"   e={e:+.4f} step {it:4d}  E={st['E']:.6f}  max|F|={st['fmax']:.5f}  "
                f"max|sig_free|={st['smax']:.5f}  sig_load={st['stress'][a, a]*EV_A3_TO_GPA:+.4f} GPa"
                f"  econv={st.get('econv')}")
            if ref_check is not None and it == 0:
                _check_reference(st, ref_check, comm)
            if on_partial and it > 0 and it % 10 == 0:
                on_partial(e, st)
        st, conv = rel.run(opt["steps"], on_step=step_cb, should_stop=should_stop)
        sig = st["stress"] * EV_A3_TO_GPA
        row = {"strain": e, **{c: float(sig[VOIGT[k]]) for k, c in enumerate(CSV_COLS[1:7])},
               "energy_eV": st["E"], "energy_per_atom_eV": st["E"] / n, "max_force_eVA": st["fmax"],
               "max_free_stress_GPa": st["smax"] if free else 0.0, "steps": info["steps"],
               "dftpy_calls": getattr(engine, "ncalls", 0) - calls0, "time_s": time.time() - t0,
               "converged": bool(conv)}
        rows.append(row)
        log(f" strain {e:+.4f}: sig_load {row[CSV_COLS[1 + a]]:+.4f} GPa  max|F| {st['fmax']:.4f}  "
            f"{'converged' if conv else 'NOT converged'} in {info['steps']} steps / {row['dftpy_calls']} calls, "
            f"{row['time_s']:.0f} s")
        if on_point:
            on_point(e, st, row)
        if not conv:
            log(" point not converged: stopping the curve (rerun to continue from its partial CONTCAR, "
                "maybe with more --steps)")
            break
        hist[round(e, 10)] = (e, rel.Q.copy(), rel.eps.copy())
    return rows


def parse_args(argv=None):
    # allow_abbrev=False: `ofdft jobscript` must see the same options as the task (e.g. --prefix)
    p = argparse.ArgumentParser(prog="ofdft sscurve", description=__doc__, allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("poscar", help="RELAXED reference structure (zero stress), VASP5 POSCAR/CONTCAR")
    p.add_argument("--pp", action="append", default=[], metavar="EL=FILE",
                   help="local PP per element (repeatable); FILE: a path or a bare name searched in --pppath, "
                        "$OFDFT_PPPATH, ~/.ofdft/pp, the current directory [registry default for --xc]")
    p.add_argument("--pppath", default=None, metavar="DIR", help="directory searched first for PP files")
    p.add_argument("--xc", choices=["pbe", "lda"], default="pbe", help="[pbe]")
    p.add_argument("--kedf", default="WT", help="[WT]")
    p.add_argument("--ecut", type=float, default=600.0, help="eV [600]")
    p.add_argument("--nr", type=int, nargs=3, default=None, help="fix the FFT grid (else from ecut)")
    p.add_argument("--axis", choices=list(AXIS), default="z", help="loading axis [z]")
    p.add_argument("--mode", choices=["tension", "strain"], default="tension",
                   help="tension = uniaxial stress (other cell components relaxed); strain = uniaxial "
                        "strain (other cell components fixed) [tension]")
    p.add_argument("--strain-step", type=float, default=0.005, help="engineering strain increment [0.005]")
    p.add_argument("--strain-max", type=float, default=0.05, help="final strain [0.05]")
    p.add_argument("--strains", type=float, nargs="+", default=None,
                   help="explicit list of strains instead of --strain-step/--strain-max (e.g. to re-check "
                        "single points with tighter thresholds, together with --start)")
    p.add_argument("--start", default=None,
                   help="CONTCAR to start the first strain point from (e.g. <prefix>_e+0.0350_CONTCAR)")
    p.add_argument("--fmax", type=float, default=0.01, help="force criterion per point, eV/A [0.01]")
    p.add_argument("--smax", type=float, default=0.05, help="criterion on the relaxed stresses, GPa [0.05]")
    p.add_argument("--steps", type=int, default=1000, help="max optimizer steps per strain point [1000]")
    p.add_argument("--econv", type=float, default=None,
                   help="density convergence, Ha/atom [1e-7; 1e-8 if --fmax < 0.01]")
    p.add_argument("--econv-loose", type=float, default=0.0, help="loose density convergence [0 = off]")
    p.add_argument("--econv-switch", type=float, default=5.0, help="loose while max|F| > switch*fmax [5]")
    p.add_argument("--precon", choices=["exp", "none"], default="none", help="[none]")
    p.add_argument("--predictor", choices=["affine", "linear"], default="affine",
                   help="start of each point: previous relaxed structure stretched affinely, or linear "
                        "extrapolation from the last two points (skipped after a slip) [affine]")
    p.add_argument("--keep-memory", action="store_true",
                   help="keep the L-BFGS memory between strain points [off]")
    p.add_argument("--regrid-tol", type=float, default=0.02,
                   help="re-choose the FFT grid when a cell length changed by more than this [0.02]")
    p.add_argument("--memory", type=int, default=20)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--stiffness", type=float, default=100.0)
    p.add_argument("--maxstep", type=float, default=0.2)
    p.add_argument("--maxstrain", type=float, default=0.01)
    p.add_argument("--prefix", default=None, help="[<POSCAR stem>_ss_<axis>_<mode>[_comp]]")
    p.add_argument("--engine", choices=["dftpy", "emt"], default="dftpy",
                   help="dftpy = OF-DFT [default]; emt = ASE EMT classical potential, TESTING / DRY RUNS ONLY")
    p.add_argument("--allow-xc-mismatch", action="store_true",
                   help="allow a PP whose name says LDA with --xc pbe (or GGA with --xc lda)")
    p.add_argument("--allow-unrelaxed-reference", action="store_true",
                   help="skip the reference checks (status file and forces/stresses at strain 0)")
    p.add_argument("--overwrite", action="store_true",
                   help="start a new curve although <prefix>.log/_params.json/_status.json of another run exist "
                        "(they are replaced). Never discards a curve: when <prefix>.csv exists the curve continues")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def reference_status_path(poscar):
    """<dir>/<stem>_status.json for a reference <dir>/<stem>_CONTCAR (or <stem>.<ext>)."""
    d, b = os.path.split(poscar)
    stem = b[:-len("_CONTCAR")] if b.endswith("_CONTCAR") else os.path.splitext(b)[0]
    return os.path.join(d, f"{stem}_status.json")


def _read_structure(path, what):
    if not os.path.isfile(path):
        die(f"{what} not found: {path}")
    try:
        return read_poscar(path)
    except (OSError, ValueError, IndexError) as exc:
        die(f"cannot read {what} {path} as a VASP5 POSCAR: {exc}")


def _composition(symbols):
    """'Al 31 Cu 1' in file order (consecutive runs), plus the atom count."""
    runs = []
    for s in symbols:
        if runs and runs[-1][0] == s:
            runs[-1][1] += 1
        else:
            runs.append([s, 1])
    return f"{len(symbols)} atoms ({' '.join(f'{el} {n}' for el, n in runs)})"


def _check_same_atoms(s, ref, what, path, ref_path, remedy):
    """die() unless structure s has the reference's atoms in the same order."""
    if list(s["symbols"]) != list(ref["symbols"]):
        die(f"the {what} {path} does not match the reference {ref_path}: {_composition(s['symbols'])} vs "
            f"{_composition(ref['symbols'])} (the same atoms in the same order are required); {remedy}")


def _read_restart_structure(path, what, ref, ref_path, remedy):
    """A point or partial CONTCAR of an earlier run: readable and with the reference's atoms, else die()."""
    try:
        s = read_poscar(path)
    except (OSError, ValueError, IndexError) as exc:
        die(f"cannot read the {what} {path} as a VASP5 POSCAR ({type(exc).__name__}: {exc}). Restore it, or "
            f"delete it ({remedy})")
    _check_same_atoms(s, ref, what, path, ref_path, "it belongs to another structure: use a new --prefix or run "
                      "folder for this reference")
    return s


def _basename_any(path):
    """File name of a Windows or POSIX path, whatever OS wrote it."""
    return re.split(r"[\\/]", str(path))[-1]


def _old_branch(old):
    """Loading branch of a params file without a 'branch' key (old ss_curve.py)."""
    vals = old.get("strains") or [old.get("strain_step") or 0.0]
    try:
        return "compression" if min(float(v) for v in vals) < 0 else "tension"
    except (TypeError, ValueError):
        return None


def _legacy_diff(old, params):
    """Differences between a params file of the original ss_curve.py and this run (keys of both formats;
    the reference and the PP files by name, since the old format has no checksums)."""
    diff = {k: (old.get(k), params[k]) for k in LEGACY_SHARED_KEYS if old.get(k) != params[k]}
    if params["engine"] != "dftpy":
        diff["engine"] = ("dftpy (old ss_curve.py)", params["engine"])
    old_ref = _basename_any(old.get("poscar", ""))
    if old_ref != params["poscar"]:
        diff["poscar"] = (old_ref, params["poscar"])
    old_pp = {}
    for item in old.get("pp") or []:
        el, _, name = str(item).partition("=")
        old_pp[el.strip()] = _basename_any(name.strip())
    if old_pp != params["pp"]:
        diff["pp"] = (old_pp, params["pp"])
    if _old_branch(old) != params["branch"]:
        diff["branch"] = (_old_branch(old), params["branch"])
    return diff


def _same_path(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _prepare(args, strains, branch, csv_path, par_path, status_path, log_path):
    """Rank 0: every refusal decision and every input read, before anything is written."""
    if args.fmax <= 0 or args.smax <= 0 or args.steps < 0 or args.econv <= 0:
        die("--fmax, --smax and --econv must be > 0 and --steps >= 0")
    ref = _read_structure(args.poscar, "reference structure")
    st_path = reference_status_path(args.poscar)
    if _same_path(st_path, status_path):
        die(f"--prefix {args.prefix}: the curve's status file {status_path} would replace the status file of the "
            f"reference {args.poscar}. Use another --prefix (default: <reference file name>_ss_<axis>_<mode>)")
    ref_state, ref_hint = None, ""
    if os.path.isfile(st_path):
        info = read_json(st_path)
        ref_state = info.get("state") if isinstance(info, dict) else "unreadable"
        if ref_state != "converged" and not args.allow_unrelaxed_reference:
            die(f"reference {args.poscar}: {st_path} says state '{ref_state}', not 'converged'. Relax it with "
                f"`ofdft relax --relax full` first (or --restart that relaxation), or pass "
                f"--allow-unrelaxed-reference")
        if isinstance(info, dict):
            rf, rs = info.get("fmax"), info.get("smax")
            looser = [f"{k} {v} > {mine}" for k, v, mine in (("fmax", rf, args.fmax), ("smax", rs, args.smax))
                      if isinstance(v, (int, float)) and v > mine]
            if looser:
                ref_hint = (f". Note: the reference was relaxed with looser thresholds than this curve "
                            f"({', '.join(looser)}, from {st_path})")
    restart = os.path.isfile(csv_path)
    notes, append = [], restart
    if not restart:                                # never replace another run's files without --overwrite
        leftovers = [f for f in (log_path, par_path, status_path) if os.path.exists(f)]
        if leftovers:
            old_st = read_json(status_path)
            empty_attempt = (isinstance(old_st, dict) and old_st.get("csv") == csv_path
                             and "strains_total" in old_st and not old_st.get("points_done")
                             and not os.path.exists(par_path))
            if empty_attempt:                      # our own earlier attempt that computed nothing
                append = True
                notes.append(f"an earlier attempt with this prefix ended without any point (state "
                             f"'{old_st.get('state')}'): starting again, log appended")
            elif args.overwrite:
                notes.append(f"--overwrite: replacing {', '.join(leftovers)} (there was no {csv_path})")
            else:
                die(f"{' and '.join(leftovers)} already exist but {csv_path} does not: prefix '{args.prefix}' "
                    f"belongs to another run in this folder (e.g. an `ofdft relax` with the same prefix) or its "
                    f"csv was deleted. Use another --prefix or a new run folder, or --overwrite to replace these "
                    f"files. Nothing was changed.")
    elif args.overwrite:
        notes.append(f"--overwrite has no effect: {csv_path} exists, so the curve continues (to start again use a "
                     f"new --prefix or delete the curve's files)")
    if args.engine == "dftpy":
        pp_files = ppmod.resolve(args.pp, ref["symbols"], args.xc, args.pppath,
                                 allow_xc_mismatch=args.allow_xc_mismatch)
    else:
        pp_files = {}
    params = {"version": __version__, "poscar": os.path.basename(args.poscar),
              "poscar_sha256": sha256_text(args.poscar),
              "pp": {el: os.path.basename(p) for el, p in pp_files.items()},
              "pp_sha256": {el: sha256_text(p) for el, p in pp_files.items()},
              "engine": args.engine, "xc": args.xc, "kedf": args.kedf, "ecut": args.ecut,
              "nr": None if args.nr is None else [int(v) for v in args.nr], "axis": args.axis, "mode": args.mode,
              "branch": branch, "strain_step": args.strain_step, "fmax": args.fmax, "smax": args.smax,
              "econv": args.econv, "econv_loose": args.econv_loose, "econv_switch": args.econv_switch,
              "strain_max": args.strain_max, "strains": args.strains}
    params = json.loads(json.dumps(params))
    rewrite_params = False
    if restart:
        old = read_json(par_path) if os.path.isfile(par_path) else None
        if not os.path.isfile(par_path):
            rewrite_params = True
            notes.append(f"{par_path} is missing: the options of the earlier run cannot be checked; it is "
                         f"written now with the current options")
        elif not isinstance(old, dict):
            die(f"{par_path} is unreadable; cannot check the options of {csv_path} (use a new --prefix)")
        elif "version" not in old:                 # written by the original ss_curve.py
            diff = _legacy_diff(old, params)
            if diff:
                lines = "\n".join(f"    {k}: was {a!r}, now {b!r}" for k, (a, b) in diff.items())
                die(f"{csv_path} was made by the old ss_curve.py script ({par_path} has no 'version') with "
                    f"different options:\n{lines}\n  Repeat the options of the old run (its params file lists "
                    f"them), use a new --prefix, or delete the old files")
            rewrite_params = True
            params["legacy_params"] = old
            notes.append(f"{par_path} was written by the old ss_curve.py: options checked by name; the "
                         f"reference and PP file contents cannot be verified (the old format has no checksums). "
                         f"It is rewritten in the new format (old content under 'legacy_params')")
        else:
            if "branch" not in old:
                old = dict(old, branch=_old_branch(old))
            diff = {k: (old.get(k), params[k]) for k in PARAM_KEYS if old.get(k) != params[k]}
            if diff:
                lines = "\n".join(f"    {k}: was {a!r}, now {b!r}" for k, (a, b) in diff.items())
                die(f"{csv_path} was made with different options:\n{lines}\n  Use the original options, a new "
                    f"--prefix, or delete the old files")

    def partial_path(e):
        return f"{args.prefix}_e{e:+.4f}_partial_CONTCAR"

    done, resume = [], {}
    if restart:                                    # converged points from the csv (+ their CONTCARs)
        last = {}
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                last[round(float(r["strain"]), 10)] = r          # last row per strain wins
        for key, r in last.items():
            path = f"{args.prefix}_e{key:+.4f}_CONTCAR"
            if str(r.get("converged", "")).strip().lower() == "true" and os.path.isfile(path):
                s = _read_restart_structure(path, f"CONTCAR of the converged strain {key:+.4f}", ref, args.poscar,
                                            "that strain point is then computed again")
                done.append((key, s["cell"], s["positions"]))
        donekeys = {d[0] for d in done}
        for e in strains:                          # unfinished points with a partial structure
            if round(e, 10) not in donekeys and os.path.isfile(partial_path(e)):
                s = _read_restart_structure(partial_path(e), f"partial CONTCAR of strain {e:+.4f}", ref,
                                            args.poscar, "that point then starts from the previous point")
                resume[round(e, 10)] = (s["cell"], s["positions"])
    if args.start and round(strains[0], 10) not in {d[0] for d in done} and round(strains[0], 10) not in resume:
        s = _read_structure(args.start, "--start structure")   # user-given start for the first strain point
        _check_same_atoms(s, ref, "--start structure", args.start, args.poscar,
                          "--start must be a CONTCAR of a curve from this reference")
        resume[round(strains[0], 10)] = (s["cell"], s["positions"])
    return dict(ref=ref, restart=restart, append=append, pp_files=pp_files, params=params, done=done,
                resume=resume, ref_status=(st_path, ref_state), ref_hint=ref_hint, notes=notes,
                rewrite_params=rewrite_params)


def _ref_check_text(args, strains, setup):
    """(applied?, text) for the force/stress check of the reference at strain 0 in this run."""
    if args.allow_unrelaxed_reference:
        return False, "disabled (--allow-unrelaxed-reference)"
    keys = {round(e, 10) for e in strains}
    if 0.0 not in keys:
        return False, ("NOT applied: strain 0 is not in the strain list, so the forces/stresses of the reference "
                       "are not checked (only its status file, if present)")
    if 0.0 in {d[0] for d in setup["done"]}:
        return False, "strain 0 was already done in an earlier run"
    if 0.0 in setup["resume"]:
        return False, ("NOT applied: strain 0 starts from a saved structure (--start or partial CONTCAR), not "
                       "from the reference")
    return True, (f"at strain 0 the first evaluation must have max|F| <= {2 * args.fmax:.4g} eV/A and all six "
                  f"|sigma| <= {2 * args.smax:.4g} GPa")


def _csv_has_rows(path):
    try:
        with open(path, newline="") as f:
            return len([r for r in csv.reader(f) if r]) > 1
    except OSError:
        return True


def _dftpy_info():
    try:
        import dftpy
        return f"dftpy {getattr(dftpy, '__version__', '?')} ({os.path.dirname(dftpy.__file__)})"
    except Exception:
        return "dftpy not importable"


def main(argv=None):
    args = parse_args(argv)
    stem = os.path.splitext(os.path.basename(args.poscar))[0]
    if args.strains:
        strains = [round(e, 10) + 0.0 for e in args.strains]
        if min(strains) < 0 < max(strains):
            die("--strains mixes tension (> 0) and compression (< 0) strains: run them as two curves, each with "
                "its own prefix")
        comp = min(strains) < 0                    # e.g. --strains 0 -0.005 -0.01 is compression
    else:
        strains = strain_list(args.strain_step, args.strain_max)
        comp = args.strain_step < 0
    branch = "compression" if comp else "tension"
    args.prefix = args.prefix or (f"{stem}_ss_{args.axis}_{args.mode}" + ("_comp" if comp else ""))
    econv_note = None
    if args.econv is None:
        args.econv = 1e-8 if args.fmax < 0.01 else 1e-7
        if args.fmax < 0.01:
            econv_note = f"econv auto-tightened to 1e-8 Ha/atom because fmax {args.fmax} < 0.01 eV/A"
    mp = init_mpi()
    root = mp.comm.rank == 0
    if args.engine == "emt" and mp.comm.size > 1:
        die("--engine emt is serial only (testing / dry runs): run it without mpirun")
    csv_path, par_path = f"{args.prefix}.csv", f"{args.prefix}_params.json"
    status_path, log_path = f"{args.prefix}_status.json", f"{args.prefix}.log"
    setup = gate(mp.comm, lambda: _prepare(args, strains, branch, csv_path, par_path, status_path, log_path))
    params = setup["params"]
    setup["ref_check"] = _ref_check_text(args, strains, setup)

    # ---- all checks passed (identically on every rank): from here on files are written ----
    logf = open(log_path, "a" if setup["append"] else "w") if root else None

    def log(msg=""):
        if root:
            print(msg, flush=True)
            print(msg, file=logf, flush=True)

    strain_keys = {round(e, 10) for e in strains}
    state = {"stop": None, "engine": None, "rows": [], "t0": time.time(),
             "done": {d[0] for d in setup["done"]} & strain_keys}
    status_base = {"version": __version__, "engine": args.engine, "prefix": args.prefix, "csv": csv_path,
                   "reference": os.path.basename(args.poscar), "reference_sha256": params["poscar_sha256"],
                   "axis": args.axis, "mode": args.mode, "branch": branch, "fmax": args.fmax, "smax": args.smax,
                   "reference_check": setup["ref_check"][1],
                   "strains_total": len(strain_keys), "started": now_iso(), "mpi_ranks": mp.comm.size,
                   "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}

    def status(**kw):                              # rank 0, atomic; "state" first for readability
        if root:
            done = sorted(state["done"], key=abs)
            write_json_atomic(status_path, {"state": kw.pop("state"), **status_base, "points_done": len(done),
                                            "last_strain": done[-1] if done else None,
                                            "dftpy_calls": getattr(state["engine"], "ncalls", 0),
                                            "wall_s": round(time.time() - state["t0"], 1), **kw})

    status(state="running", updated=now_iso())
    restore_term = install_term_handler() if root else (lambda: None)
    try:
        final, reason = _run(args, mp, root, setup, strains, csv_path, par_path, log, status, state, econv_note)
        status(state=final, reason=reason, rows_this_run=len(state["rows"]), finished=now_iso())
        log(f" status: {status_path} ({final})")
    except ReferenceNotRelaxed as exc:
        if root:
            msg = f"{exc}{setup['ref_hint']}"
            if not setup["done"] and not _csv_has_rows(csv_path):   # nothing computed: no restart files
                for f in (csv_path, par_path):
                    if os.path.isfile(f):
                        os.remove(f)
            log(f" ERROR: {msg}")
            status(state="reference_not_relaxed", reason=msg, finished=now_iso())
            log(f" status: {status_path} (reference_not_relaxed)")
            print(f"ERROR: {msg}", file=sys.stderr, flush=True)
        raise SystemExit(2)
    except BaseException as exc:
        if root:
            if isinstance(exc, SystemExit):
                reason = f"refused or aborted (exit code {exc.code}); see the error message in the job output"
            else:
                reason = f"{type(exc).__name__}: {exc}"
            try:
                status(state="error", reason=reason, finished=now_iso())
                log(f" ERROR: {reason}")
                log(f" status: {status_path} (error)")
            except Exception:
                pass
        raise
    finally:
        restore_term()
        if root:
            eng = state["engine"]
            if eng is not None and eng.scf_log is not None:
                eng.scf_log.close()
            logf.close()
    return 0


def _run(args, mp, root, setup, strains, csv_path, par_path, log, status, state, econv_note):
    """The strain loop (as in ss_curve.py). Returns (final state, reason) for the status file."""
    ref, restart, params = setup["ref"], setup["restart"], setup["params"]
    done, resume = setup["done"], setup["resume"]
    symbols, cell0, R0, fixed = ref["symbols"], ref["cell"], ref["positions"], ref["fixed"]

    def partial_path(e):
        return f"{args.prefix}_e{e:+.4f}_partial_CONTCAR"

    if args.engine == "emt":
        engine = EMTEngine(symbols, econv=args.econv, mp=mp)
    else:
        engine = DFTpyEngine(args, symbols, mp, verbose=args.verbose, pp_files=setup["pp_files"])
    state["engine"] = engine
    if root and args.engine == "dftpy":
        engine.scf_log = open(f"{args.prefix}_scf.log", "a" if setup["append"] else "w", buffering=1)
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
    log(f" ofdft sscurve: reference {args.poscar} ({len(symbols)} atoms), MPI ranks {mp.comm.size}")
    log(f" loading: {args.mode} along {args.axis}, strains {strains[0]:+.4f} .. {strains[-1]:+.4f} "
        f"(step {args.strain_step:+.4f}), {len(done)} already done, {len(resume)} to resume")
    log(f" per point: fmax {args.fmax} eV/A, smax {args.smax} GPa, max {args.steps} steps; precon {args.precon}, "
        f"predictor {args.predictor}, memory {'kept' if args.keep_memory else 'reset'}, "
        f"econv {args.econv} (loose {args.econv_loose or 'off'})")
    if args.engine == "emt":
        pp_txt = "none (EMT)"
    else:
        pp_txt = ", ".join(f"{el}: {engine.pp[el]} (sha256 {engine.pp_sha256[el][:12]})" for el in sorted(engine.pp))
    log(f" XC={args.xc.upper()}  KEDF={args.kedf}  ecut={args.ecut} eV  PP={pp_txt}")
    if econv_note:
        log(f" note: {econv_note}")
    for note in setup["notes"]:
        log(f" note: {note}")
    st_path, ref_state = setup["ref_status"]
    applied, check_txt = setup["ref_check"]
    if args.allow_unrelaxed_reference:
        log(" reference checks DISABLED (--allow-unrelaxed-reference)")
    else:
        log(f" reference checks: {st_path}: {ref_state or 'not present'}; force/stress check: {check_txt}")
        if not applied and check_txt.startswith("NOT applied"):
            log(" WARNING: the reference's forces and stresses are not checked in this run: make sure it is a "
                "converged `ofdft relax --relax full` structure")
    if root:
        stale = stop_requested(args.prefix)
        if stale:
            log(f" WARNING: STOP file '{stale}' exists: the curve will stop at once (delete it to run normally)")
    log("=" * 96)

    if root and not restart:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_COLS)
        write_json_atomic(par_path, params)
    elif root and setup["rewrite_params"]:         # missing, or the old ss_curve.py format
        write_json_atomic(par_path, params)

    def on_partial(e, st):
        if root:
            write_poscar(partial_path(e), symbols, st["cell"], st["R"], fixed,
                         comment=f"ss_curve strain {e:+.4f} partial E={st['E']:.6f} eV")

    def on_point(e, st, row):
        if root:
            if row["converged"]:
                write_poscar(f"{args.prefix}_e{e:+.4f}_CONTCAR", symbols, st["cell"], st["R"], fixed,
                             comment=f"ss_curve strain {e:+.4f} E={st['E']:.6f} eV")
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([row[c] for c in CSV_COLS])
                if os.path.isfile(partial_path(e)):
                    os.remove(partial_path(e))
            else:
                on_partial(e, st)
        mem = rss_gb()
        mem = mp.comm.allreduce(mem) if mp.is_mpi else mem
        log(f"   memory {mem:.1f} GB")
        state["rows"].append(row)
        if row["converged"] and round(e, 10) in {round(x, 10) for x in strains}:
            state["done"].add(round(e, 10))
        try:                                       # progress only: never let it stop the run
            status(state="running", last_point=row, updated=now_iso())
        except OSError:
            pass

    def should_stop():
        flag = stop_requested(args.prefix) if root else None
        flag = mp.comm.bcast(flag, root=0) if mp.is_mpi else flag
        if flag:
            state["stop"] = flag
        return flag

    opt = dict(axis=args.axis, mode=args.mode, strains=strains, fmax=args.fmax, smax=args.smax,
               steps=args.steps, precon=args.precon == "exp", predictor=args.predictor,
               keep_memory=args.keep_memory, regrid_tol=args.regrid_tol, alpha=args.alpha,
               stiffness=args.stiffness, memory=args.memory, maxstep=args.maxstep,
               maxstrain=args.maxstrain, econv_tight=args.econv,
               econv_loose=args.econv_loose or None, econv_switch=args.econv_switch,
               ref_check=None if args.allow_unrelaxed_reference else (2 * args.fmax, 2 * args.smax))
    t0 = time.time()
    rows = run_curve(engine, cell0, R0, symbols, fixed, opt, log=log,
                     comm=mp.comm if mp.is_mpi else None, done=done, resume=resume,
                     on_point=on_point, on_partial=on_partial, should_stop=should_stop)
    log("=" * 96)
    log(f" {sum(r['converged'] for r in rows)} strain points converged this run, {engine.ncalls} DFTpy calls, "
        f"{time.time()-t0:.0f} s. Curve: {csv_path}")
    nonconv = [r for r in rows if not r["converged"]]
    if state["stop"]:
        log(f" stopped by STOP file '{state['stop']}': delete it before restarting")
        return "stopped", f"STOP file '{state['stop']}'"
    if root and stop_requested(args.prefix):
        log(f" STOP file '{stop_requested(args.prefix)}' exists: delete it before restarting")
    if nonconv:
        r = nonconv[-1]
        return "point_not_converged", (f"strain {r['strain']:+.4f} not converged in {r['steps']} steps "
                                       f"(max|F| {r['max_force_eVA']:.4f} eV/A); rerun to continue, maybe with more --steps")
    if len(state["done"]) >= len({round(e, 10) for e in strains}):
        return "complete", "all strain points converged"
    return "stopped", "ended before all strain points were done"


if __name__ == "__main__":
    sys.exit(main())
