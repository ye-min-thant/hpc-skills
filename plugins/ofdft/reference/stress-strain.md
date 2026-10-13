# `ofdft sscurve`: quasi-static stress-strain curves

`ofdft sscurve REF_CONTCAR [options]` increases the engineering strain along one axis step by step and
relaxes the structure at every strain point with the same L-BFGS relaxer as `ofdft relax`. On the cluster
it runs through `ofdft jobscript --task sscurve --poscar REF_CONTCAR -- <sscurve options>` and
`sbatch job.sh`.

## The reference structure

- Must be a fully relaxed, stress-free structure: the `<prefix>_CONTCAR` of an `ofdft relax --relax full`
  run that finished with `state: converged`, computed with the same XC, pseudopotential and ecut.
- Upload or copy the reference CONTCAR together with its status file into the stress-strain run folder.
  The check looks for the sibling status file `<reference name minus trailing _CONTCAR>_status.json`
  (`Al_gb_CONTCAR` -> `Al_gb_status.json`); if it exists and its state is not `converged`, the run is
  refused (exit 2).
- At strain 0 the very first evaluation must have max|F| <= 2 x fmax and all six |sigma| <= 2 x smax (GPa).
  Otherwise the curve stops with status `reference_not_relaxed`; the message says which limit failed and
  how to relax the reference (`ofdft relax ... --relax full --fmax ... --smax ...`).
- A curve with tight thresholds needs a reference relaxed to at least those thresholds: a reference
  relaxed with the defaults (0.01/0.05) usually fails the check of a 0.005/0.01 curve. Relax it further
  (a `--restart` of its relaxation with the curve's `--fmax`/`--smax`) and copy the new CONTCAR and status.
- The force/stress check needs strain 0 in the list, computed from the reference. With `--strains` that
  omit 0 (the tight re-check below) or when strain 0 starts from `--start`, the log says
  `force/stress check: NOT applied` and only the status file is checked (also recorded as
  `reference_check` in the curve's status file).
- `--allow-unrelaxed-reference` skips both checks. Use it only if the user insists, and say that the curve
  then includes residual stress.

## Loading modes

- `--mode tension` (default, uniaxial stress): the box length along `--axis` is exactly (1 + e) x the
  reference length ("rigid grips", like LAMMPS `fix deform`); the five other cell components relax until
  their stresses are below smax; the shear components that involve the loading axis relax as tilts.
- `--mode strain` (uniaxial strain): all other cell components stay at the reference; only atoms relax.
- Strains: `--strain-step 0.005 --strain-max 0.05` (default) gives 0, 0.005, ..., 0.05. Or an explicit list
  `--strains e1 e2 ...` (used for re-checks, with `--start`).
- Compression: negative `--strain-step` and `--strain-max` (e.g. `--strain-step -0.005 --strain-max
  -0.05`), or a `--strains` list with negative values; the prefix gets the suffix `_comp`. A list may not
  mix tension and compression (refused): run them as two curves. The branch is stored in the params
  file, so a restart with strains of the other sign is refused.

## Per point

- Start structure (predictor): `--predictor affine` (default) stretches the previous converged point;
  `linear` extrapolates from the last two points, skipped after a slip (any atom moved > 0.1 A in the
  last increment). `--start FILE` gives the start structure of the first listed strain point.
- L-BFGS memory is reset at every point (`--keep-memory` keeps it).
- Converged when max|F| < fmax and (tension mode) the five free stresses < smax. The loading-axis stress
  is the result. A point that does not converge within `--steps` stops the curve (status
  `point_not_converged`); its latest structure is kept as a partial CONTCAR.

## Options and defaults

| Option | Default | Meaning |
|---|---|---|
| `--axis` | `z` | loading axis `x`, `y` or `z` |
| `--mode` | `tension` | `tension` or `strain` |
| `--strain-step` / `--strain-max` | 0.005 / 0.05 | strain increment / final strain (negative = compression) |
| `--strains` | none | explicit strain list instead of step/max |
| `--start` | none | CONTCAR to start the first strain point from |
| `--fmax` / `--smax` | 0.01 / 0.05 | per-point force (eV/A) / free-stress (GPa) criteria |
| `--steps` | 1000 | max optimizer steps per strain point |
| `--econv` | auto | 1e-7 Ha/atom, or 1e-8 when fmax < 0.01 |
| `--econv-loose` / `--econv-switch` | 0 (off) / 5 | loose econv while max\|F\| > switch x fmax |
| `--precon` | `none` | `exp` = Exp preconditioner (experimental) |
| `--predictor` | `affine` | `affine` or `linear` |
| `--keep-memory` | off | keep L-BFGS memory between points |
| `--regrid-tol` | 0.02 | re-choose the FFT grid when a cell length changed more than this |
| `--memory`, `--alpha`, `--stiffness`, `--maxstep`, `--maxstrain` | 20, 10, 100, 0.2, 0.01 | as in `ofdft relax` |
| `--xc`, `--kedf`, `--ecut`, `--nr`, `--pp`, `--pppath` | pbe, WT, 600, auto, registry | as in `ofdft relax` |
| `--prefix` | `<ref stem>_ss_<axis>_<mode>[_comp]` | output prefix |
| `--allow-xc-mismatch` | off | as in `ofdft relax` |
| `--allow-unrelaxed-reference` | off | skip the reference checks |
| `--overwrite` | off | start although `<prefix>.log`/`_params.json`/`_status.json` of another run exist (replaced); never discards a curve: with `<prefix>.csv` present the curve continues |
| `--engine` | `dftpy` | `emt` for tests and dry runs only |
| `--verbose` | off | show DFTpy output |

The stem keeps everything before the file extension: reference `Al_gb_CONTCAR` gives the prefix
`Al_gb_CONTCAR_ss_z_tension`.

## Outputs

| File | Content |
|---|---|
| `<prefix>.csv` | one row per converged strain point |
| `<prefix>_e+0.0100_CONTCAR` | relaxed structure at strain +0.0100 (format `e%+.4f`) |
| `<prefix>_e+0.0100_partial_CONTCAR` | latest structure of an unfinished point (used on restart) |
| `<prefix>_status.json` | `state`: `running`, `complete`, `stopped`, `point_not_converged`, `error` or `reference_not_relaxed`; `points_done`, `strains_total`, `last_strain`, `version` |
| `<prefix>.log`, `<prefix>_scf.log`, `<prefix>_params.json` | log (header with version and thresholds), DFTpy output, settings checked on restart |

CSV columns: `strain`, `sxx_GPa`, `syy_GPa`, `szz_GPa`, `syz_GPa`, `sxz_GPa`, `sxy_GPa` (positive =
tensile), `energy_eV`, `energy_per_atom_eV`, `max_force_eVA`, `max_free_stress_GPa` (0 in strain mode),
`steps`, `dftpy_calls`, `time_s`, `converged`.

Log lines: per step `e=+0.0100 step N E=... max|F|=... max|sig_free|=... sig_load=+0.4965 GPa`, and one
summary line per point (`strain +0.0100: sig_load ... converged in N steps / M calls, T s`).

A job killed by Slurm leaves the status at `running` (or `error` with a reason starting `Terminated:` when
it caught the SIGTERM); check `squeue`/`sacct` and the `.out` file, then resubmit to continue.

## Restart and stop

- Restart = run the same command again: `sbatch job.sh` in the same folder. Converged strains (CSV) are
  skipped; an unfinished point continues from its partial CONTCAR.
- Changing physics or tolerance options (pp, xc, kedf, ecut, nr, axis, mode, loading branch, strain step,
  fmax, smax, econv settings) or the reference file content with the same prefix is refused (exit 2).
  `--steps` may change: regenerate job.sh with `ofdft jobscript` in the folder, repeating ALL the `args`
  from `run.json` after `--` plus the larger `--steps` (and its `--ntasks`/`--mem`/`--time`); check that
  the printed prefix is unchanged. Leaving out e.g. the negative `--strain-step` of a compression curve
  gives the tension prefix, and the resubmitted job starts a new curve.
- A new curve never replaces the log/params/status of another run with the same prefix (e.g. a relax
  run with `--prefix` equal to the curve's): refused unless `--overwrite`. The reference's own status file
  is never replaced.
- Point CONTCARs read on restart (and `--start`) must hold the reference's atoms in the same order; an
  unreadable one is refused with its name (restore it, or delete it and the point is computed again).
- Curves started by the original `ss_curve.py` script continue too: run `ofdft sscurve` with the same
  options (`--pp` file names, xc, thresholds, econv settings, strain step; the old defaults were
  `--econv-loose 1e-6 --precon exp --steps 500`). The old params file has no checksums, so the reference
  and PP files are compared by name only; it is then rewritten in the new format (old content kept under
  `legacy_params`).
- Stop cleanly: `touch <prefix>.STOP` (or `STOP`) in the run folder; the current point stops at the next
  step and keeps a partial CONTCAR; status `stopped`. Delete the STOP file before resubmitting.

## Cost

Elastic points need ~8-40 steps, grain-boundary events or slip ~80+ steps. 13,140 atoms at 64 ranks:
~95-115 s/step; the first 6 points (0-2.5 %) took ~4 h. Size `--time` as (points x ~40 steps x s/step x
1.5) and resubmit if needed; the default is 10 days.

## Required tight re-check before reporting a peak stress or yield strain

Per-point thresholds 0.01/0.05 can overestimate the peak stress by ~40 % (knowledge.md). Before reporting a
peak stress or yield strain, re-relax the peak point and the next point with tight thresholds:

1. From the CSV find the peak strain e_p (largest loading-axis stress) and the next strain e_n.
2. New run folder (e.g. `<stem>_sscurve_check`); copy into it the reference CONTCAR, its status file, and
   `<prefix>_e+<e_p>_CONTCAR`.
3. In that folder (example: peak at 0.035, step 0.005, reference `Al_gb_CONTCAR`):

```
ofdft jobscript --task sscurve --poscar Al_gb_CONTCAR --name check_tight -- --axis z --mode tension --strains 0.035 0.04 --start Al_gb_CONTCAR_ss_z_tension_e+0.0350_CONTCAR --fmax 0.005 --smax 0.01 --prefix check_tight
```

4. Compare `check_tight.csv` with the curve at e_p and e_n. If the loading stress changes by more than the
   accuracy the user needs (a few %), report the tight values and recommend rerunning the region around the
   peak with `--fmax 0.005 --smax 0.01`.

Without this check, label peak and yield values "unverified, thresholds 0.01/0.05". Always state the
thresholds with any stress-strain result.

## Plotting

`ofdft plot <prefix>.csv [more.csv ...] [-o PNG] [--axis x|y|z]` writes `<first csv stem>.png` next to the
CSV (left: loading stress vs strain, other components dotted; right: optimizer steps per point) and prints
the peak stress, its strain and the initial slope. The title shows fmax/smax from `<prefix>_params.json`.
Needs matplotlib (in the environment.yml env). The 13,140-atom polycrystal in z tension has an initial
slope of ~50-55 GPa; fcc Al [001] single crystal ~45.6 GPa.
