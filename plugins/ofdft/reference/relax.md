# `ofdft relax`: structure relaxation

`ofdft relax POSCAR [options]` relaxes atomic positions and, optionally, the cell. On the cluster it is
never run by hand: `ofdft jobscript --task relax --poscar POSCAR -- <relax options>` writes `job.sh`, and
`sbatch job.sh` runs it (see the optimize skill). Everything after `--` is passed to `ofdft relax`
verbatim.

## Modes (`--relax`, default `full`)

| Mode | Relaxes | Converged when |
|---|---|---|
| `positions` | atoms only, cell fixed | max\|F\| < fmax |
| `iso` | atoms + volume (isotropic) | max\|F\| < fmax and \|mean normal stress + p\| < smax (shear and deviatoric stress are not relaxed) |
| `full` (= `cell`) | atoms + all six cell components | max\|F\| < fmax and max\|sigma + p\| < smax over all six components |

Use `full` for a reference structure that a stress-strain curve will start from. Atoms fixed with
selective dynamics flags in the POSCAR stay fixed.

## Options and defaults

| Option | Default | Meaning |
|---|---|---|
| `--fmax` | 0.01 | force criterion, eV/A |
| `--smax` | 0.05 | stress criterion, GPa (not used in `positions` mode) |
| `--pressure` | 0 | target hydrostatic pressure, GPa (positive = compression) |
| `--steps` | 1000 | max optimizer steps for this job |
| `--econv` | auto | density convergence, Ha/atom: 1e-7, or 1e-8 when fmax < 0.01 (logged) |
| `--econv-loose` | 0 (off) | looser econv while max\|F\| > `--econv-switch` x fmax |
| `--econv-switch` | 5 | see `--econv-loose` |
| `--precon` | `none` | `exp` = Exp preconditioner (experimental; slower on real OF-DFT, see knowledge.md) |
| `--xc` | `pbe` | `pbe` or `lda` |
| `--kedf` | `WT` | kinetic energy functional |
| `--ecut` | 600 | grid cutoff, eV |
| `--nr N1 N2 N3` | from ecut | fix the FFT grid explicitly |
| `--pp EL=FILE` | registry default | pseudopotential per element; optional for Al (`al.gga.recpot` for PBE, `al.lda.recpot` for LDA). FILE is a bare name (searched) or a path |
| `--pppath DIR` | none | searched first; then `$OFDFT_PPPATH`, `~/.ofdft/pp`, the current directory |
| `--maxstep` | 0.2 | max atomic displacement per step, A |
| `--maxstrain` | 0.01 | max strain change per step |
| `--memory` | 20 | L-BFGS memory |
| `--alpha` | 10 | initial Hessian guess, eV/A^2 |
| `--stiffness` | 100 | stiffness used to scale strain, GPa (~C11) |
| `--regrid-tol` | 0.02 | after convergence, re-choose the grid if the spacing drifted more than this and continue (0 = never) |
| `--prefix` | POSCAR stem | output file prefix |
| `--traj-every` | 10 | write a trajectory frame every N steps |
| `--restart` | off | continue from `<prefix>_CONTCAR` |
| `--overwrite` | off | start fresh even though outputs exist (destroys them) |
| `--allow-xc-mismatch` | off | allow an LDA pseudopotential with PBE or the reverse |
| `--engine` | `dftpy` | `emt` = ASE EMT toy potential, for tests and dry runs only |
| `--verbose` | off | show DFTpy output |

## Outputs (in the run folder)

| File | Content |
|---|---|
| `<prefix>.log` | header (ofdft-opt version, XC backend, PP name + sha256 prefix, thresholds) and one line per step |
| `<prefix>_CONTCAR` | latest structure, rewritten every step; NOT proof of convergence |
| `<prefix>_status.json` | authoritative result (below) |
| `<prefix>_params.json` | settings used, checked on restart |
| `<prefix>_scf.log` | DFTpy density-optimisation output (`tail -n 20` shows the step in progress) |
| `<prefix>.xyz` | extended-XYZ trajectory with forces and stress; large (~100 MB for 13k atoms) |
| `<name>_<jobid>.out` | Slurm stdout: environment check, MPI test, log echo, errors |
| `job.sh`, `run.json` | job script and its record (version, resources, command) from `ofdft jobscript` |

Log columns: `step`, `energy (eV)`, `dE (eV)`, `max|F|` (eV/A), `max|sig|` (GPa, the stress criterion of the
mode: max |sigma + p| for `full`, |mean stress + p| for `iso`),
`P (GPa)` (pressure, positive = compressive), `V/atom` (A^3), `step(A)` (largest atomic move), `ev`
(energy evaluations in the line search), `time (s)` (cumulative wall time of this job), `mem(GB)` (total
resident memory over all ranks). s/step = difference of `time (s)` between consecutive steps (skip step 0,
which starts the density from scratch).

`<prefix>_status.json`:
- while running: `{"state": "running", ...}`;
- at the end: `state` = `converged` or `not_converged`, `reason` = `converged`, `steps` (step budget
  used up) or `stop` (STOP file), plus `steps`, `dftpy_calls`, `energy_eV`, `energy_per_atom_eV`,
  `max_force_eVA`, `max_stress_GPa` (largest component of |sigma + p|, also in `iso` mode; null in
  positions mode), `fmax`, `smax`, `relax`, `version`, `pp_sha256`, `contcar`, `wall_s`, `finished`;
- on a Python error: `{"state": "error", "reason": "<exception text>"}`.
- A job killed by Slurm (time limit, out of memory, scancel) usually cannot write the file, so it stays
  `running`. `running` while the job is no longer in `squeue` means the job was killed: read the `.out`.
  If the run caught Slurm's SIGTERM in time, the state is `error` with a reason starting
  `Terminated: terminated by signal 15`: also a kill, not a bug; resubmit to continue.

## Restart rules

- Starting without `--restart` or `--overwrite` when `<prefix>.log` or `<prefix>_CONTCAR` exists is refused
  (exit 2, nothing touched).
- `--restart` continues from `<prefix>_CONTCAR` and appends to the log. The `job.sh` written by
  `ofdft jobscript` adds `--restart` automatically when `<prefix>_CONTCAR` exists, so a restart is simply
  `sbatch job.sh` again.
- `--restart` is refused when the POSCAR content differs from the one the run started from (a new
  structure under the old name would otherwise silently continue the OLD structure from the CONTCAR):
  use a new run folder or `--prefix` for the new structure.
- Physics keys (`engine`, `xc`, `kedf`, `ecut`, `nr`, `relax`, `pressure`, `pp_sha256`) must match
  `<prefix>_params.json`; otherwise the restart is refused with the list of differences. Use a new
  `--prefix` (or a new run folder) for different physics. `--overwrite` also works but destroys the old
  outputs: ask first.
- `--fmax`, `--smax`, `--econv` and `--steps` may change on restart (logged). To change them, run
  `ofdft jobscript` again in the same folder: first `cat run.json` and repeat ALL its `args` after `--`
  (and its `--ntasks`/`--mem`/`--time`/`--partition`/`--name`), changing only those options; check that
  the printed prefix is unchanged, then `sbatch job.sh`. Options left out fall back to the defaults (e.g.
  a dropped `--xc lda` is refused at restart, a dropped `--ntasks 8` becomes auto).

## Stopping cleanly

`touch <prefix>.STOP` in the run folder: the run finishes the current step, writes the CONTCAR, trajectory
and status (`not_converged`, reason `stop`) and exits. A file named `STOP` stops every run in that
folder. Delete the STOP file before restarting: `job.sh` refuses to start while one exists (and a
direct `ofdft relax` would stop again after its first evaluation).

## Typical runs (options after `--` of `ofdft jobscript --task relax --poscar <file>`)

| Goal | Options |
|---|---|
| Reference structure for stress-strain (default thresholds 0.01/0.05) | `--relax full` |
| Atoms only, fixed cell | `--relax positions` |
| Tight relaxation (econv becomes 1e-8 automatically) | `--relax full --fmax 0.005 --smax 0.01` |
| LDA instead of PBE (uses `al.lda.recpot`) | `--relax full --xc lda` |
| Under 1 GPa hydrostatic pressure | `--relax full --pressure 1.0` |
| Cost probe (2 steps) | `--relax full --steps 2` |

## Time and memory sizing

- `ofdft jobscript` defaults: `--ntasks auto` (by atom count: < 500 atoms 8 ranks, < 2000 16, < 6000 32,
  else 64), `--mem auto` (max(8, ceil(0.0065 x atoms)) GB), `--time 3-00:00:00`.
- For a new system size, run a 2-step probe first and read s/step and `mem(GB)` from its log.
- Expected steps: small near-equilibrium cells 10-50; polycrystals from an as-built structure hundreds
  (13,140 atoms: 674 steps). `--time` = s/step x expected steps x 1.5; `--mem` = probe `mem(GB)` x 1.3
  (rounded up, at least 8G).
- Reference point: 13,140 atoms at 64 ranks ~95-115 s/step, 38-43 GB.
