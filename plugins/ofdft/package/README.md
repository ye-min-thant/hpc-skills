# ofdft-opt -- command-line guide

`ofdft-opt` relaxes crystal structures and computes quasi-static stress-strain curves with
orbital-free DFT ([DFTpy](https://gitlab.com/pavanello-research-group/dftpy)), using its own
L-BFGS optimizer for atomic positions and cell together (no ASE optimizer), and writes Slurm job
scripts for a cluster. The Claude Code plugin in this repository drives exactly this command line;
everything below works without Claude.

Validated scope: aluminium with the BLPS local pseudopotentials (Huang & Carter), Wang-Teter
kinetic functional, PBE or LDA exchange-correlation, 600 eV grid cutoff, single-node MPI. Other
elements run only with a matching local pseudopotential you supply, and their results are not
validated.

## 1. Install

### Python environment

Use Miniforge (conda-forge). Never the Anaconda `defaults` channel. Create the tested environment
from `environment.yml` (in this directory) on the cluster login node (takes several minutes):

    conda env create -n ofdft -f environment.yml

An existing environment also works if it has Python >= 3.10, numpy, scipy, ase >= 3.23, DFTpy
2.1.2-2.2.x from PyPI (`python -m pip install dftpy==2.2.0`; never the DFTpy GitHub development
branch, which computes a wrong XC stress under MPI) and, for MPI runs, openmpi + mpi4py + mpi4py-fft
from conda-forge. LibXC is not needed: a NumPy LDA/PBE stand-in is vendored and used automatically
when `pylibxc` is not installed.

### The package: copy the source (recommended, no pip)

The package is plain Python. Copy `src/ofdft_opt` into a versioned directory and point
`PYTHONPATH` at it. Old versions stay, so running jobs keep their exact code:

    mkdir -p ~/.ofdft/lib/0.1.0
    cp -r src/ofdft_opt ~/.ofdft/lib/0.1.0/
    # from a PC instead:
    #   tar -C src -czf - --exclude=__pycache__ ofdft_opt | ssh <alias> 'mkdir -p ~/.ofdft/lib/0.1.0 && tar -xzf - -C ~/.ofdft/lib/0.1.0'

Then a small config file and a wrapper command (adjust the python path to your environment):

    cat > ~/.ofdft/config.sh <<'EOF'
    OFDFT_PYTHON=$HOME/miniforge3/envs/ofdft/bin/python
    OFDFT_VERSION=0.1.0
    EOF
    mkdir -p ~/.ofdft/bin
    cat > ~/.ofdft/bin/ofdft <<'EOF'
    #!/bin/bash
    source ~/.ofdft/config.sh
    export PYTHONPATH=$HOME/.ofdft/lib/$OFDFT_VERSION
    export PYTHONNOUSERSITE=1
    exec "$OFDFT_PYTHON" -m ofdft_opt "$@"
    EOF
    chmod +x ~/.ofdft/bin/ofdft

Use it as `~/.ofdft/bin/ofdft ...` (or add `~/.ofdft/bin` to your PATH yourself). Finish with

    ~/.ofdft/bin/ofdft pp fetch      # download + verify the pseudopotentials into ~/.ofdft/pp
    ~/.ofdft/bin/ofdft doctor        # must end with "doctor OK"

### The package: pip (alternative)

Inside the environment: `python -m pip install .` (from this directory; needs setuptools >= 77).
This installs the `ofdft` command. Install the conda environment first so pip does not pull
numpy/scipy/MPI packages from PyPI.

Without installing anything, from this directory: `PYTHONPATH=src python -m ofdft_opt <command>`.
Small serial runs also work on Windows/macOS/Linux PCs.

## 2. Quick start on a Slurm cluster

One run per folder. The job script is generated, never submitted for you:

    mkdir -p ~/ofdft-runs/Al_relax && cd ~/ofdft-runs/Al_relax
    cp /path/to/Al.vasp .
    ofdft jobscript --task relax --poscar Al.vasp -- --relax full
    cat job.sh                   # check resources and command
    sbatch job.sh
    tail -f Al.log               # one line per optimizer step
    cat Al_status.json           # authoritative: running / converged / not_converged / error

Stress-strain curve from the relaxed structure, in a new folder:

    mkdir -p ~/ofdft-runs/Al_sscurve && cd ~/ofdft-runs/Al_sscurve
    cp ~/ofdft-runs/Al_relax/Al_CONTCAR .
    ofdft jobscript --task sscurve --poscar Al_CONTCAR -- --axis z --mode tension --strain-step 0.005 --strain-max 0.05
    sbatch job.sh
    ofdft plot Al_CONTCAR_ss_z_tension.csv

Smoke test (minutes) with the bundled 255-atom cell (fcc Al, one vacancy):

    cp "$(ofdft version | sed 's/.*(\(.*\))/\1/')/data/al_vacancy_255.vasp" .
    ofdft jobscript --task relax --poscar al_vacancy_255.vasp --ntasks 4 --mem 8G --time 00:20:00 -- --steps 2

## 3. Commands

`ofdft <command> -h` prints all options. Exit codes: 0 = ok (also "not converged": the status file
tells), 2 = refused / bad input (nothing was overwritten), 1 = unexpected error.

### `ofdft relax POSCAR [options]`

Relaxes positions and/or the cell. Converged when max|F| < `--fmax` AND max|sigma + p| < `--smax`
(`--relax positions`: forces only; `--relax iso`: the mean normal stress |tr(sigma)/3 + p|).

| option | default | meaning |
|---|---|---|
| `--relax positions\|iso\|full\|cell` | full | what is relaxed (`cell` = `full`) |
| `--fmax` / `--smax` | 0.01 eV/A / 0.05 GPa | convergence thresholds |
| `--pressure` | 0 | target hydrostatic pressure, GPa |
| `--steps` | 1000 | max optimizer steps |
| `--pp EL=FILE` | registry default (Al: `al.gga.recpot` for PBE, `al.lda.recpot` for LDA) | pseudopotential per element (repeatable); FILE = path or bare name |
| `--pppath DIR` | - | searched before `$OFDFT_PPPATH`, `~/.ofdft/pp`, the current directory |
| `--xc pbe\|lda`, `--kedf`, `--ecut` | pbe, WT, 600 eV | physics |
| `--nr N N N` | from ecut | fixed FFT grid |
| `--econv` | 1e-7 Ha/atom (1e-8 if fmax < 0.01) | density convergence |
| `--econv-loose`, `--econv-switch` | 0 (off), 5 | looser density convergence while max\|F\| > switch x fmax |
| `--precon exp\|none` | none | Exp preconditioner (did not pay off for large OF-DFT polycrystals) |
| `--maxstep`, `--maxstrain`, `--memory`, `--alpha`, `--stiffness` | 0.2 A, 0.01, 20, 10, 100 GPa | optimizer |
| `--regrid-tol` | 0.02 | re-choose the FFT grid after convergence if the spacing drifted more |
| `--prefix` | POSCAR name without extension | output names |
| `--traj-every` | 10 | trajectory frame interval |
| `--restart` / `--overwrite` | - | continue from `<prefix>_CONTCAR` / start over existing outputs |
| `--allow-xc-mismatch` | - | allow an LDA-named PP with PBE (or the reverse) |
| `--engine dftpy\|emt` | dftpy | `emt` = classical EMT potential for tests and dry runs only |

Outputs: `<prefix>.log` (one line per step: energy, max|F|, max|sigma|, pressure, volume, time,
memory), `<prefix>_CONTCAR` (every step), `<prefix>.xyz` (trajectory), `<prefix>_scf.log`,
`<prefix>_params.json` (options and checksums), `<prefix>_status.json` (final state, energy,
max force/stress, thresholds).

Without `--restart` or `--overwrite`, existing outputs make `relax` refuse (exit 2). With
`--restart`, physics options (engine, xc, kedf, ecut, nr, relax, pressure, pseudopotential
checksums) must match `<prefix>_params.json`; thresholds and `--steps` may change.

### `ofdft sscurve REF_CONTCAR [options]`

Quasi-static curve along one axis from a RELAXED reference (zero stress). At every strain point
the atoms (and, in tension mode, the other five cell components) are relaxed.

| option | default | meaning |
|---|---|---|
| `--axis x\|y\|z` | z | loading axis |
| `--mode tension\|strain` | tension | uniaxial stress (rigid grips: box length exactly (1+e) L) or uniaxial strain |
| `--strain-step`, `--strain-max` | 0.005, 0.05 | negative values = compression (prefix gets `_comp`) |
| `--strains e1 e2 ...` + `--start CONTCAR` | - | explicit strain list, e.g. to re-check single points (negative = compression; no mixed signs) |
| `--fmax` / `--smax` / `--steps` | 0.01 / 0.05 / 1000 per point | per-point thresholds |
| `--predictor affine\|linear`, `--keep-memory` | affine, off | start of each point |
| `--allow-unrelaxed-reference` | - | skip the reference checks |
| `--overwrite` | - | start although `<prefix>.log/_params.json/_status.json` of another run exist (never discards an existing csv: the curve then continues) |
| physics, PP, econv, precon, engine options | as for `relax` | |

Default prefix: `<ref stem>_ss_<axis>_<mode>[_comp]`. Outputs: `<prefix>.csv` (one row per
converged point: strain, six stresses in GPa with positive = tensile, energy, max|F|, steps,
calls, time), `<prefix>_e+0.0100_CONTCAR` per point, `<prefix>_e+..._partial_CONTCAR` for an
unfinished point, `.log`, `_scf.log`, `_params.json`, `_status.json` (running, complete, stopped,
point_not_converged, error, reference_not_relaxed).

Reference check: a sibling status file of the reference (`Al_CONTCAR` -> `Al_status.json`), if present,
must say converged, and at strain 0 the first evaluation must have max|F| <= 2 fmax and all six
|sigma| <= 2 smax; otherwise the curve is refused (`--allow-unrelaxed-reference` skips both checks).
A curve with tighter thresholds than the reference relaxation fails this check: relax the reference to
at least the curve's `--fmax/--smax`. Without strain 0 in the list (or with `--start` at strain 0) the
force/stress check is not applied; the log says so.

Restart: the same command continues the curve; physics/tolerance options and the loading branch must
match `<prefix>_params.json` (only `--steps` may change). Curves started by the original `ss_curve.py`
script also continue with the same options (checked by name; the params file is then rewritten).

The per-point thresholds decide the yield point (EMT test: 0.01/0.05 overestimated the peak stress
by ~40% compared with 0.005/0.01). Before reporting a peak stress or yield strain, re-relax the
peak point and the next one tighter, in a new prefix:

    ofdft sscurve Al_CONTCAR --axis z --mode tension --strains 0.035 0.04 \
        --start Al_CONTCAR_ss_z_tension_e+0.0350_CONTCAR --fmax 0.005 --smax 0.01 --prefix check_tight

On a cluster, run it as a job in a new folder holding copies of `Al_CONTCAR`, its status file and the
`--start` CONTCAR: `ofdft jobscript --task sscurve --poscar Al_CONTCAR -- <the options above>`.

### `ofdft jobscript --task relax|sscurve --poscar FILE [options] [-- TASK OPTIONS]`

Writes `job.sh` (or `-o FILE`) and `run.json` into the run folder; does not submit.

| option | default |
|---|---|
| `--ntasks N\|auto` | auto by atoms: < 500 -> 8, < 2000 -> 16, < 6000 -> 32, else 64 (one node) |
| `--mem auto\|<GB>G` | auto: max(8, ceil(0.0065 x atoms)) G |
| `--time` | relax 3-00:00:00, sscurve 10-00:00:00 |
| `--partition` | cluster default |
| `--name` | `<task>_<poscar stem>` |
| `--python` | the python running `ofdft` |
| `--libdir` | the directory containing the running `ofdft_opt` (put on PYTHONPATH) |

Everything after `--` goes to the task verbatim. The POSCAR must be in the run folder. The
generated script: activates the conda env of `--python` (or uses absolute paths if that python is
not in a conda env), pins `PYTHONPATH`, sets one thread per rank, refuses to start while a STOP file
exists or another job runs in the folder, runs `ofdft doctor --brief`, checks that `mpirun` starts
exactly one MPI job of N ranks (`ofdft doctor --mpi`), and launches
`mpirun -np N python -m mpi4py -m ofdft_opt <task> ...` (an exception on one rank aborts the job
instead of hanging). Resubmitting the same `job.sh` continues the run: relax adds `--restart` when
`<prefix>_CONTCAR` exists; sscurve continues from its csv.

### `ofdft doctor [--brief] [--mpi]`

Checks the environment without initializing MPI: ofdft-opt, python, numpy, scipy, ase, dftpy
(FAIL for a git/dev build, WARN outside 2.1.2-2.2.x), mpi4py and mpi4py-fft (WARN if missing),
the SciPy shim, the XC backend, and the registry pseudopotentials (checksums). Exit 1 if anything
FAILs. `--brief` prints only WARN/FAIL lines and a summary. `--mpi` is the wire-up test, run as
`mpirun -np N python -m ofdft_opt doctor --mpi`: rank 0 prints exactly one line `N | <MPI library>`.
Many `1 | ...` lines mean mpirun and mpi4py belong to different MPI installations.

### `ofdft pp list` / `ofdft pp fetch [--dest DIR]`

`list` shows the registry files, where they are found and whether their sha256 matches. `fetch`
downloads them from the pinned commit of github.com/EACcodes/local-pseudopotentials into
`~/.ofdft/pp` and verifies them. Without internet: download on another machine and copy the files
into `~/.ofdft/pp`. Files copied from Windows (CRLF) verify too.

### `ofdft plot CSV [CSV ...] [-o PNG] [--axis x|y|z]`

Stress-strain curve(s) plus optimizer steps per point, to `<first csv stem>.png` by default. Prints
points, peak stress, total steps and hours, the initial slope (first point to the first point >= 1 %
further) and the thresholds from `<prefix>_params.json`. Needs matplotlib.

### `ofdft version`

Prints `ofdft-opt <version> (<package directory>)`.

## 4. Restart, stop, monitor

- Restart after a time limit, node failure or clean stop: `sbatch job.sh` again in the same folder.
- Stop cleanly: `touch <prefix>.STOP` (or `touch STOP`) in the run folder; the current step finishes
  and all files are written. Delete the STOP file before resubmitting.
- Convergence is in `<prefix>_status.json`, never inferred from log columns.
- Do not change physics options on restart (refused); use a new `--prefix` or a new folder.
- To change `--steps` (or relax thresholds, memory, time), regenerate `job.sh` with ALL the task options
  from `run.json` (`args`) repeated after `--`; `ofdft jobscript` warns when the prefix would change.
- Never compute on a login node beyond `doctor`, `pp`, `jobscript` and `plot`.

## 5. Cost guide (13,140-atom Al polycrystal, PBE, 600 eV)

About 95-115 s per optimizer step at 64 ranks and 38-43 GB of memory. Relaxing an as-built
polycrystal took 674 steps; grain-boundary reconstruction is the slow tail (pre-relaxing with a
classical potential helps). Stress-strain points: elastic 8-40 steps, grain-boundary events 80+.
Measure a new size with a `--steps 2` probe and set `--time` from the seconds per step.

## 6. Tests

From this directory: `python -m unittest discover -s tests -v` (stdlib unittest; DFTpy tests are
skipped when DFTpy is not installed).

Licence: MIT. Credits and how to cite (DFTpy, BLPS, Wang-Teter, Exp preconditioner): see `README.md`
and `NOTICE.md` at the repository root.
