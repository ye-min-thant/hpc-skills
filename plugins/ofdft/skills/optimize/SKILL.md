---
name: optimize
description: Use only for orbital-free DFT (OF-DFT, DFTpy, ofdft) work on a Slurm cluster - relaxing or optimizing a structure (atomic positions and/or cell relaxation) with DFTpy, computing a stress-strain curve (uniaxial tension or compression), or checking, restarting, stopping, plotting or downloading an ofdft run. Not for VASP, Quantum ESPRESSO, LAMMPS or other codes, and not for optimizing program code. For DFTpy/OF-DFT runs on the IMLab im cluster use this skill with plain ssh, not the imlab job skills or imlab MCP tools.
---

# OF-DFT relaxations and stress-strain curves (ofdft)

Runs the ofdft-opt package (DFTpy driver) on the user's Slurm cluster over SSH. Validated for Al with BLPS
pseudopotentials, WT KEDF, PBE or LDA, ecut 600 eV. On the cluster the CLI is `~/.ofdft/bin/ofdft`.

Reference files (read the one you need, when you need it):
- `${CLAUDE_PLUGIN_ROOT}/reference/relax.md` - relax modes, options, outputs, restart rules, sizing
- `${CLAUDE_PLUGIN_ROOT}/reference/stress-strain.md` - loading modes, CSV, reference check, tight re-check
- `${CLAUDE_PLUGIN_ROOT}/reference/knowledge.md` - method, convergence, costs, reference values
- `${CLAUDE_PLUGIN_ROOT}/reference/cluster-im.md` - nodes, ranks, memory, fair use
- `${CLAUDE_PLUGIN_ROOT}/reference/ssh.md` - exact ssh/scp/tar forms, Windows notes
- `${CLAUDE_PLUGIN_ROOT}/reference/troubleshooting.md` - error signature -> cause -> action

## Rules

- Run every ssh/scp/tar command with the **Bash tool** (Git Bash on Windows), never PowerShell (Windows
  PowerShell 5.1 strips embedded quotes). Always `ssh -o BatchMode=yes <alias> '...'`; Slurm commands as
  `bash -lc "..."` inside the single quotes.
- Never compute on the login node beyond `ofdft doctor/pp/jobscript/plot` (seconds). Every relaxation,
  curve or probe goes through `sbatch`. Never use `StrictHostKeyChecking=no`, never remove SSH key
  passphrases, never enter passwords.
- Ask before: submitting a job, cancelling/stopping, overwriting a run folder, downloading > 100 MB.
- Before submitting: `squeue -u $USER`; if the user already has >= 2 running jobs on the cluster, say so
  and ask.
- Scope: orbital-free DFT with DFTpy, validated for Al. Other elements only with a user-supplied matching
  local pseudopotential; results unvalidated - say so.
- For DFTpy/OF-DFT runs use this skill and plain ssh, not the imlab skills (e.g. imlab-submitting-jobs)
  or the imlab MCP tools: they fail on Windows and launch with `srun`/module MPI, which DFTpy's env
  cannot use (ssh.md).
- Start remote commands with `cd ~ && ` or a relative path, never with `/...` (Git Bash may rewrite it
  on Windows), and use only commands that end by themselves (`tail -n`, never `tail -f`).

## 1. Confirm the method

If the user did not say DFTpy, OF-DFT, orbital-free or ofdft (e.g. only "relax this POSCAR"), ask whether
they want an orbital-free DFT run with DFTpy. This skill does not run VASP, QE or LAMMPS.

## 2. Config and version

- Read `~/.ofdft/config.json` (`ssh_alias`, `remote_root`, `version`, `env_python`). Missing: run the setup
  skill (`/ofdft:setup`) first.
- One call: `ssh -o BatchMode=yes <alias> '~/.ofdft/bin/ofdft version; bash -lc "squeue -u $USER"'`.
- Compare the printed version with `version` in `${CLAUDE_PLUGIN_ROOT}/.claude-plugin/plugin.json`. If they
  differ or the wrapper is missing, run the setup upgrade (setup steps 4-6) first.

## 3. Choose the task

| The user wants | Task | Load |
|---|---|---|
| relax / optimize positions, volume or cell | `relax` | relax.md |
| stress-strain curve, tension, compression, elastic slope, yield | `sscurve` | stress-strain.md |
| status, restart, stop, plot or download of an existing run | sections Monitor ... Results below | troubleshooting.md if something failed |

Defaults are the production settings (fmax 0.01 eV/A, smax 0.05 GPa, econv 1e-7 or 1e-8 when fmax < 0.01,
`--precon none`, 1000 steps). Pass only options the user asked for or that differ from the defaults.

## 4. Check the input (on this PC, before uploading)

- Read the first 8 lines: VASP5 format with element symbols on line 6 and counts on line 7 (sum = atoms).
  No element line: ask for the elements and add it.
- Al uses the registry pseudopotential automatically. Any other element needs `--pp X=FILE` with a local
  PP that matches `--xc`, uploaded into the run folder; warn that results are unvalidated.
- Stress-strain: the reference must be the `_CONTCAR` of an `ofdft relax --relax full` run with
  `"state": "converged"` in its `<stem>_status.json` (same XC, PP and ecut), relaxed to at least the
  curve's thresholds (its `fmax`/`smax` <= the curve's; else the curve stops with
  `reference_not_relaxed`). Check that file (on the cluster if the relaxation ran there). Not converged,
  looser thresholds or no status file: offer to relax it (further) first.

## 5. Run folder

- `<remote_root>/<stem>_<task>[_<tag>]`, stem = file name without extension or trailing `_CONTCAR`, sanitised
  to `[A-Za-z0-9._-]`; tag for variants. Examples: `~/ofdft-runs/Al_gb_relax`,
  `~/ofdft-runs/Al_gb_sscurve_z`, `~/ofdft-runs/Al_gb_sscurve_z_comp`. One run per folder.
- `ssh -o BatchMode=yes <alias> 'ls -A <dir> 2>/dev/null'`. If it holds outputs and this is not a restart
  of that run, never reuse it silently: ask (new tag, restart, or overwrite).

## 6. Upload

```
ssh -o BatchMode=yes <alias> 'mkdir -p <dir>'
cd "<local folder of the file>" && scp -o BatchMode=yes "<local name>" <alias>:<dir relative to home>/<clean name>
```

Clean name = `[A-Za-z0-9._-]` only. A reference relaxed on the cluster is copied there instead, together
with its status file: `ssh -o BatchMode=yes <alias> 'cp <relax dir>/Al_gb_CONTCAR <relax dir>/Al_gb_status.json <dir>/'`.

## 7. Size the job

- `ofdft jobscript` defaults: `--ntasks auto` and `--mem auto` by atom count (cluster-im.md); `--time`
  3-00:00:00 (relax) or 10-00:00:00 (sscurve).
- For a new system size (no earlier run of a similar size), offer a 2-step probe job (ask; minutes to an
  hour):

```
ssh -o BatchMode=yes <alias> 'mkdir -p <dir>/probe && cp <dir>/<poscar> <dir>/probe/ && cd <dir>/probe && ~/.ofdft/bin/ofdft jobscript --task relax --poscar <poscar> --name probe_<stem> -- --relax full --steps 2'
ssh -o BatchMode=yes <alias> 'bash -lc "cd <dir>/probe && sbatch job.sh"'
```

  From `probe/<poscar name without extension>.log`: s/step = `time (s)` at step 2 minus step 1; memory =
  `mem(GB)`. Then
  `--time` = s/step x expected steps x 1.5 (relax: ~50 steps for small near-equilibrium cells, ~700 for
  as-built polycrystals; sscurve: points x ~40) and `--mem` = mem x 1.3, at least 8G.
- A job that runs out of time is not lost: resubmitting restarts it.

## 8. Job script and submission

In the run folder (examples; options after `--` go to the task verbatim):

```
ssh -o BatchMode=yes <alias> 'cd <dir> && ~/.ofdft/bin/ofdft jobscript --task relax --poscar Al_gb.vasp -- --relax full'
ssh -o BatchMode=yes <alias> 'cd <dir> && ~/.ofdft/bin/ofdft jobscript --task sscurve --poscar Al_gb_CONTCAR -- --axis z --mode tension --strain-step 0.005 --strain-max 0.05'
```

Add `--ntasks N`, `--mem <GB>G`, `--time D-HH:MM:SS` or `--partition NAME` before `--` when sizing gave
values. It writes `job.sh` and `run.json` and does not submit. Show the user its summary (ranks, memory,
time, command) and their `squeue -u $USER` count, ask, then:

```
ssh -o BatchMode=yes <alias> 'bash -lc "cd <dir> && sbatch job.sh"'
```

Report the job id, the run folder and the files to watch: `<prefix>_status.json`, `<prefix>.log`, and the
Slurm output `<name>_<jobid>.out` (name defaults to `<task>_<file name without extension>`). Relax prefix =
the POSCAR file name without extension; sscurve prefix = `<reference file name without extension>_ss_<axis>_<mode>[_comp]`
(`Al_gb_CONTCAR` -> `Al_gb_CONTCAR_ss_z_tension`). `ofdft jobscript` prints the prefix.

## Monitor

One ssh call per check, no more often than every few minutes:

```
ssh -o BatchMode=yes <alias> 'cd <dir> && cat <prefix>_status.json; tail -n 15 <prefix>.log; bash -lc "squeue -u $USER"'
```

- `<prefix>_status.json` is authoritative for convergence; never infer convergence from log columns.
  Relax: `running`, `converged`, `not_converged` (reason `steps` or `stop`), `error`. Sscurve: `running`,
  `complete`, `stopped`, `point_not_converged`, `error`, `reference_not_relaxed`.
- `running` but the job is gone from `squeue`, or `error` with a reason starting `Terminated:`: Slurm
  killed it (time limit, memory, scancel). Check
  `bash -lc "sacct -j <id> -o JobID,State,Elapsed,MaxRSS,ExitCode"` and `tail -n 30 <name>_<jobid>.out`,
  then diagnose with troubleshooting.md.
- Progress: relax - max|F| and max|sig| against the thresholds; grain-boundary cells have a long slow tail.
  Sscurve - `points_done` / `strains_total` and the last point line. ETA (say it is an estimate): relax =
  remaining steps x s/step, s/step from the `time (s)` column (difference between steps); sscurve =
  remaining points x ~40 steps x s/step, s/step = T / N from the per-point log lines (`converged in N
  steps / M calls, T s`), or `time_s` / `steps` of recent CSV rows (or of `last_point` in the status).
- Pending jobs: the reason column of `squeue`; free cores with `sinfo` (cluster-im.md).

## Restart

- Resubmit the same `job.sh` (ask): `ssh -o BatchMode=yes <alias> 'bash -lc "cd <dir> && sbatch job.sh"'`.
  Relax adds `--restart` automatically; sscurve continues from its CSV.
- First look for a stale STOP file (`ls <dir>`): if `<prefix>.STOP` or `STOP` exists, tell the user and
  remove it (`rm -f <prefix>.STOP STOP`, ask), or job.sh refuses to start.
- More steps (relax and sscurve), new fmax/smax/econv (relax only), more memory or time: regenerate job.sh.
  `ofdft jobscript` takes the task options and the prefix only from its arguments, so first
  `ssh ... 'cd <dir> && cat run.json'` and repeat ALL its `args` after `--` plus its `--ntasks`, `--mem`,
  `--time`, `--partition`, `--name`, changing only what is needed. Check that the printed prefix is
  unchanged (it warns when run.json had another prefix) before sbatch: another prefix starts a NEW run.
  A sscurve with new tolerances, and any changed physics, is refused (exit 2): use a new folder or
  prefix. job.sh pins the ofdft-opt version that wrote it; regenerating it moves the run to the
  installed version.

## Stop

Ask first. Clean stop: `ssh -o BatchMode=yes <alias> 'touch <dir>/<prefix>.STOP'` - the run finishes the
current step (minutes for large cells), writes its files and status, and exits. Hard stop:
`ssh -o BatchMode=yes <alias> 'bash -lc "scancel <id>"'` - the CONTCAR of the last step is intact, the
status stays `running` (or becomes `error`, reason `Terminated: ...`).

## Results

- Download (tar form in ssh.md; ask for the local folder, default: next to the input): relax
  `<prefix>_CONTCAR`, `<prefix>.log`, `<prefix>_status.json`, `<prefix>_params.json`; sscurve `<prefix>.csv`,
  `<prefix>.log`, `<prefix>_status.json`, `<prefix>_params.json`. Point CONTCARs
  (`<prefix>_e+0.0100_CONTCAR`, ...) only on request; `.xyz` only if asked (check its size; ask if > 100 MB).
- Plot on the cluster and download the PNG:
  `ssh -o BatchMode=yes <alias> 'cd <dir> && ~/.ofdft/bin/ofdft plot <prefix>.csv'` writes `<prefix>.png`.
- Relax report: converged or not (from status), energy (eV and eV/atom), max|F|, max stress, cell, steps,
  wall time, thresholds, ofdft-opt version.
- Stress-strain report: initial slope, peak stress and its strain, number of points, thresholds. Before
  reporting a peak stress or yield strain, the tight re-check in stress-strain.md is REQUIRED (re-relax the
  peak point and the next one with `--strains e1 e2 --start <prefix>_e+X_CONTCAR --fmax 0.005 --smax 0.01`
  in a new folder and prefix); otherwise label the values "unverified, thresholds 0.01/0.05".
- Always state the thresholds with any result.

## Don't

- Don't run `ofdft relax` or `ofdft sscurve` on the login node, not even small or EMT tests.
- Don't reuse a folder with outputs for a new run, and don't use `--overwrite` without asking.
- Don't edit job.sh by hand; regenerate it with `ofdft jobscript`.
- Don't `module load openmpi` or use `srun` for ofdft jobs; job.sh uses the env's own `mpirun`.
- Don't compare energies across different PP, XC, ecut, econv or ofdft-opt versions.
