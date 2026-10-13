# Troubleshooting: signature -> cause -> action

Look in this order: `<prefix>_status.json`, the end of the Slurm output `<name>_<jobid>.out`, the end of
`<prefix>.log`, then `sacct -j <id> -o JobID,State,Elapsed,MaxRSS,ExitCode`. Ask before resubmitting,
cancelling or deleting anything.

## Job killed by Slurm (status stays `running`, or `error` with a reason starting `Terminated:`)

| Signature | Cause | Action |
|---|---|---|
| `oom_kill event`, `Killed`, `signal 9`, exit code 137, sacct `OUT_OF_MEMORY` | out of memory | Read the last `mem(GB)` in the log; regenerate job.sh (see "Regenerating job.sh" below) with `--mem` = that x 1.3 (or fewer ranks), resubmit. Relax restarts from its CONTCAR; sscurve from its CSV/partial CONTCAR. |
| `CANCELLED AT ... DUE TO TIME LIMIT`, sacct `TIMEOUT` | `--time` too short | Resubmit the same `job.sh` (= restart). For the next segment regenerate it (see below) with a longer `--time` (s/step x remaining steps x 1.5). |
| sacct `CANCELLED` without time limit | scancel by the user or an admin | Ask the user; resubmit to continue. |
| status `running` but the job is not in `squeue` | any of the above | Check `sacct` and the `.out`; the status file cannot be updated after a hard kill. |
| status `error`, reason `Terminated: terminated by signal 15 ...` | the run caught Slurm's SIGTERM (time limit or scancel) | Same as the rows above: not a code error; check `sacct`, then resubmit `job.sh` to continue. |

## Refusals before the run starts (exit code 2: nothing was changed)

| Signature | Cause | Action |
|---|---|---|
| refuses to start because `<prefix>.log` or `<prefix>_CONTCAR` exists | outputs from an earlier run | Restart with `--restart` (job.sh adds it automatically when the CONTCAR exists), use a new run folder, or `--overwrite` (destroys the old outputs: ask). Log but no CONTCAR (the first job died during step 0, e.g. out of memory): regenerate job.sh (see below) adding `--restart` after `--` (it then starts from the POSCAR and appends to the log). |
| restart refused, lists parameter differences | physics or tolerance changed with the same prefix, often a job.sh regenerated without the original task options | Go back to the original options (`cat run.json`: `args`), or use a new `--prefix` / run folder. Relax allows changing fmax/smax/econv/steps; sscurve allows only `--steps`. |
| `... was made by the old ss_curve.py script ... with different options` | a curve started by the original `ss_curve.py`, continued with other options | Repeat the old options listed in its `_params.json` (the old script's defaults were `--econv-loose 1e-6 --precon exp --steps 500`; only the options in that file are checked). With the same options the curve continues and the params file is rewritten in the new format. |
| relax `--restart` refused: `... is not the structure that run ... started from` | a different POSCAR under the old file name in a used run folder | New run folder (or `--prefix`) for the new structure; `--overwrite` only on request (destroys the old outputs). |
| sscurve refused: `... already exist but <prefix>.csv does not` | the prefix belongs to another run (e.g. an `ofdft relax` with the same prefix) or the csv was deleted | Use the default prefix or another `--prefix` / run folder. `--overwrite` replaces those files: ask first. |
| sscurve refused: `does not match the reference` | `--start` or a point CONTCAR is from another structure (atom count or order) | Use a CONTCAR of a curve from this reference; a new run folder for another structure. |
| sscurve refused: `cannot read the CONTCAR ...` / `partial CONTCAR ...` | a point file was emptied or truncated | Restore it, or delete it (ask): the point is then computed again. |
| xc / pseudopotential mismatch refused | e.g. `al.lda.recpot` with `--xc pbe` | Use the matching file (`al.gga.recpot` for PBE, `al.lda.recpot` for LDA) or drop `--pp` to use the default. `--allow-xc-mismatch` only on explicit request. |
| `no default pseudopotential for X (only Al is validated); pass --pp X=FILE` | element without a registry PP | Get a matching local PP from the user, upload it to the run folder or `~/.ofdft/pp`, pass `--pp X=FILE`; results are unvalidated. |
| PP file not found / no registry files | `ofdft pp fetch` not run | `~/.ofdft/bin/ofdft pp fetch`; offline: download on a PC and copy into `~/.ofdft/pp/`, then `ofdft pp list`. |
| sha256 mismatch for a registry PP | corrupted or wrong file | Delete that file from `~/.ofdft/pp` (ask) and `ofdft pp fetch` again. |
| sscurve refused: reference status not converged | the reference relaxation did not finish | Finish it (resubmit the relax run) or relax with `ofdft relax --relax full`. |
| status `reference_not_relaxed` | residual force/stress at strain 0 > 2 x fmax / 2 x smax (the message says which). Also when the reference was relaxed with looser thresholds than the curve (e.g. relax defaults 0.01/0.05, curve 0.005/0.01) | Relax the reference with `--relax full`, the same XC/PP/ecut, and at least the curve's `--fmax`/`--smax` (a `--restart` of that relaxation with the tighter thresholds). `--allow-unrelaxed-reference` only if the user insists. |

## Errors during the run

| Signature | Cause | Action |
|---|---|---|
| `VASP4 POSCAR without element line` | POSCAR has no element-symbols line | Add the element names as line 6 (VASP5 format), upload again. |
| `DFTpy density optimization failed twice` (status `error`) | SCF failure, often after a large step | Resubmit (the CONTCAR of the last good step is intact). If it repeats, look at `<prefix>_scf.log`; try a smaller `--maxstep`. |
| occasional `WARNING: DFTpy density not converged; retrying from uniform density` | one SCF restart | Harmless if it happens rarely (count in the final line). |
| `NOT converged`, status `not_converged`, reason `steps` | step budget used up | Resubmit `job.sh` (restart); for a larger budget regenerate it (see below) with a larger `--steps`. |
| sscurve status `point_not_converged` | one strain point hit `--steps` | Rerun (continues from the partial CONTCAR), with more `--steps` if needed (regenerate job.sh, see below). Often a plastic event. |
| `ERROR: stop file .../<prefix>.STOP exists` (or `.../STOP`) in the `.out`, job ends at once; or a run by hand stops after step 0 and its last log line names the STOP file | stale STOP file | Delete it (ask; `rm -f <prefix>.STOP STOP` in the run folder), resubmit. |
| max\|F\| stuck at 0.01-0.05 eV/A for hundreds of steps, energy still dropping slowly | grain-boundary reconstruction (normal for polycrystals) | Not a failure; let it run (13k atoms needed 674 steps). |
| `warning: command substitution: ignored null byte in input` in the `.out` | from the MPI wire-up test | Harmless. |

## Environment and MPI

| Signature | Cause | Action |
|---|---|---|
| MPI test prints many lines `1 \| ...` or `COMM_WORLD.size == 1` | mpi4py not linked to the launcher's MPI | Use the env's own `mpirun` (job.sh does this); do not load an `openmpi` module; rebuild the env from environment.yml if needed. |
| `No module named 'mpi4py_fft'` / doctor WARN mpi4py_fft | missing in the env | Install into the env from conda-forge (ask): `ssh ... 'cd ~ && <conda base>/bin/conda install -y -n <env name> -c conda-forge mpi4py-fft'`, or recreate the env from environment.yml. |
| job aborts right after `ofdft doctor --brief` | a doctor FAIL | Read the FAIL lines in the `.out`; run `~/.ofdft/bin/ofdft doctor` on the login node. |
| doctor FAIL: dftpy is a dev/git version | DFTpy installed from GitHub (wrong XC stress under MPI) | Reinstall the release into the env (ask): `ssh ... 'cd ~ && <env python> -m pip install dftpy==2.2.0'`. |
| `conda activate` failed / `conda.sh` not found | env moved or deleted, or job.sh from another machine | Fix `OFDFT_PYTHON` in `~/.ofdft/config.sh` (rerun setup), then regenerate job.sh with `ofdft jobscript`. |
| `No module named ofdft_opt` | the version directory `~/.ofdft/lib/<ver>` is missing | Reinstall that version (setup install step). job.sh pins its version's directory on PYTHONPATH. |
| `ofdft version` differs from the plugin version | plugin updated, cluster not | Run the setup upgrade (install, pp, doctor). Old run folders keep their pinned version. |
| `sbatch: command not found` / `squeue: command not found` | Slurm command without login shell | Wrap in `bash -lc "..."`. |
| `/bin/bash^M: bad interpreter` or `$'\r': command not found` | job.sh has Windows CRLF line endings | `sed -i 's/\r$//' job.sh` on the cluster. |

## Regenerating job.sh (more steps, memory or time)

`ofdft jobscript` builds the task options and the prefix only from its own arguments. First
`cat run.json` in the run folder and repeat ALL its `args` after `--`, plus `--ntasks`, `--mem`, `--time`,
`--partition` and `--name` from it, changing only what is needed. Check that the printed `prefix` is the
same as before (it also warns when run.json describes another prefix) before `sbatch`: a different prefix
starts a NEW run (e.g. a compression curve regenerated without its negative `--strain-step` becomes a new
tension curve). Relax may change `--steps`, `--fmax`, `--smax`, `--econv`; sscurve only `--steps`.

## SSH

| Signature | Cause | Action |
|---|---|---|
| `Permission denied (publickey)` | key not loaded or not authorised | Check the alias and key (ssh.md); with a passphrase, the key must be in an agent. Windows: a key in the Windows agent service works only with the Windows ssh client: check `command -v ssh; ssh -V` in the Bash tool and set up the `~/bin/ssh` wrapper (ssh.md, first-time setup step 4). Never remove the passphrase. |
| a remote path arrives as `C:/Program Files/Git/home/...` | Windows ssh client called from Git Bash: MSYS rewrote an argument starting with `/` | Start the remote command with `cd ~ && `; add `export MSYS_NO_PATHCONV=1` to the `~/bin/ssh` wrapper (ssh.md). |
| `Host key verification failed` | first connection, host key not accepted | User runs `ssh <alias> hostname` once interactively in a terminal and verifies the fingerprint. Never `StrictHostKeyChecking=no`. |
| `Could not resolve hostname` / timeout | off campus without VPN/ProxyJump, or wrong alias | Connect the VPN or add ProxyJump; check `~/.ssh/config`. |
| quotes lost, command ran in `$HOME` | command went through PowerShell | Rerun with the Bash tool. |
