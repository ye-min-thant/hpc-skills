# The IMLab `im` cluster (where ofdft was tested)

Facts as documented by the lab's imlab plugin. Hardware and partitions can change: check live with `sinfo`
before relying on a number. Other Slurm clusters work the same way if the facts below are adapted.

## Hardware and partitions

| Node | CPU | Cores | Memory | Role |
|---|---|---|---|---|
| im1 | 2x AMD EPYC 7713 | 128 | ~514 GB | login node + compute |
| im2 | AMD EPYC | 64 | ~257 GB | compute |
| im3 | 2x AMD EPYC 7713 | 128 | ~257 GB | compute |

- No GPUs, no containers.
- Slurm 24.05. Partitions `main` (default), `long`, `huge`, and possibly `debug` (verify with `sinfo -s`).
  No wall-time limit on any partition, but always give a realistic `--time` so the scheduler can plan.
- Scheduling counts cores AND memory, so `--mem` matters (`ofdft jobscript` always sets it).
- Accounting is not enforced: no `--account` needed.
- Storage: `/home` (500 GB, shared) and `/data` (9.1 TB, shared); no separate scratch. Large trajectories
  (`.xyz`) of big cells are ~100 MB each: clean up or move to `/data` when space gets tight.
- Slurm commands need a login shell: always `bash -lc "squeue ..."` over ssh.

## Rules for ofdft jobs

- **The login node (im1) is for seconds-long commands only**: `ofdft doctor`, `ofdft pp`, `ofdft jobscript`,
  `ofdft plot`, file operations, `sbatch`/`squeue`. Every relaxation or stress-strain run goes through
  `sbatch`, even a 2-step probe.
- **One node per job.** The job script launches the conda environment's own `mpirun` with
  `python -m mpi4py`; multi-node runs with conda Open MPI are untested. Do not `module load openmpi` and
  do not use `srun` for ofdft (this overrides the generic advice of the imlab job-submission skill).
- **Ranks by size** (`--ntasks auto`): < 500 atoms 8, < 2000 atoms 16, < 6000 atoms 32, else 64. 64 ranks
  fit on every node. More ranks than this rarely pay off for small FFT grids.
- **Memory**: `--mem auto` = max(8, ceil(0.0065 x atoms)) GB; measured 38-43 GB for 13,140 atoms at 64 ranks.
- **Fair use**: the cluster has three nodes shared by the whole lab. Keep at most 2 large jobs (32+ ranks
  or multi-day) running per person unless the lab agrees otherwise. Check `squeue -u $USER` before every
  submission and tell the user if they already have 2 or more running jobs.

## Seeing free capacity

```
ssh -o BatchMode=yes <alias> 'bash -lc "sinfo -N -o \"%N %P %T %C %m\""'
```

`%C` prints CPUs as allocated/idle/other/total, `%m` the memory in MB. `squeue` without `-u` shows
everyone's jobs. A pending job with reason `Resources` or `Priority` is waiting for cores or memory.

## Software

- No Python, conda or DFTpy module on the cluster. Each person uses a Miniforge environment (conda-forge
  only) created from `package/environment.yml`, or an existing env with DFTpy 2.2.0, mpi4py and
  mpi4py-fft from conda-forge. The login node has been used to build such environments.
- The cluster's own `openmpi/*` modules are for other codes; do not mix them with the conda env.

## Access

Key-based SSH only. Off campus, use the campus VPN or a `ProxyJump` through a host you have an account on.
See ssh.md. The lab's imlab plugin also offers MCP tools for this cluster, but they fail on Windows
(`os.WUNTRACED`), so ofdft uses plain `ssh`/`scp` from the Bash tool.
