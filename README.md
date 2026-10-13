# hpc-skills

A Claude Code plugin marketplace for running materials-science simulations on a Slurm cluster by asking
Claude. It currently holds one plugin, **ofdft**: orbital-free DFT (DFTpy) structure relaxation and
stress-strain curves. Claude prepares the job, submits it over SSH, watches it, restarts it after a time
limit and plots the results; the physics runs in a small Python package (`ofdft-opt`) that the plugin
installs in your cluster account.

Validated scope: aluminium with BLPS local pseudopotentials, the Wang-Teter kinetic energy functional and
PBE or LDA exchange-correlation, single-node MPI runs, tested on the IMLab `im` cluster. Other elements run
only with a pseudopotential you supply, and the results are unvalidated.

## Prerequisites

On your PC:
- Claude Code (CLI) or the Claude desktop app (Code tab).
- Windows: [Git for Windows](https://git-scm.com/download/win). Claude's Bash tool needs Git Bash, and all
  cluster commands run there (never in PowerShell).
- Key-based SSH to the cluster under an alias in `~/.ssh/config` that works without any prompt:
  `ssh -o BatchMode=yes <alias> 'echo ok'` must print `ok`. Run `ssh <alias>` once by hand in a terminal
  to accept the host key. A passphrase-protected key must be loaded in an ssh-agent. Windows with the key
  in the Windows "OpenSSH Authentication Agent" service: Git Bash's own ssh cannot use that agent, so add
  `~/bin/ssh` and `~/bin/scp` wrappers that run the Windows client with `MSYS_NO_PATHCONV=1`
  ([recipe](plugins/ofdft/reference/ssh.md#first-time-ssh-setup-each-person-once), step 4).
- Network access to the cluster: on campus, or off campus through the campus VPN or a `ProxyJump` host you
  have an account on.

On the cluster:
- Miniforge (conda-forge), so setup can create the environment, or an existing environment with DFTpy
  2.2.0, mpi4py and mpi4py-fft from conda-forge.

## Install

```
claude plugin marketplace add ye-min-thant/hpc-skills
claude plugin install ofdft@hpc-skills
```

Desktop app: in the Code tab, open the plugin settings, add the marketplace `ye-min-thant/hpc-skills` and
install `ofdft` (or run the two commands above in a terminal once).

## First run

Start a Claude Code session and run once:

```
/ofdft:setup
```

It checks your SSH alias, finds or creates the Python environment on the cluster (asking first), copies
the `ofdft-opt` package to `~/.ofdft/` on the cluster, downloads the pseudopotentials, runs `ofdft doctor`,
optionally submits a 4-rank smoke test, and saves your settings in `~/.ofdft/config.json` on your PC.

## Example prompts

- "Relax `Al_gb.vasp` with OF-DFT (DFTpy) on the cluster, full cell relaxation."
- "Relax the positions only of `Al_vacancy.vasp` with ofdft, LDA."
- "Run an ofdft z-tension stress-strain curve up to 5 % from `Al_gb_CONTCAR`."
- "How is my ofdft relaxation of Al_gb doing?"
- "Restart the ofdft run that hit the time limit."
- "Plot the stress-strain curve and download the CSV and the PNG."
- "Stop the ofdft stress-strain run cleanly."

Claude always shows the resources and the command and asks before it submits, stops or overwrites
anything. You can also call the main skill directly: `/ofdft:optimize`.

## Update

```
claude plugin marketplace update hpc-skills
claude plugin update ofdft@hpc-skills
```

Then run `/reload-plugins` in an open session (or start a new one) and ask Claude to
"upgrade ofdft on the cluster". The new version is installed next to the old one, so queued and running
jobs keep the version they were written for.

## Uninstall

```
claude plugin uninstall ofdft@hpc-skills
claude plugin marketplace remove hpc-skills
```

Files on the cluster stay until you remove them yourself: the package and pseudopotentials in `~/.ofdft/`,
and your run folders (default `~/ofdft-runs/`). On your PC, delete `~/.ofdft/config.json`.

## Using the CLI without Claude

Everything Claude does goes through the `ofdft` command (`relax`, `sscurve`, `jobscript`, `doctor`, `pp`,
`plot`, `version`). See [plugins/ofdft/package/README.md](plugins/ofdft/package/README.md) for installation
and usage by hand.

## How to cite

If you publish results obtained with this plugin, cite the underlying methods:

- DFTpy: X. Shao, K. Jiang, W. Mi, A. Genova, M. Pavanello, WIREs Comput. Mol. Sci. 11, e1482 (2021).
- BLPS pseudopotentials: C. Huang, E. A. Carter, Phys. Chem. Chem. Phys. 10, 7109 (2008).
- Wang-Teter kinetic energy functional: L.-W. Wang, M. P. Teter, Phys. Rev. B 45, 13196 (1992).
- Exp preconditioner (only if you used `--precon exp`): D. Packwood et al., J. Chem. Phys. 144, 164109
  (2016).

Third-party credits and licences: [NOTICE.md](NOTICE.md).

## Licence

MIT, see [LICENSE](LICENSE). The BLPS pseudopotentials are not part of this repository; `ofdft pp fetch`
downloads them from their original source under their own licence (CC BY-ND 4.0).

## Status and limitations

- Version 0.1: aluminium only, single-node MPI (the conda environment's own `mpirun`), tested on the
  IMLab `im` cluster. Other Slurm clusters should work but are untested.
- Peak stresses and yield strains depend strongly on the per-point convergence thresholds; the plugin
  requires a tight re-check before reporting them.
- Issues and suggestions: https://github.com/ye-min-thant/hpc-skills/issues
