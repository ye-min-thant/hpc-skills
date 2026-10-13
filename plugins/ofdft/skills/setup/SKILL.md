---
name: setup
description: Use once per person before the first orbital-free DFT (OF-DFT / DFTpy / ofdft) run, or when an ofdft task finds ~/.ofdft/config.json, the SSH alias, the cluster Python environment or the ofdft-opt package missing or outdated (including "upgrade ofdft on the cluster" after a plugin update). Checks SSH, sets up a conda-forge DFTpy environment, installs ofdft-opt on the Slurm cluster, fetches pseudopotentials and runs ofdft doctor.
---

# ofdft setup (once per person; upgrade after a plugin update)

Installs the ofdft-opt package that ships with this plugin into the user's account on their Slurm cluster
and records the connection details on this PC. With an existing DFTpy environment this takes a few
minutes. Reference files: `${CLAUDE_PLUGIN_ROOT}/reference/ssh.md` (command forms, first-time SSH),
`${CLAUDE_PLUGIN_ROOT}/reference/cluster-im.md`, `${CLAUDE_PLUGIN_ROOT}/reference/troubleshooting.md`.

## Rules for every step

- Run every ssh/scp/tar command with the **Bash tool** (Git Bash on Windows), never PowerShell (Windows
  PowerShell 5.1 strips embedded quotes). Always `ssh -o BatchMode=yes <alias> '...'`; Slurm commands as
  `bash -lc "..."` inside the single quotes.
- Never compute on the login node beyond `ofdft doctor/pp/jobscript/plot` (seconds). Never use
  `StrictHostKeyChecking=no`, never remove SSH key passphrases, never enter passwords.
- Ask before: submitting a job, cancelling/stopping, overwriting a run folder, downloading > 100 MB,
  installing Miniforge, creating a conda environment, editing any dotfile.
- Before submitting: `squeue -u $USER`; if the user already has >= 2 running jobs on the cluster, say so
  and ask.
- Scope: orbital-free DFT with DFTpy, validated for Al (BLPS, WT KEDF, PBE/LDA). Other elements only with a
  user-supplied matching local pseudopotential; results unvalidated - say so.

**Full setup** = steps 1-8. **Upgrade** (plugin updated, cluster still on the old version) = steps 4-6
with `<env python>` taken from the local config (or `cat ~/.ofdft/config.sh` on the cluster), then update
`version` in the local config (step 8).

## 1. Local checks

- In the Bash tool: `echo ok; command -v ssh scp tar`. If the Bash tool is missing (Windows without Git for
  Windows), stop and ask the user to install Git for Windows and restart Claude Code.
- `cat ~/.ofdft/config.json 2>/dev/null` - reuse its values as defaults; ask only about what is missing
  or changing.
- Read `version` from `${CLAUDE_PLUGIN_ROOT}/.claude-plugin/plugin.json`; call it `<ver>` below.

## 2. SSH alias

Ask for the SSH alias (the `Host` name in `~/.ssh/config`) and the run root on the cluster (default
`~/ofdft-runs`). Test:

```
ssh -o BatchMode=yes <alias> 'echo ok; hostname'
```

If it does not print `ok`, stop and explain (details in ssh.md, "First-time SSH setup"): key-based login
with an alias in `~/.ssh/config`; off campus a VPN or `ProxyJump`; run `ssh <alias>` once interactively in
a Git Bash terminal to accept the host key (Claude cannot answer that prompt); a passphrase-protected key
must be loaded in an ssh-agent. Windows with the key in the Windows agent service: Git Bash's own ssh
cannot use it, so bare `ssh`/`scp` must run the Windows client through `~/bin/ssh` and `~/bin/scp`
wrappers (ssh.md, first-time setup step 4; check `command -v ssh; ssh -V` -> `OpenSSH_for_Windows`).
Retest when the user says it works.

## 3. Python environment on the cluster

Ask whether the user already has a conda env on the cluster with DFTpy 2.2.0, mpi4py and mpi4py-fft from
conda-forge.

**Existing env** - get its python path and check it (presence only: importing mpi4py_fft would start MPI
on the login node; the real import is tested by the job's `doctor --mpi`):

```
ssh -o BatchMode=yes <alias> 'cd ~ && <python> -' <<'EOF'
import importlib.util, dftpy
print("dftpy", dftpy.__version__)
print("missing:", [m for m in ("mpi4py", "mpi4py_fft") if importlib.util.find_spec(m) is None])
EOF
```

Accept 2.1.2-2.2.x. A dev/git DFTpy gives a wrong XC stress under MPI: create a new env instead.

**New env** from `${CLAUDE_PLUGIN_ROOT}/package/environment.yml` (conda-forge only, pinned to the tested
versions; needs Miniforge):

1. Find conda: `ssh -o BatchMode=yes <alias> 'ls -d ~/miniforge3 ~/mambaforge 2>/dev/null; bash -lc "command -v conda"'`.
   If there is none, offer to install Miniforge (ask; ~100 MB download on the cluster; `-b` does not touch
   `~/.bashrc`):
   ```
   ssh -o BatchMode=yes <alias> 'cd ~ && curl -fsSLo Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && bash Miniforge3.sh -b -p ~/miniforge3 && rm Miniforge3.sh'
   ```
   With an Anaconda/Miniconda base, tell the user that the env file lists `nodefaults`, so the Anaconda
   `defaults` channel is not used; Miniforge is still preferred.
2. Upload the env file from its own directory with a relative name:
   ```
   ssh -o BatchMode=yes <alias> 'mkdir -p ~/.ofdft'
   cd "${CLAUDE_PLUGIN_ROOT}/package" && scp -o BatchMode=yes environment.yml <alias>:.ofdft/environment-<ver>.yml
   ```
3. Ask first (several minutes of solver time on the login node, a few GB in the home directory). If an env
   named `ofdft` already exists, ask whether to check and reuse it or pick another name. Run it detached so
   a dropped connection cannot leave a half-built env, then poll the log every minute or two until it
   ends:
   ```
   ssh -o BatchMode=yes <alias> 'nohup <conda base>/bin/conda env create -n ofdft -f ~/.ofdft/environment-<ver>.yml > ~/.ofdft/env-create.log 2>&1 < /dev/null &'
   ssh -o BatchMode=yes <alias> 'tail -n 5 ~/.ofdft/env-create.log'
   ```
4. The env python is `<conda base>/envs/ofdft/bin/python`; run the import check above.

Use the absolute path (`/home/...`, not `~`) as `<env python>` from here on.

## 4. Install ofdft-opt `<ver>`

`<ver>` is `version` in `${CLAUDE_PLUGIN_ROOT}/.claude-plugin/plugin.json`. Versions are installed side by
side in `~/.ofdft/lib/<ver>/`; queued and running jobs keep the version pinned in their job.sh.

1. `ssh -o BatchMode=yes <alias> 'ls ~/.ofdft/lib/ 2>/dev/null; test -d ~/.ofdft/lib/<ver>/ofdft_opt && echo present'`.
   If `present`, skip the upload: an installed version is never overwritten.
2. Upload with a tar pipe (the plugin path works with `tar -C`, not with scp):
   ```
   tar -C "${CLAUDE_PLUGIN_ROOT}/package/src" -czf - --exclude=__pycache__ ofdft_opt | ssh -o BatchMode=yes <alias> 'mkdir -p ~/.ofdft/lib/<ver> && tar -xzf - -C ~/.ofdft/lib/<ver>'
   ```
3. Write the config and the wrapper. Fill in `<env python>` and `<ver>`; keep the lines flush left
   exactly as below (the quoted heredoc keeps `$` literal, and `EOF` must start the line):

```
ssh -o BatchMode=yes <alias> 'mkdir -p ~/.ofdft/bin && cat > ~/.ofdft/config.sh' <<'EOF'
OFDFT_PYTHON=<env python>
OFDFT_VERSION=<ver>
EOF
ssh -o BatchMode=yes <alias> 'cat > ~/.ofdft/bin/ofdft && chmod +x ~/.ofdft/bin/ofdft' <<'EOF'
#!/bin/bash
source ~/.ofdft/config.sh
export PYTHONPATH=$HOME/.ofdft/lib/$OFDFT_VERSION
export PYTHONNOUSERSITE=1
exec "$OFDFT_PYTHON" -m ofdft_opt "$@"
EOF
```

4. `ssh -o BatchMode=yes <alias> '~/.ofdft/bin/ofdft version'` must print `ofdft-opt <ver> (...)`.

All skills call the wrapper by its full path `~/.ofdft/bin/ofdft`. Adding `~/.ofdft/bin` to PATH in
`~/.bashrc` is optional and only with the user's consent.

## 5. Pseudopotentials

```
ssh -o BatchMode=yes <alias> '~/.ofdft/bin/ofdft pp fetch'
```

Downloads the BLPS Al files (`al.gga.recpot`, `al.lda.recpot`, < 1 MB) into `~/.ofdft/pp` and verifies their
sha256; valid files are skipped. If the cluster has no internet access: ask, download both files on this
PC into a new empty folder from the URLs in the fetch error message (also in `REGISTRY` in
`${CLAUDE_PLUGIN_ROOT}/package/src/ofdft_opt/pp.py`), then
`cd "<that folder>" && scp -o BatchMode=yes al.gga.recpot al.lda.recpot <alias>:.ofdft/pp/` and check with
`~/.ofdft/bin/ofdft pp list`.

## 6. Doctor

```
ssh -o BatchMode=yes <alias> '~/.ofdft/bin/ofdft doctor'
```

It must exit 0 (no FAIL lines); fix each FAIL with troubleshooting.md ("Environment and MPI"). Treat a
missing `mpi4py_fft` as a must-fix even though it is only a WARN: no MPI run works without it. Report the
other WARN lines. Doctor does not start MPI on the login node.

## 7. Optional MPI smoke test (ask; 4 ranks, a few minutes)

```
ssh -o BatchMode=yes <alias> 'mkdir -p <remote_root>/_smoketest && ls -A <remote_root>/_smoketest'
```

If the folder already holds files from an earlier test, ask before clearing it. Then:

```
ssh -o BatchMode=yes <alias> 'cd <remote_root>/_smoketest && cp ~/.ofdft/lib/<ver>/ofdft_opt/data/al_vacancy_255.vasp . && ~/.ofdft/bin/ofdft jobscript --task relax --poscar al_vacancy_255.vasp --ntasks 4 --mem 8G --time 00:20:00 -- --steps 2'
ssh -o BatchMode=yes <alias> 'bash -lc "squeue -u $USER"'
ssh -o BatchMode=yes <alias> 'bash -lc "cd <remote_root>/_smoketest && sbatch job.sh"'
```

After a few minutes:

```
ssh -o BatchMode=yes <alias> 'cd <remote_root>/_smoketest && tail -n 40 relax_al_vacancy_255_*.out; cat al_vacancy_255_status.json'
```

Pass: `doctor OK`; the MPI test prints exactly one line starting with `4 |`; the log header says
`MPI ranks: 4`; steps 0-2 are printed; exit code 0; status `not_converged` with reason `steps` (expected
with `--steps 2`). Otherwise diagnose with troubleshooting.md.

## 8. Save the local config

```
mkdir -p ~/.ofdft && cat > ~/.ofdft/config.json <<'EOF'
{"ssh_alias": "<alias>", "remote_root": "<remote_root>", "version": "<ver>", "env_python": "<env python>"}
EOF
```

Finish with a short summary (alias, env python, DFTpy version, ofdft-opt version, doctor result, smoke
test) and example requests, e.g. "Relax Al_gb.vasp with OF-DFT, full cell relaxation" or "Run a z-tension
stress-strain curve with ofdft from Al_gb_CONTCAR".

## Don't

- Don't edit `~/.bashrc` or other dotfiles without asking.
- Don't install into the conda base env, and don't use the Anaconda `defaults` channel.
- Don't pip-install ofdft-opt or extra packages into the env; the package is copied, not installed.
- Don't overwrite or delete an installed `~/.ofdft/lib/<ver>`; jobs may be using it.
- Don't run relaxations on the login node; even the smoke test goes through sbatch.
