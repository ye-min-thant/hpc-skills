# SSH, scp and tar: exact command forms

All cluster access in ofdft is plain `ssh`/`scp`/`tar` run from Claude's **Bash tool** (Git Bash on
Windows). Below, `<alias>` is the user's SSH alias from `~/.ofdft/config.json`, and run folders live under
`<remote_root>` (default `~/ofdft-runs`).

## Rules

- **Bash tool only, never PowerShell.** Windows PowerShell 5.1 strips embedded double quotes when it calls
  native programs: `ssh <alias> 'bash -lc "cd X && sbatch job.sh"'` silently becomes
  `bash -lc cd X && sbatch job.sh`, and sbatch then runs in `$HOME`.
- **Always `ssh -o BatchMode=yes`** so a missing key or unknown host fails immediately instead of waiting
  for a prompt that nobody can answer.
- **Never** add `StrictHostKeyChecking=no`, never remove a key passphrase, never type or store passwords.
- **Batch queries** into one ssh call and poll a running job no more often than every few minutes (off
  campus each call may be two logins).
- **Never start the remote command with an absolute path** (`'/home/...'`): on Windows, Git Bash may
  rewrite it to `C:/Program Files/Git/home/...` (see Windows notes). Start with `cd ~ && ` instead:
  `ssh -o BatchMode=yes <alias> 'cd ~ && /home/me/miniforge3/envs/ofdft/bin/python -V'`.
- Every command must end by itself: `tail -n 20`, never `tail -f` or `watch` (the Bash tool would hang).
- On the cluster the CLI is the wrapper `~/.ofdft/bin/ofdft` (not on PATH unless the user added it).

## Command forms

Plain commands (files, ofdft CLI):

```
ssh -o BatchMode=yes <alias> 'echo ok; hostname'
ssh -o BatchMode=yes <alias> '~/.ofdft/bin/ofdft version'
ssh -o BatchMode=yes <alias> 'cd ~/ofdft-runs/Al_gb_relax && ~/.ofdft/bin/ofdft jobscript --task relax --poscar Al_gb.vasp -- --relax full'
ssh -o BatchMode=yes <alias> 'cd ~/ofdft-runs/Al_gb_relax && cat Al_gb_status.json; tail -n 20 Al_gb.log'
```

Slurm commands need a login shell; wrap them in `bash -lc "..."` inside the single quotes:

```
ssh -o BatchMode=yes <alias> 'bash -lc "cd ~/ofdft-runs/Al_gb_relax && sbatch job.sh"'
ssh -o BatchMode=yes <alias> 'bash -lc "squeue -u $USER"'
ssh -o BatchMode=yes <alias> 'bash -lc "sacct -j 123456 -o JobID,JobName,State,Elapsed,MaxRSS,ExitCode"'
ssh -o BatchMode=yes <alias> 'bash -lc "scancel 123456"'
```

Quoting: the outer single quotes keep the local shell from touching the command; the inner double quotes
group it for `bash -lc`. Never put a single quote inside; escape inner double quotes as `\"`.

Write a small text file on the cluster (local heredoc fed to `cat` on the remote side):

```
ssh -o BatchMode=yes <alias> 'cat > ~/ofdft-runs/notes.txt' <<'EOF'
text written verbatim
EOF
```

## Uploading and downloading

- **scp: `cd` to the local folder first and use relative local names.** In Git Bash a path such as
  `C:/Users/...` is read by scp as host `C`. Remote paths are relative to the cluster home directory (no
  `~`), or absolute.

```
cd "/c/Users/me/structures" && scp -o BatchMode=yes Al_gb.vasp <alias>:ofdft-runs/Al_gb_relax/Al_gb.vasp
cd "/c/Users/me/results" && scp -o BatchMode=yes <alias>:ofdft-runs/Al_gb_relax/Al_gb_CONTCAR .
```

- Several files in one go (download), with tar:

```
cd "/c/Users/me/results" && ssh -o BatchMode=yes <alias> 'cd ~/ofdft-runs/Al_gb_relax && tar -czf - Al_gb.log Al_gb_CONTCAR Al_gb_status.json' | tar -xzf -
```

- Check sizes before downloading (`.xyz` trajectories can be ~100 MB; ask before anything > 100 MB):
  `ssh -o BatchMode=yes <alias> 'cd ~/ofdft-runs/Al_gb_relax && ls -la'`.
- Package upload (setup): a tar pipe, which works with the `C:/...` plugin path and never nests folders:

```
tar -C "<plugin root>/package/src" -czf - --exclude=__pycache__ ofdft_opt | ssh -o BatchMode=yes <alias> 'mkdir -p ~/.ofdft/lib/<ver> && tar -xzf - -C ~/.ofdft/lib/<ver>'
```

- Sanitise remote file names to `[A-Za-z0-9._-]` (spaces, brackets and non-ASCII characters break the
  quoted commands): `Al GB (1).vasp` -> `Al_GB_1_.vasp`.

## Windows notes

- Git Bash path forms: `C:\Users\me\x` is `/c/Users/me/x` or `C:/Users/me/x`. Quote paths with spaces.
- `${CLAUDE_PLUGIN_ROOT}` expands to a `C:/...` path: fine for `cd` and `tar -C`, not for scp.
- Two ssh clients: Git Bash's own `/usr/bin/ssh` (used by bare `ssh` unless something comes first on
  PATH) and the Windows OpenSSH client `C:\Windows\System32\OpenSSH\ssh.exe`. Both work with BatchMode,
  but each has its own agent: **only the Windows client can use the Windows "OpenSSH Authentication
  Agent" service**. Check which one the Bash tool runs: `command -v ssh; ssh -V` (`OpenSSH_for_Windows`
  = the Windows client). To make bare `ssh`/`scp` use the Windows client, see step 4 below.
- With the Windows client called from Git Bash, MSYS rewrites any argument that starts with `/` into a
  Windows path (`'/home/u/env/bin/python ...'` becomes `'C:/Program Files/Git/home/u/...'`). The
  wrappers below set `MSYS_NO_PATHCONV=1` to stop this; still start remote commands with `cd ~ && `.
- A job script edited on Windows may get CRLF line endings (`/bin/bash^M: bad interpreter`); fix on the
  cluster with `sed -i 's/\r$//' job.sh`. Scripts written by `ofdft jobscript` on the cluster are LF.

## First-time SSH setup (each person, once)

1. Key pair (keep a passphrase): `ssh-keygen -t ed25519`; get the public key onto the cluster account (lab
   administrator or `ssh-copy-id`).
2. Alias in `~/.ssh/config` (Windows: `C:\Users\<you>\.ssh\config`):

```
Host imlab
    HostName <login-node address from the lab>
    User <your cluster user name>
    IdentityFile ~/.ssh/id_ed25519
```

   Off campus: connect the campus VPN, or add `ProxyJump <jump host you have an account on>` to the block
   (every connection then logs in twice).
3. Accept the host key: run `ssh imlab hostname` once **interactively in a Git Bash terminal** (not through
   Claude), compare the fingerprint with a lab member, answer `yes`. With ProxyJump this asks for both hosts.
4. Passphrase-protected key: Claude's Bash tool cannot type the passphrase. Load the key into an ssh-agent
   that new shells can reach (macOS: the keychain; Linux: the desktop session agent). Windows:
   - The user (not Claude) loads the key into the Windows agent service once: the service "OpenSSH
     Authentication Agent" must be running (an administrator sets it to start automatically:
     `Set-Service ssh-agent -StartupType Automatic; Start-Service ssh-agent`), then in PowerShell
     `C:\Windows\System32\OpenSSH\ssh-add.exe $env:USERPROFILE\.ssh\id_ed25519` (asks for the passphrase
     once; the service keeps the key).
   - Git Bash's own ssh cannot reach that service, so make bare `ssh`/`scp` in Git Bash (and in
     Claude's Bash tool) run the Windows client. In a Git Bash terminal (Git Bash puts `~/bin` first on
     PATH):

```
mkdir -p ~/bin
printf '#!/bin/sh\nexport MSYS_NO_PATHCONV=1\nexec /c/Windows/System32/OpenSSH/ssh.exe "$@"\n' > ~/bin/ssh
printf '#!/bin/sh\nexport MSYS_NO_PATHCONV=1\nexec /c/Windows/System32/OpenSSH/scp.exe "$@"\n' > ~/bin/scp
chmod +x ~/bin/ssh ~/bin/scp
```

   Restart Claude, then check in the Bash tool: `command -v ssh` prints `.../bin/ssh` and `ssh -V`
   prints `OpenSSH_for_Windows_...`. The same `~/.ssh/config` is used (`C:\Users\<you>\.ssh\config`).
   - Alternative without the Windows client: a Git Bash `ssh-agent` on a fixed socket, started from
     `~/.bashrc` (e.g. `export SSH_AUTH_SOCK=~/.ssh/agent.sock`, start `ssh-agent -a "$SSH_AUTH_SOCK"`
     if `ssh-add -l` cannot connect), and `ssh-add` once per login in a Git Bash terminal.
5. Test in the Bash tool: `ssh -o BatchMode=yes imlab 'echo ok; hostname'` must print `ok` and the host
   name without any prompt.

## Why not the imlab MCP tools

The lab's imlab plugin tools fail on Windows (its SSH library calls `os.WUNTRACED`, which does not exist on
Windows), cap transfers at 5 MB, and their job template uses module MPI with `srun`, while DFTpy needs the
conda env's own `mpirun`. ofdft therefore uses plain ssh on every OS.
