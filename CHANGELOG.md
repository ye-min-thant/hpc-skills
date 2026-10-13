# Changelog

All notable changes to the `ofdft` plugin and the `ofdft-opt` package. The plugin version
(`plugins/ofdft/.claude-plugin/plugin.json`) and the package version are always the same.

## 0.1.0 - 2026-10-13

First public release.

### Plugin
- `/ofdft:setup`: SSH check, conda-forge environment (pinned `environment.yml`), side-by-side install of
  `ofdft-opt` in `~/.ofdft/lib/<version>` on the cluster, pseudopotential download, `ofdft doctor`,
  optional MPI smoke test, local `~/.ofdft/config.json`.
- `/ofdft:optimize`: relaxation and stress-strain runs over SSH - input checks, run folders, optional
  cost probe, job script, submission, monitoring through `<prefix>_status.json`, restart, clean stop,
  downloads, plots, and the required tight re-check before reporting peak stresses.
- Reference notes: method and findings, relax and stress-strain options, cluster facts, SSH command
  forms, troubleshooting.

### Package `ofdft-opt` (from the original `dftpy_relax.py` / `ss_curve.py` scripts)
- One command `ofdft` with `relax`, `sscurve`, `jobscript`, `doctor`, `pp`, `plot`, `version`.
- New production defaults: `--precon none`, `--econv-loose 0`, `--fmax 0.01`, `--smax 0.05`,
  `--steps 1000`; `--econv` 1e-7 Ha/atom, automatically 1e-8 when fmax < 0.01.
- Safety: refuses to overwrite existing outputs without `--restart`/`--overwrite`; `<prefix>_params.json`
  checked on restart (relax and sscurve); `<prefix>_status.json` with the final verdict; per-run
  `<prefix>.STOP`; reference check before a stress-strain curve; XC/pseudopotential mismatch check;
  MPI abort hook so a crash on one rank cannot hang a job.
- Stress-strain curves started by the original `ss_curve.py` continue with `ofdft sscurve` and the same
  options (the old params file is checked by name and rewritten in the new format).
- Pseudopotential registry with sha256 verification (line-ending independent) and `ofdft pp fetch`;
  BLPS files are no longer distributed with the code.
- `ofdft jobscript` writes Slurm scripts with resources sized by atom count, environment checks and the
  MPI wire-up test; `ofdft doctor` checks the installation.
- XC stand-in and SciPy shim are imported automatically by the package.
- `ofdft plot`: labels from file names, output next to the CSV, initial slope from the actual strains,
  thresholds in the title.
