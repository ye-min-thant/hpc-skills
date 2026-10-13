# ofdft knowledge base: method, optimizer, findings

Load this when the user asks why a setting is what it is, how to judge a result, or how much a run
will cost. Numbers below were measured with ofdft-opt's algorithms on Al (BLPS, WT, PBE, ecut 600 eV).

## Method

- Orbital-free DFT with DFTpy (PyPI 2.2.0 on the cluster; 2.1.2-2.2.x accepted). The electron density is
  optimised directly (truncated Newton); there are no orbitals and no k-points.
- Kinetic energy: Wang-Teter (WT) nonlocal KEDF (`--kedf WT`, default).
- Exchange-correlation: PBE (`--xc pbe`, default) or LDA (`--xc lda`). If real LibXC bindings
  (`pylibxc`) are installed they are used; otherwise ofdft-opt uses its vendored NumPy stand-in
  (`XC_BACKEND = "ofdft-standin"`, shown in the log header and by `ofdft doctor`).
- Pseudopotentials: BLPS local pseudopotentials (Huang & Carter 2008), `al.gga.recpot` for PBE and
  `al.lda.recpot` for LDA. They are downloaded and sha256-checked by `ofdft pp fetch`. Mixing a GGA file
  with `--xc lda` (or the reverse) is refused unless `--allow-xc-mismatch`.
- Grid cutoff 600 eV (`--ecut 600`). The FFT grid follows the cell, so two different cells never share a
  grid.
- Scope: validated for Al only. Other elements need a matching local pseudopotential supplied by the user
  (`--pp X=FILE`); such results are unvalidated.

## Optimizer (`ofdft relax`, also used at every stress-strain point)

- Own L-BFGS on [reference positions, strain]; no ASE optimizer, filter or calculator.
- Strain is scaled by an elastic stiffness (`--stiffness`, 100 GPa, about C11 of Al), so cell and atomic
  degrees of freedom are equally well conditioned.
- Safeguarded line search that uses the energy and the directional derivative, so it is robust to the
  small energy noise of the density optimisation.
- The FFT grid is fixed during a run, so energy, forces and stress are exactly consistent. After
  convergence, if the grid spacing drifted by more than `--regrid-tol` (2 %), the grid is re-chosen and the
  relaxation continues.
- Under MPI the step direction is computed on rank 0 and broadcast, so all ranks stay in step.
- Why not ASE FIRE + FrechetCellFilter: the filter divides the cell "force" by N_atoms, so fmax 0.02 eV/A
  only demands |stress| < ~0.2 GPa for Al, and for 1e3-1e4 atoms the cell degrees of freedom become very
  soft and badly conditioned for FIRE.

## What "converged" means

- `--relax full`: max|F| < fmax (eV/A) AND max over the six components |sigma + p| < smax (GPa).
- `--relax iso`: max|F| < fmax AND |mean normal stress + p| < smax (only the volume relaxes).
- `--relax positions`: forces only; there is no stress criterion.
- Stress-strain, tension mode: forces and the five free stress components; the loading-axis stress is the
  result, not a criterion. Strain mode: forces only.
- The authoritative verdict is `<prefix>_status.json` (`state`, `reason`). Never decide convergence from
  the log columns: the relaxer can continue after convergence when the grid is re-chosen, and a CONTCAR is
  written every step, so an unconverged CONTCAR looks like a final one.

## Density convergence (econv, Ha/atom)

- 1e-7 is accurate enough for fmax >= 0.01 eV/A: replaying the 13k-atom SCF log gave a max energy error of
  5e-4 eV (whole cell), and each call is ~15 % faster than with 1e-8.
- For tighter thresholds use 1e-8. ofdft does this automatically when fmax < 0.01 and `--econv` is not
  given (the log says so).
- `--econv-loose` (loose econv while max|F| > 5 x fmax) is off by default; see the next section.

## Preconditioner finding (why `--precon none` is the default)

- On cheap EMT benchmarks the Exp preconditioner (Packwood et al. 2016) cut steps by 27-40 %.
- On real OF-DFT it did not help. 13,140-atom Al polycrystal, 64 ranks, precon exp + loose econv: same
  energy as the run without preconditioner up to step ~200, then much slower; at step 500 the energy was
  ~9 eV higher than the run without preconditioner at the same step (~11 eV above the converged
  minimum), with max|F| still 0.05-0.15 eV/A. In the tail max|F| hovers around the econv switch, so econv
  toggles between loose and tight.
- 202-atom DFTpy polycrystal: precon + loose econv converged in fewer steps (108 vs 134) but ended in a
  higher local minimum (+19 meV/atom). Different optimizers can end in different local minima of a
  polycrystal; faster convergence is not the lowest grain-boundary energy.
- Default therefore: `--precon none --econv-loose 0`. Use `--precon exp` only as an experiment.

## Cost data (im cluster, AMD EPYC 7713, conda-forge Open MPI)

- 13,140-atom Al polycrystal, 64 ranks: ~95-115 s per optimizer step, 38-43 GB total memory (all ranks).
  48 ranks: ~175 s/step.
- Relaxation of the as-built polycrystal to 0.01/0.05: 674 steps (~110,000 s of compute, several
  restarts). max|F| < 0.05 at step 121 but < 0.02 only at step 619, while the energy kept dropping ~22 eV
  (1.7 meV/atom): slow grain-boundary reconstruction is the expensive tail. Pre-relaxing with an EAM
  potential or a short MD anneal before OF-DFT shortens it.
- Stress-strain points: elastic points need ~8-40 optimizer steps; grain-boundary events or slip ~80+
  steps (on EMT, plastic events cost 150-330 calls each and dominate the curve).
- Memory grows roughly linearly with atoms: ~3 MB per atom measured at 64 ranks. `ofdft jobscript
  --mem auto` requests about twice that (max(8, 0.0065 GB x atoms)).

## Threshold sensitivity (stress-strain)

- EMT, small grain-boundary-rich cell: per-point fmax 0.01 / smax 0.05 versus 0.005 / 0.01 gave the elastic
  part within ~5 %, but the loose thresholds overestimated the peak stress by ~40 % and delayed yield. The
  tight run cost 2.7x more steps.
- Hence: never report a peak stress or yield strain without the tight re-check (stress-strain.md), and
  always state the thresholds used.

## DFTpy pitfalls (handled by ofdft-opt; know them when reading other scripts)

- DFTpy's ASE calculator (`api4ase.DFTpyCalculator`) sums the already-reduced energy over MPI ranks again:
  energies are N_ranks times too large in ASE-based MPI runs (2.1.x and 2.2.0; forces and stress are fine).
  ofdft-opt does not use that calculator.
- The DFTpy GitHub dev branch (commit fbc47b4) gives a wrong XC stress under MPI. Use the PyPI release;
  `ofdft doctor` fails on dev/local versions.
- Memory leak per DFTpy call: fixed by a `gc.collect()` after every call.
- Serial runs restart the density from uniform when the FFT grid changes (DFTpy's Fourier remap mis-maps
  some half-grid shape changes); MPI runs interpolate it. Only costs a few extra SCF iterations, once.

## Comparing energies

Compare total energies only between runs with the same PP (sha256), XC, KEDF, ecut, econv and ofdft-opt
version. Compute bulk references (e.g. for grain-boundary energies) with exactly the same settings.

## Reference values (this setup)

| System | Quantity | Value |
|---|---|---|
| fcc Al, 4 atoms, PBE | a0 / B0 / E0 | 4.0626 A / 78.6 GPa / -57.183641 eV/atom |
| fcc Al, 4 atoms, LDA | a0 / B0 | 3.985 A / 85 GPa |
| fcc Al, 32 atoms, uniaxial stress along [001] | slope at 1-2 % strain | ~45.6 GPa |
| 13,140-atom Al polycrystal, z tension | initial slope (0-1 %) | ~50-55 GPa |

Use these to sanity-check a new installation or a suspicious result.
