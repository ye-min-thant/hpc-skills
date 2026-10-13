# Third-party notices

The code in this repository is MIT-licensed (see LICENSE) unless stated below.

## BLPS pseudopotentials (not included)

The bulk-derived local pseudopotentials (BLPS) `al.gga.recpot` and `al.lda.recpot` are **not** distributed
with this repository. `ofdft pp fetch` downloads them from the Carter group's repository
https://github.com/EACcodes/local-pseudopotentials (pinned commit
`7a0dc7a1236a1d0a86f8a71c89999079bd14edec`), licensed under CC BY-ND 4.0, and verifies their sha256.
Please cite: C. Huang and E. A. Carter, Phys. Chem. Chem. Phys. 10, 7109 (2008).

## DFTpy (dependency, not vendored)

DFTpy is MIT-licensed and installed as a dependency from PyPI. Please cite: X. Shao, K. Jiang, W. Mi,
A. Genova, M. Pavanello, WIREs Comput. Mol. Sci. 11, e1482 (2021).

## Vendored XC stand-in (`ofdft_opt/_vendor/pylibxc`)

An original NumPy implementation of the exchange-correlation functionals used here: LDA exchange,
Perdew-Zunger 1981 LDA correlation, and PBE exchange and correlation (J. P. Perdew, K. Burke, M. Ernzerhof,
Phys. Rev. Lett. 77, 3865 (1996)) with PW92 correlation (J. P. Perdew, Y. Wang, Phys. Rev. B 45, 13244
(1992)). It is not LibXC code; it only mimics the small part of the `pylibxc` API that DFTpy calls, and it
is used only when the real LibXC bindings are not installed. MIT, as the rest of this repository.

## SciPy line-search shim (`ofdft_opt/_shim.py`)

Original code (MIT). It adapts DFTpy 2.1.x to the line-search interface of newer SciPy releases and does
nothing on DFTpy 2.2.

## Other dependencies (not vendored)

NumPy, SciPy, mpi4py and mpi4py-fft (BSD-style licences), ASE (LGPL-2.1-or-later), Open MPI and FFTW from
conda-forge, and matplotlib (optional, for plots) are installed separately under their own licences.
