"""ofdft_opt.core -- simultaneous relaxation of atomic positions and cell with DFTpy (OF-DFT),
without ASE optimizers, filters or calculators (library part of the former dftpy_relax.py).

Why not ASE's FIRE + FrechetCellFilter for large cells:
  * the filter divides the cell "force" by N_atoms, so fmax = 0.02 eV/A only demands
    |stress| < ~0.2 GPa for Al, and for N ~ 1e3-1e4 the cell degrees of freedom become
    extremely soft/ill-conditioned for FIRE;
  * DFTpy's ASE calculator re-chooses the FFT grid whenever the cell changes, which makes
    energy and (fixed-grid) analytic stress slightly inconsistent.

What this driver does instead:
  * L-BFGS on [reference positions, strain], strain scaled by an elastic stiffness so the
    cell and the atoms are equally conditioned;
  * safeguarded line search using energy AND directional derivative (robust to energy noise);
  * FFT grid kept fixed during the whole run (energy, forces and stress exactly consistent);
  * separate convergence: max|F| < fmax (eV/A) AND max|sigma + p| < smax (GPa);
  * VASP POSCAR in, CONTCAR written every step, extended-XYZ trajectory, restart, STOP file,
    selective dynamics, target pressure, MPI (srun/mpirun with mpi4py + mpi4py-fft).

Engines: DFTpyEngine (OF-DFT) and EMTEngine (ASE EMT classical potential, for tests and dry runs
only). Both map (cell, positions, need_stress) -> (E eV, F eV/A, stress eV/A^3, ASE sign convention).
The command-line drivers are ofdft_opt.relax (`ofdft relax`) and ofdft_opt.sscurve (`ofdft sscurve`).
"""
import contextlib
import gc
import io
import os
import sys
import time

import numpy as np

from . import pp as ppmod
from .runinfo import die, install_mpi_abort_hook, sha256_text, _mpi_module

EV_A3_TO_GPA = 160.21766208
HA_TO_EV = 27.211386245988
BOHR_TO_A = 0.529177210903


def rss_gb():
    """Current resident memory of this process in GB (Linux /proc, Windows API; 0 if unknown)."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
                       [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.WorkingSetSize / 1e9
    except Exception:
        return 0.0


# ----------------------------------------------------------------------------------------
# POSCAR / XYZ I/O (no ASE)
# ----------------------------------------------------------------------------------------
def read_poscar(path):
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f]
    comment = lines[0].strip()
    scale = float(lines[1].split()[0])
    cell = np.array([[float(v) for v in lines[i].split()[:3]] for i in (2, 3, 4)])
    i = 5
    tokens = lines[i].split()
    if all(t.lstrip("+-").isdigit() for t in tokens):
        raise ValueError(f"{path}: VASP4 POSCAR without element line; add the element names (line 6)")
    species = tokens
    counts = [int(v) for v in lines[i + 1].split()[:len(species)]]
    i += 2
    selective = lines[i].strip()[:1].lower() == "s"
    if selective:
        i += 1
    direct = lines[i].strip()[:1].lower() in ("d",)
    i += 1
    n = sum(counts)
    raw = [lines[i + k].split() for k in range(n)]
    pos = np.array([[float(v) for v in r[:3]] for r in raw])
    fixed = None
    if selective:
        fixed = np.array([[flag.upper().startswith("F") for flag in r[3:6]] for r in raw], dtype=bool)
    if scale < 0:                                  # negative scale = target volume
        scale = (-scale / abs(np.linalg.det(cell))) ** (1.0 / 3.0)
    cell = cell * scale
    pos = pos @ cell if direct else pos * scale
    symbols = [s for s, c in zip(species, counts) for _ in range(c)]
    return dict(comment=comment, cell=cell, positions=pos, symbols=symbols, fixed=fixed)


def write_poscar(path, symbols, cell, positions, fixed=None, comment="dftpy_relax"):
    runs = []                                      # keep atom order: consecutive runs of species
    for s in symbols:
        if runs and runs[-1][0] == s:
            runs[-1][1] += 1
        else:
            runs.append([s, 1])
    frac = positions @ np.linalg.inv(cell)
    frac -= np.floor(frac)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(comment + "\n 1.0\n")
        for v in cell:
            f.write(" %21.16f %21.16f %21.16f\n" % tuple(v))
        f.write(" " + " ".join(r[0] for r in runs) + "\n")
        f.write(" " + " ".join(str(r[1]) for r in runs) + "\n")
        if fixed is not None:
            f.write("Selective dynamics\n")
        f.write("Direct\n")
        for k, p in enumerate(frac):
            line = " %19.16f %19.16f %19.16f" % tuple(p)
            if fixed is not None:
                line += " " + " ".join("F" if x else "T" for x in fixed[k])
            f.write(line + "\n")
    os.replace(tmp, path)                          # atomic: never leaves a half-written file


def append_xyz(path, symbols, cell, positions, forces, energy, stress):
    lat = " ".join("%.10f" % v for v in cell.ravel())
    st = " ".join("%.10e" % v for v in stress.ravel())
    with open(path, "a") as f:
        f.write(f"{len(symbols)}\n")
        f.write(f'Lattice="{lat}" Properties=species:S:1:pos:R:3:forces:R:3 '
                f'energy={energy:.10f} stress="{st}" pbc="T T T"\n')
        for s, p, fo in zip(symbols, positions, forces):
            f.write("%-2s %16.10f %16.10f %16.10f %14.8f %14.8f %14.8f\n" % (s, *p, *fo))


# ----------------------------------------------------------------------------------------
# DFTpy engine: (cell, positions) -> energy, forces, stress, with a FIXED FFT grid
# ----------------------------------------------------------------------------------------
def _pp_config(pp_files):
    """(pppath, {el: name relative to pppath}) for DFTpy, which opens pppath + os.sep + name.
    One directory: pppath = that directory, names = basenames. Several directories: pppath = their
    common parent and relative names (works on POSIX and Windows, unlike absolute names)."""
    dirs = {os.path.dirname(p) for p in pp_files.values()}
    if len(dirs) == 1:
        return dirs.pop(), {el: os.path.basename(p) for el, p in pp_files.items()}
    try:
        root = os.path.commonpath(list(pp_files.values()))
    except ValueError:
        die("the pseudopotential files are on different drives; copy them into one directory "
            "(and pass --pppath DIR)")
    return root, {el: os.path.relpath(p, root) for el, p in pp_files.items()}


class DFTpyEngine:
    def __init__(self, args, symbols, mp, verbose=False, pp_files=None):
        from dftpy.config.config import DefaultOption, OptionFormat
        from dftpy.ions import Ions
        from dftpy.interface import ConfigParser, OptimizeDensityConf
        from dftpy.constants import FORCE_CONV, STRESS_CONV
        self._Ions, self._parse, self._optimize = Ions, ConfigParser, OptimizeDensityConf
        self._fconv = FORCE_CONV["Ha/Bohr"]["eV/A"]
        self._sconv = STRESS_CONV["Ha/Bohr3"]["eV/A3"]
        self.symbols, self.mp, self.verbose = symbols, mp, verbose
        self.nr = None if args.nr is None else np.array(args.nr, dtype=int)
        self.rho, self.pseudo, self.last_cell = None, None, None
        self.ncalls, self.time, self.nwarn = 0, 0.0, 0
        self.scf_log = None                        # file object (rank 0) for DFTpy's SCF output
        self.econv = args.econv                    # current density convergence (Ha/atom)
        self.grid_cell = None                      # cell for which the FFT grid was chosen

        # pseudopotentials: {element: absolute path}, resolved by ofdft_opt.pp (registry defaults,
        # search path, checksum and PP/XC checks) unless given
        if pp_files is None:
            pp_files = ppmod.resolve(getattr(args, "pp", None) or [], symbols, args.xc,
                                     getattr(args, "pppath", None),
                                     allow_xc_mismatch=getattr(args, "allow_xc_mismatch", False))
        missing = sorted(set(symbols) - set(pp_files))
        if missing:
            die(f"No pseudopotential for {missing}: use --pp EL=FILE")
        pp_files = {el: os.path.abspath(pp_files[el]) for el in sorted(set(symbols))}
        for el, path in pp_files.items():
            if not os.path.isfile(path):
                die(f"PP file not found: {path}")
        if args.xc == "pbe":
            try:
                import pylibxc  # noqa: F401  real LibXC bindings or the vendored NumPy stand-in
            except ImportError:
                die("PBE needs pylibxc: the NumPy stand-in should be vendored in ofdft_opt/_vendor/pylibxc "
                    "(reinstall/recopy the ofdft_opt package) or install the real pylibxc")
        pppath, ppname = _pp_config(pp_files)

        def make(calctype):
            c = DefaultOption()
            c["PATH"]["pppath"] = pppath
            for el in sorted(set(symbols)):
                c["PP"][el] = ppname[el]
            c["JOB"]["calctype"] = calctype
            c["KEDF"]["kedf"] = args.kedf
            c["EXC"]["libxc"] = "gga_x_pbe gga_c_pbe" if args.xc == "pbe" else "lda_x lda_c_pz"
            c["GRID"]["ecut"] = args.ecut
            c["OPT"]["method"] = "TN"
            c["OPT"]["econv"] = args.econv
            c["OPT"]["maxiter"] = 500
            c["OUTPUT"]["stress"] = False
            c["OUTPUT"]["time"] = False
            return OptionFormat(c)
        self.conf = {True: make("Energy Force Stress"), False: make("Energy Force")}
        self.pp_files = pp_files                   # {el: absolute path}
        self.pp = {el: os.path.basename(p) for el, p in pp_files.items()}
        self.pp_sha256 = {el: sha256_text(p) for el, p in pp_files.items()}

    def reset_grid(self):
        """Re-choose the grid from ecut at the next call. Under MPI the current density is kept and
        interpolated onto the new grid by DFTpy (field2distrib), so the SCF does not restart. In serial
        runs DFTpy remaps it with a Fourier routine that mis-maps some half-grid shape changes, so
        there the density restarts from uniform (a few more SCF iterations, once)."""
        self.nr, self.pseudo, self.last_cell, self.grid_cell = None, None, None, None
        if not self.mp.is_mpi:
            self.rho = None
        for c in self.conf.values():
            c["GRID"]["nr"] = None

    def set_econv(self, econv):
        """Change the density convergence (Ha/atom) for the next DFTpy calls."""
        if econv != self.econv:
            for c in self.conf.values():
                c["OPT"]["econv"] = econv
            self.econv = econv

    def spacing_drift(self, cell):
        """Largest relative change of a cell-vector length since the grid was chosen."""
        if self.grid_cell is None:
            return 0.0
        return float(np.abs(np.linalg.norm(cell, axis=1) / np.linalg.norm(self.grid_cell, axis=1) - 1).max())

    def _run(self, conf, ions, grid, rhoini):
        """One DFTpy density optimization. DFTpy's output (SCF iterations) goes live to the
        scf log file (and to stdout with --verbose); reports whether the density failed to converge."""
        from dftpy.constants import environ
        sink = io.StringIO()
        outs = [o for o in (sys.stdout if self.verbose else None, self.scf_log) if o is not None]
        if self.scf_log is not None:
            self.scf_log.write(f"\n===== DFTpy call {self.ncalls + 1}  ({time.strftime('%Y-%m-%d %H:%M:%S')})\n")
            self.scf_log.flush()

        class _Tee(io.TextIOBase):
            def write(_s, t):
                sink.write(t)
                for o in outs:
                    o.write(t)
                return len(t)

            def flush(_s):
                for o in outs:
                    o.flush()
        stream = _Tee() if outs else sink
        old_out = environ.get("STDOUT", None)
        environ["STDOUT"] = stream                 # DFTpy's sprint writes here, not to sys.stdout
        try:
            with contextlib.redirect_stdout(stream):
                config, others = self._parse(conf, ions=ions, rhoini=rhoini, pseudo=self.pseudo,
                                             grid=grid, mp=self.mp)
                res = self._optimize(config, **others)
        finally:
            if old_out is None:
                environ.pop("STDOUT", None)
            else:
                environ["STDOUT"] = old_out
        txt = sink.getvalue()                      # same messages in DFTpy 2.1.x and 2.2.x
        bad = ("Density Optimization NOT Converged" in txt) or ("Not converged, but reached max steps" in txt)
        if self.mp.is_mpi:
            bad = self.mp.comm.bcast(bad, root=0)  # sprint prints on rank 0 only
        return res, bad

    def __call__(self, cell, positions, need_stress):
        t0 = time.time()
        if self.mp.is_mpi:                         # all ranks use rank 0's geometry
            cell, positions = self.mp.comm.bcast((cell, positions), root=0)
        conf = self.conf[need_stress]
        conf["GRID"]["nr"] = None if self.nr is None else self.nr.copy()
        ions = self._Ions(symbols=self.symbols, positions=positions, cell=cell, pbc=True, units="ase")
        grid = None
        if self.rho is not None and self.last_cell is not None and np.allclose(cell, self.last_cell, atol=1e-12):
            grid = self.rho.grid
        res, bad = self._run(conf, ions, grid, self.rho)
        if bad and self.rho is not None:           # retry once from a uniform density, same grid
            self.nwarn += 1
            if self.mp.comm.rank == 0:
                print(" WARNING: DFTpy density not converged; retrying from uniform density",
                      file=sys.stderr, flush=True)
            res, bad = self._run(conf, ions, res["density"].grid, None)
        if bad:
            raise RuntimeError("DFTpy density optimization failed twice: energy/forces/stress unreliable. "
                               "The CONTCAR of the last good step is intact; rerun with --restart "
                               "(try --verbose, or a looser --econv).")
        self.rho = res["density"]
        self.pseudo = res.get("pseudo", None)
        self.last_cell = cell.copy()
        if self.nr is None:
            self.nr = np.array(self.rho.grid.nrR if hasattr(self.rho.grid, "nrR") else self.rho.grid.nr)
            self.grid_cell = cell.copy()
        # DFTpy's TotalFunctional already all-reduces the energy over MPI ranks (2.1.x and 2.2.x);
        # summing it again (as dftpy.api.api4ase does) would multiply it by the number of ranks.
        energy = float(res["energypotential"]["TOTAL"].energy) * HA_TO_EV
        forces = np.array(res["forces"]["TOTAL"]) * self._fconv
        stress = np.array(res["stress"]["TOTAL"]) * self._sconv if need_stress else np.zeros((3, 3))
        # DFTpy objects (grid, functionals, fields) form reference cycles; free the previous step's
        # large arrays now instead of letting them pile up until Python's next full collection.
        del res
        gc.collect()
        self.ncalls += 1
        self.time += time.time() - t0
        return energy, forces, stress


# ----------------------------------------------------------------------------------------
# EMT engine (ASE classical potential): same interface, for tests and dry runs ONLY
# ----------------------------------------------------------------------------------------
class EMTEngine:
    """ASE's EMT potential behind the DFTpyEngine interface -- TESTS AND DRY RUNS ONLY, not OF-DFT.

    EMTEngine(symbols, mp=None, econv=None, ...); the DFTpyEngine-style call EMTEngine(args, symbols, mp)
    also works (econv then comes from args.econv).
    __call__(cell, positions, need_stress) -> (E eV, F eV/A (N,3), stress eV/A^3 (3,3)); the stress is
    ASE's (positive = tensile), the same convention as DFTpyEngine (the Relaxer uses it unchanged).
    Serial only: refuses to run with more than one MPI rank."""

    def __init__(self, symbols, mp=None, econv=None, verbose=False, pp_files=None, args=None):
        seq = (list, tuple, np.ndarray)
        if not isinstance(symbols, seq) and isinstance(mp, seq):   # called as (args, symbols[, mp])
            args, symbols, mp = symbols, mp, econv
            econv = None
        if econv is None and args is not None:
            econv = getattr(args, "econv", None)
        size = getattr(getattr(mp, "comm", None), "size", 1)
        MPI = _mpi_module()
        if MPI is not None:
            size = max(size, MPI.COMM_WORLD.Get_size())
        if size > 1:
            raise RuntimeError("the EMT engine is serial only (tests / dry runs); run it without mpirun")
        from ase import Atoms
        from ase.calculators.emt import EMT
        self.symbols, self.mp, self.verbose = list(symbols), mp, verbose
        self.at = Atoms(self.symbols, pbc=True)
        self.at.calc = EMT()
        self.ncalls, self.time, self.nwarn = 0, 0.0, 0
        self.econv = econv
        self.scf_log = None
        self.nr = np.array([0, 0, 0])
        self.pp, self.pp_sha256, self.pp_files = {}, {}, {}

    def set_econv(self, econv):
        self.econv = econv                         # stored only (EMT has no SCF)

    def reset_grid(self):
        pass

    def spacing_drift(self, cell):
        return 0.0

    def __call__(self, cell, positions, need_stress):
        t0 = time.time()
        self.at.set_cell(cell, scale_atoms=False)
        self.at.positions = positions
        energy = float(self.at.get_potential_energy())
        forces = np.array(self.at.get_forces())
        stress = np.array(self.at.get_stress(voigt=False)) if need_stress else np.zeros((3, 3))
        self.ncalls += 1
        self.time += time.time() - t0
        return energy, forces, stress


# ----------------------------------------------------------------------------------------
# Optimizer: L-BFGS on x = [reference positions Q (A), L * strain]
#   cell = cell0 (I+eps)^T,  R = Q (I+eps)^T,  minimize H = E + p V
# ----------------------------------------------------------------------------------------
VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


class ExpPrecon:
    """'Exp' preconditioner of Packwood et al., J. Chem. Phys. 144, 164109 (2016).

    A sparse spring network over neighbouring atoms approximates the Hessian:
        P_ij = -mu * exp(-A (r_ij / r_nn - 1))   (i != j, r_ij < r_cut)
        P_ii = -sum_j P_ij + mu * c_stab
    Solving P d = g turns forces into collective displacements, so soft modes (grain boundaries,
    long-wavelength motions) relax in few steps while stiff bonds keep small steps.
    The same N x N matrix acts on the x, y and z components. mu is set so that the mean diagonal
    equals the atomic stiffness guess alpha; L-BFGS rescales it automatically afterwards."""

    def __init__(self, A=3.0, r_cut_factor=2.0, c_stab=0.1, rebuild_disp=0.3, rebuild_strain=0.005):
        self.A, self.r_cut_factor, self.c_stab = A, r_cut_factor, c_stab
        self.rebuild_disp, self.rebuild_strain = rebuild_disp, rebuild_strain
        self.P = self.Minv_diag = self.R_built = self.cell_built = None
        self.r_nn = None
        self.nbuild = 0

    def needs_rebuild(self, R, cell):
        if self.P is None:
            return True
        if np.linalg.norm(R - self.R_built, axis=1).max() > self.rebuild_disp:
            return True
        return np.abs(cell - self.cell_built).max() > self.rebuild_strain * np.abs(self.cell_built).max()

    def build(self, R, cell, alpha):
        import scipy.sparse as sp
        from ase import Atoms
        from ase.neighborlist import neighbor_list
        n = len(R)
        at = Atoms(numbers=np.ones(n, int), positions=R, cell=cell, pbc=True)
        if self.r_nn is None:                      # nearest-neighbour distance of the structure
            guess = 1.5 * (abs(np.linalg.det(cell)) / n) ** (1.0 / 3.0)
            i, d = neighbor_list("id", at, guess)
            dmin = np.full(n, np.inf)
            np.minimum.at(dmin, i, d)
            self.r_nn = float(np.median(dmin[np.isfinite(dmin)]))
        i, j, d = neighbor_list("ijd", at, self.r_cut_factor * self.r_nn)
        w = np.exp(-self.A * (d / self.r_nn - 1.0))
        diag = np.bincount(i, weights=w, minlength=n) + self.c_stab
        mu = alpha / diag.mean()
        W = sp.coo_matrix((w, (i, j)), shape=(n, n)).tocsr()   # (i,j) and (j,i) both listed: symmetric
        self.P = (mu * (sp.diags(diag) - W)).tocsr()
        self.Minv_diag = 1.0 / (mu * diag)
        self.R_built, self.cell_built = R.copy(), cell.copy()
        self.nbuild += 1

    def solve(self, G, fixed=None):
        """Solve P X = G column by column (conjugate gradients, Jacobi-preconditioned).
        Fixed atom components are removed from the system (reduced solve), and stay 0."""
        import scipy.sparse as sp
        from scipy.sparse.linalg import cg
        X = np.zeros_like(G)
        for k in range(G.shape[1]):
            f = np.ones(len(G), bool) if fixed is None else ~fixed[:, k]
            if not f.any():
                continue
            A = self.P if f.all() else self.P[f][:, f]
            M = sp.diags(self.Minv_diag[f])
            try:
                X[f, k], info = cg(A, G[f, k], rtol=1e-8, atol=0.0, maxiter=2000, M=M)
            except TypeError:                      # SciPy < 1.12 uses 'tol'
                X[f, k], info = cg(A, G[f, k], tol=1e-8, maxiter=2000, M=M)
        return X


class Relaxer:
    """L-BFGS on x = [reference positions Q (A), L * free strain components].
    mode: 'positions' (cell fixed), 'iso' (volume only), 'full' (all 6 strain components) or
    'custom' (free = list of Voigt indices; the other components stay at eps0, as used for
    strain-controlled loading in ss_curve.py).
    axis (custom mode, uniaxial loading): 'rigid grips' gauge as in LAMMPS fix deform -- row
    `axis` of F = I + eps stays at (1+e) e_axis, the loading-axis shears are F[other, axis]
    (F[axis, other] = 0). Then the box elongation equals the imposed strain exactly and a
    stationary point has exactly uniaxial Cauchy stress. Without axis, shears are symmetric."""

    def __init__(self, engine, cell, positions, symbols, fixed, mode, fmax, smax, pressure_gpa,
                 alpha=10.0, stiffness_gpa=100.0, memory=20, maxstep=0.2, maxstrain=0.01,
                 etol_per_atom=1e-6, log=print, free=None, eps0=None, precon=None, comm=None,
                 econv_tight=None, econv_loose=None, econv_switch=5.0, axis=None):
        self.engine, self.symbols, self.mode, self.log = engine, symbols, mode, log
        self.axis = axis
        self.n = len(positions)
        self.cell0 = np.array(cell, float)
        self.Q = np.array(positions, float)
        self.eps = np.zeros((3, 3)) if eps0 is None else np.array(eps0, float)
        self.eps_fixed = self.eps.copy()           # components that are not optimized
        self.fixed = np.zeros((self.n, 3), bool) if fixed is None else fixed
        self.fmax, self.smax = fmax, smax
        self.p = pressure_gpa / EV_A3_TO_GPA
        self.alpha, self.memory = alpha, memory
        self.maxstep, self.maxstrain = maxstep, maxstrain
        self.etol = etol_per_atom * self.n
        self.iso = mode == "iso"
        if mode == "custom":
            self.free = sorted(free or [])
        else:
            self.free = {"positions": [], "iso": [], "full": list(range(6))}[mode]
        self.ncell = 1 if self.iso else len(self.free)
        self.want_stress = self.ncell > 0 or smax is not None or mode == "custom"
        V = abs(np.linalg.det(self.cell0))
        # strain scaling: cell stiffness V*C/L^2 equal to the atomic stiffness guess alpha
        self.L = np.sqrt(V * (stiffness_gpa / EV_A3_TO_GPA) / alpha) if self.ncell else 1.0
        self.precon = precon
        self.comm = comm if (comm is not None and comm.size > 1) else None
        self.root = self.comm is None or self.comm.rank == 0
        self.econv_tight, self.econv_loose, self.econv_switch = econv_tight, econv_loose, econv_switch
        self.adaptive = bool(econv_loose) and econv_tight is not None and hasattr(engine, "set_econv")
        self.last_fmax = None
        self.S, self.Y = [], []
        self.nevals = 0

    # --- geometry <-> optimization vector ---
    def _F(self, eps):
        return np.eye(3) + eps

    def _slot(self, k):
        """Matrix entry (i, j) of eps for free Voigt component k, and whether it is a one-sided
        (column `axis`) shear of the rigid-grips gauge rather than a symmetric pair."""
        i, j = VOIGT[k]
        if self.axis is not None and i != j and self.axis in (i, j):
            return (j if i == self.axis else i), self.axis, True   # F[other, a] free, F[a, other] = 0
        return i, j, False

    def x_of(self, Q, eps):
        e = [eps[0, 0]] if self.iso else [eps[self._slot(k)[:2]] for k in self.free]
        return np.concatenate([Q.ravel(), self.L * np.array(e, float)])

    def geom_of(self, x):
        Q = x[:3 * self.n].reshape(self.n, 3)
        e = x[3 * self.n:] / self.L
        if self.iso:
            eps = e[0] * np.eye(3)
        else:
            eps = self.eps_fixed.copy()
            for k, v in zip(self.free, e):
                i, j, col = self._slot(k)
                eps[i, j] = v
                eps[j, i] = 0.0 if col else v
        F = self._F(eps)
        return Q, eps, self.cell0 @ F.T, Q @ F.T

    # --- energy and gradient in x ---
    def evaluate(self, x):
        Q, eps, cell, R = self.geom_of(x)
        E, forces, stress = self.engine(cell, R, self.want_stress)
        self.nevals += 1
        V = abs(np.linalg.det(cell))
        H = E + self.p * V
        F = self._F(eps)
        gQ = -(forces @ F)
        gQ[self.fixed] = 0.0
        g = [gQ.ravel()]
        if self.ncell:
            G = V * (stress + self.p * np.eye(3)) @ np.linalg.inv(F).T     # dH/dF at fixed Q
            if self.iso:
                ge = [np.trace(G)]
            else:
                ge = []
                for k in self.free:
                    i, j, col = self._slot(k)
                    ge.append(G[i, j] if (col or i == j) else G[i, j] + G[j, i])
            g.append(np.array(ge) / self.L)
        g = np.concatenate(g)
        fr = forces.copy()
        fr[self.fixed] = 0.0
        fmax = np.linalg.norm(fr, axis=1).max()
        sig = (stress + self.p * np.eye(3)) * EV_A3_TO_GPA
        if self.iso:
            smax = abs(np.trace(sig) / 3.0)
        elif self.free:
            smax = max(abs(sig[VOIGT[k]]) for k in self.free)
        else:
            smax = np.abs(sig).max() if self.want_stress else 0.0
        return dict(x=x, H=H, E=E, g=g, forces=forces, stress=stress, cell=cell, R=R,
                    fmax=fmax, smax=smax, V=V, econv=getattr(self.engine, "econv", None))

    def converged(self, st):
        ok = st["fmax"] < self.fmax
        if self.ncell:
            ok = ok and st["smax"] < self.smax
        return ok

    # --- adaptive density convergence ---
    def _set_econv(self, fmax_now):
        if self.adaptive:
            loose = fmax_now is None or fmax_now > self.econv_switch * self.fmax
            self.engine.set_econv(self.econv_loose if loose else self.econv_tight)

    def _is_loose(self, st):
        return self.adaptive and st["econv"] is not None and st["econv"] > self.econv_tight

    # --- preconditioned L-BFGS ---
    def Minv(self, q):
        """Apply the inverse preconditioner: Exp spring network for atoms, 1/alpha for the cell."""
        r = q / self.alpha
        if self.precon is not None:
            na = 3 * self.n
            X = self.precon.solve(q[:na].reshape(self.n, 3), self.fixed)
            X[self.fixed] = 0.0
            r[:na] = X.ravel()
        return r

    def direction(self, g):
        q = g.copy()
        a = []
        for s, y in reversed(list(zip(self.S, self.Y))):
            rho = 1.0 / y.dot(s)
            ai = rho * s.dot(q)
            q -= ai * y
            a.append((rho, ai))
        if self.S:                                  # H0 = gamma * M^-1, gamma from the latest pair
            s, y = self.S[-1], self.Y[-1]
            gamma = s.dot(y) / y.dot(self.Minv(y))
        else:
            gamma = 1.0
        r = gamma * self.Minv(q)
        for (s, y), (rho, ai) in zip(zip(self.S, self.Y), reversed(a)):
            b = rho * y.dot(r)
            r += s * (ai - b)
        return -r

    def limit(self, d):
        dq = d[:3 * self.n].reshape(self.n, 3)
        scale = 1.0
        mq = np.linalg.norm(dq, axis=1).max() if self.n else 0.0
        if mq > self.maxstep:
            scale = min(scale, self.maxstep / mq)
        if self.ncell:
            me = np.abs(d[3 * self.n:]).max() / self.L
            if me > self.maxstrain:
                scale = min(scale, self.maxstrain / me)
        return d * scale

    def _step_direction(self, st):
        """Computed on rank 0 only and broadcast, so every MPI rank takes the identical step."""
        d = None
        if self.root:
            if self.precon is not None and self.precon.needs_rebuild(st["R"], st["cell"]):
                self.precon.build(st["R"], st["cell"], self.alpha)
            d = self.direction(st["g"])
            if d.dot(st["g"]) >= 0:                  # not a descent direction: reset memory
                self.S, self.Y = [], []
                d = -self.Minv(st["g"])
            d = self.limit(d)
        return self.comm.bcast(d, root=0) if self.comm is not None else d

    def run(self, steps, on_step=None, should_stop=lambda: False):
        self._set_econv(self.last_fmax)
        st = self.evaluate(self.x_of(self.Q, self.eps))
        if on_step:
            on_step(0, st, None, 1)
        for it in range(1, steps + 1):
            if self.converged(st) and self._is_loose(st):   # confirm convergence at tight econv
                self.engine.set_econv(self.econv_tight)
                st = self.evaluate(st["x"])
            if self.converged(st) or should_stop():
                break
            self._set_econv(st["fmax"])
            d = self._step_direction(st)
            dg0 = d.dot(st["g"])
            new = self.evaluate(st["x"] + d)
            nls = 1
            dH, dg1 = new["H"] - st["H"], d.dot(new["g"])
            if dH > 1e-4 * dg0 + self.etol and dg1 > 0:
                # overshoot confirmed by energy AND slope: secant step on the directional derivative
                a = float(np.clip(dg0 / (dg0 - dg1), 0.1, 0.9))
                d = a * d
                new = self.evaluate(st["x"] + d)
                nls += 1
                if new["H"] - st["H"] > self.etol:
                    self.S, self.Y = [], []
            s, y = new["x"] - st["x"], new["g"] - st["g"]
            if s.dot(y) > 1e-12 * np.linalg.norm(s) * np.linalg.norm(y):
                self.S.append(s)
                self.Y.append(y)
                if len(self.S) > self.memory:
                    self.S.pop(0)
                    self.Y.pop(0)
            st = new
            if on_step:
                on_step(it, st, d, nls)
        if self.converged(st) and self._is_loose(st):   # converged on the last step at loose econv
            self.engine.set_econv(self.econv_tight)
            st = self.evaluate(st["x"])
        Q, eps, _, _ = self.geom_of(st["x"])
        self.Q, self.eps = Q, eps
        self.last_fmax = st["fmax"]
        return st, self.converged(st)


# ----------------------------------------------------------------------------------------
# MPI set-up
# ----------------------------------------------------------------------------------------
class _SerialComm:
    """Minimal serial communicator (used only when DFTpy is not installed: EMT engine)."""
    rank, size = 0, 1

    def bcast(self, data, root=0):
        return data

    def allreduce(self, data, *args, **kwargs):
        return data

    def barrier(self):
        pass


class _SerialMP:
    def __init__(self):
        self.comm = _SerialComm()

    @property
    def is_mpi(self):
        return False

    @property
    def is_root(self):
        return True


def init_mpi():
    """Set up DFTpy's MPI communicator when launched with several tasks; returns dftpy.mpi.mp
    (a minimal serial stand-in if DFTpy is not importable and only one task was launched)."""
    # Number of tasks the launcher started (SLURM_NTASKS alone is set in every sbatch job, even serial ones)
    launched = max(int(os.environ.get(k) or 1) for k in ("SLURM_STEP_NUM_TASKS", "OMPI_COMM_WORLD_SIZE",
                                                          "PMI_SIZE", "PMIX_SIZE"))
    try:
        from dftpy.mpi import pmi, mp
    except ImportError:
        if launched > 1:
            sys.exit(f"{launched} MPI tasks were launched but dftpy is not importable for {sys.executable}")
        return _SerialMP()
    if pmi.size > 1 or launched > 1:
        try:
            from mpi4py import MPI
            mp.comm = MPI.COMM_WORLD
        except ImportError:
            if launched > 1:
                sys.exit(f"{launched} MPI tasks were launched but mpi4py is not installed for {sys.executable}")
    if launched > 1 and mp.comm.size == 1:
        sys.exit(f"{launched} tasks were launched but MPI.COMM_WORLD.size == 1: mpi4py is not linked to the MPI "
                 "of this launcher (use the mpirun of the MPI mpi4py was built with, or srun --mpi=pmix / pmi2)")
    if mp.comm.size > 1:
        install_mpi_abort_hook()                   # an exception on one rank aborts the whole job
        import mpi4py_fft  # noqa: F401  (DFTpy needs it for distributed FFTs; fail early and clearly)
    return mp
