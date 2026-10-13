"""Serial DFTpy reference values (4-atom fcc Al, PBE, BLPS gga, WT KEDF, ecut 600 eV).

Skipped when dftpy is not importable or no copy of al.gga.recpot is available (searched in
$OFDFT_TEST_PP_DIR, the original DFTpy-al folder, then the ofdft search path). The PP file is copied
as is (the original folder has CRLF line endings: sha verification normalises them) into a temporary
$OFDFT_PPPATH. Reference numbers: see the build spec, section 9 (single point ~5 s, relax ~15 s)."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np  # noqa: E402

import ofdft_opt  # noqa: E402  (must come before any dftpy import: shim + XC stand-in)

HAVE_DFTPY = importlib.util.find_spec("dftpy") is not None
HAVE_ASE = importlib.util.find_spec("ase") is not None
SOURCE_DIR = os.environ.get("OFDFT_TEST_SRC_DIR") or os.path.expanduser(os.path.join("~", "DFTpy-al"))
GGA_SHA = "76034a6ea04586f32c2c13d9b4a828a83799638ab2d3b87c330afc653be09806"

A0_PBE = 4.0626                 # A
E0_PBE = -57.18364              # eV/atom at a = A0_PBE
E_TOL = 3e-4                    # eV/atom
STRESS_TOL = 0.05               # GPa
A_TOL = 0.003                   # A


def _sha_lf(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def find_gga_pp():
    cands = [os.environ.get("OFDFT_TEST_PP_DIR"), SOURCE_DIR]
    for d in cands:
        p = os.path.join(d, "al.gga.recpot") if d else None
        if p and os.path.isfile(p) and _sha_lf(p) == GGA_SHA:
            return p
    try:
        from ofdft_opt import pp
        p = pp.find("al.gga.recpot")
    except BaseException:
        p = None
    if p and _sha_lf(p) == GGA_SHA:
        return p
    return None


GGA_PP = find_gga_pp()


def run_main(main, argv, cwd):
    out = io.StringIO()
    old = os.getcwd()
    os.chdir(cwd)
    code = 0
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                rc = main(argv)
                code = rc if isinstance(rc, int) else 0
            except SystemExit as e:
                code = 0 if e.code is None else (e.code if isinstance(e.code, int) else 1)
                if not isinstance(e.code, (int, type(None))):
                    out.write(str(e.code))
    finally:
        os.chdir(old)
    return code, out.getvalue()


def fcc_al(a, rattle=0.0, seed=1):
    from ase.build import bulk
    at = bulk("Al", "fcc", a=a, cubic=True)
    if rattle:
        at.rattle(rattle, seed=seed)
    return at.get_chemical_symbols(), at.cell.array.copy(), at.positions.copy()


def read_json(path):
    with open(path) as f:
        return json.load(f)


@unittest.skipUnless(HAVE_DFTPY, "dftpy not importable")
@unittest.skipUnless(HAVE_ASE, "ase not installed")
@unittest.skipIf(GGA_PP is None, "al.gga.recpot not available (set OFDFT_TEST_PP_DIR)")
class TestDFTpySmall(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp = self._tmp.name
        self.ppdir = os.path.join(self.tmp, "pp")
        self.run = os.path.join(self.tmp, "run")
        os.makedirs(self.ppdir)
        os.makedirs(self.run)
        shutil.copyfile(GGA_PP, os.path.join(self.ppdir, "al.gga.recpot"))     # bytes as found (maybe CRLF)
        self.env = mock.patch.dict(os.environ, {"OFDFT_PPPATH": self.ppdir})
        self.env.start()
        for k in ("SLURM_STEP_NUM_TASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE"):
            os.environ.pop(k, None)

    def tearDown(self):
        self.env.stop()
        self._tmp.cleanup()

    def write_poscar(self, name, a, rattle=0.0):
        from ofdft_opt import core
        symbols, cell, pos = fcc_al(a, rattle)
        core.write_poscar(os.path.join(self.run, name), symbols, cell, pos, comment=f"fcc Al a={a}")
        return name

    def test_single_point_engine(self):
        from ofdft_opt import core, pp, relax
        symbols, cell, pos = fcc_al(A0_PBE)
        args = relax.parse_args([os.path.join(self.run, "unused.vasp")])
        args.econv = 1e-8
        pp_files = pp.resolve([], symbols, "pbe", None)
        self.assertEqual(os.path.basename(pp_files["Al"]), "al.gga.recpot")
        mp = core.init_mpi()
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            eng = core.DFTpyEngine(args, symbols, mp, verbose=False, pp_files=pp_files)
            E, F, S = eng(cell, pos, True)
        self.assertEqual(eng.pp, {"Al": "al.gga.recpot"})
        self.assertEqual(eng.pp_sha256, {"Al": GGA_SHA})
        self.assertEqual(eng.ncalls, 1)
        self.assertAlmostEqual(E / 4, E0_PBE, delta=E_TOL)
        self.assertLess(np.abs(F).max(), 1e-3)
        self.assertLess(np.abs(S).max() * core.EV_A3_TO_GPA, STRESS_TOL)
        self.assertEqual(len(eng.nr), 3)

    def test_single_point_cli(self):
        from ofdft_opt import relax
        poscar = self.write_poscar("Al4.vasp", A0_PBE)
        code, out = run_main(relax.main, [poscar, "--steps", "0", "--prefix", "sp"], self.run)
        self.assertEqual(code, 0, out)
        st = read_json(os.path.join(self.run, "sp_status.json"))
        self.assertEqual(st["state"], "converged", st)
        self.assertAlmostEqual(st["energy_per_atom_eV"], E0_PBE, delta=E_TOL)
        self.assertLess(st["max_stress_GPa"], STRESS_TOL)
        self.assertEqual(st["pp_sha256"], {"Al": GGA_SHA})
        par = read_json(os.path.join(self.run, "sp_params.json"))
        self.assertEqual(par["engine"], "dftpy")
        self.assertEqual(par["xc"], "pbe")
        self.assertEqual(par["pp"], {"Al": "al.gga.recpot"})
        self.assertEqual(par["pp_sha256"], {"Al": GGA_SHA})
        self.assertAlmostEqual(par["econv"], 1e-7)
        self.assertEqual(par["fmax"], 0.01)
        self.assertEqual(par["smax"], 0.05)
        with open(os.path.join(self.run, "sp.log")) as f:
            log = f.read()
        head = log.split("energy (eV)")[0]
        self.assertIn(f"ofdft-opt {ofdft_opt.__version__}", head)
        self.assertIn(ofdft_opt.XC_BACKEND, head)
        self.assertIn("al.gga.recpot", head)
        self.assertIn(GGA_SHA[:12], head)

    def test_full_relax_from_420(self):
        from ofdft_opt import core, relax
        poscar = self.write_poscar("Al420.vasp", 4.20, rattle=0.02)
        code, out = run_main(relax.main, [poscar, "--prefix", "r420"], self.run)
        self.assertEqual(code, 0, out)
        st = read_json(os.path.join(self.run, "r420_status.json"))
        self.assertEqual(st["state"], "converged", st)
        self.assertEqual(st["reason"], "converged")
        self.assertLess(st["max_force_eVA"], 0.01)
        self.assertLess(st["max_stress_GPa"], 0.05)
        s = core.read_poscar(os.path.join(self.run, "r420_CONTCAR"))
        lengths = np.linalg.norm(s["cell"], axis=1)
        np.testing.assert_allclose(lengths, A0_PBE, atol=A_TOL)
        self.assertAlmostEqual(abs(np.linalg.det(s["cell"])) ** (1.0 / 3.0), A0_PBE, delta=A_TOL)
        self.assertAlmostEqual(st["energy_per_atom_eV"], E0_PBE, delta=E_TOL)

    def test_sscurve_continues_curve_of_old_script(self):
        """A curve started by the original ss_curve.py (params without 'version', absolute poscar path,
        pp as ["Al=al.gga.recpot"]) continues with `ofdft sscurve` and the same options (run_ss.sh)."""
        import csv
        from ofdft_opt import sscurve
        ref = self.write_poscar("r420_CONTCAR", A0_PBE)
        prefix = "r420_CONTCAR_ss_z_tension"
        legacy = {"poscar": "/home/someone/old/run/r420_CONTCAR", "pp": ["Al=al.gga.recpot"], "xc": "pbe",
                  "kedf": "WT", "ecut": 600.0, "nr": None, "axis": "z", "mode": "tension", "strain_step": 0.01,
                  "fmax": 0.01, "smax": 0.05, "econv": 1e-07, "econv_loose": 0.0, "econv_switch": 5.0}
        with open(os.path.join(self.run, prefix + "_params.json"), "w") as f:
            json.dump(legacy, f, indent=1)
        with open(os.path.join(self.run, prefix + ".csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(sscurve.CSV_COLS)
            for e in (0.0, 0.01):
                w.writerow([e] + [0.0] * 6 + [-228.7, -57.18, 0.001, 0.0, 2, 3, 1.0, True])
                shutil.copyfile(os.path.join(self.run, ref), os.path.join(self.run, f"{prefix}_e{e:+.4f}_CONTCAR"))
        opts = ["--pp", "Al=al.gga.recpot", "--precon", "none", "--econv", "1e-7", "--econv-loose", "0",
                "--fmax", "0.01", "--smax", "0.05", "--steps", "1000", "--strain-step", "0.01", "--strain-max", "0.01"]
        code, out = run_main(sscurve.main, [ref, *opts], self.run)
        self.assertEqual(code, 0, out)
        st = read_json(os.path.join(self.run, prefix + "_status.json"))
        self.assertEqual(st["state"], "complete", st)
        self.assertEqual(st["dftpy_calls"], 0, "both points were done by the old script")
        par = read_json(os.path.join(self.run, prefix + "_params.json"))
        self.assertEqual(par["version"], ofdft_opt.__version__)
        self.assertEqual(par["pp_sha256"], {"Al": GGA_SHA})
        self.assertEqual(par["legacy_params"], legacy)
        # rewritten in the new format: the same command keeps working, a changed option is refused
        code, out = run_main(sscurve.main, [ref, *opts], self.run)
        self.assertEqual(code, 0, out)
        code, out = run_main(sscurve.main, [ref, *opts, "--fmax", "0.005"], self.run)
        self.assertEqual(code, 2, out)


if __name__ == "__main__":
    unittest.main()
