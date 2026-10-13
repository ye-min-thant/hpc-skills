"""EMT engine (testing/dry-run stand-in for DFTpy): relaxation and uniaxial loading through
core.Relaxer / sscurve.run_curve and through the `ofdft relax` / `ofdft sscurve` CLIs."""
import contextlib
import csv
import importlib.util
import inspect
import io
import json
import os
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np  # noqa: E402

import ofdft_opt  # noqa: E402

HAVE_ASE = importlib.util.find_spec("ase") is not None
if HAVE_ASE:
    from ase.build import bulk  # noqa: E402
    from ase.calculators.emt import EMT  # noqa: E402

from ofdft_opt import core  # noqa: E402

EMT_A0 = 3.9943          # EMT lattice constant of fcc Al (A), measured with this relaxer at 0.001/0.005


def fcc_al(a, rep=1, rattle=0.0, seed=1):
    at = bulk("Al", "fcc", a=a, cubic=True)
    if rep > 1:
        at = at.repeat(rep)
    if rattle:
        at.rattle(rattle, seed=seed)
    return at.get_chemical_symbols(), at.cell.array.copy(), at.positions.copy()


def write_fcc_poscar(path, a, rep=1, rattle=0.0, seed=1):
    symbols, cell, pos = fcc_al(a, rep, rattle, seed)
    core.write_poscar(path, symbols, cell, pos, comment=f"fcc Al a={a}")
    return path


def serial_mp():
    comm = types.SimpleNamespace(size=1, rank=0, bcast=lambda obj, root=0: obj, allreduce=lambda obj, *a, **k: obj,
                                 Barrier=lambda: None, barrier=lambda: None)
    return types.SimpleNamespace(is_mpi=False, comm=comm, size=1, rank=0, is_root=True)


def make_emt_engine(symbols):
    """core.EMTEngine, constructed from whatever of (args, symbols, mp, verbose) its signature asks for."""
    from ofdft_opt import relax
    kwargs = {}
    for name, p in inspect.signature(core.EMTEngine).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if name == "symbols":
            kwargs[name] = list(symbols)
        elif name == "args":
            args = relax.parse_args(["dummy_POSCAR", "--engine", "emt"])
            if getattr(args, "econv", None) is None:
                args.econv = 1e-7
            kwargs[name] = args
        elif name == "mp":
            kwargs[name] = serial_mp()
        elif name == "verbose":
            kwargs[name] = False
        elif p.default is p.empty:
            raise AssertionError(f"unexpected required EMTEngine parameter {name!r}")
    return core.EMTEngine(**kwargs)


def curve_opt(strains, mode="tension", fmax=0.01, smax=0.05, steps=500, axis="z"):
    return dict(axis=axis, mode=mode, strains=list(strains), fmax=fmax, smax=smax, steps=steps,
                precon=False, predictor="affine", keep_memory=False, regrid_tol=0.0, alpha=10.0,
                stiffness=100.0, memory=20, maxstep=0.2, maxstrain=0.01, econv_tight=None,
                econv_loose=None, econv_switch=5.0, ref_check=None)


def run_main(main, argv, cwd):
    """Call a command's main(argv) in-process inside cwd. Returns (exit code, stdout+stderr text)."""
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
                if e.code is None:
                    code = 0
                elif isinstance(e.code, int):
                    code = e.code
                else:
                    code = 1
                    out.write(str(e.code))
            except ModuleNotFoundError as e:
                if (e.name or "").split(".")[0] in ("dftpy", "mpi4py"):
                    raise unittest.SkipTest(f"CLI needs {e.name}: {e}")
                raise
    finally:
        os.chdir(old)
    return code, out.getvalue()


def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@unittest.skipUnless(HAVE_ASE, "ase not installed")
class TestEMTEngine(unittest.TestCase):
    def test_matches_ase_emt_and_interface(self):
        symbols, cell, pos = fcc_al(4.05, rep=1, rattle=0.03)
        eng = make_emt_engine(symbols)
        self.assertEqual(eng.ncalls, 0)
        E, F, S = eng(cell, pos, True)
        ref = bulk("Al", "fcc", a=4.05, cubic=True)
        ref.cell = cell
        ref.positions = pos
        ref.calc = EMT()
        self.assertAlmostEqual(float(E), ref.get_potential_energy(), places=10)
        np.testing.assert_allclose(F, ref.get_forces(), atol=1e-10)
        np.testing.assert_allclose(S, ref.get_stress(voigt=False), atol=1e-12)   # ASE convention (+ = tensile)
        self.assertEqual(np.shape(F), (4, 3))
        self.assertEqual(np.shape(S), (3, 3))
        E2, F2, S2 = eng(cell * 1.01, pos * 1.01, False)
        self.assertEqual(np.shape(S2), (3, 3))
        self.assertEqual(eng.ncalls, 2)
        self.assertGreaterEqual(eng.time, 0.0)
        self.assertEqual(eng.nwarn, 0)
        self.assertIsNone(eng.scf_log)
        self.assertEqual(len(np.asarray(eng.nr)), 3)
        self.assertEqual(eng.pp, {})
        self.assertEqual(eng.pp_sha256, {})
        eng.set_econv(1e-9)
        self.assertEqual(eng.econv, 1e-9)
        eng.reset_grid()
        self.assertEqual(eng.spacing_drift(cell * 1.1), 0.0)
        E3, _, _ = eng(cell, pos, True)
        self.assertAlmostEqual(float(E3), float(E), places=10)

    def test_tension_gives_positive_stress(self):
        symbols, cell, pos = fcc_al(EMT_A0)
        eng = make_emt_engine(symbols)
        F = np.diag([1.0, 1.0, 1.01])
        _, _, S = eng(cell @ F.T, pos @ F.T, True)
        self.assertGreater(S[2, 2], 0.0)


@unittest.skipUnless(HAVE_ASE, "ase not installed")
class TestRelaxerEMT(unittest.TestCase):
    def test_full_relax_converges(self):
        symbols, cell, pos = fcc_al(4.10, rep=2, rattle=0.02)
        eng = make_emt_engine(symbols)
        rel = core.Relaxer(eng, cell, pos, symbols, None, "full", 0.01, 0.05, 0.0)
        st, conv = rel.run(300)
        self.assertTrue(conv)
        self.assertLess(st["fmax"], 0.01)
        self.assertLess(st["smax"], 0.05)
        self.assertLess(np.abs(st["stress"]).max() * core.EV_A3_TO_GPA, 0.05)
        a = np.linalg.norm(st["cell"], axis=1) / 2
        np.testing.assert_allclose(a, EMT_A0, atol=0.01)

    def test_full_relax_tight_from_420(self):
        symbols, cell, pos = fcc_al(4.20, rattle=0.02)
        eng = make_emt_engine(symbols)
        rel = core.Relaxer(eng, cell, pos, symbols, None, "full", 0.001, 0.005, 0.0)
        st, conv = rel.run(300)
        self.assertTrue(conv)
        np.testing.assert_allclose(np.linalg.norm(st["cell"], axis=1), EMT_A0, atol=0.002)

    def test_positions_mode_keeps_cell(self):
        symbols, cell, pos = fcc_al(4.05, rep=2, rattle=0.05)
        eng = make_emt_engine(symbols)
        rel = core.Relaxer(eng, cell, pos, symbols, None, "positions", 0.01, None, 0.0)
        st, conv = rel.run(300)
        self.assertTrue(conv)
        np.testing.assert_array_equal(st["cell"], cell)
        self.assertLess(st["fmax"], 0.01)

    def test_iso_mode_scales_uniformly(self):
        symbols, cell, pos = fcc_al(4.15)
        eng = make_emt_engine(symbols)
        rel = core.Relaxer(eng, cell, pos, symbols, None, "iso", 0.01, 0.05, 0.0)
        st, conv = rel.run(300)
        self.assertTrue(conv)
        k = st["cell"][0, 0] / cell[0, 0]
        np.testing.assert_allclose(st["cell"], k * cell, atol=1e-12)
        self.assertAlmostEqual(st["cell"][0, 0], EMT_A0, delta=0.01)

    def test_fixed_atoms_do_not_move(self):
        symbols, cell, pos = fcc_al(4.05, rep=2, rattle=0.05)
        fixed = np.zeros((len(pos), 3), bool)
        fixed[0] = True
        fixed[5, 2] = True
        eng = make_emt_engine(symbols)
        rel = core.Relaxer(eng, cell, pos, symbols, fixed, "positions", 0.01, None, 0.0)
        st, conv = rel.run(300)
        self.assertTrue(conv)
        np.testing.assert_array_equal(st["R"][0], pos[0])
        self.assertEqual(st["R"][5, 2], pos[5, 2])
        self.assertGreater(np.abs(st["R"] - pos).max(), 1e-3)


@unittest.skipUnless(HAVE_ASE, "ase not installed")
class TestRunCurveEMT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ofdft_opt import sscurve
        cls.sscurve = sscurve
        cls.symbols, cell, pos = fcc_al(4.05, rep=2, rattle=0.02)
        eng = make_emt_engine(cls.symbols)
        rel = core.Relaxer(eng, cell, pos, cls.symbols, None, "full", 0.001, 0.005, 0.0)
        st, conv = rel.run(500)
        assert conv, "EMT reference relaxation failed"
        cls.cell0, cls.R0 = st["cell"].copy(), st["R"].copy()

    def run_points(self, strains, **kw):
        eng = make_emt_engine(self.symbols)
        cells = {}
        rows = self.sscurve.run_curve(eng, self.cell0, self.R0, self.symbols, None, curve_opt(strains, **kw),
                                      log=lambda *a, **k: None,
                                      on_point=lambda e, st, row: cells.__setitem__(round(e, 10), st["cell"].copy()))
        return rows, cells

    def test_tension_curve_rigid_grips(self):
        rows, cells = self.run_points([0.0, 0.005, 0.01], fmax=0.005, smax=0.005)
        self.assertEqual([r["strain"] for r in rows], [0.0, 0.005, 0.01])
        self.assertTrue(all(r["converged"] for r in rows))
        s = [r["szz_GPa"] for r in rows]
        self.assertLess(abs(s[0]), 0.05)
        self.assertGreater(s[1], 0.0)
        self.assertGreater(s[2], s[1])
        slope = s[2] / 0.01
        self.assertTrue(10.0 < slope < 200.0, f"initial slope {slope:.1f} GPa outside the sane range")
        for r in rows:                                   # lateral and shear stresses relaxed
            for c in ("sxx_GPa", "syy_GPa", "syz_GPa", "sxz_GPa", "sxy_GPa"):
                self.assertLess(abs(r[c]), 0.005 + 1e-9, (r["strain"], c, r[c]))
            self.assertLessEqual(r["max_free_stress_GPa"], 0.005)
        for e, cell in cells.items():
            # rigid grips: the z column of the cell (box length along the axis) is exactly (1+e) x reference
            np.testing.assert_allclose(cell[:, 2], (1 + e) * self.cell0[:, 2], rtol=0, atol=1e-12)
        # lateral contraction happened (Poisson effect) at 1 %
        self.assertLess(np.linalg.norm(cells[0.01][0]), np.linalg.norm(self.cell0[0]))
        self.assertTrue(set(self.sscurve.CSV_COLS) <= set(rows[0]))

    def test_uniaxial_strain_mode(self):
        rows, cells = self.run_points([0.0, 0.01], mode="strain")
        self.assertTrue(all(r["converged"] for r in rows))
        F = np.diag([1.0, 1.0, 1.01])
        np.testing.assert_allclose(cells[0.01], self.cell0 @ F.T, atol=1e-12)
        r = rows[-1]
        self.assertGreater(r["szz_GPa"], 0.0)
        self.assertGreater(r["sxx_GPa"], 0.0)            # laterally constrained: tensile lateral stress
        self.assertGreater(r["szz_GPa"], r["sxx_GPa"])

    def test_compression(self):
        strains = self.sscurve.strain_list(-0.005, -0.01)
        self.assertEqual(strains, [0.0, -0.005, -0.01])
        rows, cells = self.run_points(strains, fmax=0.005, smax=0.005)
        s = [r["szz_GPa"] for r in rows]
        self.assertTrue(all(r["converged"] for r in rows))
        self.assertLess(s[1], 0.0)
        self.assertLess(s[2], s[1])
        np.testing.assert_allclose(cells[-0.01][:, 2], 0.99 * self.cell0[:, 2], atol=1e-12)

    def test_strain_list(self):
        self.assertEqual(self.sscurve.strain_list(0.005, 0.02), [0.0, 0.005, 0.01, 0.015, 0.02])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.sscurve.strain_list(0.005, -0.02)


@unittest.skipUnless(HAVE_ASE, "ase not installed")
class TestCLIWithEMT(unittest.TestCase):
    STATUS_KEYS = {"state", "reason", "steps", "dftpy_calls", "energy_eV", "energy_per_atom_eV", "max_force_eVA",
                   "max_stress_GPa", "fmax", "smax", "relax", "version", "pp_sha256", "contcar", "wall_s",
                   "finished"}
    PARAM_KEYS = {"version", "engine", "xc", "kedf", "ecut", "nr", "relax", "pressure", "pp", "pp_sha256",
                  "poscar", "poscar_sha256", "fmax", "smax", "econv"}

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def p(self, name):
        return os.path.join(self.tmp, name)

    def relax(self, argv):
        from ofdft_opt import relax
        return run_main(relax.main, argv, self.tmp)

    def sscurve(self, argv):
        from ofdft_opt import sscurve
        return run_main(sscurve.main, argv, self.tmp)

    def test_relax_cli_converges_and_writes_status(self):
        write_fcc_poscar(self.p("Al.vasp"), 4.10, rep=2, rattle=0.02)
        code, out = self.relax(["Al.vasp", "--engine", "emt"])
        self.assertEqual(code, 0, out)
        st = read_json(self.p("Al_status.json"))
        self.assertTrue(self.STATUS_KEYS <= set(st), self.STATUS_KEYS - set(st))
        self.assertEqual(st["state"], "converged")
        self.assertEqual(st["reason"], "converged")
        self.assertEqual(st["version"], ofdft_opt.__version__)
        self.assertEqual(st["relax"], "full")
        self.assertEqual(st["fmax"], 0.01)                 # production defaults
        self.assertEqual(st["smax"], 0.05)
        self.assertLess(st["max_force_eVA"], 0.01)
        self.assertIsNotNone(st["max_stress_GPa"])
        self.assertLess(st["max_stress_GPa"], 0.05)
        self.assertAlmostEqual(st["energy_per_atom_eV"] * 32, st["energy_eV"], places=6)
        self.assertEqual(os.path.basename(st["contcar"]), "Al_CONTCAR")
        s = core.read_poscar(self.p("Al_CONTCAR"))
        np.testing.assert_allclose(np.linalg.norm(s["cell"], axis=1) / 2, EMT_A0, atol=0.01)
        par = read_json(self.p("Al_params.json"))
        self.assertTrue(self.PARAM_KEYS <= set(par), self.PARAM_KEYS - set(par))
        self.assertEqual(par["engine"], "emt")
        self.assertEqual(par["poscar"], "Al.vasp")
        self.assertAlmostEqual(par["econv"], 1e-7)
        with open(self.p("Al.log")) as f:
            log = f.read()
        head = log.split("energy (eV)")[0]                 # header block, before the step table
        self.assertIn(f"ofdft-opt {ofdft_opt.__version__}", head)
        self.assertIn("emt", head.lower())
        self.assertIn("Al_status.json", log)
        self.assertIn("CONVERGED", log)

    def test_relax_cli_tight_fmax_tightens_econv(self):
        write_fcc_poscar(self.p("Al.vasp"), 4.10, rattle=0.02)
        code, out = self.relax(["Al.vasp", "--engine", "emt", "--fmax", "0.005", "--smax", "0.01"])
        self.assertEqual(code, 0, out)
        self.assertAlmostEqual(read_json(self.p("Al_params.json"))["econv"], 1e-8)
        self.assertEqual(read_json(self.p("Al_status.json"))["state"], "converged")

    def test_relax_cli_positions_mode_status(self):
        write_fcc_poscar(self.p("Al.vasp"), 4.05, rep=2, rattle=0.03)
        code, out = self.relax(["Al.vasp", "--engine", "emt", "--relax", "positions", "--prefix", "pos"])
        self.assertEqual(code, 0, out)
        st = read_json(self.p("pos_status.json"))
        self.assertEqual(st["state"], "converged")
        self.assertEqual(st["relax"], "positions")
        self.assertIsNone(st["max_stress_GPa"])
        s0, s1 = core.read_poscar(self.p("Al.vasp")), core.read_poscar(self.p("pos_CONTCAR"))
        np.testing.assert_allclose(s1["cell"], s0["cell"], atol=1e-12)

    def test_relax_cli_step_limit(self):
        write_fcc_poscar(self.p("Al.vasp"), 4.20, rep=2, rattle=0.05)
        code, out = self.relax(["Al.vasp", "--engine", "emt", "--steps", "1"])
        self.assertEqual(code, 0, out)                      # not converged is not an error: the status file tells
        st = read_json(self.p("Al_status.json"))
        self.assertEqual(st["state"], "not_converged")
        self.assertEqual(st["reason"], "steps")
        self.assertEqual(st["steps"], 1)

    def test_tension_curve_through_clis(self):
        write_fcc_poscar(self.p("Al.vasp"), 4.05, rep=2, rattle=0.02)
        code, out = self.relax(["Al.vasp", "--engine", "emt"])
        self.assertEqual(code, 0, out)
        self.assertEqual(read_json(self.p("Al_status.json"))["state"], "converged")
        code, out = self.sscurve(["Al_CONTCAR", "--engine", "emt", "--axis", "z", "--mode", "tension",
                                  "--strain-step", "0.005", "--strain-max", "0.01", "--prefix", "ss"])
        self.assertEqual(code, 0, out)
        from ofdft_opt import sscurve
        with open(self.p("ss.csv"), newline="") as f:
            header = next(csv.reader(f))
        self.assertEqual(header, list(sscurve.CSV_COLS))
        rows = read_csv(self.p("ss.csv"))
        self.assertEqual([round(float(r["strain"]), 6) for r in rows], [0.0, 0.005, 0.01])
        self.assertTrue(all(r["converged"].strip().lower() == "true" for r in rows))
        s = [float(r["szz_GPa"]) for r in rows]
        self.assertLess(abs(s[0]), 0.1)
        self.assertGreater(s[1], 0.0)
        self.assertGreater(s[2], s[1])
        self.assertTrue(10.0 < s[2] / 0.01 < 200.0, s)
        ref = core.read_poscar(self.p("Al_CONTCAR"))
        for e in (0.0, 0.005, 0.01):
            c = core.read_poscar(self.p(f"ss_e{e:+.4f}_CONTCAR"))
            np.testing.assert_allclose(c["cell"][:, 2], (1 + e) * ref["cell"][:, 2], rtol=0, atol=1e-9)
        st = read_json(self.p("ss_status.json"))
        self.assertEqual(st["state"], "complete")
        self.assertEqual(st["points_done"], 3)
        self.assertEqual(st["strains_total"], 3)
        self.assertEqual(st["version"], ofdft_opt.__version__)
        par = read_json(self.p("ss_params.json"))
        self.assertEqual(par["poscar"], "Al_CONTCAR")
        self.assertIn("poscar_sha256", par)
        self.assertEqual(par["version"], ofdft_opt.__version__)
        self.assertIn("pp_sha256", par)


if __name__ == "__main__":
    unittest.main()
