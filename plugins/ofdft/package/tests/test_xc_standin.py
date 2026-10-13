"""Vendored NumPy XC stand-in (ofdft_opt/_vendor/pylibxc) and the package import side effects."""
import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np  # noqa: E402

import ofdft_opt  # noqa: E402

VENDOR = os.path.join(ofdft_opt.PKG_DIR, "_vendor")
SOURCE_DIR = os.environ.get("OFDFT_TEST_SRC_DIR") or os.path.expanduser(os.path.join("~", "DFTpy-al"))
FUNCTIONALS = ["lda_x", "lda_c_pz", "gga_x_pbe", "gga_c_pbe"]


def load_standin():
    """The vendored functional module, loaded from its file (independent of whether a real pylibxc exists)."""
    path = os.path.join(VENDOR, "pylibxc", "functional.py")
    spec = importlib.util.spec_from_file_location("_ofdft_test_standin_functional", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def norm(p):
    return os.path.normcase(os.path.realpath(p))


class TestStandinDerivatives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_standin()
        # densities from vacuum-like to compressed metal (bohr^-3); reduced gradients s^2 ~ sigma / rho^(8/3)
        cls.rho = np.array([1e-4, 1e-3, 0.005, 0.02, 0.05, 0.1, 0.3, 1.0, 3.0])
        cls.sigma = cls.rho ** (8.0 / 3.0) * np.array([0.05, 0.1, 0.5, 1.0, 2.0, 0.3, 5.0, 0.01, 0.2])

    def energy_density(self, f, rho, sigma):
        """f(rho, sigma) = rho * eps_xc, from zk only (no derivative code involved)."""
        out = f.compute({"rho": rho, "sigma": sigma}, do_exc=True, do_vxc=False)
        self.assertNotIn("vrho", out)
        return out["zk"][:, 0] * rho

    def test_output_shapes(self):
        for name in FUNCTIONALS:
            f = self.m.LibXCFunctional(name, "unpolarized")
            out = f.compute({"rho": self.rho, "sigma": self.sigma})
            n = len(self.rho)
            self.assertEqual(out["zk"].shape, (n, 1), name)
            self.assertEqual(out["vrho"].shape, (n, 1), name)
            if name.startswith("gga"):
                self.assertEqual(out["vsigma"].shape, (n, 1), name)
            else:
                self.assertNotIn("vsigma", out)
            for v in out.values():
                self.assertTrue(np.all(np.isfinite(v)), name)
            self.assertTrue(np.all(out["zk"] < 0), name)       # x and c energies are negative

    def test_vrho_matches_finite_differences(self):
        for name in FUNCTIONALS:
            f = self.m.LibXCFunctional(name, "unpolarized")
            v = f.compute({"rho": self.rho, "sigma": self.sigma})["vrho"][:, 0]
            h = 1e-5 * self.rho
            fd = (self.energy_density(f, self.rho + h, self.sigma) -
                  self.energy_density(f, self.rho - h, self.sigma)) / (2 * h)
            err = np.max(np.abs(fd - v) / np.abs(v))
            self.assertLess(err, 1e-6, f"{name}: vrho vs central difference, max rel. error {err:.2e}")

    def test_vsigma_matches_finite_differences(self):
        for name in ("gga_x_pbe", "gga_c_pbe"):
            f = self.m.LibXCFunctional(name, "unpolarized")
            v = f.compute({"rho": self.rho, "sigma": self.sigma})["vsigma"][:, 0]
            h = 1e-4 * self.sigma
            fd = (self.energy_density(f, self.rho, self.sigma + h) -
                  self.energy_density(f, self.rho, self.sigma - h)) / (2 * h)
            err = np.max(np.abs(fd - v) / np.abs(v))
            self.assertLess(err, 1e-6, f"{name}: vsigma vs central difference, max rel. error {err:.2e}")
            self.assertTrue(np.all(v < 0) if name == "gga_x_pbe" else np.all(v > 0), name)

    def test_vsigma_at_zero_gradient(self):
        zero = np.zeros_like(self.rho)
        for name in ("gga_x_pbe", "gga_c_pbe"):
            f = self.m.LibXCFunctional(name, "unpolarized")
            v = f.compute({"rho": self.rho, "sigma": zero})["vsigma"][:, 0]
            h = 1e-7 * self.rho ** (8.0 / 3.0)
            f0, f1, f2 = (self.energy_density(f, self.rho, zero + k * h) for k in (0, 1, 2))
            fd = (-3 * f0 + 4 * f1 - f2) / (2 * h)               # one-sided, second order
            err = np.max(np.abs(fd - v) / np.abs(v))
            self.assertLess(err, 1e-5, f"{name}: vsigma(sigma=0) max rel. error {err:.2e}")

    def test_known_values(self):
        rs = 2.0
        rho = np.array([3.0 / (4.0 * np.pi * rs ** 3)])
        zk = {n: self.m.LibXCFunctional(n).compute({"rho": rho, "sigma": np.zeros(1)})["zk"][0, 0]
              for n in FUNCTIONALS}
        # LDA exchange: eps_x = -(3/4)(9/(4 pi^2))^(1/3) / rs = -0.4581652932831429 / rs  (Ha)
        self.assertAlmostEqual(zk["lda_x"], -0.4581652932831429 / rs, places=10)
        # PBE exchange reduces to LDA exchange for zero gradient
        self.assertAlmostEqual(zk["gga_x_pbe"], zk["lda_x"], places=12)
        # Perdew-Zunger 1981, rs >= 1 branch: gamma / (1 + beta1 sqrt(rs) + beta2 rs)
        self.assertAlmostEqual(zk["lda_c_pz"], -0.1423 / (1 + 1.0529 * np.sqrt(rs) + 0.3334 * rs), places=10)
        # PBE correlation at zero gradient = PW92 correlation, eps_c(rs=2) = -0.04476 Ha
        self.assertAlmostEqual(zk["gga_c_pbe"], -0.04476, delta=2e-5)
        # LDA exchange potential: v_x = (4/3) eps_x
        v = self.m.LibXCFunctional("lda_x").compute({"rho": rho})["vrho"][0, 0]
        self.assertAlmostEqual(v, 4.0 / 3.0 * zk["lda_x"], places=10)

    def test_pbe_exchange_enhancement_limit(self):
        rho = np.array([0.05])
        x_lda = self.m.LibXCFunctional("lda_x").compute({"rho": rho})["zk"][0, 0]
        pbe = self.m.LibXCFunctional("gga_x_pbe")
        huge = pbe.compute({"rho": rho, "sigma": np.array([1e12])})["zk"][0, 0]
        self.assertAlmostEqual(huge / x_lda, 1.804, places=4)     # F_x -> 1 + kappa

    def test_low_density_and_negative_sigma(self):
        f = self.m.LibXCFunctional("gga_c_pbe")
        out = f.compute({"rho": np.array([0.0, 1e-13, 0.1]), "sigma": np.array([0.0, 0.0, 0.01])})
        for k in ("zk", "vrho", "vsigma"):
            self.assertEqual(out[k][0, 0], 0.0)
            self.assertEqual(out[k][1, 0], 0.0)
            self.assertNotEqual(out[k][2, 0], 0.0)
        a = f.compute({"rho": np.array([0.1]), "sigma": np.array([-1e-6])})
        b = f.compute({"rho": np.array([0.1]), "sigma": np.array([0.0])})
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])

    def test_accepts_grid_shaped_input(self):
        f = self.m.LibXCFunctional("gga_x_pbe")
        rho = np.full((4, 3, 2), 0.02)
        out = f.compute({"rho": rho, "sigma": np.full((4, 3, 2), 1e-4)})
        self.assertEqual(out["zk"].shape, (24, 1))

    def test_errors(self):
        with self.assertRaises(KeyError):
            self.m.LibXCFunctional("mgga_x_scan", "unpolarized")
        with self.assertRaises(NotImplementedError):
            self.m.LibXCFunctional("lda_x", "polarized")


class TestPackageImport(unittest.TestCase):
    def test_version_and_dirs(self):
        self.assertEqual(ofdft_opt.__version__, "0.1.0")
        self.assertEqual(norm(ofdft_opt.PKG_DIR), norm(os.path.dirname(ofdft_opt.__file__)))
        self.assertEqual(norm(ofdft_opt.LIB_DIR), norm(os.path.dirname(ofdft_opt.PKG_DIR)))
        self.assertIn(ofdft_opt.XC_BACKEND, ("libxc", "ofdft-standin"))

    def test_no_top_level_pylibxc_published(self):
        self.assertFalse(os.path.exists(os.path.join(ofdft_opt.LIB_DIR, "pylibxc")))
        self.assertTrue(os.path.isfile(os.path.join(VENDOR, "pylibxc", "__init__.py")))
        self.assertTrue(os.path.isfile(os.path.join(VENDOR, "pylibxc", "functional.py")))

    def test_backend_consistent_with_pylibxc_location(self):
        import pylibxc  # importable after `import ofdft_opt` (real LibXC bindings or the stand-in)
        from_vendor = norm(pylibxc.__file__).startswith(norm(VENDOR) + os.sep)
        standin = from_vendor or ofdft_opt.XC_STANDIN_COPY is not None
        self.assertEqual(ofdft_opt.XC_BACKEND, "ofdft-standin" if standin else "libxc")
        if from_vendor:
            from pylibxc.functional import LibXCFunctional
            out = LibXCFunctional("gga_x_pbe", "unpolarized").compute({"rho": np.array([0.01]),
                                                                       "sigma": np.array([1e-6])})
            self.assertEqual(set(out), {"zk", "vrho", "vsigma"})

    def test_fresh_interpreter_import_side_effects(self):
        """In a clean interpreter (cwd without any pylibxc): no dftpy import, shim loaded, stand-in appended."""
        code = (
            "import importlib.util, json, os, sys\n"
            "had_real = importlib.util.find_spec('pylibxc') is not None\n"
            "import ofdft_opt\n"
            "vendor = os.path.join(ofdft_opt.PKG_DIR, '_vendor')\n"
            "idx = [i for i, p in enumerate(sys.path) if os.path.normcase(os.path.abspath(p)) == "
            "os.path.normcase(os.path.abspath(vendor))]\n"
            "site = [i for i, p in enumerate(sys.path) if 'site-packages' in p or 'dist-packages' in p]\n"
            "import pylibxc\n"
            "print(json.dumps(dict(had_real=had_real, backend=ofdft_opt.XC_BACKEND, version=ofdft_opt.__version__,\n"
            "    dftpy_loaded=any(m == 'dftpy' or m.startswith('dftpy.') for m in sys.modules),\n"
            "    shim_loaded='ofdft_opt._shim' in sys.modules,\n"
            "    shim_patched=getattr(sys.modules.get('ofdft_opt._shim'), 'PATCHED', None),\n"
            "    vendor_idx=idx, site_idx=site, pylibxc=os.path.abspath(pylibxc.__file__), vendor=vendor)))\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = SRC
        env.pop("PYTHONHOME", None)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            r = subprocess.run([sys.executable, "-c", code], cwd=tmp, env=env, capture_output=True,
                               text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        info = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual(info["version"], "0.1.0")
        self.assertFalse(info["dftpy_loaded"], "import ofdft_opt must not import dftpy")
        self.assertTrue(info["shim_loaded"], "ofdft_opt must import its scipy dcsrch shim first")
        self.assertIn(info["shim_patched"], (True, False))
        if info["had_real"]:
            self.assertEqual(info["backend"], "libxc")
            self.assertFalse(norm(info["pylibxc"]).startswith(norm(info["vendor"])))
        else:
            self.assertEqual(info["backend"], "ofdft-standin")
            self.assertTrue(norm(info["pylibxc"]).startswith(norm(info["vendor"]) + os.sep))
            self.assertEqual(len(info["vendor_idx"]), 1)
            # appended, not prepended: a real pylibxc installed later in site-packages still wins
            self.assertGreater(info["vendor_idx"][0], 0)
            if info["site_idx"]:
                self.assertGreater(info["vendor_idx"][0], max(info["site_idx"]))


class TestStandinCopyDetection(unittest.TestCase):
    """A pylibxc/ folder in the current directory (old run folders hold a copy of the stand-in) shadows
    the vendored one: it must be reported as the stand-in (a copy), never as real LibXC."""

    PROBE = ("import json, ofdft_opt\n"
             "print(json.dumps(dict(backend=ofdft_opt.XC_BACKEND, origin=ofdft_opt.XC_ORIGIN,\n"
             "    copy=ofdft_opt.XC_STANDIN_COPY, text=ofdft_opt.xc_description())))\n")

    def probe(self, setup):
        env = dict(os.environ)
        env["PYTHONPATH"] = SRC
        env.pop("PYTHONHOME", None)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            dst = os.path.join(tmp, "pylibxc")
            setup(dst)
            r = subprocess.run([sys.executable, "-c", self.PROBE], cwd=tmp, env=env, capture_output=True,
                               text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            info = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertEqual(norm(info["origin"]), norm(dst), "the copy in the cwd is what DFTpy would import")
        return info

    def test_identical_copy_in_cwd(self):
        def setup(dst):
            os.makedirs(dst)
            for name in ("__init__.py", "functional.py"):        # CRLF, as copied from Windows
                with open(os.path.join(VENDOR, "pylibxc", name), "rb") as f:
                    data = f.read().replace(b"\n", b"\r\n")
                with open(os.path.join(dst, name), "wb") as f:
                    f.write(data)
        info = self.probe(setup)
        self.assertEqual(info["backend"], "ofdft-standin")
        self.assertEqual(info["copy"], "identical")
        self.assertIn("copy at", info["text"])

    def test_modified_copy_in_cwd(self):
        def setup(dst):
            shutil.copytree(os.path.join(VENDOR, "pylibxc"), dst)
            with open(os.path.join(dst, "functional.py"), "a", encoding="utf-8") as f:
                f.write("\n# locally modified\n")
        info = self.probe(setup)
        self.assertEqual(info["backend"], "ofdft-standin")
        self.assertEqual(info["copy"], "modified")
        self.assertIn("MODIFIED", info["text"])

    def test_other_pylibxc_is_libxc(self):
        def setup(dst):
            os.makedirs(dst)
            with open(os.path.join(dst, "__init__.py"), "w") as f:
                f.write('"""Python bindings of LibXC (stand-in for the test)."""\n')
        info = self.probe(setup)
        self.assertEqual(info["backend"], "libxc")
        self.assertIsNone(info["copy"])


def _code_without_docstring(path):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    body = tree.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return [ast.dump(n) for n in body]


@unittest.skipUnless(os.path.isdir(SOURCE_DIR), "original DFTpy-al scripts not available")
class TestVerbatimCopies(unittest.TestCase):
    """The vendored files are copies of the originals (LF line endings)."""

    def _same_text(self, original, copy):
        if not os.path.isfile(original):
            self.skipTest(f"{original} missing")
        with open(original, "rb") as f:
            want = f.read().replace(b"\r\n", b"\n")
        with open(copy, "rb") as f:
            got = f.read()
        self.assertNotIn(b"\r", got)
        self.assertEqual(got, want, f"{copy} differs from {original}")

    def test_pylibxc_files(self):
        for name in ("__init__.py", "functional.py"):
            self._same_text(os.path.join(SOURCE_DIR, "pylibxc", name), os.path.join(VENDOR, "pylibxc", name))

    def test_shim_code(self):
        original = os.path.join(SOURCE_DIR, "scipy_dcsrch_shim.py")
        if not os.path.isfile(original):
            self.skipTest("scipy_dcsrch_shim.py missing")
        copy = os.path.join(ofdft_opt.PKG_DIR, "_shim.py")
        with open(copy, "rb") as f:
            self.assertNotIn(b"\r", f.read())
        self.assertEqual(_code_without_docstring(copy), _code_without_docstring(original),
                         "_shim.py code (docstring aside) must equal scipy_dcsrch_shim.py")


if __name__ == "__main__":
    unittest.main()
