"""POSCAR / XYZ I/O (ofdft_opt.core) and the shared run-file helpers (ofdft_opt.runinfo)."""
import contextlib
import datetime
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np  # noqa: E402

import ofdft_opt  # noqa: E402,F401  (package import first: shim + XC stand-in before anything else)
from ofdft_opt import core, runinfo  # noqa: E402

# Optional local checkout of the original scripts (only used for "copied verbatim" checks; skipped if absent)
SOURCE_DIR = os.environ.get("OFDFT_TEST_SRC_DIR") or os.path.expanduser(os.path.join("~", "DFTpy-al"))


@contextlib.contextmanager
def chdir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def wrap_frac(positions, cell):
    f = np.asarray(positions) @ np.linalg.inv(cell)
    f -= np.floor(f)
    f[np.isclose(f, 1.0, atol=1e-12)] = 0.0
    return f


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def path(self, name):
        return os.path.join(self.tmp, name)

    def write_text(self, name, text, newline="\n"):
        p = self.path(name)
        with open(p, "w", newline=newline) as f:
            f.write(text)
        return p


class TestPoscar(TempDirCase):
    CELL = np.array([[4.1, 0.0, 0.0],
                     [0.3, 3.9, 0.0],
                     [-0.2, 0.4, 4.3]])

    def test_roundtrip_selective_dynamics_species_runs(self):
        symbols = ["Al", "Al", "Ni", "Al", "Ni"]
        frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.25, 0.75, 0.5],
                         [0.9, 0.1, 0.35], [0.6, 0.2, 0.8]])
        pos = frac @ self.CELL
        pos[3] += self.CELL[0]                      # outside the box: written wrapped
        fixed = np.array([[True, True, True], [False, False, False], [True, False, True],
                          [False, True, False], [False, False, True]])
        p = self.path("SD_POSCAR")
        core.write_poscar(p, symbols, self.CELL, pos, fixed, comment="sd test")
        self.assertFalse(os.path.exists(p + ".tmp"), "atomic write left a temp file")
        with open(p) as f:
            text = f.read()
        self.assertIn("Selective dynamics", text)
        self.assertIn("Direct", text)
        s = core.read_poscar(p)
        self.assertEqual(s["comment"], "sd test")
        self.assertEqual(s["symbols"], symbols)     # order of the runs Al Al | Ni | Al | Ni kept
        np.testing.assert_allclose(s["cell"], self.CELL, atol=1e-12)
        np.testing.assert_allclose(wrap_frac(s["positions"], s["cell"]), wrap_frac(pos, self.CELL), atol=1e-12)
        self.assertIsNotNone(s["fixed"])
        np.testing.assert_array_equal(s["fixed"], fixed)
        # written fractional coordinates are wrapped into [0, 1)
        f = s["positions"] @ np.linalg.inv(s["cell"])
        self.assertTrue(np.all(f >= -1e-12) and np.all(f < 1.0 + 1e-12))
        # species line keeps consecutive runs
        lines = text.splitlines()
        self.assertEqual(lines[5].split(), ["Al", "Ni", "Al", "Ni"])
        self.assertEqual(lines[6].split(), ["2", "1", "1", "1"])

    def test_roundtrip_without_selective_dynamics(self):
        symbols = ["Al"] * 4
        frac = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], float)
        p = self.path("plain_POSCAR")
        core.write_poscar(p, symbols, self.CELL, frac @ self.CELL)
        s = core.read_poscar(p)
        self.assertIsNone(s["fixed"])
        with open(p) as fh:
            self.assertNotIn("Selective", fh.read())
        core.write_poscar(self.path("again"), s["symbols"], s["cell"], s["positions"], s["fixed"])
        s2 = core.read_poscar(self.path("again"))
        np.testing.assert_allclose(s2["cell"], s["cell"], atol=1e-13)
        np.testing.assert_allclose(wrap_frac(s2["positions"], s2["cell"]), wrap_frac(frac @ self.CELL, self.CELL),
                                   atol=1e-12)

    def test_negative_scale_is_target_volume(self):
        text = ("negative scale\n"
                "  -64.0\n"
                "  2.0 0.0 0.0\n"
                "  0.0 2.0 0.0\n"
                "  0.0 0.0 2.0\n"
                "  Al\n"
                "  2\n"
                "Direct\n"
                "  0.0 0.0 0.0\n"
                "  0.5 0.5 0.5\n")
        s = core.read_poscar(self.write_text("NEG", text))
        self.assertAlmostEqual(abs(np.linalg.det(s["cell"])), 64.0, places=10)
        np.testing.assert_allclose(s["cell"], 4.0 * np.eye(3), atol=1e-12)
        np.testing.assert_allclose(s["positions"], [[0, 0, 0], [2, 2, 2]], atol=1e-12)
        # round trip keeps the scaled structure (written with scale 1.0)
        core.write_poscar(self.path("NEG_out"), s["symbols"], s["cell"], s["positions"])
        s2 = core.read_poscar(self.path("NEG_out"))
        np.testing.assert_allclose(s2["cell"], s["cell"], atol=1e-12)
        np.testing.assert_allclose(wrap_frac(s2["positions"], s2["cell"]), wrap_frac(s["positions"], s["cell"]),
                                   atol=1e-12)

    def test_negative_scale_triclinic(self):
        cell = np.array([[3.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.5, 0.5, 3.5]])
        rows = "\n".join("  %.12f %.12f %.12f" % tuple(v) for v in cell)
        text = f"tri\n -100.0\n{rows}\n Al\n 1\nDirect\n 0.25 0.25 0.25\n"
        s = core.read_poscar(self.write_text("TRI", text))
        self.assertAlmostEqual(abs(np.linalg.det(s["cell"])), 100.0, places=9)
        k = (100.0 / abs(np.linalg.det(cell))) ** (1.0 / 3.0)
        np.testing.assert_allclose(s["cell"], k * cell, atol=1e-12)
        np.testing.assert_allclose(s["positions"][0], 0.25 * (k * cell).sum(axis=0), atol=1e-12)

    def test_cartesian_with_scale(self):
        text = ("cartesian\n"
                "  1.5\n"
                "  2.0 0.0 0.0\n"
                "  0.0 2.0 0.0\n"
                "  0.0 0.0 2.0\n"
                "  Al Cu\n"
                "  1 1\n"
                "Cartesian\n"
                "  0.0 0.0 0.0\n"
                "  1.0 1.0 1.0\n")
        s = core.read_poscar(self.write_text("CART", text))
        self.assertEqual(s["symbols"], ["Al", "Cu"])
        np.testing.assert_allclose(s["cell"], 3.0 * np.eye(3), atol=1e-12)
        np.testing.assert_allclose(s["positions"], [[0, 0, 0], [1.5, 1.5, 1.5]], atol=1e-12)
        self.assertIsNone(s["fixed"])
        # Cartesian in -> Direct out -> same Cartesian positions
        core.write_poscar(self.path("CART_out"), s["symbols"], s["cell"], s["positions"])
        s2 = core.read_poscar(self.path("CART_out"))
        np.testing.assert_allclose(wrap_frac(s2["positions"], s2["cell"]), wrap_frac(s["positions"], s["cell"]),
                                   atol=1e-12)

    def test_cartesian_selective_dynamics(self):
        text = ("cart sd\n 1.0\n 4 0 0\n 0 4 0\n 0 0 4\n Al\n 2\n"
                "Selective dynamics\nCartesian\n"
                " 0.0 0.0 0.0 F F F\n"
                " 2.0 2.0 0.0 T T F\n")
        s = core.read_poscar(self.write_text("CART_SD", text))
        np.testing.assert_allclose(s["positions"], [[0, 0, 0], [2, 2, 0]], atol=1e-12)
        np.testing.assert_array_equal(s["fixed"], [[True, True, True], [False, False, True]])
        core.write_poscar(self.path("CART_SD_out"), s["symbols"], s["cell"], s["positions"], s["fixed"])
        s2 = core.read_poscar(self.path("CART_SD_out"))
        np.testing.assert_array_equal(s2["fixed"], s["fixed"])
        np.testing.assert_allclose(wrap_frac(s2["positions"], s2["cell"]), wrap_frac(s["positions"], s["cell"]),
                                   atol=1e-12)

    def test_crlf_file_reads_like_lf(self):
        text = "crlf\n 1.0\n 4 0 0\n 0 4 0\n 0 0 4\n Al\n 2\nDirect\n 0 0 0\n 0.5 0.5 0.5\n"
        a = core.read_poscar(self.write_text("LF", text))
        b = core.read_poscar(self.write_text("CRLF", text, newline="\r\n"))
        self.assertEqual(a["symbols"], b["symbols"])
        self.assertEqual(a["comment"], b["comment"])
        np.testing.assert_array_equal(a["cell"], b["cell"])
        np.testing.assert_array_equal(a["positions"], b["positions"])

    def test_vasp4_without_element_line_is_rejected(self):
        text = "vasp4\n 1.0\n 4 0 0\n 0 4 0\n 0 0 4\n 2\nDirect\n 0 0 0\n 0.5 0.5 0.5\n"
        with self.assertRaises(ValueError):
            core.read_poscar(self.write_text("V4", text))

    def test_bundled_vacancy_structure(self):
        p = os.path.join(ofdft_opt.PKG_DIR, "data", "al_vacancy_255.vasp")
        self.assertTrue(os.path.isfile(p), p)
        with open(p, "rb") as f:
            raw = f.read()
        self.assertNotIn(b"\r", raw)
        s = core.read_poscar(p)
        self.assertEqual(len(s["symbols"]), 255)
        self.assertEqual(set(s["symbols"]), {"Al"})
        np.testing.assert_allclose(s["cell"], 16.2504 * np.eye(3), atol=1e-9)
        src = os.path.join(SOURCE_DIR, "test_255", "POSCAR_test")
        if os.path.isfile(src):                     # copied verbatim (LF) from the original test structure
            with open(src, "rb") as f:
                self.assertEqual(raw, f.read().replace(b"\r\n", b"\n"))


class TestXyz(TempDirCase):
    def test_append_xyz_two_frames(self):
        cell = np.diag([4.0, 4.1, 4.2])
        cell[2, 0] = 0.3
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.5]])
        forces = np.array([[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]])
        stress = np.array([[1, 2, 3], [2, 5, 6], [3, 6, 9]], float) * 1e-3
        p = self.path("traj.xyz")
        core.append_xyz(p, ["Al", "Ni"], cell, pos, forces, -12.5, stress)
        core.append_xyz(p, ["Al", "Ni"], cell, pos + 0.1, forces, -12.6, stress)
        with open(p) as f:
            lines = f.read().splitlines()
        self.assertEqual(len(lines), 8)
        self.assertEqual(lines[0].strip(), "2")
        self.assertIn('Lattice="', lines[1])
        self.assertIn("energy=-12.5", lines[1])
        if importlib.util.find_spec("ase") is None:
            self.skipTest("ase not installed")
        import ase.io
        frames = ase.io.read(p, index=":", format="extxyz")
        self.assertEqual(len(frames), 2)
        a = frames[1]
        self.assertEqual(a.get_chemical_symbols(), ["Al", "Ni"])
        np.testing.assert_allclose(a.cell.array, cell, atol=1e-9)
        np.testing.assert_allclose(a.positions, pos + 0.1, atol=1e-9)
        self.assertAlmostEqual(a.get_potential_energy(), -12.6, places=8)
        np.testing.assert_allclose(a.get_forces(), forces, atol=1e-7)
        np.testing.assert_allclose(a.get_stress(voigt=False), stress, rtol=1e-8, atol=1e-14)


class TestRunInfo(TempDirCase):
    def test_sha256_text_normalises_crlf(self):
        body = "line one\nline two\n\nlast"
        lf = self.write_text("lf.txt", body)
        crlf = self.write_text("crlf.txt", body, newline="\r\n")
        with open(crlf, "rb") as f:
            self.assertIn(b"\r\n", f.read())
        want = hashlib.sha256(body.encode()).hexdigest()
        self.assertEqual(runinfo.sha256_text(lf), want)
        self.assertEqual(runinfo.sha256_text(crlf), want)
        self.assertEqual(runinfo.sha256_bytes(body.encode()), want)
        self.assertEqual(runinfo.sha256_bytes(b""), hashlib.sha256(b"").hexdigest())

    def test_json_atomic_roundtrip(self):
        p = self.path("x_status.json")
        obj = {"state": "running", "steps": 3, "energy_eV": -1.25, "pp_sha256": {"Al": "ab" * 32},
               "max_stress_GPa": None}
        runinfo.write_json_atomic(p, obj)
        self.assertEqual(runinfo.read_json(p), obj)
        with open(p) as f:
            self.assertEqual(json.load(f), obj)
        obj["state"] = "converged"
        runinfo.write_json_atomic(p, obj)           # replaces the old content
        self.assertEqual(runinfo.read_json(p)["state"], "converged")
        leftovers = [n for n in os.listdir(self.tmp) if n != "x_status.json"]
        self.assertEqual(leftovers, [], "atomic write left temporary files")

    def test_read_json_missing_returns_default(self):
        self.assertIsNone(runinfo.read_json(self.path("missing.json")))
        self.assertEqual(runinfo.read_json(self.path("missing.json"), default={"a": 1}), {"a": 1})

    def test_stop_requested(self):
        with chdir(self.tmp):
            self.assertIsNone(runinfo.stop_requested("job"))
            open("other.STOP", "w").close()
            self.assertIsNone(runinfo.stop_requested("job"))
            open("job.STOP", "w").close()
            self.assertEqual(runinfo.stop_requested("job"), "job.STOP")
            os.remove("job.STOP")
            open("STOP", "w").close()
            self.assertEqual(runinfo.stop_requested("job"), "STOP")
            open("job.STOP", "w").close()
            self.assertIn(runinfo.stop_requested("job"), ("job.STOP", "STOP"))

    def test_now_iso(self):
        t = runinfo.now_iso()
        self.assertIsInstance(t, str)
        datetime.datetime.fromisoformat(t.replace("Z", "+00:00"))


if __name__ == "__main__":
    unittest.main()
