"""CLI safety rails, exercised with the EMT engine (seconds, no DFTpy calculation):
existing-output refusal, --restart/--overwrite, physics-parameter checks, xc/pp mismatch, STOP files,
status files, the stress-strain reference check and the stress-strain parameter check."""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import ofdft_opt  # noqa: E402
from ofdft_opt import core  # noqa: E402

HAVE_ASE = importlib.util.find_spec("ase") is not None
SOURCE_DIR = os.environ.get("OFDFT_TEST_SRC_DIR") or os.path.expanduser(os.path.join("~", "DFTpy-al"))
PP_SHA = {"al.gga.recpot": "76034a6ea04586f32c2c13d9b4a828a83799638ab2d3b87c330afc653be09806",
          "al.lda.recpot": "c2fd4300aa39aa2466149a315e1c7cd87aac6615032db5ded8f237e9c40df803"}


def registry_copy(name):
    for d in (os.environ.get("OFDFT_TEST_PP_DIR"), SOURCE_DIR):
        p = os.path.join(d, name) if d else None
        if p and os.path.isfile(p):
            with open(p, "rb") as f:
                if hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest() == PP_SHA[name]:
                    return p
    return None


def fcc_al(a, rattle=0.0, seed=1):
    from ase.build import bulk
    at = bulk("Al", "fcc", a=a, cubic=True)
    if rattle:
        at.rattle(rattle, seed=seed)
    return at.get_chemical_symbols(), at.cell.array.copy(), at.positions.copy()


def write_fcc_poscar(path, a, rattle=0.0, seed=1):
    symbols, cell, pos = fcc_al(a, rattle, seed)
    core.write_poscar(path, symbols, cell, pos, comment=f"fcc Al a={a}")
    return path


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


def snapshot(d, names=None):
    res = {}
    for n in sorted(os.listdir(d)):
        p = os.path.join(d, n)
        if os.path.isfile(p) and (names is None or n in names):
            with open(p, "rb") as f:
                res[n] = (os.stat(p).st_mtime_ns, hashlib.sha256(f.read()).hexdigest())
    return res


def read_json(path):
    with open(path) as f:
        return json.load(f)


def csv_data_rows(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class CLICase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp = self._tmp.name
        self.env = mock.patch.dict(os.environ)
        self.env.start()
        for k in ("SLURM_STEP_NUM_TASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE"):
            os.environ.pop(k, None)

    def tearDown(self):
        self.env.stop()
        self._tmp.cleanup()

    def p(self, name):
        return os.path.join(self.tmp, name)

    def touch(self, name):
        open(self.p(name), "w").close()

    def relax(self, argv):
        from ofdft_opt import relax
        return run_main(relax.main, argv, self.tmp)

    def sscurve(self, argv):
        from ofdft_opt import sscurve
        return run_main(sscurve.main, argv, self.tmp)

    def log(self, prefix):
        with open(self.p(prefix + ".log")) as f:
            return f.read()


@unittest.skipUnless(HAVE_ASE, "ase not installed")
class TestRelaxGuards(CLICase):
    OUTPUTS = ("Al.log", "Al_CONTCAR", "Al_params.json", "Al_status.json")

    def setUp(self):
        super().setUp()
        write_fcc_poscar(self.p("Al.vasp"), 4.10, rattle=0.02)

    def first_run(self, *extra):
        code, out = self.relax(["Al.vasp", "--engine", "emt", *extra])
        self.assertEqual(code, 0, out)
        for n in self.OUTPUTS:
            self.assertTrue(os.path.isfile(self.p(n)), n)
        self.assertEqual(read_json(self.p("Al_status.json"))["state"], "converged")

    def test_refuses_existing_outputs(self):
        self.first_run()
        before = snapshot(self.tmp)
        code, out = self.relax(["Al.vasp", "--engine", "emt"])
        self.assertEqual(code, 2, out)
        self.assertIn("restart", out.lower())
        self.assertEqual(snapshot(self.tmp), before, "a refused run must not touch any file")

    def test_refuses_when_only_contcar_exists(self):
        shutil.copyfile(self.p("Al.vasp"), self.p("Al_CONTCAR"))
        before = snapshot(self.tmp)
        code, out = self.relax(["Al.vasp", "--engine", "emt"])
        self.assertEqual(code, 2, out)
        self.assertEqual(snapshot(self.tmp), before)

    def test_refuses_when_only_log_exists(self):
        with open(self.p("Al.log"), "w") as f:
            f.write("old log\n")
        before = snapshot(self.tmp)
        code, out = self.relax(["Al.vasp", "--engine", "emt"])
        self.assertEqual(code, 2, out)
        self.assertEqual(snapshot(self.tmp), before)

    def test_other_prefix_is_independent(self):
        self.first_run()
        code, out = self.relax(["Al.vasp", "--engine", "emt", "--prefix", "second"])
        self.assertEqual(code, 0, out)
        self.assertEqual(read_json(self.p("second_status.json"))["state"], "converged")

    def test_overwrite_starts_fresh(self):
        self.first_run()
        code, out = self.relax(["Al.vasp", "--engine", "emt", "--overwrite"])
        self.assertEqual(code, 0, out)
        self.assertEqual(self.log("Al").count("CONVERGED after"), 1, "--overwrite starts a new log")
        self.assertEqual(read_json(self.p("Al_status.json"))["state"], "converged")

    def test_restart_same_physics_new_tolerances(self):
        self.first_run()
        code, out = self.relax(["Al.vasp", "--engine", "emt", "--restart", "--fmax", "0.005", "--smax", "0.01",
                                "--steps", "200"])
        self.assertEqual(code, 0, out)
        log = self.log("Al")
        self.assertEqual(log.count("CONVERGED after"), 2, "--restart appends to the log")
        st = read_json(self.p("Al_status.json"))
        self.assertEqual(st["state"], "converged")
        self.assertEqual(st["fmax"], 0.005)
        self.assertEqual(st["smax"], 0.01)
        self.assertLess(st["max_force_eVA"], 0.005)
        self.assertLess(st["max_stress_GPa"], 0.01)

    def test_restart_refused_on_physics_change(self):
        self.first_run()
        keep = ("Al_CONTCAR", "Al_params.json", "Al_status.json")
        before = snapshot(self.tmp, keep)
        for extra, key in ((["--xc", "lda"], "xc"), (["--relax", "positions"], "relax"),
                           (["--pressure", "1.0"], "pressure"), (["--ecut", "500"], "ecut")):
            code, out = self.relax(["Al.vasp", "--engine", "emt", "--restart", *extra])
            self.assertEqual(code, 2, f"{extra}: {out}")
            self.assertIn(key, out, f"the refusal should list the differing key {key!r}: {out}")
            self.assertEqual(snapshot(self.tmp, keep), before, extra)

    def test_stop_file_with_prefix(self):
        self.touch("Al.STOP")
        code, out = self.relax(["Al.vasp", "--engine", "emt"])
        self.assertEqual(code, 0, out)
        st = read_json(self.p("Al_status.json"))
        self.assertEqual(st["state"], "not_converged")
        self.assertEqual(st["reason"], "stop")
        tail = "\n".join(self.log("Al").splitlines()[-3:])
        self.assertIn("Al.STOP", tail, "the final log lines must name the STOP file")
        self.assertTrue(os.path.isfile(self.p("Al_CONTCAR")))

    def test_plain_stop_file(self):
        self.touch("STOP")
        code, out = self.relax(["Al.vasp", "--engine", "emt", "--prefix", "plain"])
        self.assertEqual(code, 0, out)
        st = read_json(self.p("plain_status.json"))
        self.assertEqual(st["reason"], "stop")
        self.assertEqual(st["state"], "not_converged")
        self.assertIn("STOP", "\n".join(self.log("plain").splitlines()[-3:]))

    def test_status_error_on_exception(self):
        def boom(self_, cell, positions, need_stress):
            raise RuntimeError("boom from the test engine")
        raised = False
        with mock.patch.object(core.EMTEngine, "__call__", boom):
            try:
                code, out = self.relax(["Al.vasp", "--engine", "emt", "--prefix", "err"])
            except RuntimeError:
                raised = True
                code = 1
        self.assertTrue(raised or code == 1, "an unexpected error must not look like success")
        st = read_json(self.p("err_status.json"))
        self.assertEqual(st["state"], "error")
        self.assertIn("boom from the test engine", st["reason"])

    def test_dispatch_through_cli_main(self):
        from ofdft_opt import cli
        code, out = run_main(cli.main, ["relax", "Al.vasp", "--engine", "emt", "--prefix", "viacli"], self.tmp)
        self.assertEqual(code, 0, out)
        self.assertEqual(read_json(self.p("viacli_status.json"))["state"], "converged")

    def test_restart_refused_when_poscar_replaced(self):
        """A new structure under the old POSCAR name must not silently continue the OLD structure."""
        code, out = self.relax(["Al.vasp", "--engine", "emt", "--steps", "2"])
        self.assertEqual(code, 0, out)
        write_fcc_poscar(self.p("Al.vasp"), 3.95, rattle=0.02)
        keep = ("Al.log", "Al_CONTCAR", "Al_params.json", "Al_status.json")
        before = snapshot(self.tmp, keep)
        code, out = self.relax(["Al.vasp", "--engine", "emt", "--restart"])
        self.assertEqual(code, 2, out)
        self.assertIn("OLD structure", out)
        self.assertEqual(snapshot(self.tmp, keep), before, "a refused restart must not touch any file")

    def test_abbreviated_options_rejected(self):
        """`--pref` must not silently act as --prefix (job.sh would watch the wrong files)."""
        code, out = self.relax(["Al.vasp", "--engine", "emt", "--pref", "myp"])
        self.assertEqual(code, 2, out)
        self.assertFalse(os.path.exists(self.p("myp.log")))
        self.assertFalse(os.path.exists(self.p("Al.log")))


@unittest.skipUnless(HAVE_ASE, "ase not installed")
class TestXcPPMismatch(CLICase):
    """Name-based xc/pp consistency check, refused before any DFTpy calculation (--steps 0 bounds the cost
    if the refusal were missing)."""

    def setUp(self):
        super().setUp()
        write_fcc_poscar(self.p("Al.vasp"), 4.0626)
        os.environ["OFDFT_PPPATH"] = self.tmp

    def test_custom_lda_file_with_pbe_refused(self):
        with open(self.p("Al_lda_custom.recpot"), "w") as f:
            f.write("not a real pseudopotential\n")
        code, out = self.relax(["Al.vasp", "--pp", "Al=Al_lda_custom.recpot", "--xc", "pbe", "--steps", "0",
                                "--pppath", self.tmp, "--prefix", "mm"])
        self.assertEqual(code, 2, out)
        self.assertFalse(os.path.exists(self.p("mm_CONTCAR")))

    def test_registry_files_with_wrong_xc_refused(self):
        for name, xc in (("al.lda.recpot", "pbe"), ("al.gga.recpot", "lda")):
            src = registry_copy(name)
            if src is None:
                self.skipTest(f"no local copy of {name}")
            shutil.copyfile(src, self.p(name))
            prefix = f"mm_{xc}"
            code, out = self.relax(["Al.vasp", "--pp", f"Al={name}", "--xc", xc, "--steps", "0",
                                    "--pppath", self.tmp, "--prefix", prefix])
            self.assertEqual(code, 2, f"{name} with --xc {xc}: {out}")
            self.assertFalse(os.path.exists(self.p(prefix + "_CONTCAR")))


@unittest.skipUnless(HAVE_ASE, "ase not installed")
class TestSscurveGuards(CLICase):
    def setUp(self):
        super().setUp()
        write_fcc_poscar(self.p("Al.vasp"), 4.10, rattle=0.02)

    def make_reference(self):
        code, out = self.relax(["Al.vasp", "--engine", "emt"])
        self.assertEqual(code, 0, out)
        self.assertEqual(read_json(self.p("Al_status.json"))["state"], "converged")
        return "Al_CONTCAR"

    def curve(self, ref, prefix, *extra, strains=("--strain-step", "0.005", "--strain-max", "0.005")):
        return self.sscurve([ref, "--engine", "emt", *strains, "--prefix", prefix, *extra])

    def test_relaxed_reference_runs(self):
        ref = self.make_reference()
        code, out = self.curve(ref, "ok")
        self.assertEqual(code, 0, out)
        st = read_json(self.p("ok_status.json"))
        self.assertEqual(st["state"], "complete")
        self.assertEqual(st["points_done"], 2)
        self.assertEqual(len(csv_data_rows(self.p("ok.csv"))), 2)

    def test_reference_with_unconverged_status_refused(self):
        self.make_reference()
        shutil.copyfile(self.p("Al_CONTCAR"), self.p("X_CONTCAR"))
        with open(self.p("X_status.json"), "w") as f:
            json.dump({"state": "not_converged", "reason": "steps"}, f)
        code, out = self.curve("X_CONTCAR", "ssX")
        self.assertEqual(code, 2, out)
        self.assertEqual(csv_data_rows(self.p("ssX.csv")), [])
        code, out = self.curve("X_CONTCAR", "ssX2", "--allow-unrelaxed-reference")
        self.assertEqual(code, 0, out)
        self.assertEqual(read_json(self.p("ssX2_status.json"))["state"], "complete")

    def test_reference_with_running_status_refused(self):
        self.make_reference()
        shutil.copyfile(self.p("Al_CONTCAR"), self.p("Y_CONTCAR"))
        with open(self.p("Y_status.json"), "w") as f:
            json.dump({"state": "running"}, f)
        code, out = self.curve("Y_CONTCAR", "ssY")
        self.assertEqual(code, 2, out)

    def test_unrelaxed_reference_detected_at_first_evaluation(self):
        write_fcc_poscar(self.p("U.vasp"), 4.10)           # zero forces by symmetry, but ~GPa stress
        code, out = self.curve("U.vasp", "ssU", strains=("--strains", "0", "0.005"))
        self.assertIn(code, (0, 2), out)
        st = read_json(self.p("ssU_status.json"))
        self.assertIn("reference_not_relaxed", (st.get("state"), st.get("reason")), st)
        self.assertIn("ofdft relax", out)
        self.assertEqual(csv_data_rows(self.p("ssU.csv")), [])
        self.assertFalse(os.path.exists(self.p("ssU_e+0.0000_CONTCAR")))
        code, out = self.curve("U.vasp", "ssU2", "--allow-unrelaxed-reference", strains=("--strains", "0", "0.005"))
        self.assertEqual(code, 0, out)
        self.assertEqual(read_json(self.p("ssU2_status.json"))["state"], "complete")
        self.assertEqual(len(csv_data_rows(self.p("ssU2.csv"))), 2)

    def test_params_mismatch_refused(self):
        ref = self.make_reference()
        code, out = self.curve(ref, "ssP")
        self.assertEqual(code, 0, out)
        par = read_json(self.p("ssP_params.json"))
        self.assertEqual(par["poscar"], "Al_CONTCAR")
        self.assertEqual(par["version"], ofdft_opt.__version__)
        self.assertIn("poscar_sha256", par)
        self.assertIn("pp_sha256", par)
        keep = ("ssP.csv", "ssP_params.json")
        before = snapshot(self.tmp, keep)
        code, out = self.curve(ref, "ssP", "--fmax", "0.005")
        self.assertEqual(code, 2, out)
        self.assertIn("fmax", out)
        self.assertEqual(snapshot(self.tmp, keep), before)
        # identical options: restart detected from the csv, nothing left to do
        code, out = self.curve(ref, "ssP")
        self.assertEqual(code, 0, out)
        self.assertEqual(len(csv_data_rows(self.p("ssP.csv"))), 2)
        self.assertEqual(read_json(self.p("ssP_status.json"))["state"], "complete")

    def test_changed_reference_content_refused(self):
        ref = self.make_reference()
        code, out = self.curve(ref, "ssR")
        self.assertEqual(code, 0, out)
        s = core.read_poscar(self.p(ref))
        core.write_poscar(self.p(ref), s["symbols"], s["cell"] * 1.001, s["positions"] * 1.001,
                          comment="modified reference")
        code, out = self.curve(ref, "ssR")
        self.assertEqual(code, 2, out)

    # ---- review fixes ----
    def test_compression_strain_list_gets_comp_prefix(self):
        """--strains 0 -0.005 -0.01 is compression: own _comp prefix, never appended to the tension curve."""
        ref = self.make_reference()
        code, out = self.sscurve([ref, "--engine", "emt", "--strain-step", "0.005", "--strain-max", "0.01"])
        self.assertEqual(code, 0, out)
        tension = self.p("Al_CONTCAR_ss_z_tension.csv")
        rows_before = csv_data_rows(tension)
        code, out = self.sscurve([ref, "--engine", "emt", "--strains", "0", "-0.005", "-0.01"])
        self.assertEqual(code, 0, out)
        self.assertEqual(csv_data_rows(tension), rows_before, "compression points appended to the tension csv")
        comp = csv_data_rows(self.p("Al_CONTCAR_ss_z_tension_comp.csv"))
        self.assertEqual(sorted(float(r["strain"]) for r in comp), [-0.01, -0.005, 0.0])
        self.assertEqual(read_json(self.p("Al_CONTCAR_ss_z_tension_comp_params.json"))["branch"], "compression")
        self.assertEqual(read_json(self.p("Al_CONTCAR_ss_z_tension_status.json"))["last_strain"], 0.01,
                         "the tension curve's status file must not be overwritten")

    def test_other_branch_with_same_prefix_refused(self):
        ref = self.make_reference()
        code, out = self.curve(ref, "ssB")
        self.assertEqual(code, 0, out)
        before = snapshot(self.tmp, ("ssB.csv", "ssB_params.json", "ssB_status.json"))
        code, out = self.curve(ref, "ssB", strains=("--strains", "0", "-0.005"))
        self.assertEqual(code, 2, out)
        self.assertIn("branch", out)
        self.assertEqual(snapshot(self.tmp, ("ssB.csv", "ssB_params.json", "ssB_status.json")), before)

    def test_mixed_sign_strain_list_refused(self):
        ref = self.make_reference()
        code, out = self.curve(ref, "ssM", strains=("--strains", "-0.005", "0", "0.005"))
        self.assertEqual(code, 2, out)
        self.assertFalse(os.path.exists(self.p("ssM.log")))

    def test_prefix_of_another_run_refused(self):
        """sscurve --prefix equal to a relax prefix must not replace the relax log/params/status."""
        ref = self.make_reference()
        shutil.copyfile(self.p(ref), self.p("B_CONTCAR"))
        with open(self.p("B_status.json"), "w") as f:
            json.dump({"state": "converged"}, f)
        keep = ("Al.log", "Al_params.json", "Al_status.json", "Al_CONTCAR")
        before = snapshot(self.tmp, keep)
        code, out = self.curve("B_CONTCAR", "Al")
        self.assertEqual(code, 2, out)
        self.assertIn("--overwrite", out)
        self.assertEqual(snapshot(self.tmp, keep), before, "the relax files must stay untouched")
        self.assertFalse(os.path.exists(self.p("Al.csv")))
        # the reference's own status file is never replaced, not even with --overwrite
        code, out = self.curve(ref, "Al", "--overwrite")
        self.assertEqual(code, 2, out)
        self.assertEqual(snapshot(self.tmp, keep), before)
        # --overwrite replaces the leftovers of another run (here: not the reference's)
        code, out = self.curve("B_CONTCAR", "Al", "--overwrite")
        self.assertEqual(code, 0, out)
        self.assertEqual(read_json(self.p("Al_status.json"))["state"], "complete")

    def test_rerun_after_reference_not_relaxed(self):
        """The leftovers of a refused curve (log + status, nothing computed) do not block a rerun."""
        write_fcc_poscar(self.p("U.vasp"), 4.10)
        code, out = self.curve("U.vasp", "ssU", strains=("--strains", "0", "0.005"))
        self.assertEqual(read_json(self.p("ssU_status.json"))["state"], "reference_not_relaxed", out)
        self.assertIn("max|sigma|", out)
        self.assertIn("exceeds the limit", out)
        code, out = self.curve("U.vasp", "ssU", "--allow-unrelaxed-reference", strains=("--strains", "0", "0.005"))
        self.assertEqual(code, 0, out)
        self.assertEqual(read_json(self.p("ssU_status.json"))["state"], "complete")
        self.assertIn("reference_not_relaxed", self.log("ssU"), "the log of the refused attempt is kept")

    def test_reference_message_names_limit_and_looser_relaxation(self):
        """Reference relaxed with the default thresholds but a curve with tighter ones: the message says
        which limit failed and that the reference must be relaxed to at least the curve's thresholds."""
        write_fcc_poscar(self.p("T.vasp"), 4.10)           # zero forces by symmetry, ~GPa stress
        with open(self.p("T_status.json"), "w") as f:
            json.dump({"state": "converged", "fmax": 0.01, "smax": 0.05}, f)
        code, out = self.curve("T.vasp", "ssT", "--fmax", "0.005", "--smax", "0.01",
                               strains=("--strains", "0", "0.005"))
        self.assertEqual(code, 2, out)
        reason = read_json(self.p("ssT_status.json"))["reason"]
        self.assertIn("max|sigma| over all six components", reason)
        self.assertNotIn("max|F| = ", reason.split("(max|F|")[0], "forces are within the limit")
        self.assertIn("--fmax 0.005 --smax 0.01", reason)
        self.assertIn("looser thresholds", reason)

    def test_reference_force_check_not_applied_is_logged(self):
        ref = self.make_reference()
        code, out = self.curve(ref, "ssN", strains=("--strains", "0.005", "0.01"))
        self.assertEqual(code, 0, out)
        self.assertIn("NOT applied", self.log("ssN"))
        self.assertTrue(read_json(self.p("ssN_status.json"))["reference_check"].startswith("NOT applied"))
        code, out = self.curve(ref, "ssN0")
        self.assertEqual(code, 0, out)
        self.assertFalse(read_json(self.p("ssN0_status.json"))["reference_check"].startswith("NOT"))

    def test_start_structure_with_other_atoms_refused(self):
        ref = self.make_reference()
        from ase.build import bulk
        at = bulk("Al", "fcc", a=4.05)                     # 1 atom instead of 4
        core.write_poscar(self.p("other.vasp"), at.get_chemical_symbols(), at.cell.array, at.positions)
        code, out = self.curve(ref, "ssW", "--start", "other.vasp", strains=("--strains", "0.05", "0.1"))
        self.assertEqual(code, 2, out)
        self.assertIn("does not match the reference", out)
        for n in ("ssW.csv", "ssW.log", "ssW_params.json", "ssW_status.json"):
            self.assertFalse(os.path.exists(self.p(n)), n)

    def test_unreadable_point_contcar_refused_cleanly(self):
        ref = self.make_reference()
        code, out = self.curve(ref, "ssE")
        self.assertEqual(code, 0, out)
        open(self.p("ssE_e+0.0050_CONTCAR"), "w").close()  # emptied
        code, out = self.curve(ref, "ssE")
        self.assertEqual(code, 2, out)
        self.assertIn("cannot read", out)
        self.assertIn("ssE_e+0.0050_CONTCAR", out)
        os.remove(self.p("ssE_e+0.0050_CONTCAR"))          # as the message says: the point is recomputed
        code, out = self.curve(ref, "ssE")
        self.assertEqual(code, 0, out)
        self.assertEqual(read_json(self.p("ssE_status.json"))["state"], "complete")
        self.assertTrue(os.path.isfile(self.p("ssE_e+0.0050_CONTCAR")))

    def test_abbreviated_options_rejected(self):
        ref = self.make_reference()
        code, out = self.sscurve([ref, "--engine", "emt", "--strain-max", "0.005", "--pref", "ab"])
        self.assertEqual(code, 2, out)
        self.assertFalse(os.path.exists(self.p("ab.log")))

    def test_stop_file(self):
        ref = self.make_reference()
        self.touch("ssS.STOP")
        code, out = self.curve(ref, "ssS")
        self.assertEqual(code, 0, out)
        st = read_json(self.p("ssS_status.json"))
        self.assertEqual(st["state"], "stopped")
        self.assertEqual(st["points_done"], 0)
        self.assertEqual(csv_data_rows(self.p("ssS.csv")), [])
        os.remove(self.p("ssS.STOP"))
        code, out = self.curve(ref, "ssS")
        self.assertEqual(code, 0, out)
        self.assertEqual(read_json(self.p("ssS_status.json"))["state"], "complete")
        self.assertEqual(len(csv_data_rows(self.p("ssS.csv"))), 2)


LEGACY_PARAMS = {                  # as written by the original ss_curve.py (run_ss.sh options)
    "poscar": "C:\\old\\run\\folder\\R_CONTCAR", "pp": ["Al=Al_test.recpot"], "xc": "pbe", "kedf": "WT",
    "ecut": 600.0, "nr": None, "axis": "z", "mode": "tension", "strain_step": 0.005, "fmax": 0.01, "smax": 0.05,
    "econv": 1e-07, "econv_loose": 0.0, "econv_switch": 5.0}


@unittest.skipUnless(HAVE_ASE, "ase not installed")
class TestSscurveLegacyParams(CLICase):
    """A curve started by the original ss_curve.py continues with `ofdft sscurve` and the same options
    (checks only: _prepare decides before anything is written, so no DFTpy is needed)."""

    def setUp(self):
        super().setUp()
        write_fcc_poscar(self.p("R_CONTCAR"), 4.05)
        with open(self.p("Al_test.recpot"), "w") as f:
            f.write("dummy local pseudopotential (never read: no calculation here)\n")
        prefix = "R_CONTCAR_ss_z_tension"
        with open(self.p(prefix + "_params.json"), "w") as f:
            json.dump(LEGACY_PARAMS, f, indent=1)
        from ofdft_opt import sscurve
        with open(self.p(prefix + ".csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(sscurve.CSV_COLS)
            w.writerow([0.0] + [0.0] * 6 + [-1.0, -0.25, 0.001, 0.0, 1, 2, 3.0, True])
        shutil.copyfile(self.p("R_CONTCAR"), self.p(prefix + "_e+0.0000_CONTCAR"))

    def prepare(self, *extra, pp="Al_test.recpot"):
        """sscurve's restart checks for the options of run_ss.sh -> (exit code, output, setup or None)."""
        from ofdft_opt import sscurve
        argv = ["R_CONTCAR", "--pp", f"Al={pp}", "--pppath", self.tmp, "--precon", "none",
                "--econv", "1e-7", "--econv-loose", "0", "--strain-step", "0.005", "--strain-max", "0.01", *extra]
        args = sscurve.parse_args(argv)
        args.prefix = p = "R_CONTCAR_ss_z_tension"
        holder = {}

        def call(_argv):
            holder["setup"] = sscurve._prepare(args, sscurve.strain_list(args.strain_step, args.strain_max),
                                               "tension", f"{p}.csv", f"{p}_params.json", f"{p}_status.json",
                                               f"{p}.log")
        code, out = run_main(call, None, self.tmp)
        return code, out, holder.get("setup")

    def test_same_options_continue(self):
        code, out, setup = self.prepare()
        self.assertEqual(code, 0, out)
        self.assertTrue(setup["restart"])
        self.assertTrue(setup["rewrite_params"])
        self.assertEqual([d[0] for d in setup["done"]], [0.0])
        self.assertEqual(setup["params"]["legacy_params"], LEGACY_PARAMS)
        self.assertEqual(setup["params"]["version"], ofdft_opt.__version__)
        self.assertTrue(any("old ss_curve.py" in n for n in setup["notes"]))

    def test_different_options_refused_with_legacy_message(self):
        for extra, key in ((["--fmax", "0.005"], "fmax"), (["--econv-loose", "1e-6"], "econv_loose"),
                           (["--engine", "emt"], "engine"), (["--axis", "x"], "axis")):
            code, out, _ = self.prepare(*extra)
            self.assertEqual(code, 2, f"{extra}: {out}")
            self.assertIn("old ss_curve.py", out)
            self.assertIn(f"{key}: was", out)

    def test_other_pp_file_or_reference_refused(self):
        shutil.copyfile(self.p("Al_test.recpot"), self.p("Al_other.recpot"))
        code, out, _ = self.prepare(pp="Al_other.recpot")
        self.assertEqual(code, 2, out)
        self.assertIn("pp: was", out)
        par = read_json(self.p("R_CONTCAR_ss_z_tension_params.json"))
        par["poscar"] = "/home/someone/runs/Other_CONTCAR"
        with open(self.p("R_CONTCAR_ss_z_tension_params.json"), "w") as f:
            json.dump(par, f)
        code, out, _ = self.prepare()
        self.assertEqual(code, 2, out)
        self.assertIn("poscar: was 'Other_CONTCAR'", out)


class TestVersionCommand(unittest.TestCase):
    def test_cli_version(self):
        from ofdft_opt import cli
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            code, out = run_main(cli.main, ["version"], tmp)
        self.assertEqual(code, 0, out)
        line = out.strip().splitlines()[-1]
        self.assertTrue(line.startswith(f"ofdft-opt {ofdft_opt.__version__} ("), line)
        inside = line[line.index("(") + 1:line.rindex(")")]
        self.assertEqual(os.path.normcase(os.path.realpath(inside)),
                         os.path.normcase(os.path.realpath(ofdft_opt.PKG_DIR)))

    def test_unknown_command(self):
        from ofdft_opt import cli
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            code, out = run_main(cli.main, ["no-such-command"], tmp)
        self.assertNotEqual(code, 0, out)

    def test_python_dash_m(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = SRC
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            r = subprocess.run([sys.executable, "-m", "ofdft_opt", "version"], cwd=tmp, env=env,
                               capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.strip().startswith(f"ofdft-opt {ofdft_opt.__version__}"), r.stdout)


if __name__ == "__main__":
    unittest.main()
