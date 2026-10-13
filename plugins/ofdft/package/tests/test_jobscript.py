"""`ofdft jobscript`: Slurm header and auto resources, run.json, bash syntax, and a dry run of the generated
script with stub `python`/`mpirun` executables (checks what would really be launched, independent of how
the script is written)."""
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import ofdft_opt  # noqa: E402
from ofdft_opt import jobscript  # noqa: E402

PY = "/opt/conda/envs/ofdft/bin/python"          # a cluster-style path (need not exist)
LIB = "/home/someone/.ofdft/lib/0.1.0"

STUB_PYTHON = """#!/bin/bash
echo "python $* | PYTHONPATH=${PYTHONPATH-} PYTHONNOUSERSITE=${PYTHONNOUSERSITE-} OMP=${OMP_NUM_THREADS-} \
OPENBLAS=${OPENBLAS_NUM_THREADS-} MKL=${MKL_NUM_THREADS-} PYTHONHOME=${PYTHONHOME-unset}" >> "$OFDFT_STUB_LOG"
exit 0
"""
STUB_MPIRUN = """#!/bin/bash
echo "mpirun $*" >> "$OFDFT_STUB_LOG"
n=1
if [ "$1" = "-np" ] || [ "$1" = "-n" ]; then n="$2"; shift 2; fi
case " $* " in
    *" --mpi "*|*"import "*) echo "$n | stub MPI library"; exit 0 ;;
esac
exec "$@"
"""


def find_bash():
    cands = [os.environ.get("OFDFT_TEST_BASH")]
    if os.name == "nt":                            # Git for Windows; never the WSL launcher in System32
        for base in (os.environ.get("ProgramW6432"), os.environ.get("ProgramFiles"), r"C:\Program Files"):
            if base:
                cands += [os.path.join(base, "Git", "bin", "bash.exe"),
                          os.path.join(base, "Git", "usr", "bin", "bash.exe")]
    w = shutil.which("bash")
    if w and not (os.name == "nt" and "system32" in w.lower()):
        cands.append(w)
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


BASH = find_bash()


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


def write_dummy_poscar(path, n, element="Al"):
    """VASP5 POSCAR with n atoms (positions irrelevant: only the atom count matters here)."""
    L = (16.6 * n) ** (1.0 / 3.0)
    with open(path, "w", newline="\n") as f:
        f.write(f"dummy {n} atoms\n 1.0\n {L:.6f} 0 0\n 0 {L:.6f} 0\n 0 0 {L:.6f}\n {element}\n {n}\nDirect\n")
        for k in range(n):
            f.write("%.6f %.6f %.6f\n" % ((k * 0.618034) % 1, (k * 0.414214) % 1, (k * 0.732051) % 1))
    return path


def sbatch_opts(text):
    opts = {}
    for line in text.splitlines():
        m = re.match(r"^#SBATCH\s+--([A-Za-z-]+)(?:[=\s]\s*(\S+))?", line)
        if m:
            opts[m.group(1)] = m.group(2)
    return opts


def code_lines(text):
    return [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def slash(s):
    return s.replace("\\", "/")


class JobscriptCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def p(self, *names):
        return os.path.join(self.tmp, *names)

    def gen(self, argv, script="job.sh", cwd=None):
        code, out = run_main(jobscript.main, argv, cwd or self.tmp)
        self.assertEqual(code, 0, out)
        path = os.path.join(cwd or self.tmp, script)
        self.assertTrue(os.path.isfile(path), f"{script} not written: {out}")
        with open(path, "rb") as f:
            raw = f.read()
        self.assertNotIn(b"\r", raw, "job script must have LF line endings")
        text = raw.decode()
        self.assertTrue(text.startswith("#!") and "bash" in text.splitlines()[0], text.splitlines()[0])
        return text, out

    def run_json(self, folder=None):
        with open(os.path.join(folder or self.tmp, "run.json")) as f:
            return json.load(f)

    def assert_bash_syntax(self, text):
        if BASH is None:
            return
        r = subprocess.run([BASH, "-n"], input=text.encode(), capture_output=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr.decode(errors="replace"))


class TestAutoResources(JobscriptCase):
    def test_auto_ntasks_by_atom_count(self):
        for n, want in ((4, 8), (499, 8), (500, 16), (1999, 16), (2000, 32), (5999, 32), (6000, 64)):
            write_dummy_poscar(self.p(f"s{n}.vasp"), n)
            text, _ = self.gen(["--task", "relax", "--poscar", f"s{n}.vasp", "--ntasks", "auto", "--mem", "auto",
                                "--python", PY, "--libdir", LIB, "-o", f"job_{n}.sh"], script=f"job_{n}.sh")
            self.assertEqual(sbatch_opts(text).get("ntasks-per-node"), str(want), f"{n} atoms")
            self.assertEqual(int(self.run_json()["ntasks"]), want, f"{n} atoms")

    def test_auto_mem_by_atom_count(self):
        for n, want in ((255, 8), (3000, 20), (7000, 46), (13140, 86)):
            write_dummy_poscar(self.p(f"m{n}.vasp"), n)
            text, _ = self.gen(["--task", "relax", "--poscar", f"m{n}.vasp", "--ntasks", "auto", "--mem", "auto",
                                "--python", PY, "--libdir", LIB, "-o", f"job_{n}.sh"], script=f"job_{n}.sh")
            mem = sbatch_opts(text).get("mem")
            self.assertEqual(mem, f"{want}G", f"{n} atoms: --mem={mem}")
            self.assertEqual(re.sub(r"[^0-9]", "", str(self.run_json()["mem"])), str(want))


class TestScriptContent(JobscriptCase):
    def setUp(self):
        super().setUp()
        write_dummy_poscar(self.p("Al.vasp"), 108)

    def test_header_and_run_json(self):
        passthrough = ["--relax", "full", "--fmax", "0.01", "--steps", "2"]
        text, out = self.gen(["--task", "relax", "--poscar", "Al.vasp", "--ntasks", "4", "--mem", "16G",
                              "--time", "0-02:00:00", "--partition", "long", "--python", PY, "--libdir", LIB,
                              "--", *passthrough])
        o = sbatch_opts(text)
        self.assertEqual(o.get("job-name"), "relax_Al")
        self.assertEqual(o.get("nodes"), "1")
        self.assertEqual(o.get("ntasks-per-node"), "4")
        self.assertEqual(o.get("cpus-per-task"), "1")
        self.assertEqual(o.get("mem"), "16G")
        self.assertEqual(o.get("time"), "0-02:00:00")
        self.assertEqual(o.get("partition"), "long")
        self.assertEqual(o.get("output"), "%x_%j.out")
        t = slash(text)
        self.assertIn(f"ofdft-opt {ofdft_opt.__version__}", t)          # header: generated by ...
        self.assertRegex(t, r'cd "?\$\{?SLURM_SUBMIT_DIR')
        self.assertIn("doctor --brief", t)
        self.assertIn("/opt/conda/envs/ofdft/bin/mpirun", t)
        self.assertFalse(any(ln.strip().startswith("sbatch") for ln in code_lines(t)), "must not submit itself")
        self.assertTrue(out.strip(), "a short summary is printed")
        self.assert_bash_syntax(text)

        rj = self.run_json()
        for k in ("version", "task", "poscar", "poscar_sha256", "ntasks", "mem", "time", "python", "libdir",
                  "created"):
            self.assertIn(k, rj)
        self.assertEqual(rj["version"], ofdft_opt.__version__)
        self.assertEqual(rj["task"], "relax")
        self.assertEqual(rj["poscar"], "Al.vasp")
        with open(self.p("Al.vasp"), "rb") as f:
            self.assertEqual(rj["poscar_sha256"], hashlib.sha256(f.read()).hexdigest())
        self.assertEqual(int(rj["ntasks"]), 4)
        self.assertEqual(rj["time"], "0-02:00:00")
        self.assertEqual(slash(rj["python"]), PY)
        self.assertEqual(slash(rj["libdir"]), LIB)
        self.assertIn(passthrough, [v for v in rj.values() if isinstance(v, list)], "passthrough args in run.json")

    def test_default_times_and_names(self):
        text, _ = self.gen(["--task", "relax", "--poscar", "Al.vasp", "--python", PY, "--libdir", LIB])
        self.assertEqual(sbatch_opts(text).get("time"), "3-00:00:00")
        self.assertNotIn("partition", sbatch_opts(text))
        write_dummy_poscar(self.p("Al_CONTCAR"), 108)
        text, _ = self.gen(["--task", "sscurve", "--poscar", "Al_CONTCAR", "--python", PY, "--libdir", LIB,
                            "-o", "ss.sh", "--", "--strain-step", "0.005", "--strain-max", "0.05"], script="ss.sh")
        o = sbatch_opts(text)
        self.assertEqual(o.get("time"), "10-00:00:00")
        self.assertEqual(o.get("job-name"), "sscurve_Al_CONTCAR")
        self.assertEqual(self.run_json()["task"], "sscurve")
        self.assert_bash_syntax(text)

    def test_name_is_sanitised(self):
        text, _ = self.gen(["--task", "relax", "--poscar", "Al.vasp", "--name", "my run!(1)", "--python", PY,
                            "--libdir", LIB])
        name = sbatch_opts(text).get("job-name")
        self.assertRegex(name, r"^[A-Za-z0-9._-]+$")
        self.assertTrue(name.startswith("my"))
        self.assert_bash_syntax(text)

    def test_output_path_and_run_json_location(self):
        os.makedirs(self.p("sub"))
        shutil.copyfile(self.p("Al.vasp"), self.p("sub", "Al.vasp"))       # the POSCAR lives in the run folder
        self.gen(["--task", "relax", "--poscar", "Al.vasp", "--python", PY, "--libdir", LIB,
                  "-o", os.path.join("sub", "j.sh")], script=os.path.join("sub", "j.sh"))
        self.assertTrue(os.path.isfile(self.p("sub", "run.json")), "run.json goes next to the job script")

    def test_defaults_python_and_libdir(self):
        text, _ = self.gen(["--task", "relax", "--poscar", "Al.vasp", "--ntasks", "2"])
        rj = self.run_json()
        norm = lambda p: os.path.normcase(os.path.abspath(p))  # noqa: E731
        self.assertEqual(norm(rj["python"]), norm(sys.executable))
        self.assertEqual(norm(rj["libdir"]), norm(ofdft_opt.LIB_DIR))
        self.assertIn(slash(ofdft_opt.LIB_DIR), slash(text))
        self.assert_bash_syntax(text)


class TestConda(JobscriptCase):
    def setUp(self):
        super().setUp()
        write_dummy_poscar(self.p("Al.vasp"), 32)

    def fake_conda(self, base, env=None):
        prefix = os.path.join(base, "envs", env) if env else base
        os.makedirs(os.path.join(prefix, "bin"), exist_ok=True)
        os.makedirs(os.path.join(prefix, "conda-meta"), exist_ok=True)       # every conda env has it
        py = os.path.join(prefix, "bin", "python")
        open(py, "w").close()
        os.makedirs(os.path.join(base, "etc", "profile.d"), exist_ok=True)
        open(os.path.join(base, "etc", "profile.d", "conda.sh"), "w").close()
        return py, prefix

    def check_activates(self, py, prefix):
        text, _ = self.gen(["--task", "relax", "--poscar", "Al.vasp", "--python", py, "--libdir", LIB])
        code = "\n".join(code_lines(slash(text)))
        self.assertIn("etc/profile.d/conda.sh", code)
        self.assertIn("conda activate", code)
        self.assertIn(slash(prefix), code)
        self.assert_bash_syntax(text)

    def test_named_env(self):
        py, prefix = self.fake_conda(self.p("miniforge3"), "ofdft")
        self.check_activates(py, prefix)

    def test_base_env(self):
        py, prefix = self.fake_conda(self.p("mf_base"))
        self.check_activates(py, prefix)

    def test_no_conda_uses_absolute_paths(self):
        text, _ = self.gen(["--task", "relax", "--poscar", "Al.vasp", "--python", PY, "--libdir", LIB])
        self.assertFalse(any("conda activate" in ln for ln in code_lines(text)))
        self.assertIn(PY, slash(text))


@unittest.skipIf(BASH is None, "bash not found")
class TestBashSyntax(JobscriptCase):
    def test_variants(self):
        write_dummy_poscar(self.p("Al.vasp"), 255)
        for i, extra in enumerate((["--ntasks", "auto"], ["--ntasks", "1"], ["--ntasks", "64", "--partition", "main"])):
            for task in ("relax", "sscurve"):
                name = f"j{i}_{task}.sh"
                text, _ = self.gen(["--task", task, "--poscar", "Al.vasp", *extra, "-o", name, "--",
                                    "--fmax", "0.01", "--smax", "0.05"], script=name)
                r = subprocess.run([BASH, "-n"], input=text.encode(), capture_output=True, timeout=60)
                self.assertEqual(r.returncode, 0, f"{name}: {r.stderr.decode(errors='replace')}")


@unittest.skipIf(BASH is None, "bash not found")
class TestDryRun(JobscriptCase):
    """Run the generated script with bash, with stub python/mpirun that only log their arguments."""

    def setUp(self):
        super().setUp()
        self.run_dir = self.p("run")
        os.makedirs(self.run_dir)
        write_dummy_poscar(os.path.join(self.run_dir, "Al.vasp"), 108)
        bindir = self.p("env", "bin")
        os.makedirs(bindir)
        for name, body in (("python", STUB_PYTHON), ("mpirun", STUB_MPIRUN)):
            with open(os.path.join(bindir, name), "w", newline="\n") as f:
                f.write(body)
            os.chmod(os.path.join(bindir, name), 0o755)
        self.py = slash(os.path.join(bindir, "python"))
        self.stub_log = self.p("calls.log")

    def make(self, task, poscar, ntasks, passthrough=(), script="job.sh"):
        argv = ["--task", task, "--poscar", poscar, "--ntasks", str(ntasks), "--mem", "8G",
                "--python", self.py, "--libdir", LIB, "-o", script]
        if passthrough:
            argv += ["--", *passthrough]
        self.gen(argv, script=script, cwd=self.run_dir)
        return script

    def execute(self, script, ntasks):
        if os.path.exists(self.stub_log):
            os.remove(self.stub_log)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "OFDFT_"))}
        env.update(SLURM_NTASKS=str(ntasks), SLURM_JOB_ID="4242", OFDFT_STUB_LOG=slash(self.stub_log),
                   PYTHONHOME="/bogus/home", PYTHONPATH="/bogus/path")
        r = subprocess.run([BASH, script], cwd=self.run_dir, env=env, capture_output=True, timeout=120)
        out = r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
        self.assertEqual(r.returncode, 0, out)
        with open(self.stub_log) as f:
            calls = [ln.rstrip("\n") for ln in f]
        return calls, out

    def task_runs(self, calls, task):
        """Python invocations that run the task: [(argv after the interpreter, environment part)]."""
        res = []
        for ln in calls:
            if not ln.startswith("python "):
                continue
            args, _, envpart = ln[len("python "):].partition(" | ")
            toks = args.split()
            if "ofdft_opt" in toks and task in toks:
                res.append((toks, envpart))
        return res

    def check_env(self, envpart):
        self.assertIn(f"PYTHONPATH={LIB} ", envpart + " ")
        self.assertIn("PYTHONNOUSERSITE=1", envpart)
        self.assertIn("OMP=1", envpart)
        self.assertIn("OPENBLAS=1", envpart)
        self.assertIn("MKL=1", envpart)
        self.assertIn("PYTHONHOME=unset", envpart)

    def test_relax_mpi(self):
        pt = ["--relax", "full", "--fmax", "0.01", "--steps", "2"]
        script = self.make("relax", "Al.vasp", 4, pt)
        calls, out = self.execute(script, 4)
        doctor = [ln for ln in calls if ln.startswith("python ") and "doctor --brief" in ln]
        self.assertTrue(doctor, f"doctor --brief not run: {calls}")
        self.check_env(doctor[0].partition(" | ")[2])
        mpi = [ln.split() for ln in calls if ln.startswith("mpirun ") and "relax" in ln.split()]
        self.assertEqual(len(mpi), 1, calls)
        self.assertEqual(mpi[0][1:3], ["-np", "4"])
        self.assertEqual(slash(mpi[0][3]), self.py)
        self.assertEqual(mpi[0][4:], ["-m", "mpi4py", "-m", "ofdft_opt", "relax", "Al.vasp", *pt])
        runs = self.task_runs(calls, "relax")
        self.assertEqual(len(runs), 1, calls)
        self.assertEqual(runs[0][0], ["-m", "mpi4py", "-m", "ofdft_opt", "relax", "Al.vasp", *pt])
        self.check_env(runs[0][1])
        # resubmitting after a stop/time limit continues from the CONTCAR
        write_dummy_poscar(os.path.join(self.run_dir, "Al_CONTCAR"), 108)
        calls, out = self.execute(script, 4)
        runs = self.task_runs(calls, "relax")
        self.assertEqual(runs[0][0], ["-m", "mpi4py", "-m", "ofdft_opt", "relax", "Al.vasp", *pt, "--restart"])

    def test_relax_single_task(self):
        script = self.make("relax", "Al.vasp", 1, ["--relax", "positions"])
        calls, out = self.execute(script, 1)
        self.assertFalse([ln for ln in calls if ln.startswith("mpirun ") and "relax" in ln.split()], calls)
        runs = self.task_runs(calls, "relax")
        self.assertEqual(len(runs), 1, calls)
        self.assertEqual(runs[0][0], ["-m", "ofdft_opt", "relax", "Al.vasp", "--relax", "positions"])
        self.check_env(runs[0][1])

    def test_relax_prefix_from_passthrough(self):
        script = self.make("relax", "Al.vasp", 1, ["--prefix", "myrun"])
        write_dummy_poscar(os.path.join(self.run_dir, "Al_CONTCAR"), 108)   # another prefix's file: ignored
        runs = self.task_runs(self.execute(script, 1)[0], "relax")
        self.assertEqual(runs[0][0], ["-m", "ofdft_opt", "relax", "Al.vasp", "--prefix", "myrun"])
        write_dummy_poscar(os.path.join(self.run_dir, "myrun_CONTCAR"), 108)
        runs = self.task_runs(self.execute(script, 1)[0], "relax")
        self.assertEqual(runs[0][0], ["-m", "ofdft_opt", "relax", "Al.vasp", "--prefix", "myrun", "--restart"])

    def test_sscurve_never_adds_restart(self):
        write_dummy_poscar(os.path.join(self.run_dir, "Al_CONTCAR"), 108)
        pt = ["--strain-step", "0.005", "--strain-max", "0.05"]
        script = self.make("sscurve", "Al_CONTCAR", 4, pt, script="ss.sh")
        calls, out = self.execute(script, 4)
        runs = self.task_runs(calls, "sscurve")
        self.assertEqual(len(runs), 1, calls)
        self.assertEqual(runs[0][0], ["-m", "mpi4py", "-m", "ofdft_opt", "sscurve", "Al_CONTCAR", *pt])
        self.check_env(runs[0][1])


class TestInputsAndRegeneration(JobscriptCase):
    """Review fixes: compression prefix from a --strains list, mixed-sign lists, control characters,
    and the warning when a regenerated job.sh would not continue the run described by run.json."""

    def setUp(self):
        super().setUp()
        write_dummy_poscar(self.p("Al_CONTCAR"), 108)
        self.base = ["--task", "sscurve", "--poscar", "Al_CONTCAR", "--python", PY, "--libdir", LIB]

    def test_compression_strain_list_prefix(self):
        _, out = self.gen(self.base + ["--", "--strains", "0", "-0.005", "-0.01"])
        self.assertEqual(self.run_json()["prefix"], "Al_CONTCAR_ss_z_tension_comp")
        self.assertIn("prefix Al_CONTCAR_ss_z_tension_comp", out)
        self.assertEqual(jobscript.task_prefix("sscurve", "Al_CONTCAR", ["--strains", "0", "0.01"]),
                         "Al_CONTCAR_ss_z_tension")

    def test_mixed_sign_strain_list_refused(self):
        code, out = run_main(jobscript.main, self.base + ["--", "--strains", "-0.005", "0", "0.005"], self.tmp)
        self.assertEqual(code, 2, out)
        self.assertFalse(os.path.exists(self.p("job.sh")))

    def test_control_characters_refused(self):
        write_dummy_poscar(self.p("Al.vasp"), 108)
        for argv in (["--task", "relax", "--poscar", "Al.vasp", "--python", PY, "--libdir", LIB, "--",
                      "--prefix", "p\necho INJECTED_LINE"],
                     ["--task", "relax", "--poscar", "Al.vasp", "--name", "a\rb", "--python", PY, "--libdir", LIB]):
            code, out = run_main(jobscript.main, argv, self.tmp)
            self.assertEqual(code, 2, out)
            self.assertIn("control characters", out)
            self.assertFalse(os.path.exists(self.p("job.sh")))

    def test_regenerate_with_partial_options_warns(self):
        """Regenerating a compression curve's job.sh with only `-- --steps 2000` would start a NEW tension
        curve on resubmit: say so, and list the original task options."""
        orig = ["--axis", "z", "--strain-step", "-0.005", "--strain-max", "-0.05"]
        self.gen(self.base + ["--ntasks", "8", "--mem", "20G", "--time", "1-00:00:00", "--", *orig])
        self.assertEqual(self.run_json()["prefix"], "Al_CONTCAR_ss_z_tension_comp")
        _, out = self.gen(self.base + ["--mem", "30G", "--", "--steps", "2000"])
        self.assertIn("WARNING", out)
        self.assertIn("NEW run", out)
        self.assertIn(" ".join(orig), out)
        self.assertIn("mem 20G -> 30G", out)
        self.assertIn("time 1-00:00:00 -> 10-00:00:00", out)

    def test_regenerate_with_all_options_only_notes_changes(self):
        orig = ["--axis", "z", "--strain-step", "0.005", "--strain-max", "0.05"]
        self.gen(self.base + ["--ntasks", "8", "--mem", "20G", "--", *orig])
        _, out = self.gen(self.base + ["--ntasks", "8", "--mem", "30G", "--", *orig, "--steps", "2000"])
        self.assertNotIn("NEW run", out)
        self.assertIn("mem 20G -> 30G", out)
        self.assertIn("--steps 2000", out)


if __name__ == "__main__":
    unittest.main()
