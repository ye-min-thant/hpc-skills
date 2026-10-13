"""Pseudopotential registry, search order, CRLF-normalised sha verification, resolve() and fetch().

No network: fetch() is only exercised with a mocked urlopen."""
import contextlib
import hashlib
import io
import os
import shutil
import socket
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import ofdft_opt  # noqa: E402,F401
from ofdft_opt import pp, runinfo  # noqa: E402

COMMIT = "7a0dc7a1236a1d0a86f8a71c89999079bd14edec"
SHA = {"al.gga.recpot": "76034a6ea04586f32c2c13d9b4a828a83799638ab2d3b87c330afc653be09806",
       "al.lda.recpot": "c2fd4300aa39aa2466149a315e1c7cd87aac6615032db5ded8f237e9c40df803"}
SOURCE_DIR = os.environ.get("OFDFT_TEST_SRC_DIR") or os.path.expanduser(os.path.join("~", "DFTpy-al"))


def norm(p):
    return os.path.normcase(os.path.realpath(p))


def _locate_registry_copy(name):
    """A local copy of a registry PP file to test with (read only): $OFDFT_TEST_PP_DIR, the original
    DFTpy-al folder, or wherever ofdft finds it. Returns its path only if its (normalised) sha is right."""
    cands = [os.environ.get("OFDFT_TEST_PP_DIR"), SOURCE_DIR]
    for d in cands:
        if d and os.path.isfile(os.path.join(d, name)):
            p = os.path.join(d, name)
            if runinfo.sha256_text(p) == SHA[name]:
                return os.path.abspath(p)
    try:
        p = pp.find(name)
    except BaseException:
        p = None
    if p and runinfo.sha256_text(p) == SHA[name]:
        return os.path.abspath(p)
    return None


_REGISTRY_COPIES = {name: _locate_registry_copy(name) for name in SHA}   # decided once, before any test


def find_registry_copy(name):
    return _REGISTRY_COPIES.get(name)


def read_lf(path):
    with open(path, "rb") as f:
        return f.read().replace(b"\r\n", b"\n")


def call(fn, *args, **kwargs):
    """Run fn, capturing stdout/stderr. Returns (result, exit_code or None, combined output)."""
    out, err = io.StringIO(), io.StringIO()
    code, res = None, None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            res = fn(*args, **kwargs)
        except SystemExit as e:
            if e.code is None:
                code = 0
            elif isinstance(e.code, int):
                code = e.code
            else:
                code = 1
                err.write(str(e.code))
    return res, code, out.getvalue() + err.getvalue()


class PPCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp = self._tmp.name
        self.old_cwd = os.getcwd()
        self.cwd = self.mkdir("cwd")
        os.chdir(self.cwd)
        self.env = mock.patch.dict(os.environ)
        self.env.start()
        os.environ.pop("OFDFT_PPPATH", None)

    def tearDown(self):
        self.env.stop()
        os.chdir(self.old_cwd)
        self._tmp.cleanup()

    def mkdir(self, name):
        p = os.path.join(self.tmp, name)
        os.makedirs(p, exist_ok=True)
        return p

    def put(self, d, name, data=b"dummy\n"):
        p = os.path.join(d, name)
        with open(p, "wb") as f:
            f.write(data)
        return p

    def copy_registry(self, name, d, crlf=None):
        src = find_registry_copy(name)
        if src is None:
            self.skipTest(f"no local copy of {name} (set OFDFT_TEST_PP_DIR)")
        dst = os.path.join(d, name)
        if crlf is None:
            shutil.copyfile(src, dst)                    # as found (the original folder has CRLF files)
        else:
            data = read_lf(src)
            with open(dst, "wb") as f:
                f.write(data.replace(b"\n", b"\r\n") if crlf else data)
        return dst


class TestRegistry(unittest.TestCase):
    def test_registry_entries(self):
        self.assertEqual(set(pp.REGISTRY), {"al.gga.recpot", "al.lda.recpot"})
        for name, xc, sub in (("al.gga.recpot", "pbe", "GGA"), ("al.lda.recpot", "lda", "LDA")):
            e = pp.REGISTRY[name]
            self.assertEqual(e["element"], "Al")
            self.assertEqual(e["xc"], xc)
            self.assertEqual(e["sha256"], SHA[name])
            self.assertEqual(e["url"], "https://raw.githubusercontent.com/EACcodes/local-pseudopotentials/"
                                       f"{COMMIT}/BLPS/{sub}/reci/{name}")

    def test_default_pp(self):
        self.assertEqual(pp.DEFAULT_PP, {("Al", "pbe"): "al.gga.recpot", ("Al", "lda"): "al.lda.recpot"})


class TestShaVerification(PPCase):
    def test_verify_crlf_and_lf_copies(self):
        for crlf in (True, False):
            d = self.mkdir("crlf" if crlf else "lf")
            p = self.copy_registry("al.gga.recpot", d, crlf=crlf)
            with open(p, "rb") as f:
                self.assertEqual(b"\r\n" in f.read(), crlf)
            self.assertEqual(runinfo.sha256_text(p), SHA["al.gga.recpot"])
            self.assertEqual(pp.verify(p), ("al.gga.recpot", True))

    def test_verify_as_copied_from_original_folder(self):
        p = self.copy_registry("al.gga.recpot", self.mkdir("orig"))
        self.assertEqual(pp.verify(p), ("al.gga.recpot", True))

    def test_verify_lda(self):
        p = self.copy_registry("al.lda.recpot", self.mkdir("lda"))
        self.assertEqual(pp.verify(p), ("al.lda.recpot", True))

    def test_verify_corrupted_registry_file(self):
        d = self.mkdir("bad")
        p = self.put(d, "al.gga.recpot", b"not the BLPS file\n")
        self.assertEqual(pp.verify(p), ("al.gga.recpot", False))

    def test_verify_non_registry_file(self):
        p = self.put(self.mkdir("other"), "cu.custom.recpot", b"whatever\n")
        name, ok = pp.verify(p)
        self.assertEqual(name, "cu.custom.recpot")
        self.assertIsNone(ok)


class TestSearchPath(PPCase):
    def test_order(self):
        d0, d1, d2 = self.mkdir("d0"), self.mkdir("d1"), self.mkdir("d2")
        os.environ["OFDFT_PPPATH"] = os.pathsep.join([d1, d2])
        sp = [norm(p) for p in pp.search_path(d0)]
        self.assertEqual(sp[:3], [norm(d0), norm(d1), norm(d2)])
        self.assertEqual(sp[-1], norm(self.cwd), "the current directory is searched last")
        home = norm(os.path.expanduser(os.path.join("~", ".ofdft", "pp")))
        if home in sp:
            self.assertEqual(sp.index(home), len(sp) - 2, "~/.ofdft/pp comes after $OFDFT_PPPATH, before cwd")
        # without --pppath the environment comes first
        sp = [norm(p) for p in pp.search_path(None)]
        self.assertEqual(sp[:2], [norm(d1), norm(d2)])
        self.assertNotIn(norm(d0), sp)

    def test_find_first_match_wins(self):
        d0, d1, d2 = self.mkdir("d0"), self.mkdir("d1"), self.mkdir("d2")
        name = "zz_ofdft_test.recpot"
        f1, f2 = self.put(d1, name), self.put(d2, name)
        os.environ["OFDFT_PPPATH"] = os.pathsep.join([d1, d2])
        self.assertEqual(norm(pp.find(name)), norm(f1))
        f0 = self.put(d0, name)
        self.assertEqual(norm(pp.find(name, d0)), norm(f0))
        os.remove(f1)
        self.assertEqual(norm(pp.find(name)), norm(f2))
        self.assertIsNone(pp.find("zz_does_not_exist.recpot", d0))

    def test_find_in_cwd(self):
        name = "zz_cwd_only.recpot"
        f = self.put(self.cwd, name)
        self.assertEqual(norm(pp.find(name)), norm(f))
        self.assertTrue(os.path.isabs(pp.find(name)))


class TestResolve(PPCase):
    def test_default_pbe_and_lda_from_pppath(self):
        d = self.mkdir("pp")
        g = self.copy_registry("al.gga.recpot", d)
        res = pp.resolve([], ["Al"] * 4, "pbe", d)
        self.assertEqual(set(res), {"Al"})
        self.assertEqual(norm(res["Al"]), norm(g))
        self.assertTrue(os.path.isabs(res["Al"]))
        l_ = self.copy_registry("al.lda.recpot", d)
        res = pp.resolve([], ["Al"], "lda", d)
        self.assertEqual(norm(res["Al"]), norm(l_))

    def test_default_found_via_environment(self):
        d = self.mkdir("envpp")
        g = self.copy_registry("al.gga.recpot", d)
        os.environ["OFDFT_PPPATH"] = d
        res = pp.resolve([], ["Al", "Al"], "pbe")
        self.assertEqual(norm(res["Al"]), norm(g))

    def test_pppath_beats_environment(self):
        d_env, d_arg = self.mkdir("envpp"), self.mkdir("argpp")
        self.copy_registry("al.gga.recpot", d_env)
        g = self.copy_registry("al.gga.recpot", d_arg, crlf=False)
        os.environ["OFDFT_PPPATH"] = d_env
        self.assertEqual(norm(pp.resolve([], ["Al"], "pbe", d_arg)["Al"]), norm(g))

    def test_explicit_bare_name_is_searched(self):
        d = self.mkdir("pp")
        f = self.put(d, "al_custom.recpot", b"custom local pseudopotential\n")
        res, code, out = call(pp.resolve, ["Al=al_custom.recpot"], ["Al"], "pbe", d)
        self.assertIsNone(code, out)
        self.assertEqual(norm(res["Al"]), norm(f))

    def test_explicit_path_used_as_is(self):
        d = self.mkdir("somewhere")
        f = self.put(d, "al_custom.recpot", b"custom\n")
        res, code, out = call(pp.resolve, [f"Al={f}"], ["Al"], "pbe", None)
        self.assertIsNone(code, out)
        self.assertEqual(norm(res["Al"]), norm(f))

    def test_explicit_path_must_exist(self):
        missing = os.path.join(self.tmp, "nowhere", "al.gga.recpot")
        _, code, out = call(pp.resolve, [f"Al={missing}"], ["Al"], "pbe", None)
        self.assertEqual(code, 2, out)

    def test_bare_name_not_found(self):
        _, code, out = call(pp.resolve, ["Al=zz_not_there.recpot"], ["Al"], "pbe", self.mkdir("empty"))
        self.assertEqual(code, 2, out)

    def test_no_default_for_other_elements(self):
        d = self.mkdir("pp")
        self.copy_registry("al.gga.recpot", d)
        _, code, out = call(pp.resolve, [], ["Al", "Cu"], "pbe", d)
        self.assertEqual(code, 2, out)
        self.assertIn("no default pseudopotential for Cu", out)
        # an explicit file for Cu is accepted (non-registry: allowed, sha recorded, warned)
        f = self.put(d, "cu_custom.recpot", b"cu\n")
        res, code, out = call(pp.resolve, ["Cu=cu_custom.recpot"], ["Al", "Cu"], "pbe", d)
        self.assertIsNone(code, out)
        self.assertEqual(norm(res["Cu"]), norm(f))
        self.assertIn("Al", res)

    def test_corrupted_registry_file_refused(self):
        d = self.mkdir("pp")
        self.put(d, "al.gga.recpot", b"truncated download\n")
        _, code, out = call(pp.resolve, [], ["Al"], "pbe", d)
        self.assertEqual(code, 2, out)
        _, code, out = call(pp.resolve, ["Al=al.gga.recpot"], ["Al"], "pbe", d)
        self.assertEqual(code, 2, out)

    def test_crlf_registry_copy_accepted(self):
        d = self.mkdir("pp")
        g = self.copy_registry("al.gga.recpot", d, crlf=True)
        res, code, out = call(pp.resolve, ["Al=al.gga.recpot"], ["Al"], "pbe", d)
        self.assertIsNone(code, out)
        self.assertEqual(norm(res["Al"]), norm(g))

    def test_xc_mismatch(self):
        d = self.mkdir("pp")
        self.put(d, "Al_lda_custom.recpot", b"x\n")
        self.put(d, "al_gga_custom.recpot", b"x\n")
        for spec, xc in (("Al=Al_lda_custom.recpot", "pbe"), ("Al=al_gga_custom.recpot", "lda")):
            _, code, out = call(pp.resolve, [spec], ["Al"], xc, d)
            self.assertEqual(code, 2, f"{spec} with xc={xc} must be refused: {out}")
            res, code, out = call(pp.resolve, [spec], ["Al"], xc, d, allow_xc_mismatch=True)
            self.assertIsNone(code, out)
            self.assertIn("Al", res)
        # matching names are fine
        res, code, out = call(pp.resolve, ["Al=Al_lda_custom.recpot"], ["Al"], "lda", d)
        self.assertIsNone(code, out)

    def test_xc_mismatch_registry_files(self):
        d = self.mkdir("pp")
        self.copy_registry("al.lda.recpot", d)
        _, code, out = call(pp.resolve, ["Al=al.lda.recpot"], ["Al"], "pbe", d)
        self.assertEqual(code, 2, out)
        res, code, out = call(pp.resolve, ["Al=al.lda.recpot"], ["Al"], "pbe", d, allow_xc_mismatch=True)
        self.assertIsNone(code, out)


class FakeResponse(io.BytesIO):
    status = 200

    def getcode(self):
        return 200

    @property
    def headers(self):
        return {"Content-Length": str(len(self.getvalue()))}

    def info(self):
        return self.headers


class TestFetch(PPCase):
    def patched(self, fake):
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(urllib.request, "urlopen", fake))

        def no_network(*a, **k):
            raise OSError("network access is disabled in the tests (urlopen is mocked)")
        stack.enter_context(mock.patch.object(urllib.request, "urlretrieve", no_network))
        stack.enter_context(mock.patch.object(socket, "getaddrinfo", no_network))
        stack.enter_context(mock.patch.object(socket, "create_connection", no_network))
        if hasattr(pp, "urlopen"):
            stack.enter_context(mock.patch.object(pp, "urlopen", fake))
        return stack

    def serving(self, payload):
        calls = []

        def fake(url, *args, **kwargs):
            u = url if isinstance(url, str) else getattr(url, "full_url", str(url))
            calls.append((u, args, kwargs))
            name = u.rsplit("/", 1)[-1]
            if name not in payload:
                raise urllib.error.HTTPError(u, 404, "not found", {}, None)
            return FakeResponse(payload[name])
        return fake, calls

    def test_fetch_downloads_verifies_writes_lf(self):
        src = find_registry_copy("al.gga.recpot")
        if src is None:
            self.skipTest("no local copy of al.gga.recpot")
        lf = read_lf(src)
        dest = os.path.join(self.tmp, "dest")
        fake, calls = self.serving({"al.gga.recpot": lf})
        with self.patched(fake):
            res, code, out = call(pp.fetch, dest=dest, names=["al.gga.recpot"])
        self.assertIn(code, (None, 0), out)
        self.assertEqual(len(calls), 1)
        url, args, kwargs = calls[0]
        self.assertEqual(url, pp.REGISTRY["al.gga.recpot"]["url"])
        self.assertTrue("timeout" in kwargs or len(args) >= 2, "urlopen must be called with a timeout")
        p = os.path.join(dest, "al.gga.recpot")
        self.assertTrue(os.path.isfile(p))
        with open(p, "rb") as f:
            data = f.read()
        self.assertNotIn(b"\r", data)
        self.assertEqual(hashlib.sha256(data).hexdigest(), SHA["al.gga.recpot"])
        self.assertEqual(pp.verify(p), ("al.gga.recpot", True))
        self.assertIsInstance(res, list)
        self.assertIn(norm(p), [norm(x) for x in res])
        leftovers = [n for n in os.listdir(dest) if "tmp" in n.lower() or "part" in n.lower()]
        self.assertEqual(leftovers, [], "no temporary files left behind")

        # already present and valid -> no second download
        def must_not_download(*a, **k):
            raise AssertionError("valid file must not be downloaded again")
        with self.patched(must_not_download):
            res, code, out = call(pp.fetch, dest=dest, names=["al.gga.recpot"])
        self.assertIn(code, (None, 0), out)
        self.assertEqual(pp.verify(p), ("al.gga.recpot", True))

    def test_fetch_normalises_crlf_download(self):
        src = find_registry_copy("al.gga.recpot")
        if src is None:
            self.skipTest("no local copy of al.gga.recpot")
        crlf = read_lf(src).replace(b"\n", b"\r\n")
        dest = os.path.join(self.tmp, "dest")
        fake, _ = self.serving({"al.gga.recpot": crlf})
        with self.patched(fake):
            _, code, out = call(pp.fetch, dest=dest, names=["al.gga.recpot"])
        self.assertIn(code, (None, 0), out)
        with open(os.path.join(dest, "al.gga.recpot"), "rb") as f:
            data = f.read()
        self.assertNotIn(b"\r", data)
        self.assertEqual(hashlib.sha256(data).hexdigest(), SHA["al.gga.recpot"])

    def test_fetch_rejects_wrong_content(self):
        dest = os.path.join(self.tmp, "dest")
        fake, _ = self.serving({"al.gga.recpot": b"<html>rate limited</html>\n"})
        raised = None
        with self.patched(fake):
            try:
                _, code, out = call(pp.fetch, dest=dest, names=["al.gga.recpot"])
            except Exception as exc:              # an exception is an acceptable way to fail
                raised, code, out = exc, None, str(exc)
        self.assertTrue(raised is not None or code not in (None, 0) or "sha" in out.lower(),
                        f"a wrong download must be reported: {out}")
        self.assertFalse(os.path.exists(os.path.join(dest, "al.gga.recpot")),
                         "a file with the wrong sha must not be installed")

    def test_fetch_offline_message(self):
        dest = os.path.join(self.tmp, "dest")

        def offline(*a, **k):
            raise urllib.error.URLError("[Errno 11001] getaddrinfo failed")
        with self.patched(offline):
            try:
                _, code, out = call(pp.fetch, dest=dest, names=["al.gga.recpot"])
            except Exception as exc:
                out = str(exc)
        self.assertIn("internet", out.lower(), f"offline error should explain how to copy the file: {out}")
        self.assertFalse(os.path.exists(os.path.join(dest, "al.gga.recpot")))


class TestPPCommand(PPCase):
    def test_list(self):
        d = self.mkdir("pp")
        self.copy_registry("al.gga.recpot", d)
        os.environ["OFDFT_PPPATH"] = d
        res, code, out = call(pp.main, ["list"])
        self.assertIn(code, (None, 0), out)
        self.assertIn(res, (None, 0), out)
        self.assertIn("al.gga.recpot", out)
        self.assertIn("al.lda.recpot", out)


if __name__ == "__main__":
    unittest.main()
