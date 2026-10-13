"""Repository-level checks (stdlib unittest only; run from the repo root):

    python -m unittest discover -s tests -v

- version sync: plugin.json == pyproject.toml == ofdft_opt.__version__ == newest CHANGELOG entry
- marketplace.json / plugin.json fields
- no CR bytes in any text file (everything is LF)
- SKILL.md frontmatter has only `name` and `description`; plugin paths named in skills exist
- no personal or account-specific strings anywhere in the repository
"""
import ast
import json
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN = os.path.join(ROOT, "plugins", "ofdft")
PACKAGE = os.path.join(PLUGIN, "package")
SKIP_DIRS = {".git", "__pycache__", ".venv", "build", "dist", ".pytest_cache"}
BINARY_EXT = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ico", ".gz", ".tgz", ".zip", ".pyc", ".whl"}
# Built from pieces so that this file does not contain the strings it forbids.
FORBIDDEN = ["ye" + "min", "lab" + "server", "@m." + "titech", "@" + "isct"]
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def load_json(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return json.load(f)


def repo_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info"))
        for name in sorted(filenames):
            yield os.path.join(dirpath, name)


def text_files():
    for path in repo_files():
        if os.path.splitext(path)[1].lower() in BINARY_EXT:
            continue
        with open(path, "rb") as f:
            data = f.read()
        if b"\0" in data[:8192]:
            continue
        yield path, data


def rel(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def plugin_version():
    return load_json("plugins", "ofdft", ".claude-plugin", "plugin.json")["version"]


def pyproject_version():
    path = os.path.join(PACKAGE, "pyproject.toml")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        import tomllib
        return tomllib.loads(text)["project"]["version"]
    except ImportError:                            # Python 3.10: minimal parse of the [project] table
        m = re.search(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
        v = re.search(r'^version\s*=\s*"([^"]+)"', m.group(1) if m else "", re.M)
        return v.group(1) if v else None


def package_version():
    """__version__ from ofdft_opt/__init__.py without importing it (no numpy/scipy needed)."""
    path = os.path.join(PACKAGE, "src", "ofdft_opt", "__init__.py")
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__version__" for t in node.targets):
            return ast.literal_eval(node.value)
    return None


def frontmatter(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        return None, text
    fields = {}
    for line in m.group(1).splitlines():
        km = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if km:
            fields[km.group(1)] = km.group(2).strip()
    return fields, m.group(2)


def skill_files():
    base = os.path.join(PLUGIN, "skills")
    return sorted(os.path.join(base, d, "SKILL.md") for d in os.listdir(base)
                  if os.path.isfile(os.path.join(base, d, "SKILL.md")))


class TestVersions(unittest.TestCase):
    def test_plugin_version_is_semver(self):
        self.assertRegex(plugin_version(), SEMVER)

    def test_pyproject_matches_plugin(self):
        self.assertEqual(pyproject_version(), plugin_version())

    def test_package_matches_plugin(self):
        self.assertEqual(package_version(), plugin_version())

    def test_changelog_top_entry_matches_plugin(self):
        with open(os.path.join(ROOT, "CHANGELOG.md"), encoding="utf-8") as f:
            m = re.search(r"^## \[?(\d+\.\d+\.\d+)", f.read(), re.M)
        self.assertIsNotNone(m, "CHANGELOG.md has no '## X.Y.Z' entry")
        self.assertEqual(m.group(1), plugin_version())


class TestManifests(unittest.TestCase):
    def test_marketplace(self):
        mk = load_json(".claude-plugin", "marketplace.json")
        self.assertEqual(mk["name"], "hpc-skills")
        self.assertEqual(mk["owner"]["name"], "ye-min-thant")
        entries = {p["name"]: p for p in mk["plugins"]}
        self.assertIn("ofdft", entries)
        entry = entries["ofdft"]
        self.assertEqual(entry["source"], "./plugins/ofdft")
        self.assertNotIn("version", entry, "set the version in plugin.json only")
        self.assertTrue(entry.get("description"))
        for p in mk["plugins"]:
            if isinstance(p["source"], str):
                src = os.path.normpath(os.path.join(ROOT, p["source"]))
                self.assertTrue(os.path.isfile(os.path.join(src, ".claude-plugin", "plugin.json")), p["source"])
                self.assertEqual(load_json(os.path.relpath(src, ROOT), ".claude-plugin", "plugin.json")["name"],
                                 p["name"])

    def test_plugin(self):
        pj = load_json("plugins", "ofdft", ".claude-plugin", "plugin.json")
        self.assertEqual(pj["name"], "ofdft")
        self.assertRegex(pj["version"], SEMVER)
        self.assertTrue(pj.get("description"))
        self.assertEqual(pj["author"]["name"], "ye-min-thant")
        self.assertEqual(pj["license"], "MIT")
        self.assertEqual(pj["repository"], "https://github.com/ye-min-thant/hpc-skills")
        self.assertIsInstance(pj.get("keywords"), list)
        self.assertNotIn("email", pj["author"])

    def test_all_json_parses(self):
        for path, data in text_files():
            if path.endswith(".json"):
                with self.subTest(file=rel(path)):
                    json.loads(data.decode("utf-8"))


class TestTextFiles(unittest.TestCase):
    def test_no_cr_bytes(self):
        bad = [rel(p) for p, data in text_files() if b"\r" in data]
        self.assertEqual(bad, [], "files with CR bytes (must be LF only)")

    def test_gitattributes_forces_lf(self):
        with open(os.path.join(ROOT, ".gitattributes"), encoding="utf-8") as f:
            self.assertIn("eol=lf", f.read())

    def test_no_forbidden_strings(self):
        hits = []
        for path, data in text_files():
            low = data.decode("utf-8", errors="replace").lower()
            hits += [f"{rel(path)}: {s}" for s in FORBIDDEN if s in low]
        self.assertEqual(hits, [], "personal or account-specific strings found")

    def test_no_email_addresses(self):
        email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
        hits = []
        for path, data in text_files():
            hits += [f"{rel(path)}: {m.group(0)}" for m in email.finditer(data.decode("utf-8", errors="replace"))]
        self.assertEqual(hits, [], "e-mail addresses found (this is a public repository)")


class TestSkills(unittest.TestCase):
    def test_skills_exist(self):
        names = {os.path.basename(os.path.dirname(p)) for p in skill_files()}
        self.assertEqual(names, {"setup", "optimize"})

    def test_frontmatter_only_name_and_description(self):
        for path in skill_files():
            with self.subTest(skill=rel(path)):
                fields, body = frontmatter(path)
                self.assertIsNotNone(fields, "missing '---' frontmatter")
                self.assertEqual(set(fields), {"name", "description"})
                self.assertEqual(fields["name"], os.path.basename(os.path.dirname(path)))
                self.assertTrue(fields["description"])
                self.assertLessEqual(len(fields["description"]), 1024)
                self.assertLess(len(body.splitlines()), 300, "keep SKILL.md bodies short")

    def test_plugin_paths_in_skills_exist(self):
        for path in skill_files():
            with open(path, encoding="utf-8") as f:
                text = f.read()
            for m in re.finditer(r"\$\{CLAUDE_PLUGIN_ROOT\}/([A-Za-z0-9_./-]+)", text):
                target = m.group(1).rstrip(".")
                with self.subTest(skill=rel(path), path=target):
                    self.assertTrue(os.path.exists(os.path.join(PLUGIN, target)), target)


def read_text(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


class TestPackaging(unittest.TestCase):
    def test_package_has_license_copy(self):
        """setuptools only packs licence files from the project directory: keep a copy there."""
        self.assertEqual(read_text("plugins", "ofdft", "package", "LICENSE"), read_text("LICENSE"))


class TestWindowsSshDocs(unittest.TestCase):
    """Passphrase-protected keys on Windows: the Windows agent service works only with the Windows ssh
    client, which Claude's Bash tool reaches through ~/bin wrappers that also stop MSYS path rewriting."""

    def test_ssh_reference_has_wrapper_recipe(self):
        text = read_text("plugins", "ofdft", "reference", "ssh.md")
        self.assertIn('exec /c/Windows/System32/OpenSSH/ssh.exe "$@"', text)
        self.assertIn('exec /c/Windows/System32/OpenSSH/scp.exe "$@"', text)
        self.assertIn("MSYS_NO_PATHCONV=1", text)
        self.assertIn("OpenSSH_for_Windows", text)
        self.assertNotIn("Both work with BatchMode, but each has its own ssh-agent.", text)

    def test_setup_skill_and_readme_point_to_it(self):
        for parts in (("plugins", "ofdft", "skills", "setup", "SKILL.md"), ("README.md",)):
            with self.subTest(file="/".join(parts)):
                text = read_text(*parts)
                self.assertIn("~/bin/ssh", text)
                self.assertIn("Windows", text)

    def test_no_remote_command_starts_with_absolute_path(self):
        """`ssh ... '/home/...'` is rewritten by MSYS when ssh is the Windows client: use `cd ~ && ...`."""
        bad = []
        for path, data in text_files():
            if path.endswith(".md"):
                for m in re.finditer(r"ssh(?: -o \S+)* <alias> '(/|<[a-z ]*python>|<conda base>)",
                                     data.decode("utf-8")):
                    bad.append(f"{rel(path)}: {m.group(0)}")
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
