"""Local pseudopotentials: registry of validated files, search path, resolution, download.

Registry: bulk-derived local pseudopotentials (BLPS) of Huang & Carter, Phys. Chem. Chem. Phys. 10,
7109 (2008), from https://github.com/EACcodes/local-pseudopotentials (CC BY-ND 4.0), pinned commit.
They are NOT shipped with ofdft-opt: `ofdft pp fetch` downloads and verifies them.

Search path (first match wins): --pppath DIR, $OFDFT_PPPATH (os.pathsep-separated), ~/.ofdft/pp,
the current directory. Checksums are taken with CRLF -> LF normalisation.

  ofdft pp list                 registry files, where they are found, checksum status
  ofdft pp fetch [--dest DIR]   download the registry files [~/.ofdft/pp]
"""
import argparse
import os
import sys
import urllib.error
import urllib.request

from .runinfo import die, sha256_bytes, sha256_text, warn

_COMMIT = "7a0dc7a1236a1d0a86f8a71c89999079bd14edec"
_BASE = f"https://raw.githubusercontent.com/EACcodes/local-pseudopotentials/{_COMMIT}/BLPS"

REGISTRY = {
    "al.gga.recpot": {
        "element": "Al", "xc": "pbe",
        "url": f"{_BASE}/GGA/reci/al.gga.recpot",
        "sha256": "76034a6ea04586f32c2c13d9b4a828a83799638ab2d3b87c330afc653be09806",
    },
    "al.lda.recpot": {
        "element": "Al", "xc": "lda",
        "url": f"{_BASE}/LDA/reci/al.lda.recpot",
        "sha256": "c2fd4300aa39aa2466149a315e1c7cd87aac6615032db5ded8f237e9c40df803",
    },
}
DEFAULT_PP = {("Al", "pbe"): "al.gga.recpot", ("Al", "lda"): "al.lda.recpot"}
SOURCE = ("BLPS, Huang & Carter, PCCP 10, 7109 (2008); "
          "github.com/EACcodes/local-pseudopotentials @ " + _COMMIT[:12] + " (CC BY-ND 4.0)")


def default_dest():
    return os.path.join(os.path.expanduser("~"), ".ofdft", "pp")


def search_path(pppath=None):
    """Directories searched for bare PP file names, in order (absolute, without duplicates)."""
    dirs = []
    if pppath:
        dirs += [pppath] if isinstance(pppath, str) else list(pppath)
    dirs += [d for d in os.environ.get("OFDFT_PPPATH", "").split(os.pathsep) if d.strip()]
    dirs += [default_dest(), os.getcwd()]
    out, seen = [], set()
    for d in dirs:
        a = os.path.abspath(os.path.expanduser(d.strip()))
        key = os.path.normcase(a)
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out


def find(name, pppath=None):
    """Absolute path of the first file `name` on the search path, or None."""
    for d in search_path(pppath):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


def verify(path):
    """(registry name, ok) for a PP file: ok True/False = checksum matches/does not match the registry
    entry of that file name; a renamed registry file is recognised by its checksum.
    (basename, None) = not a registry file."""
    name = os.path.basename(path)
    sha = sha256_text(path)
    if name in REGISTRY:
        return name, sha == REGISTRY[name]["sha256"]
    for reg, entry in REGISTRY.items():
        if sha == entry["sha256"]:
            return reg, True
    return name, None


def _has_sep(fname):
    return os.sep in fname or "/" in fname or bool(os.altsep and os.altsep in fname)


def _xc_of_name(name):
    """'lda' / 'pbe' guessed from a PP file name, or None."""
    n = name.lower()
    if "_lda" in n or "lda." in n:
        return "lda"
    if "gga" in n:
        return "pbe"
    return None


def resolve(pp_args, symbols, xc, pppath=None, allow_xc_mismatch=False):
    """{element: absolute path} for every element of `symbols`.
    pp_args: ["EL=FILE", ...] (FILE with a path separator is used as given, a bare name is searched);
    elements without --pp get the registry default for `xc`. Exits 2 with a message on errors:
    missing file, no default, corrupted registry file, PP/XC mismatch (unless allow_xc_mismatch)."""
    given = {}
    for item in pp_args or []:
        el, eq, fname = str(item).partition("=")
        el, fname = el.strip().capitalize(), fname.strip()
        if not eq or not el or not fname:
            die(f"bad --pp value '{item}': use EL=FILE, e.g. --pp Al=al.gga.recpot")
        given[el] = fname
    out = {}
    for el in sorted(set(symbols)):
        if el in given:
            fname = given[el]
            if _has_sep(fname):
                path = os.path.abspath(os.path.expanduser(fname))
                if not os.path.isfile(path):
                    die(f"pseudopotential file for {el} not found: {path}")
            else:
                path = find(fname, pppath)
                if path is None:
                    die(f"pseudopotential file '{fname}' for {el} not found in the search path "
                        f"{search_path(pppath)}; pass --pppath DIR or a path, or run `ofdft pp fetch`")
        else:
            fname = DEFAULT_PP.get((el, xc))
            if fname is None:
                die(f"no default pseudopotential for {el} (only Al is validated); pass --pp {el}=FILE")
            path = find(fname, pppath)
            if path is None:
                die(f"default pseudopotential '{fname}' for {el} ({xc.upper()}) not found in the search path "
                    f"{search_path(pppath)}; run `ofdft pp fetch` (or copy the file into one of these "
                    f"directories, or pass --pp {el}=FILE)")
        reg, ok = verify(path)
        if ok is False:
            die(f"{path} has the registry name '{reg}' but its sha256 does not match the registry "
                f"(corrupted or a different file); delete it and run `ofdft pp fetch`, or rename it if it "
                f"is a different potential")
        if ok is None:
            warn(f"pseudopotential {path} for {el} is not a registry file; results not validated")
        else:
            entry = REGISTRY[reg]
            if entry["element"] != el:
                die(f"{path} is the registry file '{reg}' for {entry['element']}, not for {el}")
        guess = REGISTRY[reg]["xc"] if ok else _xc_of_name(os.path.basename(path))
        if guess is not None and guess != xc:
            msg = (f"pseudopotential {os.path.basename(path)} for {el} is for "
                   f"{'LDA' if guess == 'lda' else 'GGA/PBE'} (judged from its name/checksum), but --xc {xc}")
            if not allow_xc_mismatch:
                want = DEFAULT_PP.get((el, xc))
                die(msg + (f"; use {want}" if want else "") + " or pass --allow-xc-mismatch")
            warn(msg + " (allowed by --allow-xc-mismatch)")
        out[el] = path
    return out


class FetchError(RuntimeError):
    pass


def fetch(dest=None, names=None, timeout=60):
    """Download registry files into `dest` [~/.ofdft/pp], verify sha256, write LF bytes atomically.
    Files already present and valid are skipped. Returns the list of paths (present or written).
    Raises FetchError (offline, HTTP error, checksum mismatch)."""
    dest = os.path.abspath(os.path.expanduser(dest or default_dest()))
    names = list(names or REGISTRY)
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise FetchError(f"not registry files: {unknown} (known: {sorted(REGISTRY)})")
    os.makedirs(dest, exist_ok=True)
    out = []
    for name in names:
        entry, target = REGISTRY[name], os.path.join(dest, name)
        if os.path.isfile(target):
            if sha256_text(target) == entry["sha256"]:
                print(f"  {name}: already present and valid ({target})")
                out.append(target)
                continue
            print(f"  {name}: {target} has a wrong checksum; downloading it again")
        try:
            with urllib.request.urlopen(entry["url"], timeout=timeout) as r:
                data = r.read()
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise FetchError(f"could not download {name} from {entry['url']} ({exc}). If this machine has no "
                             f"internet access, download it on a machine with internet and copy it to {dest}")
        data = data.replace(b"\r\n", b"\n")
        sha = sha256_bytes(data)
        if sha != entry["sha256"]:
            raise FetchError(f"downloaded {name} has sha256 {sha}, expected {entry['sha256']}; not written")
        tmp = os.path.join(dest, f".{name}.{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, target)
        print(f"  {name}: downloaded and verified -> {target}")
        out.append(target)
    return out


def _list(args):
    print("pseudopotential search path (first match wins):")
    for k, d in enumerate(search_path(args.pppath), 1):
        print(f"  {k}. {d}{'' if os.path.isdir(d) else '   (missing)'}")
    print(f"registry: {SOURCE}")
    defaults = {v: k for k, v in DEFAULT_PP.items()}
    for name, entry in REGISTRY.items():
        path = find(name, args.pppath)
        if path is None:
            state, where = "MISSING", "(run: ofdft pp fetch)"
        else:
            state = "OK" if verify(path)[1] else "BAD-SHA256"
            where = path
        dflt = defaults.get(name)
        tag = f"default for {dflt[0]} {dflt[1].upper()}" if dflt else ""
        print(f"  {name:15s} {entry['element']:3s} {entry['xc'].upper():4s} {tag:20s} {state:10s} {where}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="ofdft pp", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action")
    pl = sub.add_parser("list", help="registry files, where they are found, checksum status")
    pl.add_argument("--pppath", default=None, help="extra directory searched first")
    pf = sub.add_parser("fetch", help="download and verify the registry files")
    pf.add_argument("--dest", default=None, help=f"target directory [{default_dest()}]")
    pf.add_argument("names", nargs="*", help=f"registry files to fetch [all: {' '.join(REGISTRY)}]")
    pf.add_argument("--timeout", type=float, default=60.0, help="download timeout per file, s [60]")
    args = p.parse_args(argv)
    if args.action == "list":
        return _list(args)
    if args.action == "fetch":
        unknown = [n for n in args.names if n not in REGISTRY]
        if unknown:
            print(f"ERROR: not registry files: {unknown} (known: {' '.join(REGISTRY)})", file=sys.stderr)
            return 2
        dest = os.path.abspath(os.path.expanduser(args.dest or default_dest()))
        print(f"fetching {SOURCE}\n  into {dest}")
        try:
            fetch(dest, args.names or None, timeout=args.timeout)
        except FetchError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0
    p.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
