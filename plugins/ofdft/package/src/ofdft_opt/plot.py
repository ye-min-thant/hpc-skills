"""`ofdft plot` -- plot stress-strain curves written by `ofdft sscurve`.

  ofdft plot Al_CONTCAR_ss_z_tension.csv [more.csv ...] [-o curve.png] [--axis x|y|z]

Left: loading-axis stress vs engineering strain (one line per csv) with the other five stress
components dotted grey (they should stay ~0 in tension mode). Right: optimizer steps per strain
point (cost; large values = plastic events). Positive stress = tensile. The per-point thresholds
(fmax/smax, read from <prefix>_params.json) are shown in the title: report them with any result.
Default output: <first csv without .csv>.png next to the first csv. Needs matplotlib.
"""
import argparse
import csv
import json
import os
import sys

COMP = ["sxx_GPa", "syy_GPa", "szz_GPa", "syz_GPa", "sxz_GPa", "sxy_GPa"]
SLOPE_STRAIN = 0.01          # initial slope from the first point to the first one >= 1 % further


def read_csv(path):
    """{column: [floats]} of the converged rows (last row per strain wins, file order kept)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    by_strain = {}
    for r in rows:
        conv = str(r.get("converged", "True")).strip().lower()
        if conv in ("false", "0", "no"):
            continue
        try:                                       # skips a half-written last line of a running job
            vals = {k: float(v) for k, v in r.items() if k and k != "converged"}
        except (TypeError, ValueError):
            continue
        if "strain" not in vals:
            continue
        by_strain[round(vals["strain"], 10)] = vals
    good = list(by_strain.values())
    if not good:
        return {}
    keys = [k for k in good[0] if all(k in g for g in good)]
    return {k: [g[k] for g in good] for k in keys}


def read_params(csv_path):
    """<prefix>_params.json next to <prefix>.csv, or {}."""
    base = csv_path[:-4] if csv_path.lower().endswith(".csv") else csv_path
    try:
        with open(base + "_params.json") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError):
        return {}


def initial_slope(strain, stress):
    """(slope GPa, strain_k) from point 0 to the first point >= 1 % strain away (else the last)."""
    if len(strain) < 2:
        return None, None
    k = next((i for i in range(1, len(strain)) if abs(strain[i] - strain[0]) >= SLOPE_STRAIN - 1e-12),
             len(strain) - 1)
    de = strain[k] - strain[0]
    if de == 0:
        return None, None
    return (stress[k] - stress[0]) / de, strain[k]


def label_of(path):
    b = os.path.basename(path)
    return b[:-4] if b.lower().endswith(".csv") else b


def thresholds(par):
    f, s = par.get("fmax"), par.get("smax")
    if f is None and s is None:
        return None
    return f"fmax {f} eV/A, smax {s} GPa"


def _pyplot():
    try:
        import matplotlib
    except ImportError:
        print(f"ERROR: matplotlib is not installed for {sys.executable}: install it "
              "(conda install -c conda-forge matplotlib) or copy the csv (and its _params.json) to a "
              "machine with matplotlib and run `ofdft plot` there", file=sys.stderr)
        raise SystemExit(2)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def main(argv=None):
    p = argparse.ArgumentParser(prog="ofdft plot", description=__doc__.split("\n\n")[0])
    p.add_argument("csv", nargs="+", help="<prefix>.csv file(s) written by `ofdft sscurve`")
    p.add_argument("--axis", choices=["x", "y", "z"], default=None,
                   help="loading axis [from <prefix>_params.json, else the file name, else z]")
    p.add_argument("-o", "--out", default=None, help="PNG to write [<first csv stem>.png next to it]")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    missing = [c for c in args.csv if not os.path.isfile(c)]
    if missing:
        print("ERROR: not found: " + ", ".join(missing), file=sys.stderr)
        return 2
    curves = []
    for path in args.csv:
        d = read_csv(path)
        if not d or "strain" not in d:
            print(f"{path}: no converged points yet")
            continue
        par = read_params(path)
        axis = args.axis or par.get("axis") or next((a for a in "xyz" if f"_ss_{a}_" in os.path.basename(path)), "z")
        if axis not in ("x", "y", "z"):
            axis = "z"
        curves.append((path, d, par, axis))
    if not curves:
        print("nothing to plot")
        return 0

    plt = _pyplot()
    fig, (ax, axc) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    th_all = {thresholds(c[2]) for c in curves}
    nfile = len(curves)
    for j, (path, d, par, axis) in enumerate(curves):
        load = f"s{axis}{axis}_GPa"
        strain = d["strain"]
        sig = d.get(load)
        if sig is None:
            print(f"{path}: column {load} missing, skipped")
            continue
        e = [100 * s for s in strain]                      # percent
        label = label_of(path)
        th = thresholds(par)
        leg = f"$\\sigma_{{{axis}{axis}}}$  {label}" + (f"  ({th})" if th and len(th_all) > 1 else "")
        ax.plot(e, sig, "o-", lw=2, label=leg)
        for c in COMP:
            if c != load and c in d:
                ax.plot(e, d[c], ":", lw=1, color="0.6")
        if "steps" in d:
            se = sorted(set(e))
            gap = min((b - a for a, b in zip(se, se[1:]) if b > a), default=0.5)
            w = 0.8 * gap / nfile
            axc.bar([x + (j - (nfile - 1) / 2) * w for x in e], d["steps"], width=w, label=label)
        sign = -1.0 if strain[-1] < 0 else 1.0             # compression: the peak is the most negative stress
        i = max(range(len(e)), key=lambda k: sign * sig[k])
        msg = (f"{path}: {len(e)} points, strain {e[0]:.2f} .. {e[-1]:.2f} %, peak {sig[i]:.3f} GPa at {e[i]:.2f} %")
        if "steps" in d:
            msg += f", {sum(d['steps']):.0f} steps"
        if "time_s" in d:
            msg += f", {sum(d['time_s']) / 3600:.1f} h"
        msg += f"; thresholds: {th or 'unknown (no _params.json)'}"
        print(msg)
        slope, ek = initial_slope(strain, sig)
        if slope is not None:
            print(f"   initial slope ({e[0]:.2f} .. {100 * ek:.2f} %): {slope:.1f} GPa")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("engineering strain (%)")
    ax.set_ylabel("stress (GPa, positive = tensile)")
    title = "stress-strain (dotted grey: other stress components)"
    if len(th_all) == 1 and None not in th_all:
        title += "\nper-point thresholds: " + next(iter(th_all))
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    axc.set_xlabel("engineering strain (%)")
    axc.set_ylabel("optimizer steps per point")
    axc.set_title("cost", fontsize=10)
    fig.tight_layout()
    out = args.out or os.path.splitext(args.csv[0])[0] + ".png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
