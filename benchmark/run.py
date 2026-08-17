"""Benchmark / cross-validation harness for grcwa.

Runs the same battery of physically-motivated gratings (benchmark/worker.py)
through several grcwa installations and factorization modes, in isolated
subprocesses, then:
  * cross-checks that the results agree (correctness), and
  * compares wall-clock timing,
and exports everything to benchmark/results.{json,csv}.

Variants are auto-discovered: every `benchmark/grcwa*` directory that is an
importable package becomes a suite, plus the current branch (`fork`) at the repo
root. Each variant is run with Laurent's rule, and additionally with the Pol
method (`fmm_method='pol'`) if its `obj` supports it -- so Pol columns appear
only for the versions that actually implement it (no hand-maintained list).

Drop a package into benchmark/ (e.g. `grcwa-weiliang-013`, `grcwa-codex`, ...)
and it shows up automatically; the label is the directory name with the leading
`grcwa` stripped (`grcwa-weiliang-013` -> `weiliang-013`).

On top of the grcwa family, the independent code **Ikarus** (`pip install
ikarus-rcwa`, benchmark/ikarus_suite.py) contributes three more columns --
`ikarus[Laurent]`, `ikarus[Li]` and `ikarus[NV]` -- when it is installed, and is
silently skipped when it is not. Its `[Laurent]` column joins the Laurent
cross-check, which turns that check into a *cross-codebase* one.
"""
import os
import sys
import json
import csv
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORKER = os.path.join(HERE, "worker.py")


def discover_variants():
    """Return [(label, package-parent-dir, module-name)] for every vendored
    grcwa* package in benchmark/, plus the current branch at the repo root.

    Each worker runs as a script whose sys.path[0] is benchmark/, so a vendored
    package is importable by its directory name; we still set PYTHONPATH to the
    parent dir so the choice is explicit and robust to the launch cwd. Distinct
    directory names keep the variants isolated from each other on import."""
    variants = []
    for name in sorted(os.listdir(HERE)):
        path = os.path.join(HERE, name)
        if (os.path.isdir(path) and name.startswith("grcwa") and name != "grcwa"
                and os.path.isfile(os.path.join(path, "__init__.py"))):
            label = name.removeprefix("grcwa").lstrip("-_") or name
            variants.append((label, HERE, name))
    variants.append(("fork", REPO, "grcwa"))
    return variants


VARIANTS = discover_variants()

# Ikarus factorization -> column suffix. Laurent is the direct rule (the one
# grcwa's default implements), Li the classic inverse rule, NV the normal-vector
# method (Ikarus's own default) -- the only one that is also faithful on curved
# boundaries. See benchmark/ikarus_suite.py.
IKARUS_MODES = [("laurent", "Laurent"), ("li", "Li"), ("normal", "NV")]


def run_variant(path, modname, fmm, suite="grcwa"):
    env = os.environ.copy()
    env["PYTHONPATH"] = path            # parent dir; variants kept apart by name
    env["GRCWA_MOD"] = modname
    env["FMM"] = fmm
    env["SUITE"] = suite
    print(f"run process worker: {modname or suite} ({fmm})")
    p = subprocess.run([sys.executable, WORKER], env=env,
                       capture_output=True, text=True)
    if p.returncode != 0:
        return {"_error": p.stderr.strip().splitlines()[-1] if p.stderr else "failed"}
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"_error": (p.stdout + p.stderr)[-200:]}


def airy_R(n0, n1, ns, d, freq):
    k0 = 2 * 3.141592653589793 * freq
    r01 = (n0 - n1) / (n0 + n1); r12 = (n1 - ns) / (n1 + ns)
    import cmath
    ph = cmath.exp(2j * k0 * n1 * d)
    return abs((r01 + r12 * ph) / (1 + r01 * r12 * ph)) ** 2


def _has_results(res):
    """True if a worker result dict carries at least one solved case."""
    return isinstance(res, dict) and any(
        isinstance(v, dict) and "R" in v for v in res.values())


def case_R(data, col, name):
    """Reflectance of case ``name`` in column ``col``, or None if it has none."""
    d = data.get(col)
    if isinstance(d, dict) and name in d and "R" in d[name]:
        return d[name]["R"]
    return None


def main():
    # column label -> {case: result dict}
    columns = []
    data = {}
    for label, path, mod in VARIANTS:
        res = run_variant(path, mod, "none")
        col = f"{label}[Laurent]"
        columns.append(col); data[col] = res
        # attempt Pol for every variant; keep the column only if this version
        # actually implements fmm_method='pol' (otherwise the worker skips it).
        pol = run_variant(path, mod, "pol")
        if _has_results(pol):
            colp = f"{label}[Pol]"
            columns.append(colp); data[colp] = pol

    # Ikarus: a separate codebase with its own construct, so it gets its own
    # columns rather than a variant directory. Optional -- skipped when absent.
    for fmm, suffix in IKARUS_MODES:
        res = run_variant(HERE, "", fmm, suite="ikarus")
        if _has_results(res):
            col = f"ikarus[{suffix}]"
            columns.append(col); data[col] = res
        elif fmm == IKARUS_MODES[0][0]:
            print(f"   (no ikarus columns: {res.get('_error', 'unavailable')}"
                  f"  --  pip install ikarus-rcwa)")

    # collect case order from any successful column
    case_names = []
    for col in columns:
        if isinstance(data[col], dict) and "_error" not in data[col]:
            case_names = list(data[col].keys()); break

    laurent_cols = [c for c in columns if "[Laurent]" in c]
    rows = []
    print("=" * 92)
    print("RCWA benchmark / cross-validation   (lambda = 1 um, lengths in um)")
    print("=" * 92)
    for name in case_names:
        print(f"\n{name}")
        for col in columns:
            r = data[col].get(name, {}) if isinstance(data[col], dict) else {}
            if "R" in r:
                print(f"   {col:<22} R={r['R']:.5f}  T={r['T']:.5f}  A={r['A']:+.5f}"
                      f"  R+T={r['R']+r['T']:.5f}  orders={r.get('label',''):<8} "
                      f"nG={r['nG']:<5} {r['time_ms']:8.2f} ms  [{r['mode']}]")
                rows.append(dict(case=name, column=col, label=r.get("label"),
                            **{k: r[k] for k in
                               ("R", "T", "A", "nG", "time_ms", "mode")}))
            else:
                tag = r.get("skipped") or r.get("error") or "n/a"
                print(f"   {col:<22} -- {tag}")
        # cross-check: agreement among Laurent columns. Kept separate for the
        # grcwa family (expected bit-identical) and across codebases (expected
        # only to ~1e-6, since Ikarus takes a real frequency while this battery
        # feeds grcwa the Q=1e7 complex FREQC).
        same_code = [v for v in (case_R(data, c, name) for c in laurent_cols
                                 if not c.startswith("ikarus")) if v is not None]
        if len(same_code) > 1:
            print(f"   -> grcwa Laurent cross-check: max|dR| = "
                  f"{max(same_code) - min(same_code):.2e}", end="")
        r_fork = case_R(data, "fork[Laurent]", name)
        r_ik = case_R(data, "ikarus[Laurent]", name)
        if r_fork is not None and r_ik is not None:
            print(f"   -> cross-code (fork vs ikarus, Laurent): |dR| = "
                  f"{abs(r_fork - r_ik):.2e}", end="")
        # physical sanity
        any_r = next((data[c][name] for c in columns
                      if isinstance(data[c], dict) and name in data[c]
                      and "R" in data[c][name]), None)
        if any_r is not None:
            rt = any_r["R"] + any_r["T"]
            print(f"   |  R+T={rt:.4f}  A={any_r['A']:+.4f}")
    # absolute anchor for the slab (A1_slab_air: air/Si(3.5)/air, d=0.20)
    print(f"\nA1_slab_air analytic (Airy) R = {airy_R(1.0, 3.5, 1.0, 0.20, 1.0):.5f}")

    out_json = os.path.join(HERE, "results.json")
    out_csv = os.path.join(HERE, "results.csv")
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "column", "label", "R", "T", "A", "nG", "time_ms", "mode"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\nexported: {out_json}\n          {out_csv}")


if __name__ == "__main__":
    main()
