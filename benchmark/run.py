"""Benchmark / cross-validation harness for grcwa.

Runs the same battery of physically-motivated gratings (benchmark/worker.py)
through several grcwa installations and factorization modes, in isolated
subprocesses, then:
  * cross-checks that the results agree (correctness), and
  * compares wall-clock timing,
and exports everything to benchmark/results.{json,csv}.

The "suites" compared (when their paths are available):
  1. orig-0.1.2    : weiliang's original PyPI release, BEFORE the Pol update.
  2. weiliang-0.1.3: weiliang's upstream master WITH his own Pol commits
                     (Laurent + Pol) -- the reference for the Pol method.
  3. forkmaster    : darkviruzz fork, before this work (Laurent only).
  4. fork[Laurent] : this branch, default factorization.
  5. fork[Pol]     : this branch with fmm_method='pol' (the upstream Pol
                     algorithm, ported into this branch).

Variant package locations are taken from environment variables so the harness
stays portable:
  GRCWA_ORIG_PATH       parent dir of an `orig_grcwa` package (pip download
                        grcwa==0.1.2, rename the package dir to orig_grcwa)
  GRCWA_WEILIANG_PATH   parent dir of a `wl_grcwa` package checked out at
                        weiliang's upstream master (with the Pol commits)
  GRCWA_FORKMASTER_PATH parent dir of a `grcwa` package checked out at the
                        fork's master (git archive origin/master)
The current branch (fork) is auto-detected as the repo root.
"""
import os
import sys
import json
import csv
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORKER = os.path.join(HERE, "worker.py")

# (label, package-parent path, module name, run_pol)
VARIANTS = []
# if os.environ.get("GRCWA_ORIG_PATH"):
#     VARIANTS.append(("orig-0.1.2", os.environ["GRCWA_ORIG_PATH"], "orig_grcwa", False))
# if os.environ.get("GRCWA_WEILIANG_PATH"):
#     VARIANTS.append(("weiliang-0.1.3", os.environ["GRCWA_WEILIANG_PATH"], "wl_grcwa", True))
# if os.environ.get("GRCWA_FORKMASTER_PATH"):
#     VARIANTS.append(("forkmaster", os.environ["GRCWA_FORKMASTER_PATH"], "grcwa", False))
# (label, package-parent path, module name, run_pol)
for name in sorted(os.listdir(HERE)):
    path = os.path.join(HERE, name)
    if os.path.isdir(path) and name.startswith("grcwa") and name != "grcwa":
        VARIANTS.append((
            name.removeprefix("grcwa").lstrip("-_") or name,
            path,
            name,
            False,
        ))

VARIANTS.append(('weiliang-013-POL', 'C:\\Users\\mwalther\\PycharmProjects\\grcwa\\benchmark\\grcwa-weiliang-013', 'grcwa-weiliang-013', True))
VARIANTS.append(("fork", REPO, "grcwa", True))


def run_variant(path, modname, fmm):
    env = os.environ.copy()
    env["PYTHONPATH"] = path            # isolate: only this package on the path
    env["GRCWA_MOD"] = modname
    env["FMM"] = fmm
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


def main():
    # column label -> {case: result dict}
    columns = []
    data = {}
    for label, path, mod, run_pol in VARIANTS:
        res = run_variant(path, mod, "none")
        col = f"{label}[Laurent]"
        columns.append(col); data[col] = res
        if run_pol:
            colp = f"{label}[Pol]"
            columns.append(colp); data[colp] = run_variant(path, mod, "pol")

    # collect case order from any successful column
    case_names = []
    for col in columns:
        if isinstance(data[col], dict) and "_error" not in data[col]:
            case_names = list(data[col].keys()); break

    laurent_cols = [c for c in columns if "[Laurent]" in c]
    rows = []
    print("=" * 92)
    print("grcwa benchmark / cross-validation   (lambda = 1 um, lengths in um)")
    print("=" * 92)
    for name in case_names:
        print(f"\n{name}")
        for col in columns:
            r = data[col].get(name, {}) if isinstance(data[col], dict) else {}
            if "R" in r:
                print(f"   {col:<22} R={r['R']:.5f}  T={r['T']:.5f}  A={r['A']:+.5f}"
                      f"  R+T={r['R']+r['T']:.5f}  nG={r['nG']:<4} "
                      f"{r['time_ms']:8.2f} ms  [{r['mode']}]")
                rows.append(dict(case=name, column=col, **{k: r[k] for k in
                            ("R", "T", "A", "nG", "time_ms", "mode")}))
            else:
                tag = r.get("skipped") or r.get("error") or "n/a"
                print(f"   {col:<22} -- {tag}")
        # cross-check: agreement among Laurent columns
        rvals = [data[c][name]["R"] for c in laurent_cols
                 if isinstance(data[c], dict) and name in data[c] and "R" in data[c][name]]
        if len(rvals) > 1:
            spread = max(rvals) - min(rvals)
            print(f"   -> Laurent cross-check: max|dR| = {spread:.2e}", end="")
        # physical sanity
        any_r = next((data[c][name] for c in columns
                      if isinstance(data[c], dict) and name in data[c]
                      and "R" in data[c][name]), None)
        if any_r is not None:
            rt = any_r["R"] + any_r["T"]
            print(f"   |  R+T={rt:.4f}  A={any_r['A']:+.4f}")
    # absolute anchor for the slab
    print(f"\n0D_slab analytic (Airy) R = {airy_R(1.0, 2.0, 1.0, 0.30, 1.0):.5f}")

    out_json = os.path.join(HERE, "results.json")
    out_csv = os.path.join(HERE, "results.csv")
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "column", "R", "T", "A", "nG", "time_ms", "mode"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\nexported: {out_json}\n          {out_csv}")


if __name__ == "__main__":
    main()
