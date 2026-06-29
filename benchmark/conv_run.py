"""Convergence-study harness for grcwa.

Sweeps the truncation order nG over the battery in conv_worker.py for several
grcwa installations and factorization modes (isolated subprocesses), attaches
analytic references where they exist, cross-checks, and exports everything to
benchmark/conv_results.{json,csv}.

The interesting comparison is the *factorization rule* (Laurent vs Pol): all the
Laurent codes are bit-identical, so we only run

  weiliang-0.1.3[Laurent], weiliang-0.1.3[Pol]   (independent codebase; if path set)
  fork[Laurent],           fork[Pol]              (this branch)

That gives, per structure: the convergence rate of each rule, and a check that
the two independent codebases agree for each rule.

Reference per case:
  * A1/A1b  : exact thin-film Airy result.
  * A2      : asymptotic effective-medium (form-birefringence) film -- the RCWA
              result converges close to it (finite Lambda/lambda residual), so we
              also keep the high-nG self-reference for the error-decay plot.
  * B/C     : no closed form -> a shared reference taken from the highest-nG
              Laurent run (the rule guaranteed correct in the limit). Drop in
              your external-RCWA value later by editing the 'ref' field.

Env vars (parent dir of the named package):
  GRCWA_WEILIANG_PATH  ->  wl_grcwa  (weiliang upstream master, with Pol)
The current branch (fork) is auto-detected as the repo root.
"""
import os
import sys
import json
import csv
import cmath
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORKER = os.path.join(HERE, "conv_worker.py")

# (label, package-parent path, module name)
VARIANTS = []
if os.environ.get("GRCWA_WEILIANG_PATH"):
    VARIANTS.append(("weiliang-0.1.3", os.environ["GRCWA_WEILIANG_PATH"], "wl_grcwa"))
VARIANTS.append(("fork", REPO, "grcwa"))


def run_variant(path, modname, fmm):
    env = os.environ.copy()
    env["PYTHONPATH"] = path
    env["GRCWA_MOD"] = modname
    env["FMM"] = fmm
    p = subprocess.run([sys.executable, WORKER], env=env,
                       capture_output=True, text=True)
    if p.returncode != 0:
        return {"_error": p.stderr.strip().splitlines()[-1] if p.stderr else "failed"}
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"_error": (p.stdout + p.stderr)[-300:]}


def airy_RT(n0, n1, ns, d, freq):
    """Exact reflectance/transmittance of a single film n1 (thickness d) between
    half-spaces n0 (incidence) and ns, at normal incidence."""
    k0 = 2 * cmath.pi * freq
    r01 = (n0 - n1) / (n0 + n1)
    r12 = (n1 - ns) / (n1 + ns)
    t01 = 2 * n0 / (n0 + n1)
    t12 = 2 * n1 / (n1 + ns)
    ph = cmath.exp(1j * k0 * n1 * d)
    r = (r01 + r12 * ph**2) / (1 + r01 * r12 * ph**2)
    t = (t01 * t12 * ph) / (1 + r01 * r12 * ph**2)
    R = abs(r) ** 2
    T = (ns.real / n0.real) * abs(t) ** 2 if n0.imag == 0 else None
    return R, T


def emt_indices(eps_hi, eps_lo, f):
    """0th-order effective-medium indices of a lamellar grating (fill f of hi)."""
    eps_te = f * eps_hi + (1 - f) * eps_lo                 # E || grooves
    eps_tm = 1.0 / (f / eps_hi + (1 - f) / eps_lo)         # E _|_ grooves
    return cmath.sqrt(eps_te), cmath.sqrt(eps_tm)


def analytic_ref(name, info):
    """Return dict(type, R, T, note) or None."""
    nk = info.get("nk", {})
    freq = 1.0
    if name == "A1_slab_air":
        R, T = airy_RT(1.0, 3.5, 1.0, 0.20, freq)
        return {"type": "analytic_exact", "R": R, "T": T,
                "note": "air/Si(3.5)/air slab, Airy"}
    if name == "A1b_slab_glass":
        R, T = airy_RT(1.0, 3.5, 1.5, 0.20, freq)
        return {"type": "analytic_exact", "R": R, "T": T,
                "note": "air/Si(3.5)/SiO2(1.5) slab, Airy"}
    if name in ("A2_formbiref_TE", "A2_formbiref_TM"):
        n_hi = complex(*nk["hi"]); n_lo = complex(*nk["lo"])
        n_te, n_tm = emt_indices(n_hi**2, n_lo**2, info["ff"])
        n_eff = n_te if name.endswith("TE") else n_tm
        R, T = airy_RT(1.0, n_eff, 1.0, info["d"], freq)
        return {"type": "analytic_asymptotic", "R": R, "T": T,
                "n_eff": n_eff.real,
                "note": f"EMT film n_eff={n_eff.real:.3f} (asymptotic, finite "
                        f"Lambda/lambda residual)"}
    return None


def main():
    # column label -> worker output {case: {info, sweep}}
    columns, data = [], {}
    for label, path, mod in VARIANTS:
        for fmm, tag in (("none", "Laurent"), ("pol", "Pol")):
            col = f"{label}[{tag}]"
            columns.append(col)
            data[col] = run_variant(path, mod, fmm)

    # case order + info from the first good column
    cases, infos = [], {}
    for col in columns:
        d = data[col]
        if isinstance(d, dict) and "_error" not in d:
            cases = list(d.keys())
            infos = {c: d[c].get("info", {}) for c in cases}
            break

    laurent_cols = [c for c in columns if c.endswith("[Laurent]")]

    merged = {"meta": {"lambda_um": 1.0, "freq": 1.0,
                       "note": "lengths in um; eps=(n+ik)^2; exp(-iwt)"},
              "columns": columns, "cases": {}}

    print("=" * 96)
    print("grcwa CONVERGENCE study   R(nG) -> reference   (lambda = 1 um)")
    print("=" * 96)

    rows = []
    for case in cases:
        info = infos.get(case, {})
        ref = analytic_ref(case, info)

        # self-reference for B/C: highest-nG Laurent value (correct in the limit)
        if ref is None:
            best = None
            for col in laurent_cols:
                sw = data.get(col, {}).get(case, {}).get("sweep", [])
                good = [p for p in sw if "R" in p]
                if good:
                    best = good[-1]
            if best is not None:
                ref = {"type": "self_highnG_Laurent", "R": best["R"],
                       "T": best["T"], "at_nG": best["nG"],
                       "note": "highest-nG Laurent run (replace with external RCWA)"}

        centry = {"info": info, "ref": ref, "columns": {}}
        print(f"\n{case}   [{info.get('desc','')}]")
        if ref:
            print(f"   ref ({ref['type']}): R={ref['R']:.6f}"
                  + (f"  note: {ref['note']}" if ref.get('note') else ""))
        for col in columns:
            sw = data.get(col, {}).get(case, {}).get("sweep", [])
            skip = data.get(col, {}).get(case, {}).get("skipped")
            if skip and not sw:
                print(f"   {col:<24} -- {skip}")
                continue
            centry["columns"][col] = sw
            for p in sw:
                if "R" not in p:
                    continue
                err = abs(p["R"] - ref["R"]) if ref else None
                rows.append(dict(case=case, column=col, nG_req=p["nG_req"],
                                 nG=p["nG"], R=p["R"], T=p["T"], A=p["A"],
                                 err_R=err,
                                 time_ms=p.get("time_ms"), mode=p["mode"]))
            # compact line: error at the smallest and largest nG
            good = [p for p in sw if "R" in p]
            if good and ref:
                e0 = abs(good[0]["R"] - ref["R"])
                e1 = abs(good[-1]["R"] - ref["R"])
                print(f"   {col:<24} nG {good[0]['nG']:>4}->{good[-1]['nG']:<4}"
                      f"  |dR| {e0:.2e} -> {e1:.2e}  [{good[-1]['mode']}]")
        merged["cases"][case] = centry

    out_json = os.path.join(HERE, "conv_results.json")
    out_csv = os.path.join(HERE, "conv_results.csv")
    with open(out_json, "w") as f:
        json.dump(merged, f, indent=2)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "column", "nG_req", "nG", "R",
                                          "T", "A", "err_R", "time_ms", "mode"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nexported: {out_json}\n          {out_csv}")


if __name__ == "__main__":
    main()
