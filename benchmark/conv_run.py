"""Convergence-study harness for grcwa.

Sweeps the truncation order nG over the battery in conv_worker.py for several
grcwa installations and factorization modes (isolated subprocesses), attaches
analytic references where they exist, cross-checks, and exports everything to
benchmark/conv_results.{json,csv}.

The interesting comparison is the *factorization rule* (Laurent vs Pol). Variants
are auto-discovered the same way as in run.py: every `benchmark/grcwa*` package is
a suite, plus the current branch (`fork`). Set ``GRCWA_VARIANTS`` to a
comma-separated list of labels to run only those variants. Each is run with
Laurent, and with Pol if its `obj` supports `fmm_method='pol'` -- so Pol columns
appear only where the method is actually implemented. The Laurent codes are
mutually bit-identical, so their convergence curves overlay; the Pol curves are
what differ.

Reference per case:
  * A1/A1b  : exact thin-film Airy result.
  * A2/B/C  : external Moose reference where available (including its explicitly
              marked provisional 2D values).
  * otherwise: highest-order Ikarus normal-vector result, with highest-order
               Laurent used only when Ikarus is unavailable.
"""
import os
import sys
import json
import csv
import cmath
import math
import subprocess
import tempfile

import structures as ST
from conv_worker import (order_lists, planned_points,
                         validate_cell_resolution)
from timing_model import (annotate_estimates, build_timing_models,
                          convergence_report)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORKER = os.path.join(HERE, "conv_worker.py")
OUTPUT_DIR = os.environ.get("GRCWA_OUTPUT_DIR", HERE)
MOOSE_JSON = os.environ.get("GRCWA_MOOSE_JSON",
                            os.path.join(HERE, "moose_reference.json"))

with open(MOOSE_JSON) as f:
    MOOSE = json.load(f)
MOOSE_CASES = MOOSE.get("cases", {})

def discover_variants():
    """Auto-discover vendored grcwa* packages in benchmark/, plus the current
    branch at the repo root. Returns [(label, package-parent-dir, module-name)].
    (Same convention as run.py.)"""
    variants = []
    for name in sorted(os.listdir(HERE)):
        path = os.path.join(HERE, name)
        if (os.path.isdir(path) and name.startswith("grcwa") and name != "grcwa"
                and os.path.isfile(os.path.join(path, "__init__.py"))):
            label = name.removeprefix("grcwa").lstrip("-_") or name
            variants.append((label, HERE, name))
    variants.append(("fork", REPO, "grcwa"))
    return variants


def select_variants(variants):
    """Apply the optional comma-separated ``GRCWA_VARIANTS`` label filter."""
    value = os.environ.get("GRCWA_VARIANTS")
    if not value:
        return variants
    wanted = {label.strip() for label in value.split(",") if label.strip()}
    selected = [variant for variant in variants if variant[0] in wanted]
    missing = wanted.difference(label for label, _, _ in selected)
    if missing:
        available = ", ".join(label for label, _, _ in variants)
        raise SystemExit("Unknown GRCWA_VARIANTS label(s): "
                         f"{', '.join(sorted(missing))}. Available: {available}")
    return selected


VARIANTS = select_variants(discover_variants())

# Ikarus factorizations -> column suffixes (see benchmark/ikarus_suite.py).
IKARUS_MODES = [("laurent", "Laurent"), ("li", "Li"), ("normal", "NV")]


def run_variant(path, modname, fmm, suite="grcwa"):
    env = os.environ.copy()
    env["PYTHONPATH"] = path
    env["GRCWA_MOD"] = modname
    env["FMM"] = fmm
    env["SUITE"] = suite
    print(f"run process worker: {modname or suite} ({fmm})")
    stderr_lines = []
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as worker_stdout:
        p = subprocess.Popen(
            [sys.executable, "-u", WORKER], env=env, stdout=worker_stdout,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        for raw_line in p.stderr:
            line = raw_line.rstrip()
            stderr_lines.append(line)
            print(f"   {line}", flush=True)
        p.stderr.close()
        returncode = p.wait()
        worker_stdout.seek(0)
        stdout = worker_stdout.read()
    stderr = "\n".join(stderr_lines)
    if returncode != 0:
        return {"_error": stderr_lines[-1] if stderr_lines else "failed"}
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {"_error": (stdout + stderr)[-300:]}


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


def moose_ref(name):
    """Return the external Moose reflectance reference for ``name``, if any."""
    source = MOOSE_CASES.get(name, {})
    if source.get("ref") is None:
        return None
    ref = {"type": "external_moose", "R": source["ref"], "from": "Moose",
           "note": "external Moose RCWA reference"}
    if source.get("ref_provisional"):
        ref["ref_provisional"] = True
        ref["note"] += " (provisional highest computed value)"
    return ref


def _has_sweep(res):
    """True if a worker result carries at least one solved sweep point."""
    return isinstance(res, dict) and "_error" not in res and any(
        any("R" in p for p in c.get("sweep", []))
        for c in res.values() if isinstance(c, dict))


def _requested_columns(value):
    return {part.strip() for part in (value or "").split(",")
            if part.strip()}


def validate_worker_results(attempts, included_columns, structures, ng1d, q2d,
                            max2d, required_columns=(), fast_repeat=3,
                            fast_threshold_ms=1000.0, strict_columns=None):
    """Return completeness errors before references or convergence are derived.

    Matching is done with the requested ``q`` rather than the backend's retained
    ``nG`` because a backend may legitimately retain a nearby number of modes.
    """
    issues = []
    required = set(required_columns)
    strict = required if strict_columns is None else set(strict_columns)
    for column in sorted(required):
        result = attempts.get(column)
        if column not in included_columns:
            detail = result.get("_error") if isinstance(result, dict) else None
            issues.append("required column %s is unavailable%s" % (
                column, ": " + str(detail) if detail else ""))

    for column in included_columns:
        result = attempts.get(column)
        if not isinstance(result, dict) or result.get("_error"):
            issues.append("column %s failed: %s" % (
                column, result.get("_error", "invalid worker output")
                if isinstance(result, dict) else "invalid worker output"))
            continue
        for structure in structures:
            name = structure["name"]
            case = result.get(name)
            if not isinstance(case, dict):
                issues.append("%s / %s: missing structure" % (column, name))
                continue
            sweep = case.get("sweep", [])
            by_q = {}
            for point in sweep:
                q = point.get("q")
                if q in by_q:
                    issues.append("%s / %s / q=%s: duplicate result" % (
                        column, name, q))
                by_q[q] = point
            for q, _ in planned_points(structure, ng1d, q2d, max2d):
                point = by_q.get(q)
                prefix = "%s / %s / q=%s" % (column, name, q)
                if point is None:
                    issues.append(prefix + ": missing result")
                    continue
                if point.get("error"):
                    issues.append(prefix + ": " + str(point["error"]))
                    continue
                if point.get("skipped"):
                    allowed_legacy_anchor = (
                        point["skipped"] == "no-native-0D"
                        and column not in strict)
                    if not allowed_legacy_anchor:
                        issues.append(prefix + ": skipped: "
                                      + str(point["skipped"]))
                    continue
                if "R" not in point:
                    issues.append(prefix + ": no numerical result")
                    continue
                if point.get("timing_error"):
                    issues.append(prefix + ": " + str(point["timing_error"]))
                for field in ("R", "T", "A", "nG"):
                    value = point.get(field)
                    if (not isinstance(value, (int, float))
                            or not math.isfinite(value)):
                        issues.append(prefix + ": invalid " + field)
                elapsed = point.get("time_ms")
                if (not isinstance(elapsed, (int, float))
                        or not math.isfinite(elapsed) or elapsed <= 0):
                    issues.append(prefix + ": missing/invalid timing")
                samples = point.get("time_samples_ms", [])
                needed = (fast_repeat if samples
                          and samples[0] < fast_threshold_ms else 1)
                if len(samples) < needed:
                    issues.append(prefix + ": only %d/%d timing samples" % (
                        len(samples), needed))
    return issues


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ng1d, q2d, max2d = order_lists()
    try:
        validate_cell_resolution(ST.STRUCTURES, ng1d, q2d, max2d)
    except ValueError as exc:
        raise SystemExit("Invalid convergence configuration: %s" % exc)

    # column label -> worker output {case: {info, sweep}}
    columns, data, attempts = [], {}, {}
    mandatory_columns = set()
    for label, path, mod in VARIANTS:
        laurent_column = f"{label}[Laurent]"
        mandatory_columns.add(laurent_column)
        lau = run_variant(path, mod, "none")
        attempts[laurent_column] = lau
        if _has_sweep(lau):
            columns.append(laurent_column); data[laurent_column] = lau
        # attempt Pol; keep the column only if this version implements it
        pol_column = f"{label}[Pol]"
        pol = run_variant(path, mod, "pol")
        attempts[pol_column] = pol
        if _has_sweep(pol):
            columns.append(pol_column); data[pol_column] = pol

    # Ikarus: an independent codebase, so it contributes its own columns.
    # Optional dependency -- silently absent when it is not installed.
    for fmm, suffix in IKARUS_MODES:
        res = run_variant(HERE, "", fmm, suite="ikarus")
        column = f"ikarus[{suffix}]"
        attempts[column] = res
        if _has_sweep(res):
            columns.append(column); data[column] = res
        elif fmm == IKARUS_MODES[0][0]:
            print(f"   (no ikarus columns: {res.get('_error', 'unavailable')}"
                  f"  --  pip install ikarus-rcwa)")

    explicitly_required = _requested_columns(
        os.environ.get("GRCWA_REQUIRED_COLUMNS"))
    required_columns = mandatory_columns | explicitly_required
    fast_repeat = max(1, int(os.environ.get("GRCWA_FAST_REPEAT", "3")))
    fast_threshold_ms = max(
        0.0, float(os.environ.get("GRCWA_FAST_THRESHOLD_MS", "1000")))
    issues = validate_worker_results(
        attempts, columns, ST.STRUCTURES, ng1d, q2d, max2d,
        required_columns, fast_repeat, fast_threshold_ms,
        strict_columns=explicitly_required)
    if issues:
        print("\nERROR: convergence sweep is incomplete; cached successful "
              "points were kept for resume.", file=sys.stderr)
        for issue in issues:
            print("   - " + issue, file=sys.stderr)
        raise SystemExit(1)

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
                       "note": "lengths in um; eps=(n+ik)^2; exp(-iwt)",
                       "order_config": {
                           "GRCWA_NG1D": os.environ.get("GRCWA_NG1D"),
                           "GRCWA_NG1D_FROM_Q2D": os.environ.get(
                               "GRCWA_NG1D_FROM_Q2D"),
                           "GRCWA_Q2D": os.environ.get("GRCWA_Q2D"),
                           "GRCWA_MAX2D": os.environ.get("GRCWA_MAX2D"),
                           "resolved_ng1d": ng1d,
                           "resolved_q2d": q2d,
                       },
                       "timing_config": {
                           "fast_repeat": int(os.environ.get(
                               "GRCWA_FAST_REPEAT", "3")),
                           "fast_threshold_ms": float(os.environ.get(
                               "GRCWA_FAST_THRESHOLD_MS", "1000")),
                       },
                       "required_columns": sorted(required_columns),
                       "complete": True},
              "columns": columns, "cases": {}}

    print("=" * 96)
    print("grcwa CONVERGENCE study   R(nG) -> reference   (lambda = 1 um)")
    print("=" * 96)

    for case in cases:
        info = infos.get(case, {})
        ref = analytic_ref(case, info)

        # Keep the exact 0D anchors exact. For patterned cases, prefer the
        # independent Moose value over the asymptotic EMT approximation or a
        # self-reference from one of the compared solvers.
        if ref is None or ref.get("type") == "analytic_asymptotic":
            ref = moose_ref(case)

        # With no exact or Moose reference, use Ikarus's faithful normal-vector
        # result at the highest order run. Laurent remains an availability
        # fallback for environments without Ikarus.
        if ref is None:
            def _best_of(cols):
                best, src = None, None
                for col in cols:
                    good = [p for p in data.get(col, {}).get(case, {})
                            .get("sweep", []) if "R" in p]
                    if good:
                        cand = max(good, key=lambda p: p["nG"])
                        if best is None or cand["nG"] > best["nG"]:
                            best, src = cand, col
                return best, src

            best, src = _best_of(["ikarus[NV]"])
            kind, note = "external_ikarus_NV", ("Ikarus normal-vector "
                                                 "at the highest order run")
            if best is None:
                best, src = _best_of(laurent_cols)
                kind = "self_highnG_Laurent"
                note = "highest-order Laurent fallback (Ikarus unavailable)"
            if best is not None:
                ref = {"type": kind, "R": best["R"], "T": best["T"],
                       "at_nG": best["nG"], "from": src, "note": note}
                if kind == "external_ikarus_NV":
                    ref["ref_provisional"] = True

        centry = {"info": info, "ref": ref, "columns": {}}
        print(f"\n{case}   [{info.get('desc','')}]")
        if ref:
            print(f"   ref ({ref['type']}): R={ref['R']:.6f}"
                  + (f"  note: {ref['note']}" if ref.get('note') else ""))
        for col in columns:
            sw = data.get(col, {}).get(case, {}).get("sweep", [])
            skip = next((p.get("skipped") for p in sw if p.get("skipped")), None)
            good = [p for p in sw if "R" in p]
            if not good:
                if skip:
                    print(f"   {col:<24} -- {skip}")
                continue
            centry["columns"][col] = sw
            for p in good:
                err = abs(p["R"] - ref["R"]) if ref else None
                p["err_R"] = err
            # compact line: error at the smallest and largest order count
            lo = min(good, key=lambda p: p["nG"])
            hi = max(good, key=lambda p: p["nG"])
            if ref:
                e0 = abs(lo["R"] - ref["R"]); e1 = abs(hi["R"] - ref["R"])
                print(f"   {col:<24} nG {lo['nG']:>5}->{hi['nG']:<5}"
                      f"  |dR| {e0:.2e} -> {e1:.2e}  [{hi['mode']}]")
        merged["cases"][case] = centry

    timing_models = build_timing_models(merged["cases"], columns)
    annotate_estimates(merged["cases"], columns, timing_models)
    tolerance = float(os.environ.get("GRCWA_CONV_TOL", "1e-4"))
    convergence = convergence_report(merged["cases"], columns, tolerance)
    merged["meta"]["convergence_tolerance"] = tolerance
    merged["meta"]["timing_note"] = (
        "time_ms is the minimum measured solve; time_est_ms is grouped by "
        "suite/factorization/dimension/order and smoothed monotonically")
    merged["timing_models"] = timing_models
    merged["convergence"] = convergence

    rows = []
    for case, centry in merged["cases"].items():
        for col in columns:
            for p in centry["columns"].get(col, []):
                if "R" not in p:
                    continue
                rows.append(dict(
                    case=case, column=col, q=p.get("q"), label=p.get("label"),
                    nG=p["nG"], R=p["R"], T=p["T"], A=p["A"],
                    err_R=p.get("err_R"), time_ms=p.get("time_ms"),
                    time_est_ms=p.get("time_est_ms"),
                    timing_runs=p.get("timing_runs"),
                    physics_cached=p.get("physics_cached", False),
                    timing_cached=p.get("timing_cached", False), mode=p["mode"]))

    out_json = os.path.join(OUTPUT_DIR, "conv_results.json")
    out_csv = os.path.join(OUTPUT_DIR, "conv_results.csv")
    out_convergence = os.path.join(OUTPUT_DIR, "conv_convergence.csv")
    with open(out_json, "w") as f:
        json.dump(merged, f, indent=2)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "case", "column", "q", "label", "nG", "R", "T", "A", "err_R",
            "time_ms", "time_est_ms", "timing_runs", "physics_cached",
            "timing_cached", "mode"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    convergence_fields = [
        "case", "column", "reference_type", "reference_provisional",
        "tolerance", "status", "nG", "confirmed_by_nG", "q", "label", "R",
        "err_R", "time_ms", "time_est_ms"]
    with open(out_convergence, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=convergence_fields, extrasaction="ignore")
        w.writeheader()
        for row in convergence:
            w.writerow(row)

    print(f"\nConvergence threshold: |R - R_ref| <= {tolerance:.0e}")
    for row in convergence:
        if row["status"] == "converged":
            detail = (f"nG={row['nG']} (confirmed by {row['confirmed_by_nG']}), "
                      f"estimated {row['time_est_ms']:.2f} ms")
        elif row["status"] == "provisional_crossing":
            detail = f"provisional crossing at nG={row['nG']}"
        else:
            detail = "not reached"
        print(f"   {row['case']:<26} {row['column']:<22} {detail}")

    print(f"\nexported: {out_json}\n          {out_csv}"
          f"\n          {out_convergence}")


if __name__ == "__main__":
    main()
