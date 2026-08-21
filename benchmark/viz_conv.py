"""A convergence run (``conv_results.json``) as tidy arrays.

ONE source per input, no fallback chain: each path is the environment variable
if set, otherwise the location ``run_overnight.bat`` and ``moose_csv_to_json.py``
actually write to.  A missing file is an error naming the command that produces
it -- the figures never silently fall back to an older run, because a plate that
quietly plots stale data is worse than no plate.

    GRCWA_CONV_JSON           benchmark/conv_results.json    <- conv_run.py
    GRCWA_MOOSE_JSON          benchmark/moose_reference.json <- moose_csv_to_json.py
    GRCWA_MOOSE_TIMING_JSON   sibling moose_timing.json      <- moose_csv_to_json.py

``conv_run.py`` rewrites ``conv_results.json`` in place at the end of every
stage, so reading it while a sweep is running can catch a half-written file.
``_load`` retries a truncated read rather than failing the whole build.
"""
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONV_JSON = os.environ.get("GRCWA_CONV_JSON") or os.path.join(
    HERE, "conv_results.json")
MOOSE_JSON = os.environ.get("GRCWA_MOOSE_JSON") or os.path.join(
    HERE, "moose_reference.json")
MOOSE_TIMING_JSON = os.environ.get("GRCWA_MOOSE_TIMING_JSON") or os.path.join(
    os.path.dirname(os.path.abspath(MOOSE_JSON)), "moose_timing.json")

# kept as an alias: the plotters print the run they drew
JSON = CONV_JSON

_PRODUCER = {
    CONV_JSON: "python benchmark/conv_run.py   (or benchmark/run_overnight.bat)",
    MOOSE_JSON: "python benchmark/moose/moose_csv_to_json.py <moose_conv.csv>",
}


def _load(path, required=True, attempts=5, pause=0.4):
    """Read a JSON input, tolerating a concurrent rewrite.

    conv_run.py truncates and rewrites conv_results.json without an atomic
    rename, so a read that lands inside that window sees invalid JSON.  That is
    a transient of a few milliseconds -- retry it instead of failing a build the
    user started while the sweep was between stages.
    """
    if not os.path.exists(path):
        if not required:
            return None
        hint = _PRODUCER.get(path)
        raise SystemExit(
            "missing input: %s\n%s" % (
                path, ("produce it with:\n    %s" % hint) if hint else
                "set the matching GRCWA_* variable to an existing file"))
    last = None
    for i in range(attempts):
        try:
            with open(path) as stream:
                return json.load(stream)
        except ValueError as exc:      # truncated mid-write -- let it settle
            last = exc
            if i + 1 < attempts:
                time.sleep(pause)
    raise SystemExit(
        "could not read %s after %d attempts: %s\n"
        "If a sweep is running, wait for the stage to finish and retry."
        % (path, attempts, last))


J = _load(CONV_JSON)
MOOSE = _load(MOOSE_JSON)
_timing = _load(MOOSE_TIMING_JSON, required=False)
MOOSE_TIMING = (_timing or {}).get("cases", {})

COLUMNS = J["columns"]
CASES = J["cases"]
CONV = J.get("convergence", {})

# ---- palette: colour encodes the FACTORIZATION RULE (the physics), ----------
# dash/marker encodes the codebase.  Okabe-Ito, colour-blind safe.
RULE_COLOR = {"Laurent": "#8a8f98", "Pol": "#D55E00", "Li": "#0072B2", "NV": "#009E73"}
CODE_DASH  = {"fork": (0, ()), "ikarus": (0, (5, 2))}
CODE_MARK  = {"fork": "o", "ikarus": "^"}
MOOSE_C    = "#111418"

def rule_of(col):  return col.split("[")[-1].rstrip("]")
def code_of(col):  return col.split("[")[0]

def style(col):
    return dict(color=RULE_COLOR[rule_of(col)], linestyle=CODE_DASH[code_of(col)],
                marker=CODE_MARK[code_of(col)])

def label(col):
    return col

# ---- per-case reference -----------------------------------------------------
def ref_of(case):
    """(R_ref, type, provisional) for a case.

    A run bakes its reference into conv_results.json, but moose_reference.json
    keeps improving independently of the sweep.  When a case is already judged
    against Moose, the CURRENT Moose value wins, so the figures do not keep
    quoting a reference the reference file has since replaced.  Which case is
    judged against what stays the run's decision -- this only refreshes the
    number, never the choice.
    """
    r = CASES[case].get("ref") or {}
    R, kind = r.get("R"), r.get("type", "?")
    prov = bool(r.get("ref_provisional"))
    if kind == "external_moose":
        live = MOOSE.get("cases", {}).get(case, {})
        if live.get("ref") is not None:
            R = live["ref"]
            prov = prov or bool(live.get("ref_provisional"))
    return R, kind, prov


def ref_is_stale(case, rtol=1e-6):
    """True when the baked reference and the current Moose value disagree."""
    baked = (CASES[case].get("ref") or {}).get("R")
    live, kind, _ = ref_of(case)
    if kind != "external_moose" or baked is None or live is None:
        return False
    return abs(live - baked) > rtol * max(1.0, abs(baked))

def series(case, col):
    """(nG, R, |err|, signed err, time_ms, time_est_ms) sorted by nG."""
    pts = [p for p in CASES[case]["columns"].get(col, []) if "R" in p]
    pts.sort(key=lambda p: p["nG"])
    Rref, _, _ = ref_of(case)
    nG = np.array([p["nG"] for p in pts], float)
    R = np.array([p["R"] for p in pts], float)
    t = np.array([p.get("time_ms", np.nan) for p in pts], float)
    te = np.array([p.get("time_est_ms", np.nan) for p in pts], float)
    sgn = R - Rref
    return nG, R, np.abs(sgn), sgn, t, te

def _mkey(k):
    """Moose sweep key -> total retained orders.

    The keys are Moose's MAXIMUM order m, not an order count: it retains 2m+1
    harmonics per axis, so a 1D key "m" is 2m+1 orders and a 2D key "(mx,my)"
    is (2mx+1)(2my+1).  Same convention as plot_moose.parse_total; verified
    against the nG that moose_timing.json records per key.
    """
    k = k.strip()
    if k.startswith("("):
        a, b = k.strip("()").split(",")
        return (2 * int(a) + 1) * (2 * int(b) + 1)
    return 2 * int(k) + 1


def moose_series(case):
    """(nG, R, ref) for the external Moose sweep, or (None, None, None)."""
    m = MOOSE["cases"].get(case)
    if not m or "sweep" not in m:
        return None, None, None
    pairs = sorted((_mkey(k), v) for k, v in m["sweep"].items())
    return (np.array([p[0] for p in pairs], float),
            np.array([p[1] for p in pairs], float),
            m.get("ref"))


def moose_timed(case):
    """(time_ms, R) for the Moose points that carry a wall time.

    moose_timing.json records t_solve_s per (case, max order); only the keys it
    covers can appear on a cost axis, which is usually fewer than the sweep.
    """
    m = MOOSE["cases"].get(case)
    timing = MOOSE_TIMING.get(case)
    if not m or not timing:
        return None, None
    pairs = []
    for key, R in m["sweep"].items():
        entry = timing.get(key)
        if not entry:
            continue
        t = entry.get("t_solve_s")
        if t is None or t <= 0:
            continue
        pairs.append((t * 1e3, R))
    if not pairs:
        return None, None
    pairs.sort()
    return (np.array([p[0] for p in pairs], float),
            np.array([p[1] for p in pairs], float))

def pareto(t, e):
    """Running-minimum error against increasing cost -> the honest cost curve."""
    o = np.argsort(t)
    t2, e2 = t[o], e[o]
    best = np.minimum.accumulate(e2)
    return t2, best

PATTERNED = [c for c in CASES if CASES[c]["info"]["dim"] > 0]
ZERO_D = [c for c in CASES if CASES[c]["info"]["dim"] == 0]
ORDER = [c for c in ["A2_formbiref_TE", "A2_formbiref_TM", "B1_Si_grating_TE",
                     "B1_Si_grating_TM", "B2_HCG_TM", "B3_Au_slits_TM",
                     "C1_Si_pillars", "C1b_Si_pillars_diffract", "C2_Au_holes",
                     "D1_ikarus_hcg_TM", "D2_ikarus_cylinder_TE"] if c in CASES]


def arrival(case, col, tol=1e-4):
    """First SUSTAINED point within tol (2 consecutive), like conv_run.py.
    Returns (nG, time_est_ms, status) with status in
    {"sustained", "provisional"}, or None if the sweep never gets there."""
    nG, R, e, s, t, te = series(case, col)
    for i in range(len(e) - 1):
        if e[i] <= tol and e[i + 1] <= tol:
            return nG[i], te[i], "sustained"
    hit = np.nonzero(e <= tol)[0]
    if len(hit):
        i = hit[0]
        return nG[i], te[i], "provisional"
    return None
