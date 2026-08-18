"""Load a convergence run into tidy arrays.

Defaults to ``benchmark/night_run_2``; set ``GRCWA_CONV_RUN`` to point at
another snapshot directory containing ``conv_results.json``.
"""
import json
import os

import numpy as np

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.environ.get("GRCWA_CONV_RUN", os.path.join(BENCH, "night_run_2"))

with open(os.path.join(RUN, "conv_results.json")) as _f:
    J = json.load(_f)
with open(os.path.join(BENCH, "moose_reference.json")) as _f:
    MOOSE = json.load(_f)

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
    r = CASES[case].get("ref") or {}
    return r.get("R"), r.get("type", "?"), bool(r.get("ref_provisional"))

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
    """Moose sweep key -> total retained orders (matches plot_moose.parse_total)."""
    k = k.strip()
    if k.startswith("("):
        a, b = k.strip("()").split(",")
        return int(a) * int(b)
    return int(k)


def moose_series(case):
    m = MOOSE["cases"].get(case)
    if not m or "sweep" not in m:
        return None, None, None
    pairs = sorted((_mkey(k), v) for k, v in m["sweep"].items())
    return (np.array([p[0] for p in pairs], float),
            np.array([p[1] for p in pairs], float),
            m.get("ref"))

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
