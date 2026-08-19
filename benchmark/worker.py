"""Benchmark worker: runs the shared structure battery (benchmark/structures.py)
on ONE solver installation at a single order count and prints the results as JSON.

Run by benchmark/run.py once per (variant, factorization-mode), in a separate
process so several versions can be compared without import clashes.

Which solver is selected by SUITE:
  * ``grcwa`` (default) -- the grcwa install named by GRCWA_MOD + PYTHONPATH,
    with FMM = none|pol choosing Laurent or the Pol rule;
  * ``ikarus``          -- the independent Ikarus code via benchmark/ikarus_suite.py,
    with FMM = laurent|li|normal choosing its factorization.

All structures are evaluated at ~the same TOTAL order count for a fair
cross-version comparison: with per-axis count q (env GRCWA_Q, default 11),
  * 2D structures use a (q,q) square block      -> q**2 orders,
  * 1D structures use nG = q**2                  -> q**2 orders (well converged),
  * 0D uses nG = 1.
Units: lambda = 1 um (freq = 1); lengths in um. Materials as (n,k).
"""
import os
import json
import time

import numpy as np
import structures as ST

_SUITE = os.environ.get("SUITE", "grcwa")
_FMM = os.environ.get("FMM", "none")
FMM = None if _FMM == "none" else _FMM
REPEAT = int(os.environ.get("REPEAT", "3"))
QAXIS = int(os.environ.get("GRCWA_Q", "11"))     # per-axis order count

if _SUITE == "ikarus":
    import ikarus_suite as IK

    if not IK.available():
        # Optional cross-check dependency: report it once, cleanly, instead of
        # letting every case fail with its own ImportError.
        print(json.dumps({"_error": "ikarus not installed"}))
        raise SystemExit(0)

    def solve(s, q):
        return IK.solve(s, q, FMM)
else:
    grcwa = __import__(os.environ.get("GRCWA_MOD", "grcwa"))
    NATIVE = ST.supports_native_dim(grcwa)

    def solve(s, q):
        return ST.solve(grcwa, s, q, FMM, NATIVE)


def _qarg_label(s):
    """(q passed to structures.solve, display label) for this structure."""
    if s["dim"] == 0:
        return 1, "(0D)"
    if s["dim"] == 1:
        return QAXIS * QAXIS, str(QAXIS * QAXIS)        # 1D total-matched
    return QAXIS, "(%d,%d)" % (QAXIS, QAXIS)            # 2D square block


def run_structure(s):
    qarg, label = _qarg_label(s)
    try:
        R, T, nG, mode = solve(s, qarg)
    except Exception as e:
        return {"error": repr(e)}
    if R is None:
        return {"skipped": mode}
    best = np.inf
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        try:
            solve(s, qarg)
        except Exception as e:
            return {"error": repr(e)}
        best = min(best, time.perf_counter() - t0)
    return {"R": R, "T": T, "A": 1.0 - R - T, "nG": nG, "label": label,
            "time_ms": best * 1e3, "mode": mode}


results = {s["name"]: run_structure(s) for s in ST.STRUCTURES}
print(json.dumps(results))
