"""Benchmark worker: runs the shared structure battery (benchmark/structures.py)
on ONE grcwa installation (GRCWA_MOD + PYTHONPATH) at a single order count and
prints the results as JSON.

Run by benchmark/run.py once per (variant, factorization-mode), in a separate
process so several grcwa versions can be compared without import clashes.

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

_MOD = os.environ.get("GRCWA_MOD", "grcwa")
grcwa = __import__(_MOD)
import numpy as np
import structures as ST

_FMM = os.environ.get("FMM", "none")
FMM = None if _FMM == "none" else _FMM
REPEAT = int(os.environ.get("REPEAT", "3"))
QAXIS = int(os.environ.get("GRCWA_Q", "11"))     # per-axis order count

NATIVE = ST.supports_native_dim(grcwa)


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
        R, T, nG, mode = ST.solve(grcwa, s, qarg, FMM, NATIVE)
    except Exception as e:
        return {"error": repr(e)}
    if R is None:
        return {"skipped": mode}
    best = np.inf
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        try:
            ST.solve(grcwa, s, qarg, FMM, NATIVE)
        except Exception as e:
            return {"error": repr(e)}
        best = min(best, time.perf_counter() - t0)
    return {"R": R, "T": T, "A": 1.0 - R - T, "nG": nG, "label": label,
            "time_ms": best * 1e3, "mode": mode}


results = {s["name"]: run_structure(s) for s in ST.STRUCTURES}
print(json.dumps(results))
