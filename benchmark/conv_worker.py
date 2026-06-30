"""Convergence-study worker (order sweep on ONE grcwa install + ONE factorization).

For the grcwa selected by GRCWA_MOD/PYTHONPATH and FMM (none|pol), sweep the
per-axis order count q over the shared battery (benchmark/structures.py) and emit
R(q) as JSON. Driven by conv_run.py once per (variant, mode), in a subprocess.

Order convention (see structures.py): 1D uses nG=q AND nG=q**2 (per-axis and
total points); 2D uses a (q,q) square block (nG=q**2). The plot x-axis is the
total retained order count.

Env:
  GRCWA_QLIST   comma list of per-axis q (default "1,3,5,7,9,13,17,21,25")
  GRCWA_MAX2D   cap: skip 2D points whose total q**2 exceeds this (default 700)
"""
import os
import sys
import json
import time

_MOD = os.environ.get("GRCWA_MOD", "grcwa")
grcwa = __import__(_MOD)
import numpy as np
import structures as ST

_FMM = os.environ.get("FMM", "none")
FMM = None if _FMM == "none" else _FMM
REPEAT = int(os.environ.get("REPEAT", "2"))
QLIST = [int(x) for x in os.environ.get("GRCWA_QLIST", "1,3,5,7,9,13,17,21,25").split(",")]
MAX2D = int(os.environ.get("GRCWA_MAX2D", "700"))

NATIVE = ST.supports_native_dim(grcwa)


def _timed(s, q, label):
    try:
        out = ST.solve(grcwa, s, q, FMM, NATIVE)
    except Exception as e:
        return {"q": q, "label": label, "error": repr(e)}
    R, T, nG, mode = out
    if R is None:
        return {"q": q, "label": label, "skipped": mode}
    best = np.inf
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        try:
            ST.solve(grcwa, s, q, FMM, NATIVE)
        except Exception:
            break
        best = min(best, time.perf_counter() - t0)
    return {"q": q, "label": label, "nG": nG, "R": R, "T": T, "A": 1.0 - R - T,
            "time_ms": (best * 1e3) if best < np.inf else None, "mode": mode}


def run_structure(s):
    dim = s["dim"]
    sweep = []
    if dim == 0:
        sweep.append(_timed(s, 1, "(0D)"))
    elif dim == 1:
        # per-axis points (nG=q) and total points (nG=q**2)
        done = set()
        for q in QLIST:
            for nG, lab in ((q, str(q)), (q * q, str(q * q))):
                if nG in done:
                    continue
                done.add(nG)
                sweep.append(_timed(s, nG, lab))
        sweep.sort(key=lambda p: p["q"])
    else:  # 2D: (q,q) square blocks
        for q in QLIST:
            if q * q > MAX2D:
                continue
            sweep.append(_timed(s, q, "(%d,%d)" % (q, q)))
    info = {k: s[k] for k in ("group", "dim", "pol", "desc")}
    info["nk"] = {k: list(s[k]) for k in
                  ("hi", "lo", "film", "sub", "pillar", "bg") if k in s}
    for k in ("period", "ff", "d", "ax", "ay"):
        if k in s:
            info[k] = s[k]
    return {"info": info, "sweep": [p for p in sweep if "R" in p or "skipped" in p or "error" in p]}


results = {s["name"]: run_structure(s) for s in ST.STRUCTURES}
print(json.dumps(results))
