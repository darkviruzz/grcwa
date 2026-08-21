"""Export a convergence run as compact JSON for the interactive explorer.

Writes ``figures/conv_web.json``: per case, per column, the retained-order grid,
R, |R - R_ref| and the modelled solve time -- everything the browser chart needs
and nothing else, plus a ``meta`` block describing the run itself.

That meta block is what keeps the page honest about its own provenance: the
plate book header quotes the solve count, the order range, the q schedule and
whether the sweep had finished, all measured here rather than typed into the
template.
"""
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json

import viz_conv as D

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("GRCWA_VIZ_OUT", os.path.join(HERE, "figures"))
os.makedirs(OUTDIR, exist_ok=True)


def meta():
    """Describe the run the figures were drawn from."""
    m = D.J.get("meta", {})
    orders = m.get("order_config", {})
    q = orders.get("resolved_q2d") or []
    solves = sum(len(pts) for case in D.CASES.values()
                 for pts in case.get("columns", {}).values())
    every = [p["nG"] for case in D.CASES.values()
             for pts in case.get("columns", {}).values() for p in pts if "nG" in p]
    drift = sorted(c for c in D.CASES if D.ref_is_stale(c))
    return {
        "convJson": os.path.abspath(D.CONV_JSON),
        "mooseJson": os.path.abspath(D.MOOSE_JSON),
        "built": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "structures": len(D.CASES),
        "columns": len(D.COLUMNS),
        "solves": solves,
        "orderMin": min(every) if every else None,
        "orderMax": max(every) if every else None,
        "qMax": max(q) if q else None,
        "complete": bool(m.get("complete")),
        "tolerance": m.get("convergence_tolerance"),
        "refDrift": drift,
    }


def build():
    out = {"meta": meta(), "columns": D.COLUMNS, "ruleColor": D.RULE_COLOR,
           "cases": {}}
    for case in D.ORDER:
        ref, ref_type, prov = D.ref_of(case)
        info = D.CASES[case]["info"]
        entry = {"ref": ref, "refType": ref_type, "prov": bool(prov),
                 "desc": info["desc"], "dim": info["dim"], "group": info["group"],
                 "cols": {}}
        for col in D.COLUMNS:
            nG, R, err, _sgn, _traw, test = D.series(case, col)
            entry["cols"][col] = dict(
                nG=[int(v) for v in nG],
                R=[round(float(v), 9) for v in R],
                err=[float("%.4g" % v) for v in err],
                t=[float("%.4g" % v) for v in test])
        mx, mR, _mref = D.moose_series(case)
        if mx is not None:
            entry["moose"] = dict(nG=[int(v) for v in mx],
                                  R=[round(float(v), 9) for v in mR])
        out["cases"][case] = entry
    return out


if __name__ == "__main__":
    path = os.path.join(OUTDIR, "conv_web.json")
    with open(path, "w") as f:
        json.dump(build(), f, separators=(",", ":"))
    print("wrote", path)
