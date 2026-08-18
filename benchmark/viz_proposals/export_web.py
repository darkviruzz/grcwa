"""Export a convergence run as compact JSON for the interactive explorer.

Writes ``figures/conv_web.json``: per case, per column, the retained-order grid,
R, |R - R_ref| and the modelled solve time -- everything the browser chart needs
and nothing else.
"""
import json
import os

import conv_data as D

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("GRCWA_VIZ_OUT", os.path.join(HERE, "figures"))
os.makedirs(OUTDIR, exist_ok=True)


def build():
    out = {"columns": D.COLUMNS, "ruleColor": D.RULE_COLOR, "cases": {}}
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
