"""Convergence vs the external 'Moose' reference (benchmark/moose_reference.json).

Runs grcwa (Laurent and the fixed Pol) on the same structures Moose was run on
(via the shared benchmark/structures.py) and overlays all three converging to the
Moose reference. Writes moose_compare_error.png and moose_compare_raw.png and
prints a converged-value table.

Fully dynamic w.r.t. moose_reference.json: each case's `sweep` may use 1D keys
("50") or 2D keys ("(m,m)"). Convention (matching structures.py): the value is the
PER-AXIS order count, so a 1D key N -> N total orders and a 2D key (m,m) -> m*m
total orders (a m x m square block). The x-axis is the total retained order count.
The reference is the case's `ref` field, or (if absent) the highest-order sweep
value, so appending more Moose points later needs no code change. grcwa is run at
each Moose order count up to GRCWA_MAX_NG (env, default 700); beyond that only
Moose is plotted.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import grcwa
import structures as ST

grcwa.set_backend("numpy")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
MAX_NG = int(os.environ.get("GRCWA_MAX_NG", "700"))
FLOOR = 1e-7


def parse_key(key):
    """(per-axis q, total orders) for a Moose sweep key."""
    key = key.strip()
    if key.startswith("("):
        a, b = key.strip("()").split(",")
        m = int(a)
        return m, int(a) * int(b)      # m x m square block -> m*n total
    n = int(key)
    return n, n                         # 1D: n orders


M = json.load(open(os.path.join(HERE, "moose_reference.json")))
CASES = {k: v for k, v in M["cases"].items() if k in ST.STRUCT}


def moose_points(case):
    return sorted((parse_key(k)[1], v) for k, v in CASES[case]["sweep"].items())


def ref_of(case):
    c = CASES[case]
    if c.get("ref") is not None:
        return c["ref"]
    return moose_points(case)[-1][1]


# run grcwa at the Moose order counts (capped)
print("Running grcwa (Laurent + fixed Pol) via structures.py; MAX_NG=%d ..." % MAX_NG)
gr = {}
for case in CASES:
    s = ST.STRUCT[case]
    if s["dim"] == 0:
        rl = ST.solve(grcwa, s, 1, None, True)[0]
        rp = ST.solve(grcwa, s, 1, "pol", True)[0]
        gr[case] = {"laurent": [(1, rl)], "pol": [(1, rp)]}
        continue
    lau, pol = [], []
    seen = set()
    for k in CASES[case]["sweep"]:
        q, tot = parse_key(k)
        if tot > MAX_NG or q in seen:
            continue
        seen.add(q)
        rl, T, n1, _ = ST.solve(grcwa, s, q, None, True)
        rp, T, n2, _ = ST.solve(grcwa, s, q, "pol", True)
        lau.append((n1, rl)); pol.append((n2, rp))
    lau.sort(); pol.sort()
    gr[case] = {"laurent": lau, "pol": pol}

# console table
print("\n%-24s %10s %10s %10s   %10s %10s" %
      ("case", "ref", "grcwaLau", "grcwaPol", "Lau-ref", "Pol-ref"))
for case in CASES:
    ref = ref_of(case)
    rl = gr[case]["laurent"][-1][1]
    rp = gr[case]["pol"][-1][1]
    flag = " (prov.ref)" if CASES[case].get("ref_provisional") else ""
    print("%-24s %10.6f %10.6f %10.6f   %+10.2e %+10.2e%s" %
          (case, ref, rl, rp, rl - ref, rp - ref, flag))

sweep_cases = [c for c in CASES if len(CASES[c]["sweep"]) > 2]
ncol = 3
nrow = int(np.ceil(len(sweep_cases) / ncol))


def grid_fig(kind):
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.0 * nrow), squeeze=False)
    for i, case in enumerate(sweep_cases):
        ax = axes[i // ncol][i % ncol]
        ref = ref_of(case)
        mp = moose_points(case)
        mx, mr = [p[0] for p in mp], [p[1] for p in mp]
        gl, gp = gr[case]["laurent"], gr[case]["pol"]
        if kind == "error":
            ax.loglog(mx, [max(abs(r - ref), FLOOR) for r in mr], "-D", color="k",
                      ms=4, lw=1.6, label="Moose (self)")
            if gl:
                ax.loglog([n for n, _ in gl], [max(abs(r - ref), FLOOR) for _, r in gl],
                          "-o", color="#8172b3", ms=4, lw=1.6, label="grcwa Laurent")
                ax.loglog([n for n, _ in gp], [max(abs(r - ref), FLOOR) for _, r in gp],
                          "--s", color="#d62728", ms=4, lw=1.6, label="grcwa Pol (fixed)")
            ax.set_ylabel("|R - R_ref|", fontsize=8)
        else:
            ax.semilogx(mx, mr, "-D", color="k", ms=4, lw=1.6, label="Moose")
            if gl:
                ax.semilogx([n for n, _ in gl], [r for _, r in gl], "-o", color="#8172b3",
                            ms=4, lw=1.6, label="grcwa Laurent")
                ax.semilogx([n for n, _ in gp], [r for _, r in gp], "--s", color="#d62728",
                            ms=4, lw=1.6, label="grcwa Pol (fixed)")
            ax.axhline(ref, color="k", ls="--", lw=0.9, alpha=0.6)
            ax.set_ylabel("R", fontsize=8)
        prov = "  (prov. ref)" if CASES[case].get("ref_provisional") else ""
        ax.set_title(case + prov, fontsize=9)
        ax.set_xlabel("total retained orders", fontsize=8)
        ax.grid(True, which="both", alpha=0.3); ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7.5)
    for j in range(len(sweep_cases), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    return fig


f1 = grid_fig("error")
f1.suptitle("Convergence to the Moose reference   |R(N) - R_ref|   "
            "(x = total retained orders)", fontsize=13, fontweight="bold")
f1.tight_layout()
f1.savefig(f"{OUT}/moose_compare_error.png", dpi=150, bbox_inches="tight")

f2 = grid_fig("raw")
f2.suptitle("Raw R: grcwa Laurent / grcwa Pol(fixed) / Moose   "
            "(black dashed = Moose reference)", fontsize=13, fontweight="bold")
f2.tight_layout()
f2.savefig(f"{OUT}/moose_compare_raw.png", dpi=150, bbox_inches="tight")

print("\nfigures written to", OUT)
