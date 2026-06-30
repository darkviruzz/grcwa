"""Convergence vs the external 'Moose' reference (benchmark/moose_reference.json).

Runs grcwa (Laurent and the fixed Pol) on the same structures Moose was run on and
overlays all three converging to the Moose reference. Writes moose_compare_error.png
(error |R(nG) - R_ref| vs total retained orders, log-log) and moose_compare_raw.png
(raw R), and prints a converged-value table.

Fully dynamic w.r.t. moose_reference.json: each case's `sweep` may use 1D keys
("50") or 2D keys ("(nGx,nGy)"); the x-axis is the total retained order count. For
2D, (nGx,nGy) is read as the max order per axis, i.e. (2*nGx+1)*(2*nGy+1) orders.
The reference is the case's `ref` field, or (if absent) the highest-order sweep
value -- so appending more Moose points later just works. grcwa is run at each
Moose order count up to GRCWA_MAX_NG (env, default 500); beyond that only Moose is
plotted. The A/B/C structure registry mirrors conv_worker.py.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import grcwa

grcwa.set_backend("numpy")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
FREQC = 1.0 * (1 + 1j / 2 / 1e7)
NX_1D, NX_2D = 2048, 256
MAX_NG = int(os.environ.get("GRCWA_MAX_NG", "500"))

AIR, SIO2, SI, AU = (1.0, 0.0), (1.5, 0.0), (3.5, 0.0), (0.3, 7.0)


def eps(nk):
    n, k = nk
    return (n + 1j * k) ** 2


# name -> geometry. pol 'p'=TM, 's'=TE. Mirrors conv_worker.py.
REG = {
    "A1_slab_air":      dict(dim=0, pol="s", film=SI, d=0.20, sub=AIR),
    "A1b_slab_glass":   dict(dim=0, pol="s", film=SI, d=0.20, sub=SIO2),
    "A2_formbiref_TE":  dict(dim=1, pol="s", hi=SI, lo=AIR, period=0.20, ff=0.5, d=0.30, sub=AIR),
    "A2_formbiref_TM":  dict(dim=1, pol="p", hi=SI, lo=AIR, period=0.20, ff=0.5, d=0.30, sub=AIR),
    "B1_Si_grating_TE": dict(dim=1, pol="s", hi=SI, lo=AIR, period=1.5, ff=0.5, d=0.50, sub=AIR),
    "B1_Si_grating_TM": dict(dim=1, pol="p", hi=SI, lo=AIR, period=1.5, ff=0.5, d=0.50, sub=AIR),
    "B2_HCG_TM":        dict(dim=1, pol="p", hi=SI, lo=AIR, period=0.80, ff=0.5, d=0.30, sub=AIR),
    "B3_Au_slits_TM":   dict(dim=1, pol="p", hi=AU, lo=AIR, period=0.50, ff=0.8, d=0.20, sub=AIR),
    "C1_Si_pillars":         dict(dim=2, pol="s", pillar=SI, bg=AIR, period=0.50, ax=0.30, ay=0.30, d=0.40, sub=SIO2),
    "C1b_Si_pillars_diffract": dict(dim=2, pol="s", pillar=SI, bg=AIR, period=1.50, ax=0.60, ay=0.60, d=0.40, sub=SIO2),
    "C2_Au_holes":           dict(dim=2, pol="s", pillar=AIR, bg=AU, period=0.60, ax=0.30, ay=0.30, d=0.20, sub=SIO2),
}


def total_orders(key):
    """Total retained orders for a Moose sweep key. '50' -> 50;
    '(nGx,nGy)' -> (2*nGx+1)*(2*nGy+1)."""
    key = key.strip()
    if key.startswith("("):
        a, b = key.strip("()").split(",")
        return (2 * int(a) + 1) * (2 * int(b) + 1)
    return int(key)


def run(case, nG, fmm):
    s = REG[case]
    if s["dim"] == 0:
        o = grcwa.obj(1, None, None, FREQC, 0., 0., verbose=0, fmm_method=fmm)
        o.Add_LayerUniform(1.0, eps(AIR))
        o.Add_LayerUniform(s["d"], eps(s["film"]))
        o.Add_LayerUniform(1.0, eps(s["sub"]))
        o.Init_Setup()
        o.MakeExcitationPlanewave(0., 0., 1., 0., order=0)
        R, T = o.RT_Solve(normalize=1)
        return float(np.real(R)), 1
    if s["dim"] == 1:
        xs = np.linspace(0, 1, NX_1D, endpoint=False)
        prof = np.where(xs < s["ff"], eps(s["hi"]), eps(s["lo"])).astype(complex)
        o = grcwa.obj(nG, [s["period"], 0], None, FREQC, 0., 0., verbose=0, fmm_method=fmm)
        o.Add_LayerUniform(1.0, eps(AIR))
        o.Add_LayerGrid(s["d"], NX_1D)
        o.Add_LayerUniform(1.0, eps(s["sub"]))
        flat = prof
    else:
        x = np.linspace(0, 1, NX_2D, endpoint=False)
        X, Y = np.meshgrid(x, x, indexing="ij")
        eg = np.ones((NX_2D, NX_2D), dtype=complex) * eps(s["bg"])
        inside = (np.abs(X - 0.5) < s["ax"] / (2 * s["period"])) & \
                 (np.abs(Y - 0.5) < s["ay"] / (2 * s["period"]))
        eg[inside] = eps(s["pillar"])
        o = grcwa.obj(nG, [s["period"], 0], [0, s["period"]], FREQC, 0., 0.,
                      verbose=0, fmm_method=fmm)
        o.Add_LayerUniform(1.0, eps(AIR))
        o.Add_LayerGrid(s["d"], NX_2D, NX_2D)
        o.Add_LayerUniform(1.0, eps(s["sub"]))
        flat = eg.flatten()
    o.Init_Setup()
    pa, sa = (1., 0.) if s["pol"] == "p" else (0., 1.)
    o.MakeExcitationPlanewave(pa, 0., sa, 0., order=0)
    o.GridLayer_geteps(flat)
    R, T = o.RT_Solve(normalize=1)
    return float(np.real(R)), int(o.nG)


M = json.load(open(os.path.join(HERE, "moose_reference.json")))
CASES = {k: v for k, v in M["cases"].items() if k in REG}
FLOOR = 1e-7


def moose_points(case):
    """Sorted [(total_orders, R)] for a case's Moose sweep."""
    return sorted((total_orders(k), v) for k, v in CASES[case]["sweep"].items())


def ref_of(case):
    c = CASES[case]
    if "ref" in c and c["ref"] is not None:
        return c["ref"]
    return moose_points(case)[-1][1]   # highest-order value


# gather grcwa data at the Moose order counts (capped)
print("Running grcwa (Laurent + fixed Pol); MAX_NG=%d ..." % MAX_NG)
gr = {}
for case in CASES:
    dim = REG[case]["dim"]
    if dim == 0:
        gr[case] = {"laurent": [(1, run(case, 1, None)[0])],
                    "pol": [(1, run(case, 1, "pol")[0])]}
        continue
    lau, pol = [], []
    for tot, _ in moose_points(case):
        if tot > MAX_NG:
            continue
        rl, n1 = run(case, tot, None)
        rp, n2 = run(case, tot, "pol")
        lau.append((n1, rl)); pol.append((n2, rp))
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
        gl = gr[case]["laurent"]; gp = gr[case]["pol"]
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
