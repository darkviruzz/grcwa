"""Convergence vs the external 'Moose' reference (benchmark/moose_reference.json).

Runs grcwa (Laurent and the fixed Pol) on the same A/B structures Moose was run
on, and overlays all three converging to the Moose nG=500 reference. Writes
moose_compare_error.png (error |R(nG) - R_Moose500| vs nG, log-log) and
moose_compare_raw.png (raw R(nG)), and prints a converged-value table.

Self-contained: the A/B structure registry mirrors conv_worker.py. Group C (2D)
is appended to moose_reference.json later.
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
NX = 2048

AIR, SIO2, SI, AU = (1.0, 0.0), (1.5, 0.0), (3.5, 0.0), (0.3, 7.0)


def eps(nk):
    n, k = nk
    return (n + 1j * k) ** 2


# A/B registry: name -> (dim, pol, params). pol 'p'=TM, 's'=TE. Mirrors conv_worker.
REG = {
    "A1_slab_air":      dict(dim=0, pol="s", film=SI, d=0.20, sub=AIR),
    "A1b_slab_glass":   dict(dim=0, pol="s", film=SI, d=0.20, sub=SIO2),
    "A2_formbiref_TE":  dict(dim=1, pol="s", hi=SI, lo=AIR, period=0.20, ff=0.5, d=0.30, sub=AIR),
    "A2_formbiref_TM":  dict(dim=1, pol="p", hi=SI, lo=AIR, period=0.20, ff=0.5, d=0.30, sub=AIR),
    "B1_Si_grating_TE": dict(dim=1, pol="s", hi=SI, lo=AIR, period=1.5, ff=0.5, d=0.50, sub=AIR),
    "B1_Si_grating_TM": dict(dim=1, pol="p", hi=SI, lo=AIR, period=1.5, ff=0.5, d=0.50, sub=AIR),
    "B2_HCG_TM":        dict(dim=1, pol="p", hi=SI, lo=AIR, period=0.80, ff=0.5, d=0.30, sub=AIR),
    "B3_Au_slits_TM":   dict(dim=1, pol="p", hi=AU, lo=AIR, period=0.50, ff=0.8, d=0.20, sub=AIR),
}


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
    xs = np.linspace(0, 1, NX, endpoint=False)
    prof = np.where(xs < s["ff"], eps(s["hi"]), eps(s["lo"])).astype(complex)
    o = grcwa.obj(nG, [s["period"], 0], None, FREQC, 0., 0., verbose=0, fmm_method=fmm)
    o.Add_LayerUniform(1.0, eps(AIR))
    o.Add_LayerGrid(s["d"], NX)
    o.Add_LayerUniform(1.0, eps(s["sub"]))
    o.Init_Setup()
    pa, sa = (1., 0.) if s["pol"] == "p" else (0., 1.)
    o.MakeExcitationPlanewave(pa, 0., sa, 0., order=0)
    o.GridLayer_geteps(prof)
    R, T = o.RT_Solve(normalize=1)
    return float(np.real(R)), int(o.nG)


M = json.load(open(os.path.join(HERE, "moose_reference.json")))
CASES = M["cases"]
FLOOR = 1e-7

# nG values to run grcwa at (match Moose's sweep where it makes sense)
NG_RUN = [5, 11, 21, 51, 101, 201, 401]

# gather grcwa data
gr = {}   # case -> {"laurent":[(nG,R)], "pol":[(nG,R)]}
print("Running grcwa (Laurent + fixed Pol) on the A/B structures ...")
for case in CASES:
    if REG[case]["dim"] == 0:
        rl, _ = run(case, 1, None)
        rp, _ = run(case, 1, "pol")
        gr[case] = {"laurent": [(1, rl)], "pol": [(1, rp)]}
        continue
    lau, pol = [], []
    for nG in NG_RUN:
        rl, n1 = run(case, nG, None)
        rp, n2 = run(case, nG, "pol")
        lau.append((n1, rl)); pol.append((n2, rp))
    gr[case] = {"laurent": lau, "pol": pol}

# ---- console table: converged values vs Moose ref ----
print("\n%-20s %10s %10s %10s   %10s %10s" %
      ("case", "Moose500", "grcwaLau", "grcwaPol", "Lau-Moose", "Pol-Moose"))
for case in CASES:
    ref = CASES[case]["ref"]
    rl = gr[case]["laurent"][-1][1]
    rp = gr[case]["pol"][-1][1]
    print("%-20s %10.6f %10.6f %10.6f   %+10.2e %+10.2e" %
          (case, ref, rl, rp, rl - ref, rp - ref))

# ---- sweep cases (have a real nG sweep in Moose) ----
sweep_cases = [c for c in CASES if len(CASES[c]["sweep"]) > 2]


def moose_sweep(case):
    d = CASES[case]["sweep"]
    pts = sorted((int(k), v) for k, v in d.items())
    return [p[0] for p in pts], [p[1] for p in pts]


# ===== Figure 1: error decay vs Moose-500 =====
ncol = 3
nrow = int(np.ceil(len(sweep_cases) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.0 * nrow), squeeze=False)
fig.suptitle("Convergence to the Moose nG=500 reference   |R(nG) - R_Moose500|",
             fontsize=13, fontweight="bold")
for i, case in enumerate(sweep_cases):
    ax = axes[i // ncol][i % ncol]
    ref = CASES[case]["ref"]
    mng, mr = moose_sweep(case)
    ax.loglog(mng, [max(abs(r - ref), FLOOR) for r in mr], "-D", color="#000000",
              ms=4, lw=1.6, label="Moose (self)")
    ng = [n for n, _ in gr[case]["laurent"]]
    ax.loglog(ng, [max(abs(r - ref), FLOOR) for _, r in gr[case]["laurent"]],
              "-o", color="#8172b3", ms=4, lw=1.6, label="grcwa Laurent")
    ax.loglog(ng, [max(abs(r - ref), FLOOR) for _, r in gr[case]["pol"]],
              "--s", color="#d62728", ms=4, lw=1.6, label="grcwa Pol (fixed)")
    ax.set_title(case, fontsize=9)
    ax.set_xlabel("nG", fontsize=8); ax.set_ylabel("|R - R_Moose500|", fontsize=8)
    ax.grid(True, which="both", alpha=0.3); ax.tick_params(labelsize=7)
    if i == 0:
        ax.legend(fontsize=7.5)
for j in range(len(sweep_cases), nrow * ncol):
    axes[j // ncol][j % ncol].axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/moose_compare_error.png", dpi=150, bbox_inches="tight")
plt.close()

# ===== Figure 2: raw R(nG) =====
fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.0 * nrow), squeeze=False)
fig.suptitle("Raw R(nG): grcwa Laurent / grcwa Pol(fixed) / Moose   "
             "(black dashed = Moose nG=500 reference)", fontsize=13, fontweight="bold")
for i, case in enumerate(sweep_cases):
    ax = axes[i // ncol][i % ncol]
    ref = CASES[case]["ref"]
    mng, mr = moose_sweep(case)
    ax.semilogx(mng, mr, "-D", color="#000000", ms=4, lw=1.6, label="Moose")
    ng = [n for n, _ in gr[case]["laurent"]]
    ax.semilogx(ng, [r for _, r in gr[case]["laurent"]], "-o", color="#8172b3",
                ms=4, lw=1.6, label="grcwa Laurent")
    ax.semilogx(ng, [r for _, r in gr[case]["pol"]], "--s", color="#d62728",
                ms=4, lw=1.6, label="grcwa Pol (fixed)")
    ax.axhline(ref, color="k", ls="--", lw=0.9, alpha=0.7)
    ax.set_title(case, fontsize=9)
    ax.set_xlabel("nG", fontsize=8); ax.set_ylabel("R", fontsize=8)
    ax.grid(True, which="both", alpha=0.3); ax.tick_params(labelsize=7)
    if i == 0:
        ax.legend(fontsize=7.5)
for j in range(len(sweep_cases), nrow * ncol):
    axes[j // ncol][j % ncol].axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/moose_compare_raw.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nfigures written to", OUT)
