"""C2 -- the cost staircase: best error reachable for a given wall-time budget."""
import os

import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import conv_data as D

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("GRCWA_VIZ_OUT", os.path.join(HERE, "figures"))
os.makedirs(OUTDIR, exist_ok=True)


def out(name):
    return os.path.join(OUTDIR, name)



plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
INK, MUTED = "#1b2733", "#5b6b7b"
TOL = [(1e-2, "1%"), (1e-3, "1e-3"), (1e-4, "1e-4")]
FLOOR = 1e-12


def panel(ax, case, xkey="time"):
    xmin, xmax = np.inf, 0
    for col in D.COLUMNS:
        nG, R, e, s, t, te = D.series(case, col)
        x = te if xkey == "time" else nG
        ok = np.isfinite(x) & (x > 0)
        x, e2 = x[ok], np.maximum(e[ok], FLOOR)
        if not len(x):
            continue
        st = D.style(col)
        xs, best = D.pareto(x, e2)
        ax.plot(x, e2, ls="none", marker=st["marker"], ms=3.0, mfc="none",
                mec=st["color"], mew=.8, alpha=.45, zorder=3)
        ax.step(xs, best, where="post", color=st["color"], ls=st["linestyle"],
                lw=2.2, zorder=5, solid_capstyle="round")
        xmin, xmax = min(xmin, xs.min()), max(xmax, xs.max())
        # first time the envelope is at or below 1e-4
        hit = np.nonzero(best <= 1e-4)[0]
        if len(hit):
            ax.plot([xs[hit[0]]], [1e-4], marker="v", ms=7, color=st["color"],
                    mec="white", mew=.8, zorder=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(xmin * .7, xmax * 1.5)
    lo, hi = ax.get_ylim()
    ax.set_ylim(max(lo, 1e-9), min(hi, 3))

    import matplotlib.transforms as mtr
    tb = mtr.blended_transform_factory(ax.transAxes, ax.transData)
    ylo, yhi = ax.get_ylim()
    for v, lab in TOL:
        if not (ylo < v < yhi):
            continue
        ax.axhline(v, color="#b9c7d4", lw=.8, ls=(0, (4, 3)), zorder=1)
        ax.text(.995, v, lab, va="center", ha="right", fontsize=7.2, color=MUTED,
                transform=tb, zorder=8,
                bbox=dict(fc="white", ec="none", alpha=.75, pad=.6))
    ax.axhspan(ylo, 1e-4, color="#eaf5ee", zorder=0)
    ax.grid(True, which="major", color="#eef2f6", lw=.7, zorder=0)
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3ced8")


fig, axes = plt.subplots(4, 3, figsize=(17.5, 13.6))
fig.subplots_adjust(left=.055, right=.955, top=.885, bottom=.085, hspace=.50, wspace=.30)
for ax in axes.flat:
    ax.axis("off")
for i, case in enumerate(D.ORDER):
    ax = axes.flat[i]; ax.axis("on")
    panel(ax, case)
    Rr, rt, prov = D.ref_of(case)
    ax.set_title(case + ("   (provisional ref)" if prov else ""), fontsize=10.5,
                 fontweight="bold", color=INK, loc="left", pad=6)
    if i // 3 == 3 or i >= len(D.ORDER) - 3:
        ax.set_xlabel("wall time of the solve [ms]", fontsize=8.6, color=MUTED)
    if i % 3 == 0:
        ax.set_ylabel("|R − R_ref|", fontsize=8.6, color=MUTED)

handles = [Line2D([], [], color=D.RULE_COLOR[D.rule_of(c)], ls=D.CODE_DASH[D.code_of(c)],
                  marker=D.CODE_MARK[D.code_of(c)], ms=5, lw=2.2, label=c)
           for c in D.COLUMNS]
handles += [Line2D([], [], color="#666", marker="v", ls="none", ms=7,
                   label="envelope first reaches 1e-4")]
fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9.6,
           bbox_to_anchor=(.5, .012))
fig.suptitle("C2 — the cost staircase: the best |ΔR| you can buy with a given wall-time budget",
             fontsize=17, fontweight="bold", color=INK, y=.966)
fig.text(.5, .925, "faint markers are the raw measurements; the thick step is the running "
         "best — a flat step means spending more time buys nothing\n"
         "colour = factorization rule (the physics), dash + marker = codebase",
         ha="center", fontsize=10.3, color=MUTED, linespacing=1.5)
fig.savefig(out("out_C2_cost_staircase.png"), dpi=112)
print("wrote out_C2_cost_staircase.png")
