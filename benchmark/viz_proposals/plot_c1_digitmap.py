"""C1 -- the DIGIT MAP: convergence as 'how many digits of R are right', as a
heat strip along the cost axis.  One glance per case: green/yellow = converged."""
import os

import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.cm as cm
import conv_data as D

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("GRCWA_VIZ_OUT", os.path.join(HERE, "figures"))
os.makedirs(OUTDIR, exist_ok=True)


def out(name):
    return os.path.join(OUTDIR, name)



plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
INK, MUTED = "#1b2733", "#5b6b7b"

LEV = [0, 1, 2, 3, 4, 5, 6, 9]
COLS = ["#3b2f5e", "#3d5a8a", "#3f86a0", "#4fae9b", "#8fc97a", "#d9e04f", "#f7e96b"]
CMAP = ListedColormap(COLS)
CMAP.set_bad("#eef1f4")
NORM = BoundaryNorm(LEV, CMAP.N)


def digits(err):
    with np.errstate(divide="ignore"):
        return np.clip(-np.log10(np.maximum(err, 1e-16)), 0, 8.99)


def strip(ax, x, dg, y0, h):
    """Draw one column's digit strip; cells span consecutive x samples."""
    xe = np.empty(len(x) + 1)
    lx = np.log10(x)
    xe[1:-1] = 10 ** ((lx[:-1] + lx[1:]) / 2)
    xe[0] = 10 ** (lx[0] - (lx[1] - lx[0]) / 2)
    xe[-1] = 10 ** (lx[-1] + (lx[-1] - lx[-2]) / 2)
    for i, v in enumerate(dg):
        ax.add_patch(Rectangle((xe[i], y0), xe[i + 1] - xe[i], h,
                               fc=CMAP(NORM(v)), ec="none", zorder=3))
    return xe[0], xe[-1]


def panel(ax, case, xaxis="orders", tol=1e-4):
    lo, hi = np.inf, 0
    for j, col in enumerate(D.COLUMNS):
        nG, R, e, s, t, te = D.series(case, col)
        if not len(nG):
            continue
        x = nG if xaxis == "orders" else te
        ok = np.isfinite(x) & (x > 0)
        a, b = strip(ax, x[ok], digits(e[ok]), -j - .86, .74)
        lo, hi = min(lo, a), max(hi, b)
        # first sustained crossing of the tolerance (2 consecutive points)
        good = e <= tol
        idx = next((i for i in range(len(good) - 1) if good[i] and good[i + 1]), None)
        if idx is not None and np.isfinite(x[idx]):
            ax.plot([x[idx]], [-j - .49], marker="v", ms=6.5, color="#111418",
                    zorder=6, clip_on=False)
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(-len(D.COLUMNS), 0)
    ax.set_yticks([-j - .49 for j in range(len(D.COLUMNS))])
    ax.set_yticklabels(D.COLUMNS, fontsize=8)
    for j, col in enumerate(D.COLUMNS):
        ax.get_yticklabels()[j].set_color(D.RULE_COLOR[D.rule_of(col)])
        ax.get_yticklabels()[j].set_fontweight("bold")
    ax.tick_params(axis="x", labelsize=8, length=3)
    ax.grid(axis="x", color="#ffffff", lw=.7, alpha=.55, zorder=5)
    for sp in ax.spines.values():
        sp.set_color("#c3ced8"); sp.set_linewidth(.7)


def figure(xaxis, fname, xlabel, headline, sub):
    cases = D.ORDER
    fig, axes = plt.subplots(4, 3, figsize=(17.5, 13.4))
    fig.subplots_adjust(left=.105, right=.985, top=.892, bottom=.105,
                        hspace=.60, wspace=.30)
    for ax in axes.flat:
        ax.axis("off")
    for i, case in enumerate(cases):
        ax = axes.flat[i]; ax.axis("on")
        panel(ax, case, xaxis)
        Rr, rt, prov = D.ref_of(case)
        ax.set_title(case + ("   (provisional ref)" if prov else ""), fontsize=10,
                     fontweight="bold", color=INK, loc="left", pad=6)
        ax.set_xlabel(xlabel, fontsize=8.2, color=MUTED, labelpad=2)
    cax = fig.add_axes([.30, .042, .40, .015])
    cb = fig.colorbar(cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax,
                      orientation="horizontal", spacing="proportional")
    cb.set_ticks([0, 1, 2, 3, 4, 5, 6, 9])
    cb.set_ticklabels(["0", "1", "2", "3", "4", "5", "6", "≥6"])
    cb.set_label("correct digits of R    (−log₁₀ |R − R_ref|)", fontsize=9.5, color=INK)
    cb.ax.tick_params(labelsize=8.5, length=2)
    fig.text(.735, .048, "▼ = first sustained |ΔR| ≤ 1e-4", fontsize=9, color="#111418")
    fig.suptitle(headline, fontsize=17, fontweight="bold", color=INK, y=.968)
    fig.text(.5, .930, sub, ha="center", fontsize=10.5, color=MUTED)
    fig.savefig(out(fname), dpi=112)
    print("wrote", fname)


figure("orders", "out_C1_digitmap_orders.png", "total retained orders",
       "C1 — the digit map: how many digits of R are right, vs truncation order",
       "one row per code+rule, one colour step per correct digit — a rule that never turns yellow never converges")
figure("time", "out_C1_digitmap_time.png", "wall time of that solve [ms]",
       "C1t — the digit map on the COST axis: digits of R bought per millisecond",
       "same data, x = wall time instead of order count — this is 'convergence over time' at a glance")
