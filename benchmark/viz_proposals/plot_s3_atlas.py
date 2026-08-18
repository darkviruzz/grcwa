"""Style S3 -- the atlas: every structure on ONE shared length scale, 2 columns."""
import os

import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import palette as V

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("GRCWA_VIZ_OUT", os.path.join(HERE, "figures"))
os.makedirs(OUTDIR, exist_ok=True)


def out(name):
    return os.path.join(OUTDIR, name)



plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
INK, MUTED, BEAM = "#1b2733", "#5b6b7b", "#e8542f"
GROUP_C = {"A": "#4c7bb0", "B": "#3f9070", "C": "#b8762e", "D": "#8a5aa8"}
WIN, ROW, HAIR, HSUB = 3.30, 0.88, 0.20, 0.17

G = sorted(V.ALL, key=lambda g: (g["period"] or 0.0))
split = 7
COLS = [G[:split], G[split:]]

fig = plt.figure(figsize=(17.5, 8.6))
gs = fig.add_gridspec(1, 2, wspace=.30, left=.105, right=.995, top=.845, bottom=.085)

def mrect(a, x, y, w, h, nk, lw=.5, z=2):
    m = V.mat(nk)
    a.add_patch(Rectangle((x, y), w, h, fc=m["color"], ec=m["edge"], lw=lw, zorder=z))
    if m["hatch"]:
        a.add_patch(Rectangle((x, y), w, h, fc="none", ec=m["edge"], hatch=m["hatch"],
                              lw=0, alpha=.55, zorder=z + .1))

for ci, chunk in enumerate(COLS):
    ax = fig.add_subplot(gs[ci])
    for i, g in enumerate(chunk):
        yb = -i * ROW
        d = g["d"]
        mrect(ax, 0, yb, WIN, HAIR, V.AIR, lw=0, z=1)
        mrect(ax, 0, yb - HSUB, WIN, HSUB, g["sub"], lw=0, z=1)
        if g["dim"] == 0:
            mrect(ax, 0, yb, WIN, d, g["hi"], z=3)
        else:
            P = g["period"]
            mrect(ax, 0, yb, WIN, d, g["lo"], z=2)
            w_hi = 2 * g["radius"] * P if g["shape"] == "circle" else (
                g["ff"] * P if g["dim"] == 1 else g["ax"])
            k = 0
            while k * P - w_hi / 2 < WIN:
                xa, xb_ = max(k * P - w_hi / 2, 0), min(k * P + w_hi / 2, WIN)
                if xb_ > xa:
                    mrect(ax, xa, yb, xb_ - xa, d, g["hi"], z=3)
                k += 1
        ax.add_patch(Rectangle((0, yb), WIN, d, fc="none", ec="#42506088", lw=.6, zorder=4))
        if g["dim"]:
            P = g["period"]
            ax.annotate("", (0, yb + d + .085), (P, yb + d + .085),
                        arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=.9,
                                        mutation_scale=6, shrinkA=0, shrinkB=0), zorder=6)
            ax.text(P + .04, yb + d + .085, f"Λ = {P:.2f} λ", fontsize=7.4, va="center",
                    color=INK, zorder=7)
        dimtxt = {0: "0D planar", 1: "1D lamellar",
                  2: "2D " + ("cylinder" if g["shape"] == "circle" else "rect")}[g["dim"]]
        ax.text(-.05, yb + d / 2 + .075, g["name"], ha="right", va="center", fontsize=9.4,
                fontweight="bold", color=GROUP_C[g["group"]])
        ax.text(-.05, yb + d / 2 - .105, f"{dimtxt} · d = {d:.2f} λ · {g['pol_name']}",
                ha="right", va="center", fontsize=7.2, color=MUTED)
        # regime pill on the right edge
        r = g["period"] or 0.0
        sub = r < 1.0
        pill = "planar" if not r else ("sub-λ" if sub else "diffracting")
        col = "#4c7bb0" if (not r or sub) else "#b8762e"
        no = V.orders_open(g)
        ax.text(WIN + .10, yb + d / 2, f"{pill}\n{no} order" + ("s" if no != 1 else ""),
                ha="left", va="center", fontsize=7.6, color=col, fontweight="bold",
                linespacing=1.35)
        ax.axhline(yb - HSUB - .12, color="#e9eef3", lw=.8, zorder=0)
    n = len(chunk)
    ybot = -(n - 1) * ROW - HSUB
    for xx in np.arange(0, WIN + .01, .5):
        ax.plot([xx, xx], [ybot - .07, ybot - .13], color=MUTED, lw=.8, zorder=5)
    ax.plot([0, WIN], [ybot - .07] * 2, color=INK, lw=1.3, zorder=5)
    for xx in range(0, int(WIN) + 1):
        ax.text(xx, ybot - .16, f"{xx} λ", ha="center", va="top", fontsize=7.8, color=INK)
    ax.set_xlim(-.02, WIN + .78)
    ax.set_ylim(-(split - 1) * ROW - HSUB - .34, HAIR + .16)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

handles = [Patch(fc=V.MAT[k]["color"], ec=V.MAT[k]["edge"], hatch=V.MAT[k]["hatch"],
                 label=V.mlabel(k)) for k in [(1.0, 0.0), (1.5, 0.0), (3.5, 0.0), (0.3, 7.0)]]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9.5,
           bbox_to_anchor=(.5, .004))
fig.suptitle("Style S3 — the structure atlas: all 13 on one common length scale, sorted by period",
             fontsize=17, fontweight="bold", color=INK, y=.965)
fig.text(.5, .903, "each row shows the same 3.3 λ-wide window, so how many periods you see IS "
         "the sub-wavelength / diffracting story — and the thicknesses are on the same scale too",
         ha="center", fontsize=10.5, color=MUTED)
fig.savefig(out("out_S3_atlas.png"), dpi=118)
print("wrote out_S3_atlas.png")
