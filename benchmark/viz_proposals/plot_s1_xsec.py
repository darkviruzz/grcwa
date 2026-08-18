import os

import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import palette as V


from draw_xsec import draw_xsec, draw_topview, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("GRCWA_VIZ_OUT", os.path.join(HERE, "figures"))
os.makedirs(OUTDIR, exist_ok=True)


def out(name):
    return os.path.join(OUTDIR, name)

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})

G = V.ALL
ncol, nrow = 4, 4
fig = plt.figure(figsize=(19, 15.5))
outer = fig.add_gridspec(nrow, ncol, hspace=.40, wspace=.16,
                         left=.02, right=.985, top=.925, bottom=.055)

GROUP_C = {"A": "#4c7bb0", "B": "#3f9070", "C": "#b8762e", "D": "#8a5aa8"}

for i, g in enumerate(G):
    r, c = divmod(i, ncol)
    cell = outer[r, c]
    if g["dim"] == 2:
        sub = cell.subgridspec(1, 2, width_ratios=[2.15, 1], wspace=.06)
        ax = fig.add_subplot(sub[0]); axt = fig.add_subplot(sub[1])
        draw_topview(axt, g)
    else:
        ax = fig.add_subplot(cell); axt = None
    fig.canvas.draw()
    draw_xsec(ax, g, title=False)
    gc = GROUP_C[g["group"]]
    import textwrap
    wrapped = textwrap.wrap(g["desc"], 58)
    ax.set_title(g["name"], fontsize=11, fontweight="bold", color=gc,
                 loc="left", pad=7 + 10.5 * len(wrapped))
    ax.text(0, 1.012, "\n".join(wrapped), transform=ax.transAxes,
            fontsize=7.4, color=MUTED, va="bottom", ha="left", linespacing=1.35)
    # badge strip
    bx = 0.0
    for kind, txt in V.badges(g):
        fc = V.BADGE_FC[kind]
        ec = V.BADGE_EC.get(kind, "#b9c7d4")
        t = ax.text(bx, -0.055, txt, transform=ax.transAxes, fontsize=7,
                    color=INK, va="top", ha="left",
                    bbox=dict(fc=fc, ec=ec, lw=.7, boxstyle="round,pad=.28"))
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(ax.transAxes.inverted())
        bx = bb.x1 + .022

handles = [Patch(fc=V.MAT[k]["color"], ec=V.MAT[k]["edge"],
                 hatch=V.MAT[k]["hatch"], label=V.mlabel(k))
           for k in [(1.0, 0.0), (1.5, 0.0), (3.5, 0.0), (0.3, 7.0)]]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
           fontsize=10, bbox_to_anchor=(.5, .006))
fig.suptitle("Style S1 — dimensioned cross-section (x–z cut), all 13 benchmark structures",
             fontsize=17, fontweight="bold", color=INK, y=.975)
fig.text(.5, .948, "every panel to its own scale, with a λ bar for calibration · "
         "beam glyph shows the polarization · 2D cases get the x–y top view the cut cannot show",
         ha="center", fontsize=10.5, color=MUTED)
fig.savefig(out("out_S1_crosssection.png"), dpi=115)
print("wrote out_S1_crosssection.png")
