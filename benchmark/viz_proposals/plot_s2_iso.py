import os

import numpy as np, matplotlib, textwrap
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import palette as V


from draw_iso import draw_iso, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("GRCWA_VIZ_OUT", os.path.join(HERE, "figures"))
os.makedirs(OUTDIR, exist_ok=True)


def out(name):
    return os.path.join(OUTDIR, name)

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
GROUP_C = {"A": "#4c7bb0", "B": "#3f9070", "C": "#b8762e", "D": "#8a5aa8"}
G = V.ALL
fig, axes = plt.subplots(4, 4, figsize=(19, 16.5))
fig.subplots_adjust(left=.015, right=.99, top=.915, bottom=.055, hspace=.42, wspace=.10)
for ax in axes.flat:
    ax.axis("off")
for i, g in enumerate(G):
    ax = axes.flat[i]
    ax.axis("on")
    draw_iso(ax, g)
    wrapped = textwrap.wrap(g["desc"], 52)
    ax.set_title(g["name"], fontsize=11, fontweight="bold",
                 color=GROUP_C[g["group"]], loc="left", pad=6 + 10.5 * len(wrapped))
    ax.text(0, 1.012, "\n".join(wrapped), transform=ax.transAxes, fontsize=7.4,
            color=MUTED, va="bottom", ha="left", linespacing=1.35)
    ax.text(0, -.02, "   ".join(t for _, t in V.badges(g)), transform=ax.transAxes,
            fontsize=7.4, color=INK, va="top", ha="left")
handles = [Patch(fc=V.MAT[k]["color"], ec=V.MAT[k]["edge"], label=V.mlabel(k))
           for k in [(1.0, 0.0), (1.5, 0.0), (3.5, 0.0), (0.3, 7.0)]]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10,
           bbox_to_anchor=(.5, .006))
fig.suptitle("Style S2 — isometric cut-away unit cell (top face = x–y mask, front face = x–z cut)",
             fontsize=17, fontweight="bold", color=INK, y=.975)
fig.text(.5, .950, "faces carry the real rasterized pattern, not a sketch · E is drawn along the axis the polarization selects",
         ha="center", fontsize=10.5, color=MUTED)
fig.savefig(out("out_S2_isometric.png"), dpi=115)
print("wrote out_S2_isometric.png")
