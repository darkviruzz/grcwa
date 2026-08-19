"""Draw the structure battery: the isometric sheet and the shared-scale atlas.

    python benchmark/plot_structures.py     -> struct_iso.png, struct_atlas.png

Geometry is read from ``structures.py``, so a figure can never drift from the
battery it draws.  Air is not a material in these figures -- it is absent: the
gaps between ridges and pillars are open space, a perforated film is cut
through, and a free-standing case gets a dashed phantom half space instead of a
substrate block.

``GRCWA_PLOT_OUTPUT_DIR`` redirects the output, as in the other plotters.
"""
import os
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

import viz_palette as V
from viz_iso import draw_iso, INK, MUTED, BEAM

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("GRCWA_PLOT_OUTPUT_DIR", HERE)
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
GROUP_C = {"A": "#4c7bb0", "B": "#3f9070", "C": "#b8762e", "D": "#8a5aa8"}
WIN, ROW, HAIR, HSUB = 3.30, 0.88, 0.20, 0.17


def _legend_handles():
    handles = [Patch(fc=V.MAT[k]["color"], ec=V.MAT[k]["edge"], hatch=V.MAT[k]["hatch"],
                     label=V.mlabel(k)) for k in [(1.5, 0.0), (3.5, 0.0), (0.3, 7.0)]]
    handles.append(Patch(fc="white", ec="#9fb0c0", ls=(0, (2, 2)),
                         label="air (n=1) — drawn as empty space"))
    return handles


def iso_sheet(structures=None, ncol=4, dpi=115, fname="struct_iso.png"):
    """One isometric solid unit cell per structure."""
    structures = structures or V.ALL
    nrow = -(-len(structures) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.75 * ncol, 4.3 * nrow))
    fig.subplots_adjust(left=.015, right=.99, top=.880, bottom=.062,
                        hspace=.46, wspace=.10)
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    for i, g in enumerate(structures):
        ax = np.atleast_1d(axes).flat[i]
        ax.axis("on")
        draw_iso(ax, g)
        desc = textwrap.wrap(g["desc"], 50)
        ax.set_title(g["name"], fontsize=11, fontweight="bold",
                     color=GROUP_C[g["group"]], loc="left", pad=6 + 10.5 * len(desc))
        ax.text(0, 1.012, "\n".join(desc), transform=ax.transAxes, fontsize=7.4,
                color=MUTED, va="bottom", ha="left", linespacing=1.35)
        badges = textwrap.wrap("  ·  ".join(t for _, t in V.badges(g)), 46)
        ax.text(0, -.015, "\n".join(badges), transform=ax.transAxes, fontsize=7.4,
                color=INK, va="top", ha="left", linespacing=1.45)
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=4, frameon=False,
               fontsize=10, bbox_to_anchor=(.5, .006))
    fig.suptitle("The benchmark battery as solid isometric unit cells",
                 fontsize=17, fontweight="bold", color=INK, y=.975)
    fig.text(.5, .935,
             "air is absent, not grey: the gaps between ridges and pillars are open space "
             "and a perforated film is cut through\n"
             "E is drawn along the axis the polarization selects, so TE-along-the-grooves "
             "vs TM-across-them reads as geometry",
             ha="center", fontsize=10.2, color=MUTED, linespacing=1.5)
    path = os.path.join(OUT, fname)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print("wrote", path)


def _mrect(a, x, y, w, h, nk, lw=.5, z=2):
    m = V.mat(nk)
    a.add_patch(Rectangle((x, y), w, h, fc=m["color"], ec=m["edge"], lw=lw, zorder=z))
    if m["hatch"]:
        a.add_patch(Rectangle((x, y), w, h, fc="none", ec=m["edge"], hatch=m["hatch"],
                              lw=0, alpha=.55, zorder=z + .1))


def atlas(dpi=118, fname="struct_atlas.png", split=7):
    """Every structure on ONE shared length scale, sorted by period."""
    G = sorted(V.ALL, key=lambda g: (g["period"] or 0.0))
    cols = [G[:split], G[split:]]
    fig = plt.figure(figsize=(17.5, 8.6))
    gs = fig.add_gridspec(1, 2, wspace=.30, left=.105, right=.995, top=.845, bottom=.085)
    for ci, chunk in enumerate(cols):
        ax = fig.add_subplot(gs[ci])
        for i, g in enumerate(chunk):
            yb, d = -i * ROW, g["d"]
            _mrect(ax, 0, yb, WIN, HAIR, V.AIR, lw=0, z=1)
            _mrect(ax, 0, yb - HSUB, WIN, HSUB, g["sub"], lw=0, z=1)
            if g["dim"] == 0:
                _mrect(ax, 0, yb, WIN, d, g["hi"], z=3)
            else:
                P = g["period"]
                _mrect(ax, 0, yb, WIN, d, g["lo"], z=2)
                w_hi = 2 * g["radius"] * P if g["shape"] == "circle" else (
                    g["ff"] * P if g["dim"] == 1 else g["ax"])
                k = 0
                while k * P - w_hi / 2 < WIN:
                    xa, xb = max(k * P - w_hi / 2, 0), min(k * P + w_hi / 2, WIN)
                    if xb > xa:
                        _mrect(ax, xa, yb, xb - xa, d, g["hi"], z=3)
                    k += 1
            ax.add_patch(Rectangle((0, yb), WIN, d, fc="none", ec="#42506088",
                                   lw=.6, zorder=4))
            if g["dim"]:
                P = g["period"]
                ax.annotate("", (0, yb + d + .085), (P, yb + d + .085),
                            arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=.9,
                                            mutation_scale=6, shrinkA=0, shrinkB=0),
                            zorder=6)
                ax.text(P + .04, yb + d + .085, f"Λ = {P:.2f} λ", fontsize=7.4,
                        va="center", color=INK, zorder=7)
            dimtxt = {0: "0D planar", 1: "1D lamellar",
                      2: "2D " + ("cylinder" if g["shape"] == "circle" else "rect")}[g["dim"]]
            ax.text(-.05, yb + d / 2 + .075, g["name"], ha="right", va="center",
                    fontsize=9.4, fontweight="bold", color=GROUP_C[g["group"]])
            ax.text(-.05, yb + d / 2 - .105,
                    f"{dimtxt} · d = {d:.2f} λ · {g['pol_name']}", ha="right",
                    va="center", fontsize=7.2, color=MUTED)
            r = g["period"] or 0.0
            pill = "planar" if not r else ("sub-λ" if r < 1 else "diffracting")
            col = "#4c7bb0" if (not r or r < 1) else "#b8762e"
            no = V.orders_open(g)
            ax.text(WIN + .10, yb + d / 2,
                    f"{pill}\n{no} order" + ("s" if no != 1 else ""), ha="left",
                    va="center", fontsize=7.6, color=col, fontweight="bold",
                    linespacing=1.35)
            ax.axhline(yb - HSUB - .12, color="#e9eef3", lw=.8, zorder=0)
        ybot = -(len(chunk) - 1) * ROW - HSUB
        for xx in np.arange(0, WIN + .01, .5):
            ax.plot([xx, xx], [ybot - .07, ybot - .13], color=MUTED, lw=.8, zorder=5)
        ax.plot([0, WIN], [ybot - .07] * 2, color=INK, lw=1.3, zorder=5)
        for xx in range(0, int(WIN) + 1):
            ax.text(xx, ybot - .16, f"{xx} λ", ha="center", va="top", fontsize=7.8,
                    color=INK)
        ax.set_xlim(-.02, WIN + .78)
        ax.set_ylim(-(split - 1) * ROW - HSUB - .34, HAIR + .16)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, bbox_to_anchor=(.5, .004))
    fig.suptitle("The structure atlas: all 13 on one common length scale, sorted by period",
                 fontsize=17, fontweight="bold", color=INK, y=.965)
    fig.text(.5, .903, "each row shows the same 3.3 λ-wide window, so how many periods you "
             "see IS the sub-wavelength / diffracting story — and the thicknesses are on the "
             "same scale too", ha="center", fontsize=10.5, color=MUTED)
    path = os.path.join(OUT, fname)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    atlas()
    iso_sheet()
