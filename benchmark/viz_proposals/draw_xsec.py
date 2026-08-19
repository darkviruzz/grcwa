"""Style S1 -- dimensioned x-z cross-section cards (+ x-y top view for 2D)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow, Circle, FancyArrowPatch
import viz_palette as V

GRID = "#c9d3dc"
INK = "#1b2733"
MUTED = "#5b6b7b"
BEAM = "#e8542f"

def _mrect(ax, x, y, w, h, nk, lw=0.8, z=2):
    m = V.mat(nk)
    r = Rectangle((x, y), w, h, facecolor=m["color"], edgecolor=m["edge"],
                  linewidth=lw, zorder=z)
    ax.add_patch(r)
    if m["hatch"]:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=m["edge"],
                               hatch=m["hatch"], linewidth=0, alpha=.55, zorder=z + .1))
    return r

def _dimh(ax, x0, x1, y, text, color=INK, fs=8, dy=0.0):
    ax.annotate("", (x0, y), (x1, y),
                arrowprops=dict(arrowstyle="<|-|>", color=color, lw=1.0,
                                mutation_scale=8, shrinkA=0, shrinkB=0), zorder=6)
    ax.text((x0 + x1) / 2, y + dy, text, ha="center", va="bottom", fontsize=fs,
            color=color, zorder=7,
            bbox=dict(fc="white", ec="none", alpha=.85, pad=.8))

def _dimv(ax, x, y0, y1, text, color=INK, fs=8):
    ax.annotate("", (x, y0), (x, y1),
                arrowprops=dict(arrowstyle="<|-|>", color=color, lw=1.0,
                                mutation_scale=8, shrinkA=0, shrinkB=0), zorder=6)
    ax.text(x, (y0 + y1) / 2, " " + text, ha="left", va="center", fontsize=fs,
            color=color, zorder=7, rotation=0)

def polarization_glyph(ax, x, y, pol, scale):
    """E-field orientation at the top of the beam: TM = in-plane arrow, TE = dot."""
    if pol == "p":   # TM: E in the plane of incidence -> lies along x here
        ax.annotate("", (x - .55 * scale, y), (x + .55 * scale, y),
                    arrowprops=dict(arrowstyle="<|-|>", color=BEAM, lw=1.4,
                                    mutation_scale=8), zorder=8)
        ax.text(x, y + .16 * scale, "E", color=BEAM, fontsize=8, ha="center",
                va="bottom", fontweight="bold")
    else:            # TE: E out of plane -> circle-dot
        ax.add_patch(Circle((x, y), .17 * scale, fc="white", ec=BEAM, lw=1.3, zorder=8))
        ax.add_patch(Circle((x, y), .05 * scale, fc=BEAM, ec=BEAM, zorder=9))
        ax.text(x + .3 * scale, y, "E", color=BEAM, fontsize=8, ha="left",
                va="center", fontweight="bold")

def draw_xsec(ax, g, nper=2.6, fs=8, title=True, aspect=None, marg=0.13):
    """True-aspect x-z cut with the thickness dimension drawn in a right margin
    (extension lines), the way a mechanical drawing does it."""
    P = g["period"] or 1.0
    W = nper * P if g["dim"] else 1.45
    d = g["d"]
    if aspect is None:
        fig = ax.figure
        pos = ax.get_position()
        aspect = (pos.width * fig.get_figwidth()) / (pos.height * fig.get_figheight())
    Wtot = W * (1 + marg)
    H = max(Wtot / aspect, d / 0.42)
    hair = 0.58 * (H - d)
    hsub = 0.42 * (H - d)
    x0, x1 = -W / 2, W / 2
    ax.set_xlim(x0, x0 + Wtot)
    ax.set_ylim(-hsub, d + hair)
    ax.set_aspect("equal")

    _mrect(ax, x0, d, W, hair, V.AIR, lw=0)
    _mrect(ax, x0, -hsub, W, hsub, g["sub"], lw=0)

    if g["dim"] == 0:
        _mrect(ax, x0, 0, W, d, g["hi"])
    else:
        _mrect(ax, x0, 0, W, d, g["lo"])
        w_hi = 2 * g["radius"] * P if g["shape"] == "circle" else (
            g["ff"] * P if g["dim"] == 1 else g["ax"])
        k0 = int(np.ceil(nper / 2)) + 1
        for k in range(-k0, k0 + 1):
            xc = k * P
            xa, xb_ = max(xc - w_hi / 2, x0), min(xc + w_hi / 2, x1)
            if xb_ > xa:
                _mrect(ax, xa, 0, xb_ - xa, d, g["hi"], z=3)
    ax.add_patch(Rectangle((x0, 0), W, d, fc="none", ec="#42506088", lw=.7, zorder=4))

    xb = x0 + .14 * W
    ytop = d + hair
    ax.annotate("", (xb, d + .05 * hair), (xb, ytop - .34 * hair),
                arrowprops=dict(arrowstyle="-|>", color=BEAM, lw=1.9,
                                mutation_scale=11), zorder=8)
    polarization_glyph(ax, xb, ytop - .19 * hair, g["pol"], min(hair, W) * .5)
    ax.annotate("", (xb + .11 * W, d + .04 * hair), (xb + .11 * W, d + .40 * hair),
                arrowprops=dict(arrowstyle="-|>", color=BEAM, lw=1.1, alpha=.65,
                                mutation_scale=8), zorder=8)
    ax.text(xb + .125 * W, d + .26 * hair, "R", color=BEAM, fontsize=fs - .5, alpha=.9)
    ax.annotate("", (xb, -.04 * hsub), (xb, -.66 * hsub),
                arrowprops=dict(arrowstyle="-|>", color=BEAM, lw=1.1, alpha=.65,
                                mutation_scale=8), zorder=8)
    ax.text(xb + .015 * W, -.62 * hsub, "T", color=BEAM, fontsize=fs - .5, alpha=.9)

    if g["dim"]:
        _dimh(ax, -P / 2, P / 2, d + .12 * hair, f"Λ = {P:.3f} λ", fs=fs - .5)

    # thickness dimension in the right margin, with extension lines
    xd = x1 + .55 * marg * W
    for yy in (0, d):
        ax.plot([x1, xd + .28 * marg * W], [yy, yy], color=INK, lw=.6, zorder=6)
    ax.annotate("", (xd, 0), (xd, d),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.0,
                                mutation_scale=8, shrinkA=0, shrinkB=0), zorder=6)
    ax.text(xd, d + .04 * hair, f"d = {d:.2f} λ", ha="center", va="bottom",
            fontsize=fs - .5, color=INK, zorder=7, rotation=90)

    # scale bar: the largest "nice" fraction of lambda that fits
    for L in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02):
        if L <= 0.62 * W:
            break
    xs = x1 - L - .04 * W
    yb = -.62 * hsub
    ax.plot([xs, xs + L], [yb] * 2, color=INK, lw=2.6, zorder=9, solid_capstyle="butt")
    for xx in (xs, xs + L):
        ax.plot([xx, xx], [yb - .08 * hsub, yb + .08 * hsub], color=INK, lw=1.2, zorder=9)
    ax.text(xs + L / 2, yb + .11 * hsub, "λ" if L == 1 else f"{L:g} λ", ha="center",
            va="bottom", fontsize=fs, color=INK, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(g["name"], fontsize=fs + 1.5, fontweight="bold", color=INK, pad=4)


def draw_topview(ax, g, nper=2.0, fs=7):
    """x-y unit cell (2D only) -- what the cross-section cannot show."""
    P = g["period"]
    W = nper * P
    ax.set_xlim(-W / 2, W / 2); ax.set_ylim(-W / 2, W / 2)
    _mrect(ax, -W / 2, -W / 2, W, W, g["lo"], lw=0)
    k0 = int(np.ceil(nper / 2)) + 1
    for i in range(-k0, k0 + 1):
        for j in range(-k0, k0 + 1):
            xc, yc = i * P, j * P
            if g["shape"] == "circle":
                m = V.mat(g["hi"])
                ax.add_patch(Circle((xc, yc), g["radius"] * P, fc=m["color"],
                                    ec=m["edge"], lw=.8, zorder=3))
            else:
                _mrect(ax, xc - g["ax"] / 2, yc - g["ay"] / 2, g["ax"], g["ay"],
                       g["hi"], z=3)
    ax.add_patch(Rectangle((-P / 2, -P / 2), P, P, fc="none", ec=BEAM, lw=1.2,
                           ls=(0, (3, 2)), zorder=6))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_color("#9fb0c0"); sp.set_linewidth(.6)
    ax.set_title("top view (x–y)", fontsize=fs, color=MUTED, pad=2)
