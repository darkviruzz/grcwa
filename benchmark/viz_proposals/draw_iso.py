"""Style S2 -- isometric cut-away of the unit cell.

The three visible faces of the patterned layer are *textured with the real
rasterized pattern*: the top face carries the x-y mask, the front face the x-z
cut, the side face the y-z cut.  So one picture is simultaneously the 3D view,
the top view and the cross-section -- nothing is schematic.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.transforms import Affine2D
from matplotlib.colors import to_rgb
import palette as V

A = np.deg2rad(30.0)
CX, SY = np.cos(A), np.sin(A)
INK = "#1b2733"
MUTED = "#5b6b7b"
BEAM = "#e8542f"
EDGE = "#2c3a49"

SH_TOP, SH_FRONT, SH_SIDE = 1.00, 0.80, 0.62


def P(x, y, z):
    return np.asarray([CX * (x - y), SY * (x + y) + z])


def _shade(rgb, f):
    return tuple(np.clip(np.asarray(rgb) * f + (1 - f) * 0.06, 0, 1))


def _img(mask, mats, f):
    """(ny,nx) integer mask -> shaded RGB image."""
    cols = np.array([_shade(to_rgb(V.mat(m)["color"]), f) for m in mats])
    return cols[mask]


def _face(ax, img, tr, z=3, interp="nearest"):
    h, w = img.shape[:2]
    ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), interpolation=interp,
              transform=tr + ax.transData, zorder=z, aspect="auto")


def _tri_top(x0, x1, y0, y1, z):
    return Affine2D.from_values(CX * (x1 - x0), SY * (x1 - x0),
                                -CX * (y1 - y0), SY * (y1 - y0),
                                CX * (x0 - y0), SY * (x0 + y0) + z)


def _tri_front(x0, x1, y0, z0, z1):
    return Affine2D.from_values(CX * (x1 - x0), SY * (x1 - x0), 0.0, (z1 - z0),
                                CX * (x0 - y0), SY * (x0 + y0) + z0)


def _tri_side(y0, y1, x0, z0, z1):
    return Affine2D.from_values(-CX * (y1 - y0), SY * (y1 - y0), 0.0, (z1 - z0),
                                CX * (x0 - y0), SY * (x0 + y0) + z0)


def _outline(ax, x0, x1, y0, y1, z0, z1, lw=.9, color=EDGE, z=6):
    c = [P(x0, y0, z0), P(x1, y0, z0), P(x1, y1, z0), P(x0, y1, z0),
         P(x0, y0, z1), P(x1, y0, z1), P(x1, y1, z1), P(x0, y1, z1)]
    for a, b in [(0, 1), (1, 2), (0, 3), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]:
        ax.plot(*zip(c[a], c[b]), color=color, lw=lw, zorder=z,
                solid_capstyle="round")


def _pattern(g, X, Y):
    """1 inside the inclusion, 0 in the background; X,Y in absolute length."""
    Lam = g["period"]
    u = (X / Lam) % 1.0 - 0.5
    v = (Y / Lam) % 1.0 - 0.5
    if g["dim"] == 1:
        return (np.abs(u) < g["ff"] / 2).astype(int)
    if g["shape"] == "circle":
        return (u ** 2 + v ** 2 <= g["radius"] ** 2).astype(int)
    return ((np.abs(u) < g["ax"] / (2 * Lam)) &
            (np.abs(v) < g["ay"] / (2 * Lam))).astype(int)


def draw_iso(ax, g, nper=2.0, N=420, fs=8.5, hsub_frac=.55, show_beam=True):
    Lam = g["period"] or 0.9
    Lx = nper * Lam
    Ly = nper * Lam if g["dim"] == 2 else 1.45 * Lam
    if g["dim"] == 0:
        Lx = Ly = 0.9
    d = g["d"]
    hs = hsub_frac * max(d, .25 * Lx)
    x0, y0, z0, z1 = 0.0, 0.0, 0.0, d
    mats = [g["lo"], g["hi"]]

    # ---- substrate block (uniform) ---------------------------------------
    air_sub = (float(g["sub"][0]), float(g["sub"][1])) == (1.0, 0.0)
    if air_sub:
        # free-standing: draw the half space as a phantom, not as a solid block
        for a, b in [((x0, y0, 0), (x0, y0, -hs)), ((Lx, y0, 0), (Lx, y0, -hs)),
                     ((x0, Ly, 0), (x0, Ly, -hs)),
                     ((x0, y0, -hs), (Lx, y0, -hs)), ((x0, y0, -hs), (x0, Ly, -hs))]:
            ax.plot(*zip(P(*a), P(*b)), color="#9fb0c0", lw=.8, ls=(0, (2, 3)), zorder=3)
    else:
        one = np.zeros((2, 2), int)
        sm = [g["sub"], g["sub"]]
        _face(ax, _img(one, sm, SH_FRONT), _tri_front(x0, Lx, y0, -hs, 0), z=2)
        _face(ax, _img(one, sm, SH_SIDE), _tri_side(y0, Ly, x0, -hs, 0), z=2)
        _outline(ax, x0, Lx, y0, Ly, -hs, 0, lw=.7, color="#5a6a7a", z=5)

    # ---- patterned (or uniform) layer, textured with the real mask -------
    if g["dim"] == 0:
        top = np.ones((2, 2), int)
        frontm = np.ones((2, 2), int)
        sidem = np.ones((2, 2), int)
    else:
        xs = np.linspace(0, Lx, N)
        ys = np.linspace(0, Ly, N)
        Xt, Yt = np.meshgrid(xs, ys)                    # top face: rows=y
        top = _pattern(g, Xt, Yt)
        frontm = np.tile(_pattern(g, xs, np.full_like(xs, y0))[None, :], (8, 1))
        sidem = np.tile(_pattern(g, np.full_like(ys, x0), ys)[None, :], (8, 1))
    _face(ax, _img(top, mats, SH_TOP), _tri_top(x0, Lx, y0, Ly, z1), z=4,
          interp="antialiased")
    _face(ax, _img(frontm, mats, SH_FRONT), _tri_front(x0, Lx, y0, z0, z1), z=4)
    _face(ax, _img(sidem, mats, SH_SIDE), _tri_side(y0, Ly, x0, z0, z1), z=4)
    _outline(ax, x0, Lx, y0, Ly, z0, z1, lw=1.0, z=6)

    # unit-cell marker on the top face
    if g["dim"] == 2:
        cell = [P(0, 0, z1), P(Lam, 0, z1), P(Lam, Lam, z1), P(0, Lam, z1)]
        ax.add_patch(Polygon(cell, closed=True, fc="none", ec=BEAM, lw=1.3,
                             ls=(0, (3, 2)), zorder=7))
    elif g["dim"] == 1:
        for xx in (0, Lam):
            ax.plot(*zip(P(xx, 0, z1), P(xx, Ly, z1)), color=BEAM, lw=1.2,
                    ls=(0, (3, 2)), zorder=7)

    # ---- incidence: k down, E along the axis the polarization selects ----
    if show_beam:
        xa, ya = .30 * Lx, .62 * Ly
        ztop = z1 + max(.85 * d, .40 * Lx)
        p0, p1 = P(xa, ya, ztop), P(xa, ya, z1 + .06 * d)
        ax.annotate("", tuple(p1), tuple(p0),
                    arrowprops=dict(arrowstyle="-|>", color=BEAM, lw=2.0,
                                    mutation_scale=12), zorder=9)
        ax.text(*(p0 + [-.02 * Lx, .02 * Lx]), "k", color=BEAM, fontsize=fs,
                fontweight="bold", ha="right", va="bottom")
        eL = .30 * Lx
        if g["pol"] == "p":            # TM: E in the plane of incidence -> x
            e0, e1 = P(xa - eL, ya, ztop), P(xa + eL, ya, ztop)
        else:                          # TE: E out of the plane -> y
            e0, e1 = P(xa, ya - eL, ztop), P(xa, ya + eL, ztop)
        ax.annotate("", tuple(e1), tuple(e0),
                    arrowprops=dict(arrowstyle="<|-|>", color="#7d3ac1", lw=1.8,
                                    mutation_scale=9), zorder=9)
        ax.text(*(e1 + [.015 * Lx, .015 * Lx]), "E", color="#7d3ac1",
                fontsize=fs, fontweight="bold", va="bottom")

    # ---- compact axes triad + lambda bar ---------------------------------
    ox, oy, tl = Lx * .62, -Ly * .34, .19 * Lx
    for vec, lab in [((tl, 0, 0), "x"), ((0, tl, 0), "y"), ((0, 0, tl), "z")]:
        a, b = P(ox, oy, -hs), P(ox + vec[0], oy + vec[1], -hs + vec[2])
        ax.annotate("", tuple(b), tuple(a),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=.9,
                                    mutation_scale=6), zorder=8)
        d_ = (b - a); d_ = d_ / (np.hypot(*d_) + 1e-12)
        ax.text(*(b + d_ * .045 * Lx), lab, color=MUTED, fontsize=fs - 2,
                ha="center", va="center")
    for L in (1.0, .5, .2, .1, .05):
        if L <= .8 * Lx:
            break
    ybar = -Ly * .34
    a, b = P(0, ybar, -hs), P(L, ybar, -hs)
    ax.plot(*zip(a, b), color=INK, lw=2.6, zorder=8, solid_capstyle="butt")
    ax.text(*((a + b) / 2 + [0, -.03 * Lx]), "λ" if L == 1 else f"{L:g} λ",
            color=INK, fontsize=fs - .5, ha="center", va="top", fontweight="bold")

    # explicit, identical framing rule for every panel
    corners = [P(0, ybar, -hs), P(Lx, ybar, -hs), P(0, Ly, -hs), P(Lx, Ly, d),
               P(0, 0, -hs * 1.25), P(ox + tl, oy, -hs)]
    if show_beam:
        corners += [P(xa - .32 * Lx, ya, ztop), P(xa + .32 * Lx, ya, ztop),
                    P(xa, ya - .32 * Lx, ztop), P(xa, ya + .32 * Lx, ztop)]
    cs = np.array(corners)
    xr = (cs[:, 0].min(), cs[:, 0].max())
    yr = (cs[:, 1].min(), cs[:, 1].max())
    px, py = .05 * (xr[1] - xr[0]), .07 * (yr[1] - yr[0])
    ax.set_xlim(xr[0] - px, xr[1] + px)
    ax.set_ylim(yr[0] - py * 1.4, yr[1] + py)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
