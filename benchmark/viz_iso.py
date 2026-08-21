"""Isometric cut-away of the unit cell, drawn as SOLIDS.

Air is not a material here -- it is absent.  Ridges, pillars and cylinders are
real prisms standing on (or floating above) the substrate, so the gaps between
them are open space rather than a pale grey filler, and a perforated film is
drawn with its holes cut through: you look into the pit and see its far walls
and the substrate at the bottom.

Projection: isometric, screen = (cos30*(x-y), sin30*(x+y)+z), viewer in the
(-x,-y,+z) octant, so every solid shows its top, its -y face and its -x face.
Depth order is the painter's algorithm on x_centre + y_centre.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import to_rgb

import viz_palette as V

A = np.deg2rad(30.0)
CX, SY = np.cos(A), np.sin(A)
INK = "#1b2733"
MUTED = "#5b6b7b"
BEAM = "#e8542f"
EFIELD = "#7d3ac1"
GHOST = "#9fb0c0"

SH_TOP, SH_FRONT, SH_SIDE = 1.00, 0.79, 0.60
SH_PIT_FLOOR, SH_PIT_FAR, SH_PIT_SIDE = 0.46, 0.40, 0.30


def P(x, y, z):
    return np.array([CX * (x - y), SY * (x + y) + z])


def _sh(rgb, f):
    return tuple(np.clip(np.asarray(rgb) * f + (1 - f) * 0.05, 0, 1))


def _poly(ax, pts, rgb, f, zorder, lw=0.6, clip=None, ec=None):
    fc = _sh(rgb, f)
    p = Polygon([P(*q) for q in pts], closed=True, facecolor=fc,
                edgecolor=ec if ec is not None else _sh(rgb, f * 0.72),
                linewidth=lw, zorder=zorder, joinstyle="round")
    ax.add_patch(p)
    if clip is not None:
        p.set_clip_path(clip)
    return p


def _box(ax, x0, x1, y0, y1, z0, z1, rgb, zorder):
    """A solid prism: its -y face, its -x face and its top are visible."""
    _poly(ax, [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
          rgb, SH_FRONT, zorder)
    _poly(ax, [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
          rgb, SH_SIDE, zorder)
    _poly(ax, [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
          rgb, SH_TOP, zorder + .05)


def _cylinder(ax, xc, yc, r, z0, z1, rgb, zorder, nstrip=18):
    """Circular pillar. The barrel is the arc whose outward normal faces the
    viewer, drawn in strips so the curvature actually reads as curvature."""
    t0, t1 = 3 * np.pi / 4, 7 * np.pi / 4          # normals pointing to -x,-y
    edges = np.linspace(t0, t1, nstrip + 1)
    step = edges[1] - edges[0]
    for a, b in zip(edges[:-1], edges[1:]):
        tm = (a + b) / 2
        b = min(b + step * 0.03, t1)
        lam = max(0.0, -0.45 * np.cos(tm) - 0.9 * np.sin(tm)) / 1.006
        f = 0.40 + 0.46 * lam
        quad = [(xc + r * np.cos(a), yc + r * np.sin(a), z1),
                (xc + r * np.cos(b), yc + r * np.sin(b), z1),
                (xc + r * np.cos(b), yc + r * np.sin(b), z0),
                (xc + r * np.cos(a), yc + r * np.sin(a), z0)]
        _poly(ax, quad, rgb, f, zorder, lw=.4, ec=_sh(rgb, f))
    # silhouette of the barrel, then the top disc
    arc = np.linspace(t0, t1, 4 * nstrip + 1)
    sil = [(xc + r * np.cos(t), yc + r * np.sin(t), z1) for t in arc]
    sil += [(xc + r * np.cos(t), yc + r * np.sin(t), z0) for t in arc[::-1]]
    ax.add_patch(Polygon([P(*q) for q in sil], closed=True, facecolor="none",
                         edgecolor=_sh(rgb, 0.34), linewidth=.6,
                         zorder=zorder + .04))
    disc = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    _poly(ax, [(xc + r * np.cos(t), yc + r * np.sin(t), z1) for t in disc],
          rgb, SH_TOP, zorder + .05, lw=.6)


def _pit(ax, x0, x1, y0, y1, ztop, zbot, rgb, floor_rgb, zorder):
    """A rectangular hole through a film: clipped to its own opening, so the
    interior never spills over the material in front of it."""
    mouth = Polygon([P(x0, y0, ztop), P(x1, y0, ztop),
                     P(x1, y1, ztop), P(x0, y1, ztop)],
                    closed=True, facecolor="none", edgecolor="none")
    ax.add_patch(mouth)
    _poly(ax, [(x0, y0, zbot), (x1, y0, zbot), (x1, y1, zbot), (x0, y1, zbot)],
          floor_rgb, SH_PIT_FLOOR, zorder, lw=.4, clip=mouth)
    _poly(ax, [(x0, y1, zbot), (x1, y1, zbot), (x1, y1, ztop), (x0, y1, ztop)],
          rgb, SH_PIT_FAR, zorder + .01, lw=.4, clip=mouth)
    _poly(ax, [(x1, y0, zbot), (x1, y1, zbot), (x1, y1, ztop), (x1, y0, ztop)],
          rgb, SH_PIT_SIDE, zorder + .02, lw=.4, clip=mouth)
    ax.add_patch(Polygon([P(x0, y0, ztop), P(x1, y0, ztop),
                          P(x1, y1, ztop), P(x0, y1, ztop)], closed=True,
                         facecolor="none", edgecolor=_sh(rgb, SH_TOP * .62),
                         linewidth=.6, zorder=zorder + .03))


def _is_air(nk):
    return (float(nk[0]), float(nk[1])) == (1.0, 0.0)


def _ghost_halfspace(ax, Lx, Ly, hs):
    """A free-standing case has no substrate -- outline the half space instead.

    The full wireframe, top face included: without the z = 0 rectangle the
    solids look like they float over nothing rather than over open space.
    """
    box = [(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)]
    edges = []
    for z in (0.0, -hs):                                  # top and bottom faces
        edges += [((box[i][0], box[i][1], z), (box[(i + 1) % 4][0],
                                               box[(i + 1) % 4][1], z))
                  for i in range(4)]
    edges += [((x, y, 0.0), (x, y, -hs)) for x, y in box]  # verticals
    for a, b in edges:
        ax.plot(*zip(P(*a), P(*b)), color=GHOST, lw=.8, ls=(0, (2, 3)), zorder=2)


def draw_iso(ax, g, nper=2, fs=8.5, show_beam=True):
    dim = g["dim"]
    Lam = g["period"] or 0.9
    # every patterned case gets the same nper x nper cell footprint, so the
    # depth along y is identical across the sheet
    Lx = Ly = 1.0 if dim == 0 else nper * Lam
    d = g["d"]
    hs = max(0.45 * d, 0.20 * Lx)
    inc_rgb = to_rgb(V.mat(g["hi"])["color"])
    bg_rgb = to_rgb(V.mat(g["lo"])["color"])
    sub_rgb = to_rgb(V.mat(g["sub"])["color"])
    air_sub = _is_air(g["sub"])

    # ---- substrate --------------------------------------------------------
    if air_sub:
        _ghost_halfspace(ax, Lx, Ly, hs)
    else:
        _box(ax, 0, Lx, 0, Ly, -hs, 0, sub_rgb, 1)

    # ---- unit-cell footprint, on the ground plane -------------------------
    if dim:
        cell = [P(0, 0, 0), P(Lam, 0, 0), P(Lam, Lam if dim == 2 else Ly, 0),
                P(0, Lam if dim == 2 else Ly, 0)]
        ax.add_patch(Polygon(cell, closed=True, fc="none", ec=BEAM, lw=1.2,
                             ls=(0, (3, 2)), zorder=2.5))

    # ---- the patterned layer, as solids -----------------------------------
    if dim == 0:
        _box(ax, 0, Lx, 0, Ly, 0, d, inc_rgb, 5)
    elif _is_air(g["hi"]):
        # inclusion is the void: a perforated film, holes cut through it
        _box(ax, 0, Lx, 0, Ly, 0, d, bg_rgb, 5)
        wx, wy = g["ax"] / 2, g["ay"] / 2
        pits = [((i + .5) * Lam, (j + .5) * Lam)
                for i in range(nper) for j in range(nper)]
        for xc, yc in sorted(pits, key=lambda c: -(c[0] + c[1])):
            _pit(ax, xc - wx, xc + wx, yc - wy, yc + wy, d, 0.0,
                 bg_rgb, sub_rgb if not air_sub else (1, 1, 1), 6)
    elif dim == 1:
        w = g["ff"] * Lam / 2
        for k in sorted(range(nper), key=lambda k: -k):
            xc = (k + .5) * Lam
            _box(ax, xc - w, xc + w, 0, Ly, 0, d, inc_rgb, 5 + (nper - k) * .1)
    else:
        cells = [(i, j) for i in range(nper) for j in range(nper)]
        for n, (i, j) in enumerate(sorted(cells, key=lambda c: -(c[0] + c[1]))):
            xc, yc = (i + .5) * Lam, (j + .5) * Lam
            z = 5 + n * .1
            if g["shape"] == "circle":
                _cylinder(ax, xc, yc, g["radius"] * Lam, 0, d, inc_rgb, z)
            else:
                _box(ax, xc - g["ax"] / 2, xc + g["ax"] / 2,
                     yc - g["ay"] / 2, yc + g["ay"] / 2, 0, d, inc_rgb, z)

    # ---- incidence: k down, E along the axis the polarization selects -----
    xa, ya, ztop = .34 * Lx, .70 * Ly, d + max(.85 * d, .42 * Lx)
    eL = .28 * Lx
    if show_beam:
        p0, p1 = P(xa, ya, ztop), P(xa, ya, d + .06 * max(d, .1))
        ax.annotate("", tuple(p1), tuple(p0),
                    arrowprops=dict(arrowstyle="-|>", color=BEAM, lw=2.0,
                                    mutation_scale=12), zorder=9)
        ax.text(*(p0 + [-.02 * Lx, .02 * Lx]), "k", color=BEAM, fontsize=fs,
                fontweight="bold", ha="right", va="bottom")
        if g["pol"] == "p":                     # TM: E in the plane of incidence
            e0, e1 = P(xa - eL, ya, ztop), P(xa + eL, ya, ztop)
        else:                                   # TE: E out of that plane
            e0, e1 = P(xa, ya - eL, ztop), P(xa, ya + eL, ztop)
        ax.annotate("", tuple(e1), tuple(e0),
                    arrowprops=dict(arrowstyle="<|-|>", color=EFIELD, lw=1.8,
                                    mutation_scale=9), zorder=9)
        ax.text(*(e1 + [.015 * Lx, .015 * Lx]), "E", color=EFIELD, fontsize=fs,
                fontweight="bold", va="bottom")

    # ---- compact triad + lambda bar ---------------------------------------
    ox, oy, tl = Lx * .62, -Ly * .36, .18 * Lx
    for vec, lab in [((tl, 0, 0), "x"), ((0, tl, 0), "y"), ((0, 0, tl), "z")]:
        a, b = P(ox, oy, -hs), P(ox + vec[0], oy + vec[1], -hs + vec[2])
        ax.annotate("", tuple(b), tuple(a),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=.9,
                                    mutation_scale=6), zorder=8)
        u = (b - a) / (np.hypot(*(b - a)) + 1e-12)
        ax.text(*(b + u * .045 * Lx), lab, color=MUTED, fontsize=fs - 2,
                ha="center", va="center")
    for L in (1.0, .5, .2, .1, .05):
        if L <= .8 * Lx:
            break
    ybar = -Ly * .36
    a, b = P(0, ybar, -hs), P(L, ybar, -hs)
    ax.plot(*zip(a, b), color=INK, lw=2.6, zorder=8, solid_capstyle="butt")
    ax.text(*((a + b) / 2 + [0, -.03 * Lx]), "λ" if L == 1 else f"{L:g} λ",
            color=INK, fontsize=fs - .5, ha="center", va="top", fontweight="bold")

    # ---- identical framing rule for every panel ---------------------------
    corners = [P(0, ybar, -hs), P(Lx, ybar, -hs), P(0, Ly, -hs), P(Lx, Ly, d),
               P(0, 0, -hs * 1.25), P(ox + tl, oy, -hs)]
    if show_beam:
        corners += [P(xa - .30 * Lx, ya, ztop), P(xa + .30 * Lx, ya, ztop),
                    P(xa, ya - .30 * Lx, ztop), P(xa, ya + .30 * Lx, ztop)]
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
