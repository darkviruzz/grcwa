"""What geometry does each suite actually solve, and what does the answer cost?

`geometry_fidelity.py` measures how far the shared mask is from the nominal
structure.  This module measures the *other* half of the same question -- how
far the Fourier coefficients each code feeds its eigenproblem are from the
coefficients of the geometry it was handed -- and it prices both errors in R.

Two error channels, not one
---------------------------
Every FFT-based RCWA turns geometry into permittivity Fourier coefficients in
two steps, and each step has its own error:

  1. **shape error** -- the pixel image is not the nominal shape.  O(1/N) when a
     boundary falls between two samples; *exactly zero* when every boundary
     falls on a cell edge.  This is what `geometry_fidelity.py` reports.
  2. **sampling error** -- the DFT of the samples is not the Fourier integral of
     the pixel image.  For a cell image (piecewise constant on
     ``[n/N, (n+1)/N)``) the two are related *exactly* by

         c_m = DFT_m . sinc(m/N) . exp(-i pi m / N)                        (*)

     so the plain FFT that grcwa, Ikarus and Moose all use overstates every
     coefficient by ``1/sinc(m/N) = 1 + (pi m/N)^2/6 + ...``.  It is O(1/N^2) at
     fixed order, it never vanishes on any grid, and it is the residual left
     over after channel 1 is closed.

The half-cell phase in (*) is a rigid translation of the layer and does not
change R at all; the sinc factor does.  Applying it (``pixel_exact()`` below)
makes the direct rule *bit-identical* across grids -- see ``grid``.

Sub-commands
------------
    coeffs   coefficient accuracy of every rasterization rule (numpy only)
    grid     R vs the eps grid N at fixed order, with and without (*)
    pol      the Pol/normal-vector tangent field vs the grid
    fill1d   the 1D fill fractions: NX_1D = 8192 cannot render ff = 0.8
    circle   the one case no grid renders exactly, against analytic coefficients

Ikarus (``pip install ikarus-rcwa``) is optional; its columns disappear without
it.  Nothing here changes `structures.py` -- it measures what is there.
"""
import argparse
import contextlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import structures as ST                                            # noqa: E402
import grcwa                                                       # noqa: E402
import grcwa.fft_funs as _ff                                       # noqa: E402
import grcwa.rcwa as _gr                                           # noqa: E402


# ===========================================================================
#  rasterization rules
# ===========================================================================
def rect_fill(N, w, h=None, rule="centre"):
    """Fill fraction per cell of a centred axis-aligned rectangle, w, h in [0,1].

    ``left``   the historical `structures.layer_mask` rect rule: samples on the
               left cell edge, strict ``<``.
    ``centre`` cell centres with ``<=`` -- what `ikarus.shapes.rectangle` and
               the circle branch of `layer_mask` already do.  Exact whenever
               ``w*N`` and ``(1-w)/2*N`` are integers.
    ``area``   the exact cell-averaged fill (grey boundary cells).
    """
    h = w if h is None else h
    if rule == "area":
        e = np.arange(N + 1) / N

        def cov(size):
            lo, hi = 0.5 - size / 2, 0.5 + size / 2
            return np.clip(np.minimum(e[1:], hi) - np.maximum(e[:-1], lo), 0, None) * N
        return cov(w)[:, None] * cov(h)[None, :]
    if rule == "centre":
        c = (np.arange(N) + 0.5) / N
        X, Y = np.meshgrid(c, c, indexing="ij")
        return ((np.abs(X - .5) <= w / 2 + 1e-12)
                & (np.abs(Y - .5) <= h / 2 + 1e-12)).astype(float)
    if rule == "left":
        x = np.linspace(0, 1, N, endpoint=False)
        X, Y = np.meshgrid(x, x, indexing="ij")
        return ((np.abs(X - .5) < w / 2) & (np.abs(Y - .5) < h / 2)).astype(float)
    raise ValueError(rule)


def _gauss(a, b, nq):
    g, w = np.polynomial.legendre.leggauss(nq)
    return (a + b) / 2 + (b - a) / 2 * g, (b - a) / 2 * w


def circle_fill(N, r, rule="centre", nq=96):
    """Fill fraction per cell of a disk of radius ``r`` centred in the cell.

    ``centre`` binary, cell centres (what `layer_mask` already does).
    ``area``   the exact cell-averaged fill: Gauss-Legendre in x per cell, with
               the cells holding the extreme points ``0.5 -+ r`` split there so
               the square-root endpoint sits on a sub-interval boundary.
    """
    if rule == "centre":
        c = (np.arange(N) + 0.5) / N
        X, Y = np.meshgrid(c, c, indexing="ij")
        return (((X - .5) ** 2 + (Y - .5) ** 2) <= r * r).astype(float)
    if rule != "area":
        raise ValueError(rule)
    edges = np.arange(N + 1) / N
    out = np.zeros((N, N))
    for i in range(N):
        a, b = edges[i], edges[i + 1]
        if b <= 0.5 - r or a >= 0.5 + r:
            continue
        pts = sorted({a, b} | {p for p in (0.5 - r, 0.5 + r) if a < p < b})
        xs, ws = [], []
        for k in range(len(pts) - 1):
            n_, w_ = _gauss(pts[k], pts[k + 1], nq)
            xs.append(n_)
            ws.append(w_)
        xs, ws = np.concatenate(xs), np.concatenate(ws)
        hh = np.sqrt(np.clip(r ** 2 - (xs - .5) ** 2, 0, None))
        ov = np.clip(np.minimum((.5 + hh)[:, None], edges[None, 1:])
                     - np.maximum((.5 - hh)[:, None], edges[None, :-1]), 0, None)
        out[i] = ws @ ov
    return out * N * N


def fill(s, N, rule):
    """Fill-fraction grid of the inclusion of battery case ``s``."""
    if s.get("shape", "rect") == "circle":
        return circle_fill(N, s["radius"], rule)
    return rect_fill(N, s["ax"] / s["period"], s["ay"] / s["period"], rule)


def eps_of(s, N, rule):
    f = fill(s, N, rule)
    return ST.eps(s["bg"]) * (1 - f) + ST.eps(s["pillar"]) * f


# exact_N, shape_transform and analytic_coeffs now live in structures.py --
# this module imports them (below) rather than keeping a second copy, per
# RASTERIZATION.md's own warning to "keep the two in step".


# ===========================================================================
#  the sinc ('pixel-exact') quadrature
# ===========================================================================
_orig_get_conv = _ff.get_conv


def _get_conv_pixel(dN, s_in, G):
    """`get_conv` times sinc(dm/Nx) sinc(dn/Ny) -- identity (*) of the docstring.

    Turns the point-sample DFT into the exact Fourier integral of the cell
    image the grid stands for.  The half-cell phase of (*) is left out: it is a
    rigid shift of the layer and cancels in R and T.
    """
    Nx, Ny = s_in.shape[0], s_in.shape[1]
    mgx = int(np.max(G[:, 0])) - int(np.min(G[:, 0]))
    mgy = int(np.max(G[:, 1])) - int(np.min(G[:, 1]))
    # `get_conv` nearest-neighbour upsamples when the grid is too coarse; the
    # sinc must refer to the grid it actually transformed.
    sx = max(int(np.ceil((mgx + 1) / Nx)), 1) if mgx >= Nx else 1
    sy = max(int(np.ceil((mgy + 1) / Ny)), 1) if mgy >= Ny else 1
    gi = G[:, 0][:, None] - G[:, 0]
    gj = G[:, 1][:, None] - G[:, 1]
    return _orig_get_conv(dN, s_in, G) * np.sinc(gi / (Nx * sx)) * np.sinc(gj / (Ny * sy))


@contextlib.contextmanager
def pixel_exact(on=True):
    """Use the exact cell-image quadrature inside grcwa for the duration."""
    _ff.get_conv = _get_conv_pixel if on else _orig_get_conv
    try:
        yield
    finally:
        _ff.get_conv = _orig_get_conv


# ===========================================================================
#  analytic Fourier coefficients -- the geometry oracle (structures.py)
# ===========================================================================
exact_N = ST.exact_N
shape_transform = ST.shape_transform
analytic_coeffs = ST.analytic_coeffs


def _toeplitz(s, G, inverse=False):
    d = np.stack([G[:, 0][:, None] - G[:, 0], G[:, 1][:, None] - G[:, 1]], axis=-1)
    return analytic_coeffs(s, d.reshape(-1, 2), inverse).reshape(d.shape[:2])


@contextlib.contextmanager
def analytic_eps(s):
    """Make grcwa's Laurent path use the analytic coefficients of ``s``."""
    keep = _gr.Epsilon_fft

    def _eps_fft(dN, eps_grid, G):
        eps_hat = _toeplitz(s, G)
        Z = np.zeros_like(eps_hat)
        return np.linalg.inv(eps_hat), np.block([[eps_hat, Z], [Z, eps_hat]])
    _gr.Epsilon_fft = _eps_fft
    try:
        yield
    finally:
        _gr.Epsilon_fft = keep


# ===========================================================================
#  solvers
# ===========================================================================
def solve_grcwa(s, q, eg, fmm=None, **kw):
    """R of case ``s`` at per-axis order ``q`` on the eps grid ``eg``."""
    kwargs = {} if fmm is None else dict(fmm_method=fmm)
    kwargs.update(kw)
    if s["dim"] == 1:
        N = eg.shape[0]
        o = grcwa.obj(q, [s["period"], 0], None, ST.FREQC, 0., 0., verbose=0, **kwargs)
        o.Add_LayerUniform(1.0, ST.eps(ST.AIR))
        o.Add_LayerGrid(s["d"], N)
        o.Add_LayerUniform(1.0, ST.eps(s["sub"]))
        o.Init_Setup(Gmethod=1)
        flat = eg
    else:
        N = eg.shape[0]
        L = s["period"]
        o = grcwa.obj(q * q, [L, 0], [0, L], ST.FREQC, 0., 0., verbose=0, **kwargs)
        o.Add_LayerUniform(1.0, ST.eps(ST.AIR))
        o.Add_LayerGrid(s["d"], N, N)
        o.Add_LayerUniform(1.0, ST.eps(s["sub"]))
        o.Init_Setup(Gmethod=1)
        flat = eg.flatten()
    pa, sa = (1., 0.) if s["pol"] == "p" else (0., 1.)
    o.MakeExcitationPlanewave(pa, 0., sa, 0., order=0)
    o.GridLayer_geteps(flat)
    R, T = o.RT_Solve(normalize=1)
    return float(np.real(R))


def have_ikarus():
    try:
        import ikarus                                              # noqa: F401
        return True
    except Exception:
        return False


def solve_ikarus(s, q, eg, rule="li"):
    """Ikarus on an arbitrary *complex* eps grid.

    Ikarus's public API wants an integer topology plus one material per index,
    so a grey grid is passed as its distinct values quantized into that list --
    exactly the grid grcwa gets, no resampling (``resolution`` is pinned).
    """
    from ikarus import RCWA
    import ikarus_suite as IK
    N = eg.shape[0]
    vals, inv = np.unique(eg, return_inverse=True)
    idx = np.sqrt(vals.astype(complex))
    idx = np.where(idx.imag < 0, -idx, idx)          # exp(-iwt): absorbers k > 0
    M = (q - 1) // 2
    per = s["period"] * IK.UNIT
    m = (M, 0) if s["dim"] == 1 else (M, M)
    rc = RCWA(period_x=per, period_y=per, n_orders=m, factorization=rule)
    rc.add_uniform_layer(np.inf, complex(*ST.AIR))
    topo = inv.reshape(eg.shape).astype(int)
    if s["dim"] == 1:
        topo = topo.reshape(N, 1) if topo.ndim == 1 else topo
    rc.add_layer(s["d"] * IK.UNIT, topo, [complex(v) for v in idx],
                 resolution=topo.shape)
    rc.add_uniform_layer(np.inf, complex(*s["sub"]))
    rc.set_source(wavelength=IK.WAVELENGTH, theta=0., phi=0.,
                  polarization="linear",
                  linear_pol_angle=90. if s["pol"] == "p" else 0.)
    _T, _R, res = rc.simulate()
    return float(np.real(res.R_total))


# ===========================================================================
#  sub-command: coeffs
# ===========================================================================
def cmd_coeffs(args):
    """How close are the coefficients each rule produces to the analytic ones?"""
    Ms = args.orders
    m = np.arange(-Ms, Ms + 1)
    MX, MY = np.meshgrid(m, m, indexing="ij")
    G = np.stack([MX.ravel(), MY.ravel()], axis=1)

    def numeric(grid, sinc):
        N = grid.shape[0]
        c = np.fft.fft2(grid) / N ** 2
        o = c[MX % N, MY % N]
        if sinc:
            o = o * np.sinc(MX / N) * np.sinc(MY / N)
        return o * np.exp(-1j * np.pi * (MX + MY) / N)

    for name in args.case or ["C1_Si_pillars", "D2_ikarus_cylinder_TE"]:
        s = ST.STRUCT[name]
        circ = s.get("shape", "rect") == "circle"
        ana = analytic_coeffs(s, G).reshape(MX.shape)
        ana_i = analytic_coeffs(s, G, inverse=True).reshape(MX.shape)
        rules = ["centre", "area"] if circ else ["left", "centre", "area"]
        print("\n%s  (%s)   rms coefficient error over the %dx%d block"
              % (name, "circle" if circ else "rect", 2 * Ms + 1, 2 * Ms + 1))
        head = ["N"] + ["%s%s" % (r, sfx) for r in rules for sfx in ("", "+sinc")]
        print(("%6s" + "%15s" * (len(head) - 1)) % tuple(head))
        for N in args.grids:
            row = [N]
            for r in rules:
                f = fill(s, N, r)
                eg = ST.eps(s["bg"]) * (1 - f) + ST.eps(s["pillar"]) * f
                for sinc in (False, True):
                    row.append(np.sqrt(np.mean(np.abs(numeric(eg, sinc) - ana) ** 2)))
            print(("%6d" + "%15.3e" * (len(row) - 1)) % tuple(row))
        print("  the same for 1/eps, which every faithful rule reads off the "
              "SAME grid:")
        head = ["N"] + ["%s%s" % (r, sfx) for r in rules for sfx in ("", "+sinc")]
        print(("%6s" + "%15s" * (len(head) - 1)) % tuple(head))
        for N in args.grids:
            row = [N]
            for r in rules:
                f = fill(s, N, r)
                eg = ST.eps(s["bg"]) * (1 - f) + ST.eps(s["pillar"]) * f
                for sinc in (False, True):
                    row.append(np.sqrt(np.mean(np.abs(numeric(1 / eg, sinc) - ana_i) ** 2)))
            print(("%6d" + "%15.3e" * (len(row) - 1)) % tuple(row))
        if not circ:
            print("  (exact grid for this case: N = %d and its multiples)"
                  % exact_N(s))


# ===========================================================================
#  sub-command: grid
# ===========================================================================
def cmd_grid(args):
    """R against the eps grid, with the geometry held exactly representable."""
    for name in args.case or ["C1_Si_pillars"]:
        s = ST.STRUCT[name]
        base = exact_N(s)
        if base is None:
            print("%s: no exact grid (curved boundary) -- use `circle`" % name)
            continue
        grids = [base * k for k in (1, 2, 4, 8)][:args.n_grids]
        print("\n%s, exactly representable geometry (grids %d x 1,2,4,...), q = %d"
              % (name, base, args.q))
        cols = ["Laurent", "Laurent+sinc"] + (
            ["ik-laurent", "ik-li", "ik-normal"] if have_ikarus() else [])
        print(("%7s" + "%16s" * len(cols)) % tuple(["N"] + cols))
        for N in grids:
            eg = eps_of(s, N, "centre")
            row = [N, solve_grcwa(s, args.q, eg, None)]
            with pixel_exact():
                row.append(solve_grcwa(s, args.q, eg, None))
            if have_ikarus():
                for rule in ("laurent", "li", "normal"):
                    row.append(solve_ikarus(s, args.q, eg, rule))
            print(("%7d" + "%16.9f" * (len(row) - 1)) % tuple(row))
        with analytic_eps(s):
            Ra = solve_grcwa(s, args.q, eps_of(s, base, "centre"), None)
        print("  analytic coefficients, no grid at all:  %.9f" % Ra)


# ===========================================================================
#  sub-command: pol
# ===========================================================================
def cmd_pol(args):
    """The tangent-field rules against the eps grid they are built from."""
    for name in args.case or ["C1_Si_pillars"]:
        s = ST.STRUCT[name]
        base = exact_N(s) or ST.NX_2D
        grids = [base * k for k in (1, 2, 4, 8)][:args.n_grids]
        print("\n%s, exact geometry, q = %d -- Pol's blur is a fraction of the period"
              % (name, args.q))
        cols = ["Pol sigma=3px", "Pol sigma=3/%d period" % base,
                "Pol default=1/12"] + (
            ["ik-normal"] if have_ikarus() else [])
        print(("%7s" + "%25s" * len(cols)) % tuple(["N"] + cols))
        for N in grids:
            eg = eps_of(s, N, "centre")
            row = [N, solve_grcwa(s, args.q, eg, "pol", pol_sigma=3.0 / N),
                   solve_grcwa(s, args.q, eg, "pol", pol_sigma=3.0 / base),
                   solve_grcwa(s, args.q, eg, "pol")]
            if have_ikarus():
                row.append(solve_ikarus(s, args.q, eg, "normal"))
            print(("%7d" + "%25.9f" * (len(row) - 1)) % tuple(row))


# ===========================================================================
#  sub-command: fill1d
# ===========================================================================
def cmd_fill1d(args):
    """NX_1D = 8192 renders ff = 0.5 exactly and ff = 0.8 not at all."""
    def prof(N, ff):
        x = (np.arange(N) + 0.5) / N
        return (x < ff).astype(float)

    for name in args.case or ["B3_Au_slits_TM"]:
        s = ST.STRUCT[name]
        ff = s["ff"]
        print("\n%s   ff = %.3f   (NX_1D=8192 -> %.1f cells)" % (name, ff, 8192 * ff))
        print("%8s %10s %16s %16s %16s"
              % ("N", "cells", "Laurent", "Laurent+sinc", "Pol default"))
        for N in args.grids:
            f = prof(N, ff)
            eg = ST.eps(s["lo"]) * (1 - f) + ST.eps(s["hi"]) * f
            a = solve_grcwa(s, args.q, eg, None)
            with pixel_exact():
                b = solve_grcwa(s, args.q, eg, None)
            c = solve_grcwa(s, args.q, eg, "pol")
            print("%8d %10.1f %16.9f %16.9f %16.9f" % (N, f.sum(), a, b, c))


# ===========================================================================
#  sub-command: circle
# ===========================================================================
def cmd_circle(args):
    """The case no grid renders exactly, against the analytic coefficients."""
    s = ST.STRUCT[args.case[0] if args.case else "D2_ikarus_cylinder_TE"]
    with analytic_eps(s):
        Ra = solve_grcwa(s, args.q, eps_of(s, 64, "centre"), None)
    print("%s  r/L = %.3f  q = %d" % (s["name"], s["radius"], args.q))
    print("grcwa Laurent on ANALYTIC coefficients (no grid): %.9f\n" % Ra)
    cols = ["grcwa-Laur", "err vs analytic"] + (["ik-li", "ik-normal"]
                                                if have_ikarus() else [])
    print(("%7s %-6s" + "%16s" * len(cols)) % tuple(["N", "rule"] + cols))
    for N in args.grids:
        for r in ("centre", "area"):
            eg = eps_of(s, N, r)
            row = [solve_grcwa(s, args.q, eg, None)]
            row.append(row[0] - Ra)
            if have_ikarus():
                for rule in ("li", "normal"):
                    row.append(solve_ikarus(s, args.q, eg, rule))
            print(("%7d %-6s" + "%16.9f" + "%16.2e" + "%16.9f" * (len(row) - 2))
                  % tuple([N, r] + row))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm, fn, gdef, qdef in (
            ("coeffs", cmd_coeffs, [128, 256, 512, 1024], 21),
            ("grid", cmd_grid, None, 21),
            ("pol", cmd_pol, None, 21),
            ("fill1d", cmd_fill1d, [8000, 8192, 10240, 16000], 201),
            ("circle", cmd_circle, [128, 256, 512], 21)):
        p = sub.add_parser(nm, help=fn.__doc__.splitlines()[0])
        p.add_argument("--case", action="append", default=None)
        p.add_argument("--q", type=int, default=qdef)
        p.add_argument("--orders", type=int, default=30,
                       help="half-width of the coefficient block (coeffs)")
        if gdef is None:
            p.add_argument("--n-grids", type=int, default=4)
        else:
            p.add_argument("--grids", type=int, nargs="+", default=gdef)
        p.set_defaults(func=fn)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
