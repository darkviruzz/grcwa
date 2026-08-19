"""How faithfully does the shared mask of :mod:`structures` represent the
battery's nominal geometry -- and how much of the answer depends on that?

Why this exists
---------------
``structures.layer_mask`` rasterizes every patterned layer once and hands the
same integer mask to grcwa and to Ikarus.  That makes the two python columns
immune to *each other's* pixel-grid artifacts, but it does not make either of
them faithful to the structure the battery is nominally defined by -- and any
code that draws its own geometry from the parameters (Moose, S4, a lab
measurement) solves the nominal structure, not the mask.

On the 1D cases this is a non-issue: ``NX_1D = 8192`` with ``ff = 0.5`` fills
exactly 4096 cells, so the mask *is* the nominal grating and Moose agrees with
the python columns to five or six digits.

On the 2D cases it is the whole story.  ``NX_2D = 256`` cannot represent the
square pillars exactly -- ``0.6 * 256 = 153.6`` and ``0.4 * 256 = 102.4`` are
not integers -- and the rect branch samples on the left cell edge with a strict
``<``, which additionally drops a pixel on each side when an edge lands exactly
on a sample (C2).  The pillars therefore come out between 0.4 % and 0.8 % away
from their nominal size, and R turns out to be about ten times more sensitive to
that than to the truncation order at the top of the sweep:

    C1_Si_pillars      one pixel of the 256 grid is worth ~0.012 in R
    C1b_..._diffract   the +0.59 % mask error is worth ~0.010 in R

which is larger than the entire disagreement with the external Moose reference.
Run this module with ``--solve`` to reproduce that statement from scratch.

Usage
-----
    python benchmark/geometry_fidelity.py              # the fidelity table
    python benchmark/geometry_fidelity.py --solve      # + R(mask) vs R(exact)
    python benchmark/geometry_fidelity.py --solve --q 41 --case C1_Si_pillars

The fidelity table needs nothing but numpy.  ``--solve`` uses Ikarus when it is
installed (``pip install ikarus-rcwa``, rules ``li`` and ``normal``) and falls
back to the fork's own Pol rule otherwise.
"""
import argparse
import json
import os
from fractions import Fraction

import numpy as np

import structures as ST

HERE = os.path.dirname(os.path.abspath(__file__))

#: Relative error in a linear feature size above which the mask is called unfit
#: to be compared against a code that draws its own geometry.  0.1 % is already
#: worth ~0.003 in R on C1 -- an order of magnitude above the truncation error
#: at the top of the sweep -- so this is a generous bound, not a tight one.
FEATURE_TOL = 1e-3


def _rect_width(mask):
    """Relative width of the centred pillar along the mask's centre row."""
    nx = mask.shape[0]
    row = mask[:, mask.shape[1] // 2]
    return float(row.sum()) / nx


def fidelity(s):
    """Nominal vs rasterized feature size of one structure.

    Returns a dict with ``kind`` ('fill', 'width' or 'area'), the nominal and
    rasterized value, and the relative error of the *linear* feature (a width,
    or the effective radius of the circle), which is what R responds to.
    """
    dim = s["dim"]
    if dim == 0:
        return dict(name=s["name"], dim=0, kind="none", nominal=None,
                    got=None, rel=0.0, grid=None, detail="uniform layer")

    mask, _ = ST.layer_mask(s)
    if dim == 1:
        nominal = s["ff"]
        got = float(mask.mean())
        return dict(name=s["name"], dim=1, kind="fill", nominal=nominal,
                    got=got, rel=got / nominal - 1.0, grid=ST.NX_1D,
                    detail="%d of %d cells (exact would be %.1f)"
                           % (int(mask.sum()), ST.NX_1D, nominal * ST.NX_1D))

    n = mask.shape[0]
    if s.get("shape", "rect") == "circle":
        nominal = s["radius"]
        got = float(np.sqrt(mask.mean() / np.pi))     # effective radius
        return dict(name=s["name"], dim=2, kind="area", nominal=nominal,
                    got=got, rel=got / nominal - 1.0, grid=n,
                    detail="area %.6f vs %.6f (a circle has no exact raster)"
                           % (mask.mean(), np.pi * nominal ** 2))

    nominal = s["ax"] / s["period"]
    got = _rect_width(mask)
    return dict(name=s["name"], dim=2, kind="width", nominal=nominal, got=got,
                rel=got / nominal - 1.0, grid=n,
                detail="%d of %d px (exact would be %.1f)"
                       % (int(round(got * n)), n, nominal * n))


def exact_grid(s, at_least=None):
    """A grid on which ``s``'s nominal feature is exactly representable.

    An axis-aligned rectangle needs only ``w * N`` to be an integer: with
    cell-centred samples its edges then fall on pixel boundaries and the binary
    mask *is* the rectangle, with no staircase at all.  A circle has no such
    grid, so this just returns a finer one.
    """
    at_least = ST.NX_2D if at_least is None else at_least
    if s.get("shape", "rect") == "circle":
        return int(4 * at_least)
    w = Fraction(s["ax"] / s["period"]).limit_denominator(10000)
    den = w.denominator
    return int(den * int(np.ceil(at_least / den)))


def exact_mask(s, n=None):
    """Cell-centred rasterization of ``s`` on :func:`exact_grid`.

    Cell-centred sampling is what ``ikarus.shapes.rectangle`` and
    ``ikarus.shapes.circle`` use, and what ``structures.layer_mask`` already
    uses for its circle; only the rect branch samples on the left cell edge.
    """
    n = exact_grid(s) if n is None else n
    c = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(c, c, indexing="ij")
    if s.get("shape", "rect") == "circle":
        inside = (X - 0.5) ** 2 + (Y - 0.5) ** 2 <= s["radius"] ** 2
    else:
        w = s["ax"] / s["period"]
        h = s["ay"] / s["period"]
        inside = (np.abs(X - 0.5) <= w / 2) & (np.abs(Y - 0.5) <= h / 2)
    return inside.astype(int)


def report(structures=None):
    """Print the fidelity table; return the rows."""
    structures = ST.STRUCTURES if structures is None else structures
    rows = [fidelity(s) for s in structures]
    print("%-26s %3s %10s %12s %12s %10s  %s"
          % ("case", "dim", "kind", "nominal", "rasterized", "rel.err", "detail"))
    for r in rows:
        if r["kind"] == "none":
            print("%-26s %3d %10s %12s %12s %10s  %s"
                  % (r["name"], r["dim"], "-", "-", "-", "-", r["detail"]))
            continue
        flag = "  <-- over tolerance" if abs(r["rel"]) > FEATURE_TOL else ""
        print("%-26s %3d %10s %12.6f %12.6f %+9.3f%%  %s%s"
              % (r["name"], r["dim"], r["kind"], r["nominal"], r["got"],
                 100 * r["rel"], r["detail"], flag))
    bad = [r for r in rows if r["kind"] != "none" and abs(r["rel"]) > FEATURE_TOL]
    print("\n%d of %d patterned layers are off by more than %.2f%%"
          % (len(bad), len([r for r in rows if r["kind"] != "none"]),
             100 * FEATURE_TOL))
    if bad:
        print("  " + ", ".join(r["name"] for r in bad))
        print("  -> these cannot be compared against a code that builds the")
        print("     geometry from the parameters instead of from this mask.")
    return rows


# ---------------------------------------------------------------------------
#  --solve: what does that rasterization error cost in R?
# ---------------------------------------------------------------------------
def _solvers():
    """Available (label, callable(structure, mask, q) -> R) pairs."""
    out = []
    try:
        import ikarus_suite                                    # noqa: F401
        from ikarus import RCWA

        def _ik(s, mask, q, factorization):
            m = (q - 1) // 2
            period = s["period"] * ikarus_suite.UNIT
            rc = RCWA(period_x=period, period_y=period, n_orders=(m, m),
                      factorization=factorization)
            rc.add_uniform_layer(np.inf, complex(*ST.AIR))
            rc.add_layer(s["d"] * ikarus_suite.UNIT, mask,
                         [complex(*nk) for nk in (s["bg"], s["pillar"])],
                         resolution=mask.shape)
            rc.add_uniform_layer(np.inf, complex(*s["sub"]))
            rc.set_source(wavelength=ikarus_suite.WAVELENGTH, theta=0.0, phi=0.0,
                          polarization="linear",
                          linear_pol_angle=90.0 if s["pol"] == "p" else 0.0)
            _T, _R, res = rc.simulate()
            return float(np.real(res.R_total))

        out.append(("ikarus[Li]", lambda s, m, q: _ik(s, m, q, "li")))
        out.append(("ikarus[NV]", lambda s, m, q: _ik(s, m, q, "normal")))
    except Exception:
        pass

    import grcwa

    def _fork(s, mask, q, fmm):
        eg = np.array([ST.eps(nk) for nk in (s["bg"], s["pillar"])],
                      dtype=complex)[mask]
        n = mask.shape[0]
        kwargs = {} if fmm is None else dict(fmm_method=fmm)
        o = grcwa.obj(q * q, [s["period"], 0], [0, s["period"]], ST.FREQC,
                      0., 0., verbose=0, **kwargs)
        o.Add_LayerUniform(1.0, ST.eps(ST.AIR))
        o.Add_LayerGrid(s["d"], n, n)
        o.Add_LayerUniform(1.0, ST.eps(s["sub"]))
        o.Init_Setup(Gmethod=1)
        pa, sa = (1., 0.) if s["pol"] == "p" else (0., 1.)
        o.MakeExcitationPlanewave(pa, 0., sa, 0., order=0)
        o.GridLayer_geteps(eg.flatten())
        R, _T = o.RT_Solve(normalize=1)
        return float(np.real(R))

    out.append(("fork[Pol]", lambda s, m, q: _fork(s, m, q, "pol")))
    return out


def _moose_ref(name, q):
    """The Moose value at the matching order, and its highest-order value."""
    path = os.path.join(HERE, "moose_reference.json")
    if not os.path.exists(path):
        return None, None
    cases = json.load(open(path)).get("cases", {})
    if name not in cases:
        return None, None
    sweep = cases[name].get("sweep", {})
    key = "(%d,%d)" % ((q - 1) // 2, (q - 1) // 2)
    return sweep.get(key), cases[name].get("ref")


def solve_compare(names, q):
    """Solve every named 2D case twice -- shared mask vs exact geometry."""
    solvers = _solvers()
    print("\nR from the shared mask against R from the nominal geometry, q = %d"
          % q)
    print("(the gap between the two columns is what a code drawing its own\n"
          " geometry -- Moose -- sees as a disagreement)\n")
    for name in names:
        s = ST.STRUCT[name]
        if s["dim"] != 2:
            print("%s: not a 2D case, skipped" % name)
            continue
        f = fidelity(s)
        shared, _ = ST.layer_mask(s)
        n_exact = exact_grid(s)
        ex = exact_mask(s, n_exact)
        moose_q, moose_ref = _moose_ref(name, q)
        print("%s   nominal %.6f, mask %.6f (%+.3f%%), exact grid %d"
              % (name, f["nominal"], f["got"], 100 * f["rel"], n_exact))
        print("   %-14s %12s %12s %12s" % ("rule", "mask 256", "exact", "delta"))
        for label, fn in solvers:
            try:
                r_mask = fn(s, shared, q)
                r_exact = fn(s, ex, q)
            except Exception as exc:                       # pragma: no cover
                print("   %-14s failed: %s" % (label, exc))
                continue
            print("   %-14s %12.6f %12.6f %+12.6f"
                  % (label, r_mask, r_exact, r_exact - r_mask))
        if moose_q is not None:
            print("   %-14s %12s %12.6f   (Moose at the same order)"
                  % ("moose", "-", moose_q))
        elif moose_ref is not None:
            print("   %-14s %12s %12.6f   (Moose, highest order it ran)"
                  % ("moose", "-", moose_ref))
        print("")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solve", action="store_true",
                    help="also solve mask vs exact geometry (slow)")
    ap.add_argument("--q", type=int, default=31,
                    help="per-axis retained orders for --solve (odd; default 31)")
    ap.add_argument("--case", action="append", default=None,
                    help="restrict --solve to this case (repeatable)")
    args = ap.parse_args()

    report()
    if args.solve:
        names = args.case or [s["name"] for s in ST.STRUCTURES if s["dim"] == 2]
        solve_compare(names, args.q)


if __name__ == "__main__":
    main()
