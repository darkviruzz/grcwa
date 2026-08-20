"""How faithfully did the shared mask of :mod:`structures` represent the
battery's nominal geometry, before it was fixed -- and how much did the
answer depend on that?

FIXED, see ``structures.layer_mask``'s current default and
``RASTERIZATION.md``.  This module now documents the HISTORY: every function
here defaults to ``legacy=True`` and reproduces the pre-fix numbers exactly,
on purpose, so the record of what was wrong -- and by how much -- does not
silently disappear under the reader. Call anything here with ``legacy=False``
(or ``report(legacy=False)``) to see the current, corrected numbers instead.
For new work, prefer ``benchmark/rasterization_study.py``, which measures the
current default and the sampling-channel fix on top of it.

Why this exists
---------------
``structures.layer_mask`` rasterizes every patterned layer once and hands the
same integer mask to grcwa and to Ikarus.  That makes the two python columns
immune to *each other's* pixel-grid artifacts, but it did not make either of
them faithful to the structure the battery is nominally defined by -- and any
code that draws its own geometry from the parameters (Moose, S4, a lab
measurement) solves the nominal structure, not the mask.

On the 1D cases this was a non-issue: ``NX_1D_LEGACY = 8192`` with
``ff = 0.5`` fills exactly 4096 cells, so the mask *is* the nominal grating
and Moose agrees with the python columns to five or six digits.

On the 2D cases it was the whole story.  ``NX_2D_LEGACY = 256`` could not
represent the square pillars exactly -- ``0.6 * 256 = 153.6`` and
``0.4 * 256 = 102.4`` are not integers -- and the rect branch sampled on the
left cell edge with a strict ``<``, which additionally dropped a pixel on
each side when an edge landed exactly on a sample (C2).  The pillars
therefore came out between 0.4 % and 0.8 % away from their nominal size, and
R turned out to be about ten times more sensitive to that than to the
truncation order at the top of the sweep:

    C1_Si_pillars      one pixel of the 256 grid is worth ~0.012 in R
    C1b_..._diffract   the +0.59 % mask error is worth ~0.010 in R

which was larger than the entire disagreement with the external Moose
reference.  Run this module with ``--solve`` to reproduce that statement from
scratch (still on the legacy mask, by design -- that is the point being
measured); the current default renders every rect case exactly, at 0.000 %.

Usage
-----
    python benchmark/geometry_fidelity.py              # the legacy fidelity table
    python benchmark/geometry_fidelity.py --current    # + the current (fixed) table
    python benchmark/geometry_fidelity.py --solve      # + R(legacy mask) vs R(exact)
    python benchmark/geometry_fidelity.py --solve --q 41 --case C1_Si_pillars

The fidelity table needs nothing but numpy.  ``--solve`` uses Ikarus when it is
installed (``pip install ikarus-rcwa``, rules ``li`` and ``normal``) and falls
back to the fork's own Pol rule otherwise.
"""
import argparse
import json
import os
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Run directly from anywhere: the battery lives here, the fork one level up.
for _p in (HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import structures as ST                                            # noqa: E402

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


def fidelity(s, legacy=True):
    """Nominal vs rasterized feature size of one structure.

    ``legacy=True`` (the default) reproduces the ORIGINAL, pre-fix rasterization
    this module was written to characterize (``structures.NX_1D_LEGACY``,
    ``NX_2D_LEGACY``, left-edge rect sampling) -- this module's whole purpose is
    documenting that historical mismatch, so its default output must not change
    just because ``structures.layer_mask`` got a better default. Pass
    ``legacy=False`` to measure the CURRENT (fixed) rasterization instead --
    see ``report(legacy=False)`` for the fixed side by side with this one.

    Returns a dict with ``kind`` ('fill', 'width' or 'area'), the nominal and
    rasterized value, and the relative error of the *linear* feature (a width,
    or the effective radius of the circle), which is what R responds to.
    """
    dim = s["dim"]
    if dim == 0:
        return dict(name=s["name"], dim=0, kind="none", nominal=None,
                    got=None, rel=0.0, grid=None, detail="uniform layer")

    mask, _ = ST.layer_mask(s, legacy=legacy)
    if dim == 1:
        nx = ST.NX_1D_LEGACY if legacy else ST.NX_1D
        nominal = s["ff"]
        got = float(mask.mean())
        return dict(name=s["name"], dim=1, kind="fill", nominal=nominal,
                    got=got, rel=got / nominal - 1.0, grid=nx,
                    detail="%d of %d cells (exact would be %.1f)"
                           % (int(mask.sum()), nx, nominal * nx))

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


def report(structures=None, legacy=True):
    """Print the fidelity table; return the rows.

    ``legacy=True`` (default) is this module's original subject: the pre-fix
    mask. Call ``report(legacy=False)`` to see the same table for the current,
    corrected default in ``structures.layer_mask`` -- on this battery every
    rect and 1D case comes back at 0.000 % there; only the circle (D2) still
    has a nonzero, unavoidable entry.
    """
    structures = ST.STRUCTURES if structures is None else structures
    rows = [fidelity(s, legacy=legacy) for s in structures]
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
    print("\n%d of %d patterned layers are off by more than %.2f%%  (%s grid)"
          % (len(bad), len([r for r in rows if r["kind"] != "none"]),
             100 * FEATURE_TOL, "legacy" if legacy else "current"))
    if bad:
        print("  " + ", ".join(r["name"] for r in bad))
        if legacy:
            print("  -> these cannot be compared against a code that builds the")
            print("     geometry from the parameters instead of from this mask.")
            print("     structures.layer_mask's CURRENT default fixes all of the")
            print("     rect cases -- run report(legacy=False) to see it.")
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
    """Solve every named 2D case twice -- the LEGACY shared mask vs exact
    geometry. This reproduces the historical measurement (mask 256 vs exact)
    that established the fix; it is not the current default any more -- see
    ``rasterization_study.py``'s ``grid``/``coeffs`` subcommands for that."""
    solvers = _solvers()
    print("\nR from the legacy shared mask against R from the nominal geometry, "
          "q = %d" % q)
    print("(the gap between the two columns is what a code drawing its own\n"
          " geometry -- Moose -- used to see as a disagreement, before the fix)\n")
    for name in names:
        s = ST.STRUCT[name]
        if s["dim"] != 2:
            print("%s: not a 2D case, skipped" % name)
            continue
        f = fidelity(s, legacy=True)
        shared, _ = ST.layer_mask(s, legacy=True)
        n_exact = exact_grid(s)
        ex = exact_mask(s, n_exact)
        moose_q, moose_ref = _moose_ref(name, q)
        print("%s   nominal %.6f, mask %.6f (%+.3f%%), exact grid %d"
              % (name, f["nominal"], f["got"], 100 * f["rel"], n_exact))
        print("   %-14s %12s %12s %12s" % ("rule", "legacy mask", "exact", "delta"))
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
            print("   moose, which builds the nominal geometry itself, at the "
                  "same order: %.6f" % moose_q)
        elif moose_ref is not None:
            print("   moose, which builds the nominal geometry itself, at the "
                  "highest order it ran: %.6f" % moose_ref)
        print("")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--current", action="store_true",
                    help="also print the fidelity table for the CURRENT "
                         "(fixed) default, alongside the legacy one")
    ap.add_argument("--solve", action="store_true",
                    help="also solve mask vs exact geometry (slow, legacy mask)")
    ap.add_argument("--q", type=int, default=31,
                    help="per-axis retained orders for --solve (odd; default 31)")
    ap.add_argument("--case", action="append", default=None,
                    help="restrict --solve to this case (repeatable)")
    args = ap.parse_args()

    report(legacy=True)
    if args.current:
        print("\n--- current default (structures.layer_mask, legacy=False) ---\n")
        report(legacy=False)
    if args.solve:
        names = args.case or [s["name"] for s in ST.STRUCTURES if s["dim"] == 2]
        solve_compare(names, args.q)


if __name__ == "__main__":
    main()
