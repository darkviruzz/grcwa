"""Reproduce -- and audit -- the Ikarus whitepaper's cross-code claims.

The whitepaper (Shelling Neto, CAVITY technologies GmbH, 2026,
doi 10.5281/zenodo.21966455; PDF in the repo root) argues that the direct
(Laurent) Fourier rule converges to the wrong answer in TM on a high-contrast
grating, and puts **grcwa** in the direct-rule column of its Table 1:

    FMMax (NORMAL)     faithful       10.0 %
    Ikarus (default)   faithful       10.0 %
    Ikarus (laurent)   direct rule    16.3 %
    torcwa             direct rule    16.3 %
    grcwa              direct rule    17.5 %

This script re-derives every line of that table from the shared battery
(structure ``D1_ikarus_hcg_TM`` in benchmark/structures.py), then does the same
for the paper's second cross-code case, the curved cylinder
``D2_ikarus_cylinder_TE``. Along the way it checks three things the table cannot
show on its own:

1. **Is the physics claim right?**  Two independent codes must agree on the
   direct-rule value -- otherwise the disagreement is a bug, not a rule.
2. **Where does 17.5 % come from?**  The paper's own grcwa harness drives a *1D*
   grating through a *square 2D lattice*, so a nominal ``nG=400`` retains only
   ~23 orders along x. Re-run it, count the x-orders, and compare against this
   fork's native-1D path at the same x-truncation.
3. **Does the table's grcwa row still hold for this fork?**  Upstream grcwa is
   Laurent-only, but this fork also ships a *fixed* Pol factorization
   (tests/test_pol_correctness.py). Whether that lands on the faithful 10 %
   decides if the table's grcwa row describes this codebase at all.

Run it directly; it needs the fork on sys.path and, for the Ikarus columns,
``pip install ikarus-rcwa`` (they are skipped when it is absent).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import grcwa                                                       # noqa: E402
import structures as ST                                            # noqa: E402
import ikarus_suite as IK                                          # noqa: E402

grcwa.set_backend("numpy")

CASE = ST.STRUCT["D1_ikarus_hcg_TM"]
CYL = ST.STRUCT["D2_ikarus_cylinder_TE"]
FAITHFUL = 0.100            # the whitepaper's "true value" (FMMax, converged)
# The paper's second cross-code claim, on the curved boundary: FMMax's NORMAL
# lands in 0.92 < R < 0.96 (its test suite's window), and Li's separable rule
# lags well behind there. Orders for the 2D case stay modest -- cost is O(M^6).
CYL_WINDOW = (0.92, 0.96)
CYL_QS = (11, 15, 21)

# The whitepaper's Table 1 quotes the direct rule "at a truncation a user would
# actually pick"; 16.3 % is Ikarus laurent at M=12, i.e. 2*12+1 = 25 orders.
Q_PRACTICAL = 25
PUBLISHED = {"FMMax (NORMAL)": (0.100, "faithful"),
             "Ikarus (default)": (0.100, "faithful"),
             "Ikarus (laurent)": (0.163, "direct rule"),
             "torcwa": (0.163, "direct rule"),
             "grcwa": (0.175, "direct rule")}


def fork_R(case, q, fmm):
    R, T, nG, _ = ST.solve(grcwa, case, q, fmm, True)
    return R, nG


def ikarus_R(case, q, factorization):
    R, T, nG, _ = IK.solve(case, q, factorization)
    return R, nG


def whitepaper_grcwa_R(nG, Nx=256):
    """The whitepaper's own grcwa harness, ported verbatim from its test suite
    (ikarus/tests/validation/grcwa_reference.py: ``grating_RT``).

    Note the square 2D lattice ``L2 = [0, 1]`` for a structure that varies only
    in x: the truncation is 2D, so most of ``nG`` is spent on ``Gy != 0`` orders
    that a y-invariant layer never couples into. Returns (R, nG_actual,
    n_x_orders_at_Gy0) -- that last number is the truncation the physics sees.
    """
    period, height = CASE["period"], CASE["d"]
    n_hi = complex(*CASE["hi"]).real
    # Their harness normalizes every length to the period and sets
    # freq = period/wavelength; here wavelength = 1, so freq = period.
    o = grcwa.obj(int(nG), [1.0, 0.0], [0.0, 1.0], period, 0.0, 0.0, verbose=0)
    o.Add_LayerUniform(1.0, 1.0)
    o.Add_LayerGrid(height / period, Nx, Nx)
    o.Add_LayerUniform(1.0, 1.0)
    o.Init_Setup()
    eps = np.ones((Nx, Nx))
    eps[: int(Nx * CASE["ff"]), :] = n_hi ** 2
    o.GridLayer_geteps(eps.flatten())
    o.MakeExcitationPlanewave(1.0, 0.0, 0.0, 0.0, order=0)          # p = TM
    R, T = o.RT_Solve(normalize=1)
    G = np.asarray(o.G)
    n_x = int(np.count_nonzero(G[:, 1] == 0))
    return float(np.real(R)), int(o.nG), n_x


def _fmt(x, w=9):
    return " " * w if x is None else f"{x:{w}.4f}"


def main():
    have_ikarus = IK.available()
    print("=" * 78)
    print("Ikarus whitepaper cross-check   (doi 10.5281/zenodo.21966455)")
    print("=" * 78)
    print(f"case   : {CASE['name']}  --  {CASE['desc']}")
    print(f"struct : period={CASE['period']:.6f}  d={CASE['d']:.6f}  "
          f"ff={CASE['ff']}  n_hi={CASE['hi'][0]}  free-standing, TM, normal "
          f"incidence")
    print("units  : lambda = 1 (the paper's 400/300 nm at 700 nm, scaled)")
    ik_tag = ("v" + IK.version() if have_ikarus
              else "NOT INSTALLED (pip install ikarus-rcwa)")
    print(f"ikarus : {ik_tag}")

    # ---------------------------------------------------------------- part 1
    print("\n" + "-" * 78)
    print("1. Convergence of each rule  (q = retained orders along x)")
    print("-" * 78)
    print(f"{'q':>5} | {'fork Laur':>9} {'fork Pol':>9} | "
          f"{'ik laurent':>10} {'ik li':>9} {'ik normal':>9}")
    sweep = {}
    for q in (5, 11, 15, 21, 25, 31, 41, 61, 101, 201):
        row = {"fork[Laurent]": fork_R(CASE, q, None)[0],
               "fork[Pol]": fork_R(CASE, q, "pol")[0]}
        if have_ikarus:
            for f, key in (("laurent", "ik[Laurent]"), ("li", "ik[Li]"),
                           ("normal", "ik[NV]")):
                row[key] = ikarus_R(CASE, q, f)[0]
        sweep[q] = row
        print(f"{q:>5} | {_fmt(row['fork[Laurent]'])} {_fmt(row['fork[Pol]'])} | "
              f"{_fmt(row.get('ik[Laurent]'), 10)} {_fmt(row.get('ik[Li]'))} "
              f"{_fmt(row.get('ik[NV]'))}")

    # ---------------------------------------------------------------- part 2
    print("\n" + "-" * 78)
    print(f"2. Table 1 re-derived   (direct rule at q = {Q_PRACTICAL}, the "
          f"paper's practical count)")
    print("-" * 78)
    wp_R, wp_nG, wp_nx = whitepaper_grcwa_R(400)
    measured = {
        "Ikarus (default)": sweep[Q_PRACTICAL].get("ik[NV]"),
        "Ikarus (laurent)": sweep[Q_PRACTICAL].get("ik[Laurent]"),
        "grcwa": wp_R,
    }
    print(f"{'row':<20} {'published':>10} {'measured':>10} {'|d|':>8}")
    for row, (pub, rule) in PUBLISHED.items():
        got = measured.get(row)
        if got is None:
            print(f"{row:<20} {pub:>10.3f} {'--':>10} {'':>8}   ({rule}; not "
                  f"re-run here)")
        else:
            print(f"{row:<20} {pub:>10.3f} {got:>10.4f} {abs(got - pub):>8.4f}"
                  f"   ({rule})")
    print(f"\n   the paper's grcwa harness: nG={wp_nG} in a SQUARE 2D lattice "
          f"-> only {wp_nx} x-orders")
    native = fork_R(CASE, wp_nx if wp_nx % 2 else wp_nx + 1, None)[0]
    print(f"   this fork, native 1D at q={wp_nx if wp_nx % 2 else wp_nx + 1} "
          f"(the same x-truncation): R = {native:.4f}")
    print(f"   -> the 17.5 % is the Laurent rule at ~{wp_nx} x-orders, not at "
          f"400; the paper's\n      'even at 400 orders' counts orders the "
          f"y-invariant layer never couples to.")

    # ---------------------------------------------------------------- part 3
    print("\n" + "-" * 78)
    print(f"3. The curved boundary  ({CYL['name']}, TE)")
    print("-" * 78)
    cyl = {}
    if have_ikarus:
        print(f"{'q':>5} {'nG':>6} | {'fork Laur':>9} {'fork Pol':>9} | "
              f"{'ik laurent':>10} {'ik li':>9} {'ik normal':>9}")
        for q in CYL_QS:
            row = {"fork[Laurent]": fork_R(CYL, q, None)[0],
                   "fork[Pol]": fork_R(CYL, q, "pol")[0],
                   "ik[Laurent]": ikarus_R(CYL, q, "laurent")[0],
                   "ik[Li]": ikarus_R(CYL, q, "li")[0],
                   "ik[NV]": ikarus_R(CYL, q, "normal")[0]}
            cyl[q] = row
            print(f"{q:>5} {q*q:>6} | {_fmt(row['fork[Laurent]'])} "
                  f"{_fmt(row['fork[Pol]'])} | {_fmt(row['ik[Laurent]'], 10)} "
                  f"{_fmt(row['ik[Li]'])} {_fmt(row['ik[NV]'])}")
        print(f"\n   the paper's window for the faithful answer here (FMMax "
              f"NORMAL): {CYL_WINDOW[0]} .. {CYL_WINDOW[1]}")
    else:
        print("   skipped (needs ikarus-rcwa)")

    # ---------------------------------------------------------------- part 4
    print("\n" + "-" * 78)
    print("4. Verdicts")
    print("-" * 78)
    hi = max(sweep)
    checks = []
    if have_ikarus:
        gap = abs(sweep[hi]["fork[Laurent]"] - sweep[hi]["ik[Laurent]"])
        checks.append(("direct rule agrees across codebases "
                       f"(fork vs ikarus at q={hi}: |dR|={gap:.1e})", gap < 1e-4))
        nv = sweep[hi]["ik[NV]"]
        checks.append((f"Ikarus faithful reproduces the published 10.0 % "
                       f"(NV at q={hi}: {nv:.4f})", abs(nv - FAITHFUL) < 5e-3))
        checks.append(("Ikarus faithful has settled by q=15 "
                       f"({sweep[15]['ik[NV]']:.4f} vs {nv:.4f})",
                       abs(sweep[15]["ik[NV]"] - nv) < 5e-3))
    lau = sweep[hi]["fork[Laurent]"]
    checks.append((f"fork Laurent is still wrong at q={hi} ({lau:.4f} vs "
                   f"{FAITHFUL:.3f}) -- the paper's point", abs(lau - FAITHFUL) > 3e-3))
    pol = sweep[hi]["fork[Pol]"]
    checks.append((f"fork Pol IS faithful: it converges to 10.0 % "
                   f"(q={hi}: {pol:.4f}) -- so Table 1's grcwa row does not\n"
                   f"      describe this fork, only the Laurent-only upstream",
                   abs(pol - FAITHFUL) < 5e-3))
    if cyl:
        qc = max(cyl)
        nv, li, cpol = (cyl[qc]["ik[NV]"], cyl[qc]["ik[Li]"], cyl[qc]["fork[Pol]"])
        checks.append((f"curved boundary: Ikarus NV lands in the paper's window "
                       f"(q={qc}: {nv:.4f} in {CYL_WINDOW})",
                       CYL_WINDOW[0] < nv < CYL_WINDOW[1]))
        checks.append((f"curved boundary: Li's separable rule lags the "
                       f"normal-vector method (Li {li:.4f} vs NV {nv:.4f})",
                       nv - li > 0.02))
        checks.append((f"curved boundary: fork Pol behaves like a normal-vector "
                       f"method, not a separable one\n      "
                       f"(Pol {cpol:.4f}: NV {nv:.4f}, Li {li:.4f})",
                       abs(cpol - nv) < abs(cpol - li)))
    ok = True
    for text, passed in checks:
        ok &= passed
        print(f"   [{'PASS' if passed else 'FAIL'}] {text}")
    print("\n" + ("all checks passed" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
