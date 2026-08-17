"""Cross-code validation against Ikarus, an independent RCWA implementation.

Ikarus (CAVITY technologies GmbH, ``pip install ikarus-rcwa``) is a separate
codebase with a different construct -- SI units, semi-infinite cover/substrate,
integer-topology layers, ``(T, R, result)`` returns. Its whitepaper
(doi 10.5281/zenodo.21966455) argues that the direct (Laurent) Fourier rule
converges to the wrong answer in TM on a high-contrast grating, and lists
**grcwa** among the direct-rule solvers that miss the true value by 60-75 %.

These tests decide how much of that applies here, using the two group-D
structures the whitepaper specifies (benchmark/structures.py):

* the claim about **Laurent** is true, and both codes agree on the wrong value
  to ~1e-6 -- so the disagreement is the rule, not a bug in either code;
* the claim about **grcwa** describes the Laurent-only upstream, *not* this fork:
  the fork's fixed Pol factorization converges to the same faithful ~0.100
  Ikarus and FMMax report;
* Ikarus's normal-vector method still converges markedly faster, and is the only
  rule here that is also faithful on a curved boundary.

Ikarus is an optional cross-check dependency, so the whole module skips when it
is not installed. The geometry comes from ``structures.layer_mask`` for both
codes, so a disagreement can never be a pixel-grid artifact.
"""
import os
import sys

import numpy as np
import pytest

import grcwa

BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "benchmark")
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)

pytest.importorskip("ikarus", reason="pip install ikarus-rcwa to run the "
                                     "cross-code checks")

import ikarus_suite as IK                                          # noqa: E402
import structures as ST                                            # noqa: E402

# The whitepaper's "true value" for the 1D case: FMMax NORMAL, converged.
FAITHFUL_1D = 0.100
GRATING = ST.STRUCT["D1_ikarus_hcg_TM"]
CYLINDER = ST.STRUCT["D2_ikarus_cylinder_TE"]


@pytest.fixture(autouse=True)
def _numpy_backend():
    grcwa.set_backend("numpy")
    yield
    try:
        grcwa.set_backend("autograd")
    except ValueError:
        pass


def fork_R(case, q, fmm):
    R, _T, _nG, _mode = ST.solve(grcwa, case, q, fmm, True)
    return R


def ikarus_R(case, q, factorization):
    R, _T, _nG, _mode = IK.solve(case, q, factorization)
    return R


# --------------------------------------------------------------- the harness
def test_ikarus_harness_reproduces_airy():
    """Sanity: the adapter's conventions (SI scaling, T-first unpacking, the
    material sign convention) are right, checked on a case with a closed form.

    A uniform slab is factorization-independent, so this isolates the wiring
    from the physics the other tests are about.
    """
    slab = ST.STRUCT["A1_slab_air"]
    R = ikarus_R(slab, 1, "normal")
    n0, n1, ns, d = 1.0, 3.5, 1.0, slab["d"]
    r01, r12 = (n0 - n1) / (n0 + n1), (n1 - ns) / (n1 + ns)
    ph = np.exp(2j * 2 * np.pi * n1 * d)
    R_airy = abs((r01 + r12 * ph) / (1 + r01 * r12 * ph)) ** 2
    assert abs(R - R_airy) < 1e-6, f"ikarus slab {R:.8f} != Airy {R_airy:.8f}"


def test_order_counts_match_between_codes():
    """Ikarus counts a MAXIMUM order per axis (2M+1 harmonics), grcwa a
    truncation count. The adapter maps them so both retain the same nG -- if that
    slips, every comparison below is silently unfair."""
    for case in (GRATING, CYLINDER):
        for q in (5, 11, 15):
            _R, _T, nG_ik, _m = IK.solve(case, q, "normal")
            _R, _T, nG_gr, _m = ST.solve(grcwa, case, q, None, True)
            assert nG_ik == nG_gr, f"{case['name']} q={q}: {nG_ik} != {nG_gr}"


def test_even_order_count_is_reported_not_rounded():
    """Ikarus can only retain an odd number of orders per axis; an even request
    must be skipped explicitly rather than quietly rounded."""
    R, _T, _nG, reason = IK.solve(GRATING, 20, "normal")
    assert R is None and "even" in reason


# ------------------------------------------- the direct rule, in two codebases
@pytest.mark.parametrize("case,q", [(GRATING, 41), (CYLINDER, 15)])
def test_direct_rule_agrees_across_codebases(case, q):
    """grcwa's Laurent and Ikarus's ``laurent`` are the same rule, so they must
    agree -- in 1D and on the curved 2D boundary alike. The residual is the
    battery's Q=1e7 complex-frequency regularization, which Ikarus has no knob
    for (see benchmark/ikarus_suite.py)."""
    gap = abs(fork_R(case, q, None) - ikarus_R(case, q, "laurent"))
    assert gap < 1e-4, f"{case['name']}: direct-rule codes disagree by {gap:.2e}"


def test_laurent_is_badly_wrong_at_a_practical_truncation():
    """The whitepaper's central claim, on the fork itself: at an order count a
    user would pick, Laurent is far above the faithful answer -- while conserving
    energy perfectly, which is why it looks trustworthy."""
    R = fork_R(GRATING, 25, None)
    assert R > 0.15, f"Laurent at q=25 is {R:.4f}, expected the wrong ~0.16"
    T = ST.solve(grcwa, GRATING, 25, None, True)[1]
    assert abs(R + T - 1.0) < 5e-3, "the point is that energy balance looks fine"


def test_laurent_crawls_toward_the_faithful_value():
    """...and it does converge, just as O(1/M): monotone from above, still off
    at 101 orders."""
    Rs = [fork_R(GRATING, q, None) for q in (25, 41, 101)]
    assert Rs[0] > Rs[1] > Rs[2] > FAITHFUL_1D
    assert abs(Rs[-1] - FAITHFUL_1D) > 3e-3


# ------------------------------------------------------- the faithful rules
def test_ikarus_faithful_reproduces_the_published_value():
    """Ikarus's normal-vector default lands on the 10.0 % its whitepaper
    publishes (and attributes to FMMax at high truncation)."""
    R = ikarus_R(GRATING, 41, "normal")
    assert abs(R - FAITHFUL_1D) < 5e-3, f"ikarus NV {R:.4f} != {FAITHFUL_1D}"


def test_ikarus_faithful_has_settled_by_15_orders():
    """The rate claim: faithful is converged by M ~ 7 (q = 15), where Laurent is
    not converged at 101."""
    assert abs(ikarus_R(GRATING, 15, "normal")
               - ikarus_R(GRATING, 41, "normal")) < 5e-3


def test_fork_pol_is_a_faithful_rule():
    """The reason Table 1's grcwa row does not describe this fork: its fixed Pol
    factorization converges to the same faithful value as Ikarus and FMMax,
    instead of to Laurent's wrong one."""
    R = fork_R(GRATING, 101, "pol")
    assert abs(R - FAITHFUL_1D) < 5e-3, f"fork Pol {R:.4f} != {FAITHFUL_1D}"
    assert abs(R - fork_R(GRATING, 101, None)) > 3e-3, "Pol must differ from Laurent"


def test_fork_pol_agrees_with_ikarus_faithful():
    """Two faithful rules in two codebases, one limit."""
    gap = abs(fork_R(GRATING, 101, "pol") - ikarus_R(GRATING, 41, "normal"))
    assert gap < 5e-3, f"faithful rules disagree by {gap:.2e}"


def test_normal_vector_beats_li_on_a_curved_boundary():
    """Why the normal-vector method exists: on a circular pillar the boundary is
    oblique to both axes, and Li's separable inverse rule lags well behind."""
    nv = ikarus_R(CYLINDER, 15, "normal")
    li = ikarus_R(CYLINDER, 15, "li")
    assert nv - li > 0.02, f"NV {nv:.4f} vs Li {li:.4f}: expected NV well ahead"


def test_fork_pol_tracks_the_normal_vector_method_on_the_cylinder():
    """The fork's Pol builds its tangent field from the rendered eps grid, so on
    the curved boundary it behaves like a normal-vector method rather than like
    Li's separable rule -- it should sit with Ikarus's NV, not with its Li."""
    pol = fork_R(CYLINDER, 15, "pol")
    nv = ikarus_R(CYLINDER, 15, "normal")
    li = ikarus_R(CYLINDER, 15, "li")
    assert abs(pol - nv) < abs(pol - li), \
        f"fork Pol {pol:.4f}: NV {nv:.4f}, Li {li:.4f}"
