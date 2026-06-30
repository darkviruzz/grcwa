"""Decide whether the Pol (S4 Eq. 51) Fourier factorization is correct.

Physics background
------------------
Laurent's rule (default) and Li's inverse rule -- the "Pol" method -- are *both*
valid Fourier factorizations of the permittivity. Li (1996) proved they converge
to the **same** exact Maxwell solution as the truncation order ``nG -> infinity``.
They differ only in *rate*:

* For **TE** polarization (E parallel to the grooves) the relevant field is
  continuous across the dielectric interfaces, so the inverse-rule correction is
  a no-op: Pol must reproduce Laurent **exactly**, at every nG.
* For **TM** polarization (E has a component normal to the interfaces, which is
  discontinuous) the inverse rule is the whole point: a correct Pol implementation
  is supposed to converge **faster** than Laurent.

What the convergence study found
--------------------------------
On a lossless high-contrast Si grating (period 1.5 um, 50% fill, t=0.5 um,
lambda=1 um, normal incidence):

* TE: Pol == Laurent to ~1e-9 (correct).
* TM: Laurent converges smoothly to R ~ 0.211 (nG 81->201: 0.193, 0.197, 0.200,
  0.202; and 0.210/0.211 by nG ~ 700-1000). The Pol code instead oscillates
  (0.83, 0.83, 0.73, 0.62 over nG 81..201) and only crawls back toward 0.211 by
  nG ~ 1000 -- i.e. it converges to the *same* physical value but *far slower*
  than Laurent, the opposite of Pol's intended benefit.

Verdict: the physically correct value is the **Laurent** one. The Pol result at
any practical nG is simply un-converged. Note that **energy conservation does not
catch this** -- both rules satisfy R+T=1 the whole way -- which is why the tests
below check *convergence* and *agreement with the Laurent limit*, not energy.

The two Pol checks are marked ``xfail(strict=True)``: they encode the known defect
and will turn the suite red (XPASS) if the Pol method is ever fixed, prompting
removal of the markers.
"""
import numpy as np
import pytest
import grcwa


@pytest.fixture(autouse=True)
def _numpy_backend():
    grcwa.set_backend('numpy')
    yield
    try:
        grcwa.set_backend('autograd')
    except ValueError:
        pass


# lossless, high-contrast 1D grating -- the regime where the factorization rule
# matters. Qabs only regularizes the Rayleigh anomaly (A ~ 1e-6).
FREQC = 1.0 * (1 + 1j / 2 / 1e7)
LAM, EPS_HI, FF, THICK, NX = 1.5, 3.5 ** 2, 0.5, 0.5, 1024
NG = [81, 121, 161, 201]
CONVERGED = 0.02     # a settled sweep varies by less than this over the top nG


def _grating_RT(nG, fmm, pol):
    """R, T of the lossless Si grating. pol='p' -> TM, 's' -> TE."""
    xs = np.linspace(0, 1, NX, endpoint=False)
    prof = np.where(xs < FF, EPS_HI, 1.0).astype(complex)
    o = grcwa.obj(nG, [LAM, 0], None, FREQC, 0., 0., verbose=0, fmm_method=fmm)
    o.Add_LayerUniform(1.0, 1.0)
    o.Add_LayerGrid(THICK, NX)
    o.Add_LayerUniform(1.0, 1.0)
    o.Init_Setup()
    pa, sa = (1., 0.) if pol == 'p' else (0., 1.)
    o.MakeExcitationPlanewave(pa, 0., sa, 0., order=0)
    o.GridLayer_geteps(prof)
    R, T = o.RT_Solve(normalize=1)
    return float(np.real(R)), float(np.real(T))


def test_TE_pol_equals_laurent():
    """TE: the tangential field is continuous, so Pol's correction is a no-op
    and it must reproduce Laurent bit-for-bit. (Confirms Pol isn't globally
    broken -- the defect is specific to the discontinuous TM case.)"""
    for nG in NG:
        R_lau, _ = _grating_RT(nG, None, 's')
        R_pol, _ = _grating_RT(nG, 'pol', 's')
        assert abs(R_lau - R_pol) < 1e-9, f"TE Pol != Laurent at nG={nG}"


def test_energy_conservation_does_not_discriminate():
    """Both rules conserve energy for this lossless structure, so R+T=1 cannot
    be used to tell a correct factorization from a non-converging one. This test
    documents that (it passes for *both* rules)."""
    for fmm in (None, 'pol'):
        R, T = _grating_RT(161, fmm, 'p')
        assert abs(R + T - 1.0) < 1e-3, f"energy not conserved for fmm={fmm}"


def test_laurent_converges_TM():
    """Laurent settles to a stable TM reflectance over the top of the sweep."""
    Rs = [_grating_RT(nG, None, 'p')[0] for nG in NG]
    assert max(Rs) - min(Rs) < CONVERGED, f"Laurent not settled: {Rs}"


@pytest.mark.xfail(strict=True,
                   reason="Pol (S4 Eq.51) as implemented does not converge for "
                          "TM: it oscillates over nG instead of settling.")
def test_pol_converges_TM():
    """A correct factorization must converge as nG grows. The Pol code instead
    oscillates for TM (R ~ 0.83, 0.83, 0.73, 0.62 over nG 81..201), so the
    spread over the top of the sweep is large -> this assertion fails."""
    Rs = [_grating_RT(nG, 'pol', 'p')[0] for nG in NG]
    assert max(Rs) - min(Rs) < CONVERGED, f"Pol not settled: {Rs}"


@pytest.mark.xfail(strict=True,
                   reason="Pol does not reach the (correct) Laurent limit at "
                          "any practical nG for TM.")
def test_pol_matches_laurent_limit_TM():
    """Both rules share the same nG->inf limit; the Laurent value at high nG is
    the trusted reference. The Pol result is nowhere near it at practical nG."""
    R_ref, _ = _grating_RT(201, None, 'p')    # Laurent ~ 0.202 (-> 0.211)
    R_pol, _ = _grating_RT(201, 'pol', 'p')   # Pol ~ 0.62 (un-converged)
    assert abs(R_pol - R_ref) < CONVERGED, f"Pol {R_pol:.3f} vs Laurent {R_ref:.3f}"
