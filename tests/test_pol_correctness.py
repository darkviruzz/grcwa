"""Correctness of the Pol (S4 Eq. 51 / Li 1996) Fourier factorization.

Background
----------
Laurent's rule (default) and Li's inverse rule -- the "Pol" method -- are both
valid factorizations and converge to the **same** exact Maxwell solution as the
truncation order ``nG -> infinity`` (Li, JOSA A 13, 1870 (1996)). They differ
only in *rate*:

* **TE** (E parallel to the grooves) is continuous across interfaces, so the
  inverse-rule correction is a no-op: Pol must reproduce Laurent **exactly**.
* **TM** (E has a discontinuous normal component) is the case the inverse rule
  is for: a correct Pol implementation converges **faster** than Laurent.

History of the fix
------------------
The first port of the Pol method did not converge for TM -- it oscillated wildly
over nG (e.g. R = 0.83, 0.83, 0.73, 0.62 ...). Two bugs were responsible, found
via a variant search on the lossless high-contrast grating below:

1. ``epsinv`` for the kp/Ez term used the reciprocal Toeplitz ``[[1/eps]]``
   instead of ``inv([[eps]])`` (Laurent's convention). This made the kp matrix
   inconsistent with the eigenproblem ``M = ep2*kp - kkT`` the eps2 correction
   sits on top of, and was the main cause of the oscillation.
2. The tangent field was **globally** max-normalized, so ``P = t t^T`` was not a
   projection (``|t| < 1`` almost everywhere) and the correction was negligible.
   Fixed with **per-pixel** unit normalization.

With both fixed, Pol now reproduces Laurent exactly for TE and converges to the
same TM limit (~0.2135 here) markedly faster than Laurent. These tests guard
that behaviour; if either bug regresses, the TM tests fail.
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
TM_LIMIT = 0.2135    # converged TM reflectance (both rules agree here)


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
    """TE field is continuous -> Pol's correction is a no-op; it must reproduce
    Laurent bit-for-bit at every nG."""
    for nG in [41, 81, 121, 201]:
        R_lau, _ = _grating_RT(nG, None, 's')
        R_pol, _ = _grating_RT(nG, 'pol', 's')
        assert abs(R_lau - R_pol) < 1e-9, f"TE Pol != Laurent at nG={nG}"


def test_energy_conservation_both_rules():
    """Lossless -> R+T=1 for both rules (Qabs gives only A ~ 1e-6)."""
    for fmm in (None, 'pol'):
        R, T = _grating_RT(161, fmm, 'p')
        assert abs(R + T - 1.0) < 5e-3, f"energy not conserved for fmm={fmm}"


def test_laurent_converges_TM():
    """Laurent settles (slowly) toward the TM limit from below."""
    Rs = [_grating_RT(nG, None, 'p')[0] for nG in [121, 161, 201]]
    assert max(Rs) - min(Rs) < 0.02
    assert Rs[0] < Rs[-1] <= TM_LIMIT + 0.01     # monotone, below the limit


def test_pol_converges_TM():
    """The fix: Pol now *converges* for TM instead of oscillating. The change
    over the top of the sweep is tiny (the old broken Pol moved by ~0.1)."""
    Rs = [_grating_RT(nG, 'pol', 'p')[0] for nG in [121, 161, 201]]
    assert max(Rs) - min(Rs) < 5e-3, f"Pol not settled for TM: {Rs}"


def test_pol_matches_laurent_limit_TM():
    """Both rules share the same nG->inf limit; Pol sits right at it."""
    R_pol, _ = _grating_RT(201, 'pol', 'p')
    assert abs(R_pol - TM_LIMIT) < 5e-3, f"Pol TM {R_pol:.4f} != limit {TM_LIMIT}"


def test_pol_converges_faster_than_laurent_TM():
    """The payoff of Pol: at a moderate nG it is closer to the converged TM
    value than Laurent (which is still crawling up from below)."""
    ref, _ = _grating_RT(281, 'pol', 'p')      # Pol is converged by here
    R_lau, _ = _grating_RT(121, None, 'p')
    R_pol, _ = _grating_RT(121, 'pol', 'p')
    assert abs(R_pol - ref) < abs(R_lau - ref)
