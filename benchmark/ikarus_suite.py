"""Ikarus backend for the shared benchmark battery (benchmark/structures.py).

Ikarus (CAVITY technologies GmbH, ``pip install ikarus-rcwa``) is an independent
RCWA/FMM code with a different *construct* from grcwa: SI units, a stack written
cover -> substrate whose first and last layers are semi-infinite (``height=inf``),
patterned layers given as an integer topology plus one material per index, a
maximum-order pair ``n_orders=(Mx, My)`` instead of a truncation count, and
``simulate()`` returning ``(T, R, result)`` -- transmission FIRST.

This module rebuilds the battery in that construct while keeping the *physics*
identical:

* geometry comes from :func:`structures.layer_mask`, so grcwa and Ikarus
  rasterize nothing of their own and see the very same pixels;
* lengths are the battery's lambda = 1 units scaled by ``UNIT`` (1 um), so
  ``freq = 1`` maps to ``wavelength = 1 um``;
* materials are passed as raw complex indices ``n + ik`` (never library names),
  which keeps them non-dispersive and the comparison scale-invariant;
* ``pol`` maps to ``linear_pol_angle`` (``s`` -> 0 = TE, ``p`` -> 90 = TM).

Order convention. Ikarus's ``n_orders`` is the *maximum* order per axis, so it
retains ``2M+1`` harmonics along an axis. The battery's per-axis count ``q`` is
that harmonic count, hence ``M = (q-1)//2`` and ``nG`` comes out equal to grcwa's:
``q`` for 1D and ``q**2`` for a 2D ``(q,q)`` block. Only odd ``q`` is
representable; an even ``q`` is reported as skipped rather than silently rounded.

Factorization modes (the ``fmm`` argument):
  ``None``/``"laurent"`` direct (Laurent) rule -- the same rule grcwa's default
                         implements, so these columns should agree;
  ``"li"``               Li's inverse rule (faithful, separable/axis-aligned);
  ``"normal"``/``"nv"``  the normal-vector method (Ikarus's default ``"auto"``),
                         faithful also on curved boundaries.

The single known physical difference from the grcwa columns: this battery feeds
grcwa a slightly complex frequency (``structures.FREQC``, Q = 1e7) to regularize
Rayleigh anomalies, while Ikarus takes a real wavelength. That is a 5e-8 relative
detuning -- far below the factorization differences under study.
"""
import numpy as np

import structures as ST

UNIT = 1e-6                      # the battery's length unit (lambda = 1 um) in metres
WAVELENGTH = UNIT / ST.FREQ      # freq = 1 -> lambda = 1 um

_FMM_ALIAS = {None: "laurent", "none": "laurent", "nv": "normal",
              "auto": "normal", "pol": None}


def available():
    """True if Ikarus is importable (it is an optional cross-check dependency)."""
    try:
        import ikarus                                        # noqa: F401
        return True
    except Exception:
        return False


def version():
    import ikarus
    return getattr(ikarus, "__version__", "?")


def _index(nk):
    """(n, k) -> the complex index Ikarus wants (exp(-iwt): absorbers have k>0)."""
    n, k = nk
    return complex(n, k)


def solve(s, q, fmm, native=True):
    """Solve structure ``s`` at per-axis order count ``q`` with factorization
    ``fmm``. Signature and return value mirror :func:`structures.solve`, so the
    benchmark workers can drive either backend.

    Returns ``(R, T, nG_actual, mode)``, or ``(None, None, None, reason)`` when
    the point is not representable in this backend.
    """
    from ikarus import RCWA

    factorization = _FMM_ALIAS.get(fmm, fmm)
    if factorization is None:                     # 'pol' is a grcwa-only rule
        return None, None, None, "no-pol"
    if factorization not in ("laurent", "li", "normal"):
        return None, None, None, f"unknown-factorization:{factorization}"

    dim = s["dim"]
    if dim == 0:
        m = (0, 0)
    else:
        if q % 2 == 0:
            return None, None, None, "even-q-not-representable"
        M = (q - 1) // 2
        m = (M, 0) if dim == 1 else (M, M)

    period = s.get("period", 1.0) * UNIT
    rc = RCWA(period_x=period, period_y=period, n_orders=m,
              factorization=factorization)
    rc.add_uniform_layer(np.inf, _index(ST.AIR))               # cover
    if dim == 0:
        rc.add_uniform_layer(s["d"] * UNIT, _index(s["film"]))
    else:
        mask, nk_pair = ST.layer_mask(s)
        # Pin the layer's own resolution to the shared mask so Ikarus uses those
        # exact pixels instead of resampling to its 4M+1 anti-aliasing grid --
        # identical geometry in both codes is the point. Safe here because every
        # mask (2048 in 1D, 256 in 2D) already exceeds 4M+1 at the orders run.
        rc.add_layer(s["d"] * UNIT, mask, [_index(nk) for nk in nk_pair],
                     resolution=mask.shape)
    rc.add_uniform_layer(np.inf, _index(s["sub"]))             # substrate

    rc.set_source(wavelength=WAVELENGTH, theta=0.0, phi=0.0,
                  polarization="linear",
                  linear_pol_angle=90.0 if s["pol"] == "p" else 0.0)
    T, R, res = rc.simulate()
    nG = (2 * m[0] + 1) * (2 * m[1] + 1)
    return float(np.real(res.R_total)), float(np.real(res.T_total)), nG, "native"
