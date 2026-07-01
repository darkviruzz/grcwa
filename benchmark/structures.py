"""Shared structure battery + solver for the grcwa benchmark and convergence study.

Single source of truth, so benchmark/worker.py (cross-version, single order count)
and benchmark/conv_worker.py (order sweep on the fork) test the SAME structures.

Order-counting convention
-------------------------
Everything is parametrized by the PER-AXIS order count ``q`` (number of retained
Fourier orders along one lattice axis; odd -> symmetric -p..p):

  * 1D structure: ``q`` orders total          (nG = q).
  * 2D structure: a ``q x q`` square block,    ``q**2`` orders total, written
    "(q,q)" (parallelogramic truncation gives exactly q x q).

So 1D ``nG=q`` and 2D ``(q,q)`` share the same per-axis resolution, while 1D
``nG=q**2`` matches the 2D ``(q,q)`` *total* order count. Plot/compare on the
total retained-order axis (q for 1D-per-axis, q**2 for 2D and 1D-total).

Materials are given as (n,k); eps = (n+ik)**2 (exp(-i w t): lossy -> Im(eps)>0).
"""
import numpy as np

FREQ = 1.0                       # lambda = 1 micron
QABS = 1e7                       # tiny loss regularizes Rayleigh anomalies
FREQC = FREQ * (1 + 1j / 2 / QABS)
NX_1D = 2048
NX_2D = 256

# materials at lambda = 1 um  (n, k)
AIR = (1.0, 0.0)
SIO2 = (1.5, 0.0)
SIN = (2.0, 0.0)
SI = (3.5, 0.0)
AU = (0.3, 7.0)                  # eps ~ -48.9 + 4.2j


def eps(nk):
    n, k = nk
    return (n + 1j * k) ** 2


# group A: analytic anchors; B: 1D gratings; C: 2D rectangular pillars
STRUCTURES = [
    dict(name="A1_slab_air", group="A", dim=0, pol="s",
         film=SI, d=0.20, sub=AIR, desc="planar Si slab in air (exact Airy)"),
    dict(name="A1b_slab_glass", group="A", dim=0, pol="s",
         film=SI, d=0.20, sub=SIO2, desc="Si slab on glass (exact Airy)"),
    dict(name="A2_formbiref_TE", group="A", dim=1, pol="s",
         hi=SI, lo=AIR, period=0.20, ff=0.5, d=0.30, sub=AIR,
         desc="deep-subwave 1D grating = birefringent film, TE (EMT)"),
    dict(name="A2_formbiref_TM", group="A", dim=1, pol="p",
         hi=SI, lo=AIR, period=0.20, ff=0.5, d=0.30, sub=AIR,
         desc="deep-subwave 1D grating = birefringent film, TM (EMT)"),
    dict(name="B1_Si_grating_TE", group="B", dim=1, pol="s",
         hi=SI, lo=AIR, period=1.5, ff=0.5, d=0.50, sub=AIR,
         desc="Si transmission grating, TE (fast baseline)"),
    dict(name="B1_Si_grating_TM", group="B", dim=1, pol="p",
         hi=SI, lo=AIR, period=1.5, ff=0.5, d=0.50, sub=AIR,
         desc="Si transmission grating, TM (slow under Laurent)"),
    dict(name="B2_HCG_TM", group="B", dim=1, pol="p",
         hi=SI, lo=AIR, period=0.80, ff=0.5, d=0.30, sub=AIR,
         desc="high-contrast subwavelength grating, TM (Li showcase)"),
    dict(name="B3_Au_slits_TM", group="B", dim=1, pol="p",
         hi=AU, lo=AIR, period=0.50, ff=0.8, d=0.20, sub=AIR,
         desc="metal slit array, TM (plasmonic/EOT; hardest 1D)"),
    dict(name="C1_Si_pillars", group="C", dim=2, pol="s",
         pillar=SI, bg=AIR, period=0.50, ax=0.30, ay=0.30, d=0.40, sub=SIO2,
         desc="Si square-pillar metasurface (subwavelength)"),
    dict(name="C1b_Si_pillars_diffract", group="C", dim=2, pol="s",
         pillar=SI, bg=AIR, period=1.50, ax=0.60, ay=0.60, d=0.40, sub=SIO2,
         desc="Si pillars, supra-wavelength (diffraction)"),
    dict(name="C2_Au_holes", group="C", dim=2, pol="s",
         pillar=AIR, bg=AU, period=0.60, ax=0.30, ay=0.30, d=0.20, sub=SIO2,
         desc="metal hole array, 2D EOT (hardest 2D)"),
]
STRUCT = {s["name"]: s for s in STRUCTURES}


def supports_native_dim(grcwa):
    """True for versions with dimensionality inference (native L2=None +
    Add_LayerGrid Ny default). Old versions fall back to degenerate-2D for 1D
    and cannot do 0D natively."""
    try:
        o = grcwa.obj(3, [1.0, 0], None, FREQC, 0., 0., verbose=0)
        o.Add_LayerUniform(1.0, 1.0)
        o.Add_LayerGrid(0.1, 4)
        o.Add_LayerUniform(1.0, 1.0)
        o.Init_Setup(Gmethod=1)
        return True
    except Exception:
        return False


def _obj(grcwa, nG, L1, L2, fmm):
    kwargs = {}
    if fmm is not None:
        kwargs["fmm_method"] = fmm
    try:
        return grcwa.obj(nG, L1, L2, FREQC, 0., 0., verbose=0, **kwargs)
    except TypeError:
        if fmm is not None:
            return None            # this version has no fmm_method -> no Pol
        return grcwa.obj(nG, L1, L2, FREQC, 0., 0., verbose=0)


def solve(grcwa, s, q, fmm, native):
    """Solve structure ``s`` at per-axis order count ``q`` with factorization
    ``fmm`` (None=Laurent, 'pol'=Pol). ``native`` is supports_native_dim(grcwa).

    Returns (R, T, nG_actual, mode) or (None, None, None, reason).
    1D uses nG=q; 2D uses a (q,q) square block (nG=q**2, parallelogramic).
    """
    dim = s["dim"]
    if dim == 0:
        if not native:
            return None, None, None, "no-native-0D"
        o = _obj(grcwa, 1, None, None, fmm)
        if o is None:
            return None, None, None, "no-pol"
        o.Add_LayerUniform(1.0, eps(AIR))
        o.Add_LayerUniform(s["d"], eps(s["film"]))
        o.Add_LayerUniform(1.0, eps(s["sub"]))
        o.Init_Setup()
        eps_flat, gmethod = None, 0
    elif dim == 1:
        xs = np.linspace(0, 1, NX_1D, endpoint=False)
        prof = np.where(xs < s["ff"], eps(s["hi"]), eps(s["lo"])).astype(complex)
        if native:
            o = _obj(grcwa, q, [s["period"], 0], None, fmm)
            if o is None:
                return None, None, None, "no-pol"
            o.Add_LayerUniform(1.0, eps(AIR))
            o.Add_LayerGrid(s["d"], NX_1D)
            o.Add_LayerUniform(1.0, eps(s["sub"]))
            mode = "native"
        else:
            o = _obj(grcwa, q, [s["period"], 0], [0, s["period"] * 1e-3], fmm)
            if o is None:
                return None, None, None, "no-pol"
            o.Add_LayerUniform(1.0, eps(AIR))
            o.Add_LayerGrid(s["d"], NX_1D, 1)
            o.Add_LayerUniform(1.0, eps(s["sub"]))
            mode = "degenerate-2D"
        o.Init_Setup(Gmethod=1)
        eps_flat = prof
    else:  # dim == 2: q x q square block via parallelogramic truncation
        Lam = s["period"]
        x = np.linspace(0, 1, NX_2D, endpoint=False)
        X, Y = np.meshgrid(x, x, indexing="ij")
        eg = np.ones((NX_2D, NX_2D), dtype=complex) * eps(s["bg"])
        inside = (np.abs(X - 0.5) < s["ax"] / (2 * Lam)) & \
                 (np.abs(Y - 0.5) < s["ay"] / (2 * Lam))
        eg[inside] = eps(s["pillar"])
        o = _obj(grcwa, q * q, [Lam, 0], [0, Lam], fmm)
        if o is None:
            return None, None, None, "no-pol"
        o.Add_LayerUniform(1.0, eps(AIR))
        o.Add_LayerGrid(s["d"], NX_2D, NX_2D)
        o.Add_LayerUniform(1.0, eps(s["sub"]))
        try:
            o.Init_Setup(Gmethod=1)      # parallelogramic -> q x q block
        except TypeError:
            o.Init_Setup()
        eps_flat = eg.flatten()
        mode = "native"

    if dim == 0:
        mode = "native"
    pa, sa = (1., 0.) if s["pol"] == "p" else (0., 1.)
    o.MakeExcitationPlanewave(pa, 0., sa, 0., order=0)
    if eps_flat is not None:
        o.GridLayer_geteps(eps_flat)
    R, T = o.RT_Solve(normalize=1)
    return float(np.real(R)), float(np.real(T)), int(o.nG), mode
