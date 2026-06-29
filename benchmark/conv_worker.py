"""Convergence-study worker.

For ONE grcwa installation (selected by GRCWA_MOD + PYTHONPATH) and ONE
factorization mode (FMM = none|pol), sweep the truncation order nG over a
battery of physically-motivated structures and emit R(nG), T(nG) as JSON.

Driven by conv_run.py once per (variant, mode), each in its own subprocess so
several grcwa versions never clash on import. Not a pytest module.

Units: c = 1, freq = 1/lambda. We work at lambda = 1 micron (freq = 1.0); all
lengths are therefore in microns. Materials are given as (n, k) and converted to
eps = (n + i k)^2 (exp(-i omega t) convention -> lossy media have Im(eps) > 0).
"""
import os
import sys
import json
import time

_MOD = os.environ.get("GRCWA_MOD", "grcwa")
grcwa = __import__(_MOD)
import numpy as np

_FMM = os.environ.get("FMM", "none")
FMM = None if _FMM == "none" else _FMM

FREQ = 1.0                       # lambda = 1 micron
QABS = 1e7                       # tiny loss regularizes Rayleigh anomalies
FREQC = FREQ * (1 + 1j / 2 / QABS)
REPEAT = int(os.environ.get("REPEAT", "2"))

NX_1D = 2048                     # 1D eps profile resolution
NX_2D = 256                      # 2D eps grid resolution (per axis)


def eps(nk):
    n, k = nk
    nc = n + 1j * k
    return nc * nc


# materials at lambda = 1 um  (n, k)
AIR = (1.0, 0.0)
SIO2 = (1.5, 0.0)
SIN = (2.0, 0.0)
SI = (3.5, 0.0)
AU = (0.3, 7.0)                  # eps ~ -48.9 + 4.2j (realistic metal at 1 um)

NG_1D = [5, 11, 21, 41, 61, 81, 121, 161, 201]
NG_2D = [13, 25, 49, 81, 121, 169, 225, 301, 401]

# --- the battery --------------------------------------------------------------
# A: analytic references (exact slab / asymptotic EMT)
# B: 1D gratings (cross-check vs an external RCWA; Laurent-vs-Pol convergence)
# C: 2D rectangular pillars
STRUCTURES = [
    dict(name="A1_slab_air", group="A", dim=0, pol="TE", theta=0.0,
         film=SI, d=0.20, sub=AIR, nG_list=[1],
         desc="planar Si slab in air (exact Airy)"),
    dict(name="A1b_slab_glass", group="A", dim=0, pol="TE", theta=0.0,
         film=SI, d=0.20, sub=SIO2, nG_list=[1],
         desc="Si slab on glass (exact Airy)"),
    dict(name="A2_formbiref_TE", group="A", dim=1, pol="TE", theta=0.0,
         hi=SI, lo=AIR, period=0.20, ff=0.5, d=0.30, sub=AIR, nG_list=NG_1D,
         desc="deep-subwave 1D grating = birefringent film, TE (EMT)"),
    dict(name="A2_formbiref_TM", group="A", dim=1, pol="TM", theta=0.0,
         hi=SI, lo=AIR, period=0.20, ff=0.5, d=0.30, sub=AIR, nG_list=NG_1D,
         desc="deep-subwave 1D grating = birefringent film, TM (EMT)"),

    dict(name="B1_Si_grating_TE", group="B", dim=1, pol="TE", theta=0.0,
         hi=SI, lo=AIR, period=1.5, ff=0.5, d=0.50, sub=AIR, nG_list=NG_1D,
         desc="Si transmission grating, TE (fast-convergence baseline)"),
    dict(name="B1_Si_grating_TM", group="B", dim=1, pol="TM", theta=0.0,
         hi=SI, lo=AIR, period=1.5, ff=0.5, d=0.50, sub=AIR, nG_list=NG_1D,
         desc="Si transmission grating, TM (slow under Laurent)"),
    dict(name="B2_HCG_TM", group="B", dim=1, pol="TM", theta=0.0,
         hi=SI, lo=AIR, period=0.80, ff=0.5, d=0.30, sub=AIR, nG_list=NG_1D,
         desc="high-contrast subwavelength grating, TM (Li showcase)"),
    dict(name="B3_Au_slits_TM", group="B", dim=1, pol="TM", theta=0.0,
         hi=AU, lo=AIR, period=0.50, ff=0.8, d=0.20, sub=AIR, nG_list=NG_1D,
         desc="metal slit array, TM (plasmonic/EOT; hardest 1D)"),

    dict(name="C1_Si_pillars", group="C", dim=2, pol="TE", theta=0.0,
         pillar=SI, bg=AIR, period=0.50, ax=0.30, ay=0.30, d=0.40, sub=SIO2,
         nG_list=NG_2D, desc="Si square-pillar metasurface (subwavelength)"),
    dict(name="C1b_Si_pillars_diffract", group="C", dim=2, pol="TE", theta=0.0,
         pillar=SI, bg=AIR, period=1.50, ax=0.60, ay=0.60, d=0.40, sub=SIO2,
         nG_list=NG_2D, desc="Si pillars, supra-wavelength (diffraction)"),
    dict(name="C2_Au_holes", group="C", dim=2, pol="TE", theta=0.0,
         pillar=AIR, bg=AU, period=0.60, ax=0.30, ay=0.30, d=0.20, sub=SIO2,
         nG_list=NG_2D, desc="metal hole array, 2D EOT (hardest 2D)"),
]


def _supports_native_dim():
    """True only for versions with dimensionality inference (native L2=None +
    Add_LayerGrid Ny default)."""
    try:
        o = grcwa.obj(3, [1.0, 0], None, FREQC, 0., 0., verbose=0)
        o.Add_LayerUniform(1.0, 1.0)
        o.Add_LayerGrid(0.1, 4)
        o.Add_LayerUniform(1.0, 1.0)
        o.Init_Setup()
        return True
    except Exception:
        return False


SUPPORTS_DIM = _supports_native_dim()


def _excite(obj, pol):
    pa, sa = (1., 0.) if pol == "TM" else (0., 1.)
    obj.MakeExcitationPlanewave(pa, 0., sa, 0., order=0)


def _mk(nG, L1, L2, theta):
    kwargs = {}
    if FMM is not None:
        kwargs["fmm_method"] = FMM
    try:
        return grcwa.obj(nG, L1, L2, FREQC, theta, 0.0, verbose=0, **kwargs)
    except TypeError:
        if FMM is not None:
            return None            # this version cannot do Pol
        return grcwa.obj(nG, L1, L2, FREQC, theta, 0.0, verbose=0)


def _build(s, nG):
    """Return (obj, eps_flat_or_None, mode) or (None, None, reason)."""
    dim = s["dim"]
    theta = s["theta"]

    if dim == 0:
        if not SUPPORTS_DIM:
            return None, None, "no-native-0D"
        obj = _mk(1, None, None, theta)
        if obj is None:
            return None, None, "no-pol"
        obj.Add_LayerUniform(1.0, eps(AIR))
        obj.Add_LayerUniform(s["d"], eps(s["film"]))
        obj.Add_LayerUniform(1.0, eps(s["sub"]))
        return obj, None, "native"

    if dim == 1:
        xs = np.linspace(0, 1, NX_1D, endpoint=False)
        prof = np.where(xs < s["ff"], eps(s["hi"]), eps(s["lo"])).astype(complex)
        if SUPPORTS_DIM:
            obj = _mk(nG, [s["period"], 0], None, theta)
            if obj is None:
                return None, None, "no-pol"
            obj.Add_LayerUniform(1.0, eps(AIR))
            obj.Add_LayerGrid(s["d"], NX_1D)
            obj.Add_LayerUniform(1.0, eps(s["sub"]))
            return obj, prof, "native"
        # degenerate-2D fallback (tiny y-period so only Gy=0 survives)
        obj = _mk(nG, [s["period"], 0], [0, s["period"] * 1e-3], theta)
        if obj is None:
            return None, None, "no-pol"
        obj.Add_LayerUniform(1.0, eps(AIR))
        obj.Add_LayerGrid(s["d"], NX_1D, 1)
        obj.Add_LayerUniform(1.0, eps(s["sub"]))
        return obj, prof, "degenerate-2D"

    if dim == 2:
        Lam = s["period"]
        obj = _mk(nG, [Lam, 0], [0, Lam], theta)
        if obj is None:
            return None, None, "no-pol"
        x = np.linspace(0, 1, NX_2D, endpoint=False)
        X, Y = np.meshgrid(x, x, indexing="ij")
        eg = np.ones((NX_2D, NX_2D), dtype=complex) * eps(s["bg"])
        inside = (np.abs(X - 0.5) < s["ax"] / (2 * Lam)) & \
                 (np.abs(Y - 0.5) < s["ay"] / (2 * Lam))
        eg[inside] = eps(s["pillar"])
        obj.Add_LayerUniform(1.0, eps(AIR))
        obj.Add_LayerGrid(s["d"], NX_2D, NX_2D)
        obj.Add_LayerUniform(1.0, eps(s["sub"]))
        return obj, eg.flatten(), mode_2d()

    return None, None, "unsupported"


def mode_2d():
    return "native"


def _solve(s, nG):
    obj, eps_flat, mode = _build(s, nG)
    if obj is None:
        return None, mode
    obj.Init_Setup()
    _excite(obj, s["pol"])
    if eps_flat is not None:
        obj.GridLayer_geteps(eps_flat)
    R, T = obj.RT_Solve(normalize=1)
    return (float(np.real(R)), float(np.real(T)), int(obj.nG)), mode


def run_structure(s):
    sweep = []
    skipped = None
    for nG in s["nG_list"]:
        try:
            out, mode = _solve(s, nG)
        except Exception as e:
            sweep.append({"nG_req": nG, "error": repr(e)})
            continue
        if out is None:
            skipped = mode
            break
        R, T, nGa = out
        best = np.inf
        for _ in range(REPEAT):
            t0 = time.perf_counter()
            try:
                _solve(s, nG)
            except Exception:
                break
            best = min(best, time.perf_counter() - t0)
        sweep.append({"nG_req": nG, "nG": nGa, "R": R, "T": T,
                      "A": 1.0 - R - T,
                      "time_ms": (best * 1e3) if best < np.inf else None,
                      "mode": mode})
    info = {k: s[k] for k in ("group", "dim", "pol", "theta", "desc")}
    info["nk"] = {k: list(s[k]) for k in
                  ("hi", "lo", "film", "sub", "pillar", "bg") if k in s}
    for k in ("period", "ff", "d", "ax", "ay"):
        if k in s:
            info[k] = s[k]
    out = {"info": info, "sweep": sweep}
    if skipped:
        out["skipped"] = skipped
    return out


results = {s["name"]: run_structure(s) for s in STRUCTURES}
print(json.dumps(results))
