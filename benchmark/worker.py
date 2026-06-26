"""Benchmark worker: runs a fixed battery of physically-motivated gratings on
ONE grcwa installation (selected via the GRCWA_MOD env var + sys.path) and
prints the results as JSON.

Run by benchmark/run.py once per (variant, factorization-mode) combination, in
a separate process so several grcwa versions can be compared without import
clashes. Not a pytest module.

Units: vacuum c = 1, so freq = 1/lambda. We work at lambda = 1 micron
(freq = 1.0); lengths are therefore in microns.
"""
import os
import sys
import json
import time

# import the requested grcwa package (name + path provided by the runner)
_MOD = os.environ.get("GRCWA_MOD", "grcwa")
grcwa = __import__(_MOD)
import numpy as np

_FMM = os.environ.get("FMM", "none")
FMM = None if _FMM == "none" else _FMM

FREQ = 1.0                      # lambda = 1 micron
QABS = 1e7                      # tiny loss regularizes Rayleigh anomalies
FREQC = FREQ * (1 + 1j / 2 / QABS)
REPEAT = int(os.environ.get("REPEAT", "3"))


# --- the battery: physically meaningful 1D/2D/0D gratings, real & complex n ---
CASES = [
    # 2D dielectric (Si) hole array -- all versions run this natively
    dict(name="2D_Si_hole_subwave", dim=2, period=0.5, eps_hi=12.25, eps_lo=1.0,
         thick=0.30, pol="TE", theta=0.0, nG=100, Nx=64, lossless=True,
         note="sub-lambda: effective medium, R+T=1"),
    dict(name="2D_Si_hole_diffract", dim=2, period=1.5, eps_hi=12.25, eps_lo=1.0,
         thick=0.30, pol="TE", theta=0.0, nG=120, Nx=64, lossless=True,
         note="supra-lambda: diffraction orders open, R+T=1"),
    dict(name="2D_metal_hole_absorb", dim=2, period=0.7, eps_hi=-10 + 1j, eps_lo=1.0,
         thick=0.10, pol="TE", theta=0.0, nG=120, Nx=64, lossless=False,
         note="lossy metal: absorption A=1-R-T>0"),
    # 1D lamellar gratings -- fork runs native (L2=None); old versions fall back
    # to a degenerate 2D cell (tiny y-period) -- the historical way to do 1D.
    dict(name="1D_Si_TE_subwave", dim=1, period=0.4, eps_hi=12.25, eps_lo=1.0,
         thick=0.30, pol="TE", theta=0.0, nG=41, Nx=256, lossless=True,
         note="sub-lambda 1D: only 0th order, R+T=1"),
    dict(name="1D_Si_TM_diffract", dim=1, period=1.5, eps_hi=12.25, eps_lo=1.0,
         thick=0.30, pol="TM", theta=np.pi/9, nG=61, Nx=256, lossless=True,
         note="supra-lambda 1D TM: diffraction, R+T=1"),
    dict(name="1D_metal_TM_absorb", dim=1, period=0.5, eps_hi=-10 + 0.3j, eps_lo=1.0,
         thick=0.10, pol="TM", theta=np.pi/9, nG=61, Nx=256, lossless=False,
         note="metal 1D TM: strong reflection + absorption"),
    # 0D planar slab -- fork runs native (TMM); old versions use nG=1
    dict(name="0D_slab", dim=0, period=None, eps_hi=4.0, eps_lo=None,
         thick=0.30, pol="TE", theta=0.0, nG=1, Nx=1, lossless=True,
         note="planar slab = TMM; compare to analytic Airy"),
]


def _supports_native_dim():
    """True only for versions with dimensionality inference (native L2=None +
    Add_LayerGrid Ny default). Old versions defer lattice checks to Init_Setup,
    so probe the whole pipeline rather than trusting construction."""
    try:
        o = grcwa.obj(3, [1.0, 0], None, FREQC, 0., 0., verbose=0)
        o.Add_LayerUniform(1.0, 1.0)
        o.Add_LayerGrid(0.1, 4)        # Ny default exists only on the fork
        o.Add_LayerUniform(1.0, 1.0)
        o.Init_Setup()
        return True
    except Exception:
        return False


SUPPORTS_DIM = _supports_native_dim()


def _excite(obj, pol):
    pa, sa = (1., 0.) if pol == "TM" else (0., 1.)
    obj.MakeExcitationPlanewave(pa, 0., sa, 0., order=0)


def _new_obj(period_spec, nG, theta):
    """Construct an obj, using native dim when supported, else a degenerate 2D
    cell. Returns (obj, mode_str, build_kwargs_ok)."""
    L1, L2, dimlabel = period_spec
    kwargs = {}
    if FMM is not None:
        kwargs["fmm_method"] = FMM
    try:
        obj = grcwa.obj(nG, L1, L2, FREQC, theta, 0.0, verbose=0, **kwargs)
        return obj, "native"
    except TypeError:
        # old signature: no fmm_method kwarg. If Pol was requested, this
        # version cannot do it -> signal skip.
        if FMM is not None:
            return None, "no-pol"
        obj = grcwa.obj(nG, L1, L2, FREQC, theta, 0.0, verbose=0)
        return obj, "native"
    except Exception:
        return None, "unsupported"


def _build(case):
    """Return (obj, eps_flat_or_None, mode). Handles native vs degenerate setup
    for 1D/0D on versions that lack dimensionality inference."""
    dim = case["dim"]
    nG = case["nG"]
    theta = case["theta"]
    Nx = case["Nx"]

    if dim == 2:
        Lam = case["period"]
        spec = ([Lam, 0], [0, Lam], "2D")
        obj, mode = _new_obj(spec, nG, theta)
        if obj is None:
            return None, None, mode
        Ny = Nx
        x = np.linspace(0, 1, Nx, endpoint=False)
        X, Y = np.meshgrid(x, x, indexing="ij")
        eg = np.ones((Nx, Ny), dtype=complex) * case["eps_hi"]
        eg[(X - 0.5) ** 2 + (Y - 0.5) ** 2 < (0.3) ** 2] = case["eps_lo"]
        obj.Add_LayerUniform(1.0, 1.0)
        obj.Add_LayerGrid(case["thick"], Nx, Ny)
        obj.Add_LayerUniform(1.0, 1.0)
        return obj, eg.flatten(), mode

    if dim == 1:
        Lam = case["period"]
        xs = np.linspace(0, 1, Nx, endpoint=False)
        prof = np.where(xs < 0.5, case["eps_hi"], case["eps_lo"]).astype(complex)
        if SUPPORTS_DIM:
            obj, mode = _new_obj(([Lam, 0], None, "1D"), nG, theta)
            if obj is None:
                return None, None, mode
            obj.Add_LayerUniform(1.0, 1.0)
            obj.Add_LayerGrid(case["thick"], Nx)
            obj.Add_LayerUniform(1.0, 1.0)
            return obj, prof, "native"
        # old versions: degenerate 2D cell, tiny y-period so only Gy=0 survives
        obj, mode = _new_obj(([Lam, 0], [0, Lam * 1e-3], "1Ddeg"), nG, theta)
        if obj is None:
            return None, None, mode
        obj.Add_LayerUniform(1.0, 1.0)
        obj.Add_LayerGrid(case["thick"], Nx, 1)
        obj.Add_LayerUniform(1.0, 1.0)
        return obj, prof, "degenerate-2D"

    if dim == 0:
        if not SUPPORTS_DIM:
            return None, None, "no-native-0D"
        obj, mode = _new_obj((None, None, "0D"), 1, theta)
        if obj is None:
            return None, None, mode
        obj.Add_LayerUniform(1.0, 1.0)
        obj.Add_LayerUniform(case["thick"], case["eps_hi"])
        obj.Add_LayerUniform(1.0, 1.0)
        return obj, None, "native"

    return None, None, "unsupported"


def _solve_once(case):
    obj, eps_flat, mode = _build(case)
    if obj is None:
        return None, mode
    obj.Init_Setup()
    _excite(obj, case["pol"])
    if eps_flat is not None:
        obj.GridLayer_geteps(eps_flat)
    R, T = obj.RT_Solve(normalize=1)
    return (float(np.real(R)), float(np.real(T)), int(obj.nG)), mode


def run_case(case):
    # warm-up + timed repeats (report the min wall time)
    try:
        out, mode = _solve_once(case)
    except Exception as e:
        return {"error": repr(e)}
    if out is None:
        return {"skipped": mode}
    best = np.inf
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        try:
            o2, _ = _solve_once(case)
        except Exception as e:
            return {"error": repr(e)}
        best = min(best, time.perf_counter() - t0)
    R, T, nG = out
    A = 1.0 - R - T
    return {"R": R, "T": T, "A": A, "nG": nG, "time_ms": best * 1e3, "mode": mode}


results = {c["name"]: run_case(c) for c in CASES}
print(json.dumps(results))
