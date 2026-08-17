"""Tests for the dimensionality inference (0D/1D/2D) and tensor-uniform layers.

All reference values are closed-form thin-film (Fresnel/Airy) results, so these
are independent analytic checks of the new code paths.
"""
import numpy as np
import pytest
import grcwa


@pytest.fixture(autouse=True)
def _numpy_backend():
    # other test modules switch the global backend to autograd on import;
    # force numpy here so these analytic comparisons are deterministic, then
    # restore autograd afterwards so those modules' tests still see it.
    grcwa.set_backend('numpy')
    yield
    try:
        grcwa.set_backend('autograd')
    except ValueError:
        pass


# ---- analytic single-film references (incident medium n0, film n1, substrate ns) ----
def _slab_R_T(n0, n1, ns, d, freq, theta, pol):
    k0 = 2 * np.pi * freq
    kx = k0 * n0 * np.sin(theta)
    kz0 = np.sqrt((k0 * n0) ** 2 - kx ** 2 + 0j)
    kz1 = np.sqrt((k0 * n1) ** 2 - kx ** 2 + 0j)
    kzs = np.sqrt((k0 * ns) ** 2 - kx ** 2 + 0j)
    if pol == 's':
        r01 = (kz0 - kz1) / (kz0 + kz1)
        r12 = (kz1 - kzs) / (kz1 + kzs)
        t01 = 2 * kz0 / (kz0 + kz1)
        t12 = 2 * kz1 / (kz1 + kzs)
    else:  # p
        r01 = (n1 ** 2 * kz0 - n0 ** 2 * kz1) / (n1 ** 2 * kz0 + n0 ** 2 * kz1)
        r12 = (ns ** 2 * kz1 - n1 ** 2 * kzs) / (ns ** 2 * kz1 + n1 ** 2 * kzs)
        t01 = 2 * n0 * n1 * kz0 / (n1 ** 2 * kz0 + n0 ** 2 * kz1)
        t12 = 2 * n1 * ns * kz1 / (ns ** 2 * kz1 + n1 ** 2 * kzs)
    ph = np.exp(1j * kz1 * d)
    r = (r01 + r12 * ph ** 2) / (1 + r01 * r12 * ph ** 2)
    t = (t01 * t12 * ph) / (1 + r01 * r12 * ph ** 2)
    R = np.abs(r) ** 2
    T = np.real(kzs / kz0) * np.abs(t) ** 2
    return R, T


def _run_0d(eps_film, d, freq, theta, pol):
    o = grcwa.obj(1, None, None, freq, theta, 0., verbose=0)
    o.Add_LayerUniform(1.0, 1.0)
    o.Add_LayerUniform(d, eps_film)
    o.Add_LayerUniform(1.0, 1.0)
    o.Init_Setup()
    pa, sa = (1., 0.) if pol == 'p' else (0., 1.)
    o.MakeExcitationPlanewave(pa, 0., sa, 0., order=0)
    return o.RT_Solve(normalize=1)


def test_0d_is_tmm_normal_and_oblique():
    freq, d, epf = 1.0, 0.3, 4.0
    nf = np.sqrt(epf)
    for theta in [0.0, np.pi/6, np.pi/3]:
        for pol in ['s', 'p']:
            R, T = _run_0d(epf, d, freq, theta, pol)
            Rref, Tref = _slab_R_T(1.0, nf, 1.0, d, freq, theta, pol)
            assert abs(R - Rref) < 1e-9, (theta, pol, R, Rref)
            assert abs(T - Tref) < 1e-9, (theta, pol, T, Tref)
            assert abs(R + T - 1.0) < 1e-9   # lossless


def test_0d_dim_inferred():
    o = grcwa.obj(7, None, None, 1.0, 0.0, 0.0, verbose=0)
    o.Add_LayerUniform(1.0, 1.0)
    o.Init_Setup()
    assert o.dim == 0
    assert o.nG == 1            # nG forced to 1 regardless of input


def test_1d_constant_grating_equals_slab():
    freq, d, epf = 1.0, 0.3, 4.0
    nf = np.sqrt(epf)
    Nx = 64
    o = grcwa.obj(41, [0.8, 0], None, freq, 0.0, 0.0, verbose=0)
    o.Add_LayerUniform(1.0, 1.0)
    o.Add_LayerGrid(d, Nx)            # Ny defaults to 1
    o.Add_LayerUniform(1.0, 1.0)
    o.Init_Setup()
    assert o.dim == 1
    o.MakeExcitationPlanewave(0., 0., 1., 0., order=0)
    o.GridLayer_geteps(np.ones(Nx) * epf)
    R, T = o.RT_Solve(normalize=1)
    Rref, Tref = _slab_R_T(1.0, nf, 1.0, d, freq, 0.0, 's')
    assert abs(R - Rref) < 1e-9
    assert abs(R + T - 1.0) < 1e-9


def _build_1d_binary(nG, Lam, theta, prof, tg):
    o = grcwa.obj(nG, [Lam, 0], None, 1.0, theta, 0., verbose=0)
    o.Add_LayerUniform(1.0, 1.0); o.Add_LayerGrid(tg, len(prof)); o.Add_LayerUniform(1.0, 2.0)
    o.Init_Setup()
    o.MakeExcitationPlanewave(1., 0., 0., 0., order=0)
    o.GridLayer_geteps(prof.copy())
    return o


def _build_2d_yinvariant(nG, Lam, theta, prof, tg):
    # tiny y-period -> circular truncation keeps only Gy=0 -> physically 1D
    o = grcwa.obj(nG, [Lam, 0], [0, Lam * 1e-3], 1.0, theta, 0., verbose=0)
    o.Add_LayerUniform(1.0, 1.0); o.Add_LayerGrid(tg, len(prof), 1); o.Add_LayerUniform(1.0, 2.0)
    o.Init_Setup()
    o.MakeExcitationPlanewave(1., 0., 0., 0., order=0)
    o.GridLayer_geteps(prof.copy())
    return o


def test_1d_matches_2d_yinvariant():
    Lam, theta, tg, M = 1.2, np.pi/12, 0.4, 15
    Nx = 200
    xs = np.linspace(0, 1, Nx, endpoint=False)
    prof = np.where(xs < 0.5, 12.0, 1.0)
    o2 = _build_2d_yinvariant(2*M+1, Lam, theta, prof, tg)
    R2, T2 = o2.RT_Solve(normalize=1)
    # match the 1D order count to whatever the 2D truncation kept
    o1 = _build_1d_binary(o2.nG, Lam, theta, prof, tg)
    R1, T1 = o1.RT_Solve(normalize=1)
    assert abs(R1 - R2) < 1e-9, (R1, R2)
    assert abs(T1 - T2) < 1e-9, (T1, T2)


def test_tensor_uniform_degenerate_equals_isotropic():
    freq, d = 1.0, 0.3
    for theta in [0.0, np.pi/5]:
        for pol in ['s', 'p']:
            Ri, Ti = _run_0d(4.0, d, freq, theta, pol)
            Ra, Ta = _run_0d([4.0, 4.0, 4.0], d, freq, theta, pol)
            assert abs(Ri - Ra) < 1e-12, (theta, pol)
            assert abs(Ti - Ta) < 1e-12


def test_tensor_uniform_uniaxial_axis_selective():
    # normal incidence: s-pol probes one in-plane axis, p-pol the other.
    freq, d = 1.0, 0.3
    epx, epy, epz = 4.0, 6.0, 5.0
    Rs, _ = _run_0d([epx, epy, epz], d, freq, 0.0, 's')
    Rp, _ = _run_0d([epx, epy, epz], d, freq, 0.0, 'p')
    Rx, _ = _slab_R_T(1.0, np.sqrt(epx), 1.0, d, freq, 0.0, 's')
    Ry, _ = _slab_R_T(1.0, np.sqrt(epy), 1.0, d, freq, 0.0, 's')
    # the two polarizations must see two different axes (epx vs epy)
    assert min(abs(Rs - Rx), abs(Rs - Ry)) < 1e-9
    assert min(abs(Rp - Rx), abs(Rp - Ry)) < 1e-9
    assert abs(Rs - Rp) > 1e-3        # genuinely anisotropic response


def test_dim_validation_errors():
    # 0D: no grid layers allowed
    o = grcwa.obj(1, None, None, 1.0, 0., 0., verbose=0)
    with pytest.raises(ValueError):
        o.Add_LayerGrid(0.2, 16)
    # 1D: grid must have Ny == 1
    o = grcwa.obj(11, [1.0, 0], None, 1.0, 0., 0., verbose=0)
    with pytest.raises(ValueError):
        o.Add_LayerGrid(0.2, 16, 4)
    # L1=None with L2 given is invalid
    with pytest.raises(ValueError):
        grcwa.obj(11, None, [0, 1.0], 1.0, 0., 0., verbose=0)
    # eps length mismatch
    o = grcwa.obj(11, [1.0, 0], None, 1.0, 0., 0., verbose=0)
    o.Add_LayerUniform(1.0, 1.0); o.Add_LayerGrid(0.2, 16); o.Add_LayerUniform(1.0, 1.0)
    o.Init_Setup()
    with pytest.raises(ValueError):
        o.GridLayer_geteps(np.ones(15))   # should be 16


def test_incident_medium_must_be_isotropic():
    o = grcwa.obj(1, None, None, 1.0, 0., 0., verbose=0)
    o.Add_LayerUniform(1.0, [2.0, 2.0, 2.0])   # anisotropic incident medium
    o.Add_LayerUniform(1.0, 1.0)
    with pytest.raises(ValueError):
        o.Init_Setup()
