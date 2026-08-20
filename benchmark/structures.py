"""Shared structure battery + solver for the grcwa benchmark and convergence study.

Single source of truth, so benchmark/worker.py (cross-version, single order count)
and benchmark/conv_worker.py (order sweep on the fork) test the SAME structures.

The geometry of every patterned layer is rasterized once by :func:`layer_mask`
and consumed by *both* backends -- grcwa (a flattened eps vector, this module's
:func:`solve`) and Ikarus (an integer topology + material list, see
benchmark/ikarus_suite.py). Neither backend rasterizes on its own, so a
disagreement *between those two* can never be a pixel-grid artifact.

Rasterization: two error channels, and this module now closes the first one
------------------------------------------------------------------------------
Turning a nominal shape into permittivity Fourier coefficients has two
independent error sources (the full derivation and the measurements behind
every number in this docstring are in ``benchmark/RASTERIZATION.md``):

  1. **shape** -- the pixel image is not the nominal shape. ``O(1/N)`` when a
     boundary falls between two samples, and *exactly zero* when every
     boundary falls on a cell edge.
  2. **sampling** -- the plain FFT every suite (grcwa, Ikarus, Moose) uses
     overstates every coefficient relative to the exact Fourier integral of
     the pixel image, by a factor that is ``O(1/N^2)`` and never vanishes on
     any finite grid, but is removable exactly (see
     ``rasterization_study.pixel_exact``) and does not depend on this module.

Until this rewrite, ``layer_mask`` used ``NX_2D = 256`` with **left-edge,
strict "<"** sampling on the rectangle branch, which does not represent this
battery's own pillar widths exactly (0.6 * 256 = 153.6 is not an integer) --
channel 1 alone cost ~0.4-0.8 % on the linear feature size and ~0.01 in R, an
order of magnitude more than the truncation error at the top of a sweep, and
was the entire 2D disagreement with an external Moose reference that builds
its geometry from the parameters instead of this mask.

``layer_mask`` now defaults to **cell-centre sampling** (matching the circle
branch, which was already exact) on a per-case grid chosen so every
axis-parallel boundary in the battery lands on a cell edge: ``NX_2D = 260``
(a multiple of 20, satisfying C1's width 0.6, C1b's 0.4 and C2's 0.5 all at
once) and ``NX_1D = 10240`` (a multiple of 20, so B3's ``ff = 0.8`` -- which
``NX_1D = 8192`` could not represent exactly -- is now exact too). Channel 1
is now exactly zero on every rectangle and 1D case in the battery; only the
circle (D2) has no exact grid at any resolution, because pi is irrational.

The pre-fix mask stays reachable for anyone who needs the old, published
numbers bit-for-bit: pass ``legacy=True`` to :func:`layer_mask`,
:func:`mask_eps` or :func:`solve`. This changes results for every 2D
structure and B3 -- any cache keyed only by ``(structure, order, rule)``
needs the rasterization mode in its key too, or it will silently mix pre- and
post-fix values.

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
from fractions import Fraction

import numpy as np

FREQ = 1.0                       # lambda = 1 micron
QABS = 1e7                       # tiny loss regularizes Rayleigh anomalies
FREQC = FREQ * (1 + 1j / 2 / QABS)

# Current (exact-geometry) grids -- see the module docstring.  Both are
# multiples of 20: NX_2D = 260 represents C1 (w=0.6), C1b (w=0.4) and C2
# (w=0.5) exactly at once; NX_1D = 10240 represents ff=0.5 and B3's ff=0.8
# exactly, and stays well above the ~7441 samples the overnight sweep's
# highest 1D order (q = 61**2 = 3721) needs (2*q - 1); get_conv upsamples
# by an integer factor beyond that, which preserves exactness (a cell-centred
# exact mask stays exact under nearest-neighbour integer upsampling).
NX_1D = 10240
NX_2D = 260

# The pre-fix grids, for bit-for-bit reproduction of every number recorded
# before this rewrite: ``layer_mask(s, legacy=True)`` and friends use these.
NX_1D_LEGACY = 8192
NX_2D_LEGACY = 256

# materials at lambda = 1 um  (n, k)
AIR = (1.0, 0.0)
SIO2 = (1.5, 0.0)
SIN = (2.0, 0.0)
SI = (3.5, 0.0)
AU = (0.3, 7.0)                  # eps ~ -48.9 + 4.2j


def eps(nk):
    n, k = nk
    return (n + 1j * k) ** 2


# The group-D cases come from the Ikarus whitepaper (CAVITY technologies GmbH,
# doi 10.5281/zenodo.21966455) and its shipped test suite, which specifies them in
# SI units at lambda = 700 nm. RCWA is scale-invariant, so they are re-expressed
# here in this battery's lambda = 1 units by dividing every length by 700 nm.
WL_D = 700e-9                    # the whitepaper's wavelength, for provenance


def _d(x_nm):
    """A group-D length: nanometres from the whitepaper -> lambda = 1 units."""
    return x_nm * 1e-9 / WL_D


# group A: analytic anchors; B: 1D gratings; C: 2D rectangular pillars;
# D: the Ikarus whitepaper's cross-code cases
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
    dict(name="D1_ikarus_hcg_TM", group="D", dim=1, pol="p",
         hi=SI, lo=AIR, period=_d(400), ff=0.5, d=_d(300), sub=AIR,
         desc="Ikarus whitepaper Fig.1/Tab.1: free-standing n=3.5 lamellar "
              "grating, TM (the factorization stress test)"),
    dict(name="D2_ikarus_cylinder_TE", group="D", dim=2, pol="s", shape="circle",
         pillar=SI, bg=AIR, period=_d(400), radius=0.30, d=_d(200), sub=AIR,
         desc="Ikarus whitepaper: free-standing n=3.5 circular pillar, TE "
              "(curved boundary oblique to both axes)"),
]
STRUCT = {s["name"]: s for s in STRUCTURES}


def exact_N(s, at_least=None):
    """Smallest grid ``N >= at_least`` on which every boundary of ``s`` is a
    cell edge, at cell-centre sampling -- ``None`` if no such grid exists
    (only true for a circle, since pi is irrational).

    A centred rectangle of relative width ``w`` is exact on ``N`` cells iff
    ``w*N`` and ``(1-w)*N/2`` are both integers, i.e. ``N`` a multiple of
    ``w``'s denominator and of twice ``(1-w)``'s. ``NX_2D`` is chosen so this
    already holds for every rect case in the battery at ``N = NX_2D`` itself.
    """
    at_least = NX_2D if at_least is None else at_least
    if s.get("shape", "rect") == "circle":
        return None
    dens = []
    for k in ("ax", "ay"):
        w = Fraction(s[k] / s["period"]).limit_denominator(10 ** 6)
        dens += [w.denominator, (Fraction(1) - w).denominator * 2]
    step = int(np.lcm.reduce(np.array(dens, dtype=np.int64)))
    return step * int(np.ceil(at_least / step))


def layer_mask(s, N=None, legacy=False):
    """Rasterize the patterned layer of ``s`` ONCE, for every backend.

    Returns ``(mask, nk_pair)``: an integer ``(nx, ny)`` array whose value
    indexes ``nk_pair`` -- ``0`` = background, ``1`` = inclusion -- with each
    entry an ``(n, k)`` material tuple. 1D layers come back as ``(nx, 1)``.

    grcwa consumes it as ``eps(nk)[mask].flatten()``, Ikarus as an integer
    topology plus a material list, so both codes see the *same pixels* and a
    disagreement can only come from the physics, never from the grid.

    ``N`` overrides the default grid (``NX_1D``/``NX_2D``, or their legacy
    values); an explicit ``N`` on the rect/circle branch does not have to be
    exact -- pass ``exact_N(s)`` for that. ``legacy=True`` reproduces the
    pre-fix rasterization (left-edge rect sampling, ``NX_2D_LEGACY = 256`` /
    ``NX_1D_LEGACY = 8192``) bit-for-bit; see the module docstring.

    Raises ValueError for dim == 0 (uniform layers have no pattern).
    """
    dim = s["dim"]
    if dim == 0:
        raise ValueError(f"{s['name']}: 0D structure has no patterned layer")
    if dim == 1:
        n = N if N is not None else (NX_1D_LEGACY if legacy else NX_1D)
        # Left-edge sampling: xs < ff fills exactly the first ff*n cells.
        # Unlike the 2D rect branch, this has only ONE boundary (at x = ff,
        # the pattern starts flush at x = 0), so it is exact whenever ff*n is
        # an integer regardless of edge- vs centre-sampling -- no separate
        # "centre" rule is needed here, only a big-enough, well-divisible n.
        xs = np.linspace(0, 1, n, endpoint=False)
        return (xs < s["ff"]).astype(int)[:, None], [s["lo"], s["hi"]]

    Lam = s["period"]
    n = N if N is not None else (NX_2D_LEGACY if legacy else NX_2D)
    if s.get("shape", "rect") == "circle":
        # Cell-centred sampling, so a centred circle stays exactly symmetric
        # under the lattice's C4v. No grid renders a circle exactly (pi is
        # irrational); this is the closest a binary raster gets.
        c = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(c, c, indexing="ij")
        inside = (X - 0.5) ** 2 + (Y - 0.5) ** 2 <= s["radius"] ** 2 + 1e-12
    elif legacy:
        # Left-edge sampling with a strict "<": kept ONLY for bit-for-bit
        # reproduction of pre-fix numbers. Not faithful -- see the module
        # docstring -- and re-breaks exactness even on an N where the
        # cell-centre rule below would be exact.
        x = np.linspace(0, 1, n, endpoint=False)
        X, Y = np.meshgrid(x, x, indexing="ij")
        inside = (np.abs(X - 0.5) < s["ax"] / (2 * Lam)) & \
                 (np.abs(Y - 0.5) < s["ay"] / (2 * Lam))
    else:
        # Cell centres, closed ("<="): exact whenever ax/Lam * n and
        # ay/Lam * n are integers, which they are for every rect case in the
        # battery at n = NX_2D (see exact_N). Matches ikarus.shapes.rectangle
        # and moose_raster_probe.cs's CellInside -- keep the three in step.
        c = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(c, c, indexing="ij")
        inside = (np.abs(X - 0.5) <= s["ax"] / (2 * Lam) + 1e-12) & \
                 (np.abs(Y - 0.5) <= s["ay"] / (2 * Lam) + 1e-12)
    return inside.astype(int), [s["bg"], s["pillar"]]


def mask_eps(s, N=None, legacy=False):
    """:func:`layer_mask` with the materials already turned into permittivity."""
    mask, nk_pair = layer_mask(s, N=N, legacy=legacy)
    return np.array([eps(nk) for nk in nk_pair], dtype=complex)[mask]


def shape_transform(s, G):
    """Analytic Fourier transform ``S(G)`` of the inclusion of ``s`` at
    integer orders ``G`` (shape ``(n, 2)``) -- no grid, no rasterization, no
    truncation. The oracle a rasterized mask is measured against.

    Rectangle of relative size ``(w, h)``: ``S(G) = w h sinc(m w) sinc(n h)``.
    Circle of relative radius ``r``: ``S(G) = pi r^2 . 2 J1(x) / x`` with
    ``x = 2 pi |G| r``. Both times ``exp(-i pi (m+n))`` for a shape centred
    in the cell.
    """
    m, n = G[:, 0].astype(float), G[:, 1].astype(float)
    if s.get("shape", "rect") == "circle":
        from scipy.special import j1
        r = s["radius"]
        x = 2 * np.pi * np.hypot(m, n) * r
        S = np.where(x == 0, np.pi * r * r,
                     np.pi * r * r * 2 * j1(np.where(x == 0, 1., x))
                     / np.where(x == 0, 1., x))
    else:
        w, h = s["ax"] / s["period"], s["ay"] / s["period"]
        S = w * h * np.sinc(m * w) * np.sinc(n * h)
    return S * np.exp(-1j * np.pi * (m + n))


def analytic_coeffs(s, G, inverse=False):
    """Exact eps (or 1/eps, ``inverse=True``) Fourier coefficients of the 2D
    patterned layer of ``s`` on integer orders ``G``: no grid, no rasterization,
    no truncation. Used as the geometry-error-free reference column in
    ``rasterization_study.py`` and the L0 coefficient test in
    ``RASTERIZATION.md``.
    """
    e_in, e_bg = eps(s["pillar"]), eps(s["bg"])
    if inverse:
        e_in, e_bg = 1 / e_in, 1 / e_bg
    z = ((G[:, 0] == 0) & (G[:, 1] == 0)).astype(float)
    return (e_in - e_bg) * shape_transform(s, G) + e_bg * z


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


def solve(grcwa, s, q, fmm, native, legacy=False):
    """Solve structure ``s`` at per-axis order count ``q`` with factorization
    ``fmm`` (None=Laurent, 'pol'=Pol). ``native`` is supports_native_dim(grcwa).
    ``legacy=True`` reproduces the pre-fix rasterization -- see the module
    docstring; every 2D and B3 value on record before this rewrite needs it to
    reproduce.

    Returns (R, T, nG_actual, mode) or (None, None, None, reason).
    1D uses nG=q; 2D uses a (q,q) square block (nG=q**2, parallelogramic).
    """
    dim = s["dim"]
    nx_1d = NX_1D_LEGACY if legacy else NX_1D
    nx_2d = NX_2D_LEGACY if legacy else NX_2D
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
        prof = mask_eps(s, legacy=legacy)[:, 0]  # shared rasterization (Nx,1)->(Nx,)
        if native:
            o = _obj(grcwa, q, [s["period"], 0], None, fmm)
            if o is None:
                return None, None, None, "no-pol"
            o.Add_LayerUniform(1.0, eps(AIR))
            o.Add_LayerGrid(s["d"], nx_1d)
            o.Add_LayerUniform(1.0, eps(s["sub"]))
            mode = "native"
        else:
            o = _obj(grcwa, q, [s["period"], 0], [0, s["period"] * 1e-3], fmm)
            if o is None:
                return None, None, None, "no-pol"
            o.Add_LayerUniform(1.0, eps(AIR))
            o.Add_LayerGrid(s["d"], nx_1d, 1)
            o.Add_LayerUniform(1.0, eps(s["sub"]))
            mode = "degenerate-2D"
        o.Init_Setup(Gmethod=1)
        eps_flat = prof
    else:  # dim == 2: q x q square block via parallelogramic truncation
        Lam = s["period"]
        eg = mask_eps(s, legacy=legacy)          # shared rasterization (Nx,Ny)
        o = _obj(grcwa, q * q, [Lam, 0], [0, Lam], fmm)
        if o is None:
            return None, None, None, "no-pol"
        o.Add_LayerUniform(1.0, eps(AIR))
        o.Add_LayerGrid(s["d"], nx_2d, nx_2d)
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
