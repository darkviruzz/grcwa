# How the three suites put geometry on a grid — and what it costs

Phase-a) write-up for the cross-solver convergence study. Nothing in
`structures.py`, `grcwa/` or the Moose scripts is changed by it; every number
below comes from `benchmark/rasterization_study.py`, which only measures.

`geometry_fidelity.py` already showed that the shared 256² mask is not the
nominal structure. This document adds the half that was missing: *even a mask
that is the nominal structure exactly* is not turned into the nominal
structure's Fourier coefficients, and that second error is what is left when
the first one is fixed. It also shows that two of the three suites have a
third, quite separate grid dependence hiding in the tangent field of their
"faithful" factorization rules.

---

## 1. There are two error channels, and the four branches conflated them

Every FFT-based RCWA takes geometry to permittivity Fourier coefficients in two
steps.

**Channel 1 — shape.** The pixel image is not the nominal shape. Its size is
`O(1/N)` when a boundary falls between two samples, and it is **exactly zero**
when every boundary falls on a cell edge. This is the channel
`geometry_fidelity.py` measures (C1 −0.39 %, C1b +0.59 %, C2 −0.78 %).

**Channel 2 — sampling.** The DFT of the samples is not the Fourier integral of
the pixel image. For a cell image — piecewise constant on `[n/N, (n+1)/N)` —
the two are related *exactly*:

```
    c_m  =  DFT_m  ·  sinc(m/N)  ·  exp(-i·pi·m/N)                        (*)
```

so the plain FFT that all three suites use overstates every coefficient by
`1/sinc(m/N) = 1 + (pi·m/N)²/6 + …`. It is `O(1/N²)` at fixed truncation order,
it never vanishes on any finite grid, and it does **not** care whether the
geometry is grid-aligned.

The `exp(-i·pi·m/N)` in (*) is a rigid half-cell translation of the layer. It
cancels in R and T at normal incidence and can be ignored; the `sinc` cannot.

Identity (*) is verified to machine precision in
`rasterization_study.py coeffs`. On `C1_Si_pillars`, rendered on a grid where
its rectangle *is* exactly representable (N a multiple of 20 — see §6a —
here 260, 520, 1040, sampled at cell centres),
the rms error of the 61×61 permittivity coefficient block against the analytic
values is:

| N | left-edge samples | cell centres | cell centres · sinc |
|---:|---:|---:|---:|
| 260 | 6.04e-03 | 1.51e-04 | **9.4e-17** |
| 520 | 3.03e-03 | 3.74e-05 | **9.5e-17** |
| 1040 | 1.52e-03 | 9.33e-06 | **9.4e-17** |

and for `1/eps`, which every faithful rule reads off the same grid, 4.9e-04 /
1.2e-05 / **8.6e-18**. Two things fall out of that table:

* the historical **left-edge** rule of `layer_mask` is not merely 0.4 % off in
  width, it is 40× worse than cell centres in the coefficients and it decays
  only as `1/N` — on an exactly representable grid it *re-breaks* the alignment
  (at N = 260 it renders 155 of 260 cells where 156 is exact);
* one multiplication by `sinc` turns the FFT into the exact Fourier integral.

In R this is not a subtlety. `C1_Si_pillars`, exact geometry, q = 21, direct
rule, grcwa:

| N | Laurent | Laurent · sinc |
|---:|---:|---:|
| 260 | 0.494744371 | **0.495296902** |
| 520 | 0.495158857 | **0.495296902** |
| 1040 | 0.495262396 | **0.495296902** |
| 2080 | 0.495288276 | **0.495296902** |

The uncorrected column converges as `1/N²` to exactly the corrected one
(Richardson from 1040/2080: 0.49529691). The corrected column is
**bit-identical on every grid** — the geometry has stopped entering the answer.
Solving the same case from the analytic rectangle coefficients instead of any
grid gives 0.495296902 as well, to 1e-12. For an axis-parallel shape on a
matched grid, the one-line `sinc` **is** the analytic method.

---

## 2. What each suite actually does

### grcwa / this fork

* Geometry never reaches grcwa. `benchmark/structures.py::layer_mask`
  rasterizes once and hands the same binary mask to grcwa and to Ikarus.
  1D: `NX_1D = 8192`, **left cell edge**, strict `<`. 2D rect: `NX_2D = 256`,
  **left cell edge**, strict `<`. 2D circle: 256², **cell centres**, `<=`.
* `fft_funs.get_conv` is a plain `fft2` of that grid, gathered at difference
  orders — channel 2 uncorrected, `O((Δm/N)²)`. It nearest-neighbour upsamples
  when the grid cannot hold every difference order, which keeps it safe but
  does not change the law.
* `Epsilon_fft_pol` builds the tangent field from *finite differences of the
  same grid*, blurred with `pol_sigma = 3.0` **pixels**. See §4.
* Strength: we own it. Any grid, any values (complex, non-binary), and both the
  `sinc` quadrature and fully analytic coefficients are small, contained
  changes.

### Ikarus

* `ikarus.shapes.*` sample **cell centres** with `<=` — already a different
  convention from `layer_mask`'s rect branch. `ikarus_suite.py` bypasses them
  and pins `resolution=mask.shape`, so Ikarus sees the battery's pixels
  verbatim. Without that pin it resamples nearest-neighbour to `max(res, 4M+1)`
  and its rasterization would change with the order.
* `core/fourier.convolution_matrix` is `fftshift(fft2(cell))/(nx·ny)` — the same
  uncorrected point-sample DFT, no `sinc`. Measured on the exact C1 rectangle at
  q = 21: `laurent` 0.494745467 → 0.495159955 → 0.495263494 for N = 260/520/1040,
  the same `1/N²` law as grcwa (and 1.1e-6 offset from it, which is the `FREQC`
  detuning).
* `li` is nearly grid-free on an axis-aligned rectangle — 0.396256166 /
  0.396256715 / 0.396256868 over the same three grids, a spread of 7e-7.
* `normal` (the normal-vector rule) is **not**: 0.399130762 / 0.400938302 /
  0.402119764 / 0.402848294 over N = 260 / 520 / 1040 / 2080 — 3.7e-3 of drift,
  and still 7.3e-4 per doubling at the end. See §4.
* A grey (non-binary) grid can still be passed through the public API by
  quantizing the distinct values into the material list, so subpixel
  experiments do not need a patched Ikarus. Patching
  `convolution_matrix` does — which is why the `sinc` fix is not available
  there and Ikarus has to be extrapolated in N instead.

### Moose

* Moose is the only one of the three that builds the **nominal** geometry from
  the parameters (`Atom(x, y, w, w, mat)` / `Atom(x, y, r, mat)`) and
  rasterizes it itself.
* It rasterizes **binary** (`moose_geometry_probe.cs` probe A: `levels = 2` at
  every resolution) and rounds **outward** — 155 px at 256 where `structures.py`
  takes 153 and 153.6 is exact. Both codes are wrong, in opposite directions.
* Its grid is `rRefinementFactorEpsFT × (2m+1)` samples per axis with the factor
  **clamped to [30, 100]**, so the sampling changes at every point of an order
  sweep, and `FFT_MODE = 1` (meant to hold the absolute grid fixed) has never
  done anything on this build.
* The refinement residual is **`1/refinement²`, not `1/refinement`.** Refitting
  the three dialog points already on record (C1, m = 10, nominal geometry:
  30 → 0.398784, 50 → 0.397764, 100 → 0.397322):

  | model | R(∞) | max residual |
  |---|---:|---:|
  | `a + b/r` | 0.396618 | 1.2e-04 |
  | `a + b/r²` | **0.397181** | **5.2e-06** |

  A two-point Richardson with `p = 2` from 30 and 100 predicts the measured
  point at 50 to 8e-6.

  > **Corrected by the probe — see §9.** The reasoning that went with this fit
  > was wrong. It said the exponent had to be channel 2 alone, because
  > `0.6·30·q` is an integer and the pillar is therefore rendered exactly.
  > `moose_raster_probe.cs` shows it is **not**: Moose's rectangle rasterizer is
  > one cell too wide per axis on *every* grid, aligned or not. The refinement
  > sequence is therefore a mixture of a `1/N` shape error and a `1/N²` sampling
  > error, it is **non-monotone** over 40/60/80/100 in three of four cases, and
  > no single power fits it. The clean three-point `p = 2` fit was fortuitous,
  > and its `R(∞) = 0.397181` is the limit of a contaminated sequence, not the
  > exact-geometry answer (which is 0.39626, measured directly in §9).
* Untested but decisive if it works: `Layer(double thickness, CaModel
  epsilonDistribution)` — a constructor that takes an explicit permittivity
  grid. See §6.

### Side by side

| | grcwa / fork | Ikarus | Moose |
|---|---|---|---|
| geometry source | external mask | external mask (pinned) or own cell-centred primitives | its own, from the nominal parameters |
| rasterization | binary, left edge (rect/1D), cell centre (circle) | binary, cell centre | binary, outward rounding |
| grid control | complete | N and values yes, transform no | refinement ∈ [30,100] only, and it is tied to the order |
| channel 1 | whatever the mask has | same mask | **+1 cell per axis, always** (§9); alignment does not help |
| channel 2 | removable **exactly** (`sinc`, one line) | `1/N²`, extrapolate | `1/refinement²`, extrapolate |
| 1/eps input | from the same grid | from the same grid | internal |
| tangent field | from the grid, blur in **pixels** | from the grid, blur = period/12 | unknown |

---

## 3. The 1D cases are not as clean as they look

`NX_1D = 8192` renders `ff = 0.5` exactly (4096 cells), which is why groups A,
B1 and B2 and `D1` agree with Moose to five or six digits. It does **not**
render `B3_Au_slits_TM`: `0.8 · 8192 = 6553.6`, so the mask is `ff = 0.800049`.

`rasterization_study.py fill1d`, q = 201, cell centres:

| N | cells | Laurent | Laurent · sinc |
|---:|---:|---:|---:|
| 8192 | 6554 (ff = 0.800049) | 0.791356829 | 0.791355573 |
| 8000 | 6400 (ff = 0.8 exact) | 0.791310799 | **0.791310128** |
| 10240 | 8192 (exact) | 0.791310540 | **0.791310128** |
| 16000 | 12800 (exact) | 0.791310297 | **0.791310128** |
| 32000 | 25600 (exact) | 0.791310170 | **0.791310128** |

The fill-fraction error is worth **4.6e-05** in R — and the residual
`B3_Au_slits_TM` disagreement with Moose on record is 4.4e-05, in the same
direction (python high, Moose low). The last 1D discrepancy in the study is a
rasterization artifact too, and `NX_1D = 10240` removes it.

---

## 4. The tangent-field rules carry a *third* grid dependence

Pol (grcwa) and the normal-vector rule (Ikarus) both build a smooth tangent
field out of the rendered permittivity grid. That field does not change the
`q → ∞` limit — but it very much changes the value at finite q, and it is
grid-dependent in a way the direct rule is not.

`C1_Si_pillars`, geometry held exactly representable, q = 21:

| N | grcwa Pol, `sigma = 3` px (current default) | grcwa Pol, `sigma` ∝ N (period/87) | Ikarus `normal` |
|---:|---:|---:|---:|
| 260 | 0.397270590 | 0.397270590 | 0.399130762 |
| 520 | 0.385865456 | 0.395534357 | 0.400938302 |
| 1040 | 0.392567850 | 0.394896877 | 0.402119764 |
| 2080 | 0.378795724 | 0.394955771 | 0.402848294 |

* grcwa's `pol_sigma` is specified **in pixels**, so the physical smoothing
  length shrinks as `1/N` and the answer does not converge at all — it swings by
  ±8e-3 with the eps grid. Expressing it as a fraction of the period reduces
  that to ~6e-5 over the same range. This is a one-line defect with a
  one-line fix, and it means **every `fork[Pol]` number on record is only
  reproducible together with its `NX`** (which changed from 2048 to 8192
  between branches).
* The 1D case shows the same thing even with the sigma fixed:
  `B3_Au_slits_TM` at q = 201 gives Pol 0.792534748 / 0.792669238 /
  0.792915125 / 0.793304484 for N = 8000 / 10240 / 16000 / 32000 — it needs a very fine grid
  before it reaches the faithful value (~0.79330). The cause is visible in
  `_compute_tangent_field_pol`: in 1D the raw tangent is `+t` at one interface
  and `−t` at the other, so the blur-and-reset harmonic extension **cancels
  between opposite normals** and leaves a spurious zero band whose position and
  width depend on the grid. Ikarus avoids exactly this with double-angle
  (orientation) diffusion. Adopting that encoding is the obvious fix.
* Ikarus's `normal` already scales its blur physically (`max(nx,ny)/12` px =
  period/12) but still drifts 3.7e-3 over N = 260 → 2080 on the square pillar
  (+1.8e-3, +1.2e-3, +7.3e-4 per step — converging, but like `N^-0.6`),
  because the double-angle field is not piecewise constant around a corner and
  therefore does **not** reduce to `li` on a 2D axis-aligned shape (only on 1D).

**Consequence for the study:** a cross-solver comparison at *fixed* truncation
order is meaningless for the tangent-field rules. Only the `q → ∞` limit is
shared, and that limit has to be reached — or extrapolated — before any
tolerance is applied.

---

## 5. The one case no grid renders: the circle

`D2_ikarus_cylinder_TE`, q = 21, against grcwa solved from the **analytic**
circle coefficients (`pi r² · 2J₁(2π|G|r)/(2π|G|r)`, no grid at all):

```
analytic coefficients:                       0.959792049
```

| N | rasterization | grcwa Laurent | error vs analytic | Ikarus `li` | Ikarus `normal` |
|---:|---|---:|---:|---:|---:|
| 128 | binary, cell centres | 0.957746172 | −2.05e-03 | 0.905643792 | 0.939529756 |
| 128 | cell-averaged | 0.960132258 | **+3.40e-04** | 0.925090705 | 0.948796597 |
| 256 | binary, cell centres | 0.960896466 | +1.10e-03 | 0.905812281 | 0.943083334 |
| 256 | cell-averaged | 0.959878286 | **+8.62e-05** | 0.915849684 | 0.945399744 |
| 512 | binary, cell centres | 0.959725624 | −6.64e-05 | 0.904746211 | 0.941705312 |
| 512 | cell-averaged | 0.959813829 | **+2.18e-05** | 0.910442553 | 0.943759375 |

The binary error is ~1e-3 in R and its **sign changes with N** — the number of
pixels a disk covers fluctuates, so there is nothing to extrapolate. Cell
averaging converges cleanly as `1/N²` (3.40e-04 → 8.62e-05 → 2.18e-05, ratios
3.94 and 3.95) and can be extrapolated. The normal-vector column splits the
same way: monotone under cell averaging (0.94880 → 0.94540 → 0.94376),
wandering under binary.

**But cell averaging is a trap for every faithful rule**, and this is the
finding that decides the question. Li, the normal-vector rule and Pol all read
`1/eps` off the *same* grid, and the cell average of `1/eps` is not one over the
cell average of `eps`. Measured on the D2 circle (rms coefficient error over the
61×61 block against the analytic values):

| N | eps, binary | eps, cell-avg | 1/eps, binary | 1/eps, cell-avg |
|---:|---:|---:|---:|---:|
| 128 | 3.11e-03 | **5.78e-04** | **2.54e-04** | 6.24e-04 |
| 256 | 9.43e-04 | **1.43e-04** | **7.70e-05** | 3.04e-04 |
| 512 | 4.01e-04 | **3.60e-05** | **3.28e-05** | 1.50e-04 |

Grey cells buy a factor 5–10 on `eps` and pay a factor 2.5–4.5 on `1/eps`,
where the arithmetic average also degrades the convergence from `1/N²` to
`1/N`. A single grey grid is therefore **not** an improvement for a code using
the inverse rule; it would only be one if `eps` and `1/eps` were supplied
separately (arithmetic mean and harmonic mean), which grcwa could be extended
to do and neither stock Ikarus nor Moose can.

The two effects do not cancel neatly, which is why this cannot be decided by
taste. In the R table above the normal-vector column *does* improve under cell
averaging — a grey boundary resolves the gradient the tangent field is built
from — while its `1/eps` input gets worse. Net, grey cells trade one error for
another and buy no reliable digit; the honest options for a curved boundary
stay (i) analytic coefficients, or (ii) binary rasterization at several N with
an extrapolation.

---

## 6. Assessment of the four candidate cures

**(a) Put every boundary on a grid point.** Correct, necessary, free — and it
must be paired with cell-centred sampling, because the *left-edge* rule breaks
the alignment again (155 of 260 where 156 is exact). It closes channel 1
exactly for every axis-parallel case, and — this is what makes it a
*cross-solver* fix — it can be satisfied in all three suites at once.

A centred rectangle of relative width `w` is exact on `N` cells iff `w·N` and
`(1−w)N/2` are both integral. For the battery that is `N` a multiple of 5
(`C1`, w = 0.6), of 10 (`C1b`, w = 0.4) and of 4 (`C2`, w = 0.5) — so **N a
multiple of 20**, and `NX_2D = 260` satisfies all three (verified: width and
area exact to 1e-16 at 260, 520 and 1040).

On the Moose side the grid is `refinement × (2m+1)` and `2m+1` is always odd, so
the same condition would read *refinement* a multiple of 20: 40, 60, 80, 100.

> **Moot on Moose's Atom path — see §9.** The probe shows Moose renders the
> rectangle one cell too wide whatever the alignment, so no refinement makes its
> own rasterizer exact. The condition still governs the *python* grid, and it
> still governs any grid handed to Moose explicitly — which, per §9, is now
> possible.

It does nothing for a circle, and by itself it still leaves channel 2. It
applies to 1D as well: `NX_1D = 8192` must become 10240 for `B3`.

**(b) Averaging / effective medium at boundary cells.** Half right and, as
usually stated, harmful — §5. Keep it as a grcwa-only option paired with a
separate harmonic-mean `1/eps` grid; do **not** make it the cross-solver
contract.

**(c) A test of nominal geometry against the solver at Δ ≤ 1e-6.** Right
instinct, wrong level. At the level of R, 1e-6 is not reachable on a sharp,
high-contrast 2D structure at any practicable order: the rules only share the
`q → ∞` limit, and at nG = 2601 (q = 51) the three faithful ones still spread
over 4.2e-04 on `C1` (0.38979 / 0.39016 / 0.38974) and 8e-04 on `C1b`
(`benchmark/README.md`) — on a *shared* mask, where the geometry cannot be the
cause.
The tolerance belongs on the **coefficients**, where it can be 1e-8 and where it
would have caught every one of the four bugs the four branches found. See §7
for the three-level test this becomes.

**(d) Analytic Fourier coefficients of the primitives.** Feasible, exact, and
worth having — but as an *oracle*, not as the common denominator:

* rect `S(G) = w_x w_y sinc(m w_x) sinc(n w_y)`, circle
  `S(G) = π r² · 2J₁(2π|G|r)/(2π|G|r)`, both times the centring phase; and for a
  two-material cell `1/eps` uses the *same* `S(G)` with `1/ε` amplitudes, so the
  inverse rule is analytic too. Implemented and verified here
  (`analytic_eps`): on the aligned C1 rectangle it agrees with the `sinc`-
  corrected FFT to 1e-12, and on the D2 circle it is the reference the table in
  §5 is measured against.
* In grcwa it is a contained change (build `eps_hat`/`eta_hat` from `S(G)`
  instead of from `get_conv`), and it removes channel 1 *and* channel 2 for the
  direct and inverse rules at once.
* It does **not** extend to everything. Li's two-step rule needs per-line 1D
  Toeplitz inverses, which are analytic only for axis-parallel shapes (an exact
  slab decomposition); for a circle it needs y-slabs — which is precisely the
  staircase Li is known to converge badly on. The tangent-field rules need the
  transform of a *smooth normal field*, which is not determined by the shape at
  all; it is a free ingredient that changes the finite-order value and not the
  limit.
* Stock Ikarus cannot take analytic coefficients (its API is integer topology +
  FFT), and Moose cannot at all. So analytic coefficients give a
  **geometry-error-free reference column**, not cross-solver agreement.

**(e) The candidate nobody proposed — and it works, see §9.** Moose's
`Layer(double thickness, CaModel epsilonDistribution)` takes an explicit
complex permittivity grid. If Moose consumes it as given, then all three suites
can be handed the *same pixel image*, the geometry drops out of the
cross-solver comparison completely (including for the circle), and the
pixel-image-to-nominal-shape gap becomes one common quantity measured once
instead of three different quantities that never cancel. **The probe says yes**,
and on the same pixel image Moose and Ikarus's Li rule agree to 7e-7. §9.

---

## 7. Proposal

### Step 0 — three questions for Moose (`moose/moose_raster_probe.cs`, run in Moose)

1. **`Layer(d, CaModel)`**: does Moose accept an explicit eps grid, does the
   result depend on the CaModel's resolution, and does
   `rRefinementFactorEpsFT` still resample it? Test: a grid-aligned rectangle
   handed as a CaModel at 260² and at 520² must give the same R as the same
   rectangle built from an `Atom`, and a deliberately asymmetric test pattern
   must come back changed.
2. **Refinement law**: refinement **40, 60, 80, 100** — the multiples of 20, so
   the geometry itself is exact (§6a) — at fixed order and nominal geometry on
   C1, C1b, C2, D2: confirm `1/refinement²` and record the Richardson `R(∞)`.
3. **Alignment**: dump `GetEpsilonDistributionsAsCaModel` at the refinement
   grids the sweep actually uses and confirm the pillar comes out at exactly
   `w·N` cells for the new (aligned) widths.

The script is written and type-checks against `moose/moose_api_stubs.cs`; see
[`moose/README.md`](moose/README.md#the-rasterization-probe) for what each probe
compares and what a failure of each one would mean. The one construction it
needs that no other script here uses is `Layer(double, CaModel)`, which has
never been exercised on a real build — if Moose rejects it, that is itself the
answer to P1, and P2/P3 are the fallback that answer needs.

### Step 1 — `structures.py` v2 (behaviour-preserving switch)

* one sampling rule everywhere: **cell centres, half-open**, so a boundary on a
  cell edge lands on exactly `w·N` cells;
* per-case grid `N` chosen so every axis-parallel boundary is a cell edge
  (`exact_N()` in the study module: 260 for C1/C1b/C2; `NX_1D = 10240`);
* `layer_mask(s, N=None)` so an N-sweep is a first-class operation, plus
  `analytic_coeffs(s, G)`;
* the current mask stays reachable (`legacy=True`) so every recorded number
  remains reproducible — the switch invalidates all 2D values on record and
  that has to be an explicit, dated flag, not a silent change.

### Step 2 — grcwa

* `sinc` quadrature as an opt-in (`obj(..., eps_quadrature="pixel")`), default
  off. One line in `get_conv`, and it makes the direct rule grid-exact.
* `pol_sigma` expressed as a fraction of the period, and the tangent field
  switched to double-angle diffusion so opposite normals stop cancelling.
  Both are backwards-compatible additions with the old behaviour still
  reachable.
* optional: analytic `eps_hat`/`eta_hat` for the battery's primitives — the
  oracle column.

### Step 3 — the test, in three levels

| level | what is compared | tolerance | why that number |
|---|---|---|---|
| **L0 geometry** | the eps and 1/eps Fourier coefficients each suite feeds its eigenproblem, against the analytic ones | **1e-8** | no solving, no truncation; the `sinc` table in §1 already reaches 1e-16 |
| **L1 physics** | R on a *band-limited* structure (eps a trigonometric polynomial, `\|m\|,\|n\| ≤ K`) | **1e-6** | every grid with `N > 2M + K` reproduces the coefficients exactly, eps is smooth so all rules coincide and convergence in q is exponential — any residual is a convention difference, not geometry |
| **L2 battery** | R on the real cases, at converged/extrapolated order | 1e-6 (1D, aligned, lossless) · 1e-5 (2D aligned rect) · 1e-4 (circle; metal; sharp corners) | set by the truncation error the physics allows, not by the geometry — `D1` already shows 1.8e-07 between the two faithful families at 625 orders, `C2` shows nothing like it at any order run so far |

L0 is the test that can actually be tight, and it is the one that would have
caught the radius-vs-diameter `Atom` constructor and the mask width directly
from the geometry, without a solve. (The percent scaling and the polarization
default were caught by the energy balance, which stays as it is — the two tests
are complementary, not alternatives.) L1 is a new case worth adding to the
battery (`E1_bandlimited`); it is the only structure on which "all three suites
agree to 1e-6" would be a statement about the solvers rather than about the
grid.

One caveat that has to be respected by the implementation: the `sinc` factor of
§1 is correct for a **pixel** input (eps piecewise constant on cells) and
**wrong** for a smooth one. The DFT of point samples of a band-limited function
is already exact, so L1 must run with the correction *off*, and inside
`Epsilon_fft_pol` it must be applied to `eps` and `1/eps` only, never to the
tangent-field projections `P_ij`, which are smooth by construction. (Measured, applying it to
`P_ij` as well moves R by ~2e-5 at N = 260 and less on finer grids — small, but
wrong on principle and free to get right.)

### Step 4 — then re-run

Everything on record for the 2D cases is invalidated by Step 1, on all three
sides. That is the cost, it is unavoidable, and it is the reason this is a gate
and not a patch.

---

## 8. Reproducing this

```bash
pip install ikarus-rcwa                       # optional, adds the Ikarus columns

python benchmark/rasterization_study.py coeffs --case C1_Si_pillars \
       --grids 260 520 1040 --orders 30
python benchmark/rasterization_study.py coeffs --case D2_ikarus_cylinder_TE
python benchmark/rasterization_study.py grid   --case C1_Si_pillars --q 21
python benchmark/rasterization_study.py pol    --case C1_Si_pillars --q 21
python benchmark/rasterization_study.py fill1d --case B3_Au_slits_TM --q 201
python benchmark/rasterization_study.py circle --q 21
```

---

## 9. What the Moose probe returned

`moose/moose_raster_probe.cs`, run on the real build (24 cores, 33 min).
Every solve came back `ok` with `R + T + A − 1` at 1e-16.

### P1: yes — and it changes the plan

**`Layer(double thickness, CaModel epsilonDistribution)` exists, compiles and
works.** Moose can be handed an explicit permittivity grid.

* **The values are `eps`, not `n + ik`.** The index variants are wildly wrong
  (`C1` 0.0366 against 0.398), and P3's dumped levels read `1.000000 |
  12.250000` for air/Si and `-48.910000 + 4.200000i | 1.000000` for gold —
  epsilon, with `+i` loss, the battery's own convention.
* **The grid is really used.** ±4 % on the width moves R by 4e-2 … 1e-1; the
  `eps` grid and the `n + ik` grid differ by 3.6e-1.
* **No transpose, no shift.** The `AN_aniso_control` (0.6 × 0.4, TE) is the row
  that could see it: Moose returns 0.893828 where python gives **0.893785** for
  that orientation and **0.656883** for the transposed one. The index
  convention is `(i, j) → (x, y)`.
* **Two mask resolutions that mean the same rectangle give bit-identical R.**
  `C1` returns 0.39625689979541406 from a 260² grid and from a 520² grid, to
  every digit; likewise `C1b` and `C2`. With the geometry exact, the mask
  resolution drops out completely.
* **The refinement still resamples an explicit grid**, by 8.5e-5 … 4.5e-4
  between `fft = 40` and `fft = 100`. That is the one term left in the budget.

### And with the same pixel image, Moose is Ikarus's Li rule

Python solved bit-for-bit the same masks (verified cell by cell, not assumed),
q = 21:

| case | Moose, mask, fft = 40 | `ikarus[li]` | Δ | `ikarus[NV]` | Δ |
|---|---:|---:|---:|---:|---:|
| `C1_Si_pillars` | 0.396256900 | 0.396256166 | **+7.3e-07** | 0.399130762 | −2.9e-03 |
| `C1b_Si_pillars_diffract` | 0.159508882 | 0.159528879 | −2.0e-05 | 0.162126121 | −2.6e-03 |
| `C2_Au_holes` | 0.672547367 | 0.672548085 | **−7.2e-07** | 0.646252195 | +2.6e-02 |
| `D2_ikarus_cylinder_TE` | 0.905398790 | 0.905812281 | −4.1e-04 | 0.943083334 | −3.8e-02 |
| `AN_aniso_control` | 0.893827841 | 0.893784640 | +4.3e-05 | 0.893513221 | +3.1e-04 |

Two things follow, and both are new.

**The cross-solver target is reached on the aligned 2D cases.** 7e-7 between two
independently written codes on a 2D structure — below the 1e-6 the study was
aiming at — the moment they are handed the same pixel image. Geometry was the
whole disagreement; nothing else had to change.

**Moose's factorization is Li's separable inverse rule**, or something
numerically indistinguishable from it on these five shapes. It is not a
normal-vector method: on `C2` it sits 2.6e-2 from NV and 7e-7 from Li, on the
curved `D2` 3.8e-2 from NV and 4e-4 from Li. That reclassifies what Moose is
*for* in this study — it is a second implementation of Li, not an independent
arbiter between Li and the normal-vector rule, and it inherits Li's slow
convergence on curved boundaries. The `D2` values that "were still climbing" are
Li climbing.

### P3: Moose's rasterizer is one cell too wide, always

Not "rounds outward" — a deterministic off-by-one, at every refinement and
order, on every rectangle:

| case | m | fft | N | cells | exact `w·N` |
|---|---:|---:|---:|---:|---:|
| `C1` | 10 | 40 | 840 | 505 | 504 |
| `C1` | 15 | 100 | 3100 | 1861 | 1860 |
| `C1b` | 10 | 40 | 840 | 337 | 336 |
| `C2` | 10 | 40 | 840 | 421 | 420 |

and the round-trip fill fractions make it an exact square: 157², 313², 105²,
131² where 156², 312², 104², 130² are exact. The circle is rasterized by a
different rule again — 18513 cells at 256² against the cell-centred 18544.

The Atom-path gap this predicts has the right sign in every case and the right
size to within a factor 0.2–1.1 (`C1` at fft = 100: predicted +1.39e-03 from
`dR/dw = +2.91` measured *in Moose*, observed +1.52e-03). The renderer is a
diagnostic, not the solver's grid, so that is as close as this can get from
outside — the mechanism is settled, the exact coefficient is not.

### P2: inconclusive, and the reason is P3

The refinement fits are poor (residuals 1e-4 … 3e-4, thirty to sixty times the
old three-point fit) and the `p = 1` / `p = 2` verdict flips between m = 10 and
m = 15. The sequences are **non-monotone** in three of four cases (`C1` at
m = 10: 0.398145, 0.397476, 0.397591, 0.397322), which no single power law can
produce. Two error channels with different exponents are being fitted at once —
plus, possibly, a third effect: an internal FFT grid rounded to a transform-
friendly size rather than exactly `r·(2m+1)` would wobble like this. Untested.

**P2 has to be re-run on the mask path**, where the shape is exact at every
refinement and only the sampling channel is left. `moose/moose_raster_probe.cs`
now has this as **P4**: `fixed` (one CaModel per case, only the refinement
varies) and `matched` (the CaModel rebuilt at `N = refinement × (2m+1)` every
time, so there is nothing left to resample). A flat `matched` result would mean
the CaModel path has no remaining grid dependence at all on axis-aligned
geometry, closing the last open channel-2 question. Not yet run.

### An independent check: the +1-cell bug is specific to the 2D Atom path

A manual UI cross-check (period 0.5, atom-width fraction 0.6, thickness 0.4,
refinement 30, m = 15, TE) corroborates P3 from a completely different angle.
`N = 30·(2·15+1) = 930` and `0.6 · 930 = 558` exactly — an aligned grid, so
there is no rounding ambiguity to begin with.

For the **1D** lamellar grating at this geometry, python at exactly 558/930
cells gives `T = 64.91491 %`, `R = 35.08494 %`, against the UI's
`T = 64.91510 %`, `R = 35.0849 %` — agreement to 4 significant figures, with
**no** +1-cell offset. So whatever internal grid Moose's 1D duty-cycle
construction uses, it is *not* the +1-cell-wide construction P3 found on the 2D
Atom path — consistent with the battery's long-standing finding that all 1D
cases already match Moose to five or six digits.

A second, independent piece of evidence from the same manual check: nudging the
width from 0.6 to 0.6001 (a 5×10⁻⁵ µm perturbation) moved the UI's `R` by
+0.045 percentage points. The *true* continuum response to that nudge is
essentially zero — solving the same nudge with the exact (sinc-corrected, no
grid at all) quadrature gives `R = 35.08489 %` at both 0.6000 and 0.6001,
identical to 5 digits — while a full one-cell jump on the 930-grid used above
is +0.462 points, about 10× the observed step. Both facts together (a real jump
where the continuum predicts none, but roughly a tenth of the 930-grid's own
pixel size) are consistent with Moose's 1D path being pixelated on some *other*,
finer internal grid (~10× the 930 estimate) that is unrelated to the
`refinement × (2m+1)` rule governing the 2D Atom renderer — not with the 1D path
being exact/analytic, and not with it sharing the 2D path's specific +1-cell
defect.

A companion **2D** manual reading at the same nominal parameters (`orders
15,15`, `fft-fac 30`) was inconclusive for a different reason: the reported
triple did not resolve into a `(T, R)` pair that sums to 1 for this lossless
stack, and none of the factorization rules this repo can construct (Laurent,
Pol, Li, NV) land anywhere near the reported R at any order tried. Rather than
force a reading, this is flagged as open — resolve via the validated
`moose_raster_probe.cs`/CaModel path (which the P1 table above already checked
digit-for-digit) rather than by hand-typing into the RCWA dialog, where the
UI shows only 6 digits and the exact column meaning for a 2D case is easy to
misread.
