# Fixing grcwa's Pol factorization — handoff

Scope: `grcwa/fft_funs.py` (`Epsilon_fft_pol`, `_compute_tangent_field_pol`) and
the one call site in `grcwa/rcwa.py` that invokes it. Nothing else in the
repository needs to change for this. This is a self-contained follow-up to
`benchmark/RASTERIZATION.md` §4 — read that section first for the measured
numbers; this document is the "how to fix it" that section pointed at.

**Do not start from a plan you write yourself — start from this one.** It
includes a derivation that has already been checked against Ikarus's reference
implementation to machine precision (§3 below), so the risky part — getting
the sign/swap conventions right — is done. What is *not* done is threading it
through grcwa's own `eps2` assembly and confirming the existing regression
suite still passes, which needs a real Python session with grcwa installed.

## 1. What "Pol" is trying to do, in one paragraph

Multiplying two functions that both jump at the same point (`eps(x)` and the
field) in a truncated Fourier series converges slowly (`O(1/M)`, Gibbs
ringing). Li (1996) showed that applying the *inverse* rule (`1/eps`, keeping
`eps·E` continuous) along the direction **perpendicular to the interface**
fixes this — but "perpendicular" has to be evaluated at *every point* of a
curved or oblique boundary, not just along a fixed x or y axis. The fix is a
smooth vector field, defined over the whole unit cell, that points along the
true local boundary normal wherever there is one. `Epsilon_fft_pol` already
does the *right thing* with such a field once it has one — the bug is entirely
in *how the field is built* from the pixel grid, in
`_compute_tangent_field_pol` (`grcwa/fft_funs.py:119-256`).

Ikarus's `normal` rule (`ikarus/core/_normalvector.py`, pip-installed at
`.../site-packages/ikarus/core/_normalvector.py` — read it directly, it is
short) solves the same problem correctly. It is the reference this handoff
ports from.

## 2. The two bugs, and where they live

### Bug A — `pol_sigma` is in pixels, not physical units

`grcwa/rcwa.py:320-334` passes `self.pol_sigma` (default `3.0`) straight into
`Epsilon_fft_pol(dN, ep_grid, self.G, pol_sigma=self.pol_sigma, ...)`, and
`_compute_tangent_field_pol` (`grcwa/fft_funs.py:205-207`) builds the Gaussian
blur kernel directly from it in pixel-frequency space:

```python
kx_freq = np.fft.fftfreq(Nx)
blur_kernel = np.exp(-2 * np.pi**2 * pol_sigma**2 * (KX**2 + KY**2))
```

`pol_sigma` pixels is a *physical* smoothing length of `pol_sigma / Nx` of the
period. Refine the eps grid (bigger `Nx`) without also scaling `pol_sigma` and
the smoothing shrinks physically — the extended field keeps changing shape
instead of settling into a fixed limit as the grid refines. Measured
(`RASTERIZATION.md` §4, `C1_Si_pillars`, exact geometry, q = 21):

| N | `pol_sigma = 3` px (current) | `pol_sigma = 3·N/260` (period-relative) |
|---:|---:|---:|
| 260 | 0.397270590 | 0.397270590 |
| 520 | 0.385865456 | 0.395534357 |
| 1040 | 0.392567850 | 0.394896877 |
| 2080 | 0.378795724 | 0.394955771 |

Scaling `pol_sigma` by `N` already cuts the spread from 1.8e-2 to ~6e-5 — this
half of the fix is small, mechanical, and already validated numerically (that
table is from a real run, not a projection).

**Fix.** `Epsilon_fft_pol` and `_compute_tangent_field_pol` need `Nx`/`Ny`
already (they use them to build the grid); change the *semantics* of
`pol_sigma` to "smoothing length as a fraction of the period" and multiply by
`min(Nx, Ny)` (or `max`, pick one and document it — Ikarus uses
`max(nx, ny) / 12.0` as its *default*, i.e. the same idea, just baked into a
default rather than exposed as a raw parameter) before building the kernel.
Keep the constructor parameter name `pol_sigma` (its default should probably
become something like `1/12` to match Ikarus's default physical smoothing
length, not `3.0`) — this is an additive, backward-*incompatible* change to
what the number means, which is fine (§5), but every caller passing an
explicit `pol_sigma` will need updating, including
`benchmark/rasterization_study.py`'s `cmd_pol`.

### Bug B — the tangent field cancels on opposite-facing interfaces

This is the deeper bug, and Bug A's fix does not cure it (see the `B3` numbers
below — those already use a period-relative sigma and it still drifts).

`_compute_tangent_field_pol` (`grcwa/fft_funs.py:190-193`) builds the *raw*
tangent as the 90°-rotated gradient of `eps`:

```python
tx_raw = -grad_y
ty_raw = grad_x
```

and then blurs `tx_raw`, `ty_raw` **directly**, as ordinary real vector
components (`grcwa/fft_funs.py:210-232`). On a 1D grating this raw field
points `+x` at the left edge of the bar and `-x` at the right edge — literally
opposite directions, separated by only the bar's width. Blurring two opposite
vectors averages them **toward zero**, producing a spurious near-zero band
inside the bar where the field should instead be a smooth, well-defined,
*non-zero* interpolation between the two edges. Where that band sits (and how
wide it is) depends on the grid, because it is an artifact of cancellation, not
of the physics — which is exactly the extra grid-dependence measured on
`B3_Au_slits_TM` (1D, q = 201, **`pol_sigma` already fixed to scale with N**,
so this residual is Bug B in isolation):

| N | 8000 | 10240 | 16000 | 32000 |
|---:|---:|---:|---:|---:|
| Pol | 0.792535 | 0.792669 | 0.792915 | 0.793305 |

still visibly climbing toward the faithful value (~0.79330, `Li`/`NV`/Moose all
agree there — see `RASTERIZATION.md` §9) at N = 32000, on the *simplest
possible* case (a straight 1D interface) — because a straight grating has
exactly two, exactly-antiparallel edges, which is the worst case for this bug,
not an edge case of it.

**The fix — double-angle (orientation) encoding, not raw-vector blurring.**
Two vectors pointing in opposite directions (`θ` and `θ + 180°`) represent the
*same undirected orientation* — an interface doesn't care which way you call
"outward". If you encode direction as the **doubled angle**
`z = (gx + i·gy)²`, then `θ` and `θ + 180°` map to the *same* `z` (since
`2(θ+180°) = 2θ + 360°`), so opposite-facing raw gradients **add instead of
cancelling** when `z` is blurred. Recover the physical projection by halving
the angle back — except you never actually need to call `angle()`/`arctan2` at
all, which matters because `grcwa/backend.py` exposes neither under the
autograd backend (checked — no `angle`, no `arctan2`, this would otherwise be
a blocker). The projection matrices `Epsilon_fft_pol` needs
(`P_xx = tx², P_yy = ty², P_xy = P_yx = tx·ty`, `grcwa/fft_funs.py:97-104`)
only need `cos(2θ)` and `sin(2θ)` — i.e. `Re(z)/|z|` and `Im(z)/|z|` — via the
half-angle identities `cos²θ = (1+cos2θ)/2`, `sin²θ = (1−cos2θ)/2`,
`sinθ·cosθ = sin(2θ)/2`. No square root of a possibly-negative number, no sign
ambiguity, no new backend primitive.

**Verified against Ikarus's own convention to machine precision** (Ikarus
additionally swaps `Pxx`↔`Pyy` relative to the naive `tx²`/`ty²` — "the
Liu-Fan diagonal-swap correction", `ikarus/core/_normalvector.py` docstring —
carry that swap over too):

```python
import numpy as np
theta = np.random.uniform(-np.pi, np.pi, size=8)   # any smooth orientation field
tx, ty = np.sin(theta), -np.cos(theta)              # ikarus's tangent_field()
Pxx_ik, Pyy_ik, Pxy_ik = ty**2, tx**2, tx*ty         # ikarus's tangent_terms()

zc = np.cos(2*theta) + 1j*np.sin(2*theta)            # = z / |z|, unit modulus
Pxx = (1 + zc.real) / 2
Pyy = (1 - zc.real) / 2
Pxy = -zc.imag / 2
# max|Pxx-Pxx_ik|, max|Pyy-Pyy_ik|, max|Pxy-Pxy_ik| are all < 1e-15
```

**The construction, concretely, replacing `grcwa/fft_funs.py:186-232`:**

1. Keep the existing gradient computation (`grad_x`, `grad_y`) as is — that
   part is fine.
2. Build the complex doubled-angle field: `z = (-grad_y + 1j*grad_x)**2`
   (using `bd`, not raw `np`, so this stays autograd-differentiable — it is
   just `+`, `-`, `*`, all of which the backend already supports on complex
   arrays).
3. Blur `z.real` and `z.imag` **separately** through the *existing* FFT-based
   `blur_kernel` machinery (`grcwa/fft_funs.py:210-232` already does exactly
   this kind of blur-and-reset for `tx_raw`/`ty_raw` — reuse the same
   iteration, just apply it to `z.real`/`z.imag` instead). This is where Bug A
   and Bug B's fixes compose: the *physical* smoothing length from §2a governs
   this same blur.
4. Recover the projections directly, no angle needed:
   `az = bd.sqrt(z.real**2 + z.imag**2)` (this is `|z|`, always ≥ 0, needs the
   same zero-floor guard the current code already has for `|t|` —
   `POL_GRAD_TOL`, `t_floor` — reuse that pattern, just applied to `|z|`
   instead of `|t|`), then:
   ```python
   P_yy = (1 + z.real / az) / 2      # note the swap vs. the naive tx**2
   P_xx = (1 - z.real / az) / 2      # -- see the Ikarus mapping above
   P_xy = P_yx = -z.imag / az / 2
   ```
5. Everything downstream of `P_xx, P_xy, P_yx, P_yy` in `Epsilon_fft_pol`
   (the `mDelta`, `E_xx`/`E_xy`/`E_yx`/`E_yy` assembly, `grcwa/fft_funs.py:
   105-116`) is **unrelated to this bug and already tested correct** (see §5)
   — do not touch it. The one thing to re-verify empirically (not assumed from
   the derivation above) is whether grcwa's `E_xx = eps_hat + mDelta @ P_xx_hat`
   convention wants `P_xx` or `P_yy` in that slot — i.e. whether the swap
   needs to also happen at the point `Epsilon_fft_pol` consumes
   `P_xx_hat`/`P_yy_hat`, or whether it cancels against a swap already present
   there. `test_pol_correctness.py`'s two invariants (below) will tell you
   immediately if this is backwards: swapped, TM will get *worse* not better,
   and TE (which must stay exactly Laurent) is actually the sharper canary —
   if TE stops reproducing Laurent exactly, the swap (or something else) is
   wrong, because on an axis-aligned 1D interface `Pxy` must be exactly 0 and
   `{Pxx,Pyy}` must exactly select the physically correct axis, with no room
   for a sign error to hide.

## 3. Why this isn't "just copy Ikarus's file"

Ikarus's `tangent_field()` uses `scipy.ndimage.gaussian_filter` — plain numpy,
not autograd-differentiable. grcwa's `Epsilon_fft_pol` docstring is explicit
that Pol has to stay differentiable through the `bd` backend "so that
gradients propagate correctly through the Pol correction during topology
optimization" — that requirement doesn't go away with this fix. The rewrite in
§2 keeps grcwa's existing FFT-based blur (`bd.fft2`/`bd.ifft2` convolution with
`blur_kernel`, already differentiable) and only changes *what* gets blurred
(the doubled-angle field instead of the raw vector) — so autograd
compatibility should be preserved by construction, but **this needs an actual
gradient check**, not just an assumption: `grcwa.set_backend('autograd')`,
build a small lossy structure, and confirm `autograd.grad` through a Pol solve
still returns finite, correct-looking numbers (finite-difference against a
`numpy`-backend solve at two nearby parameter values is the standard check —
look for how `tests/` already does this for other autograd paths, if it does).

## 4. Validation plan — hard numbers to hit

All of these are runnable today with `benchmark/rasterization_study.py` (the
`pol` and `fill1d` subcommands already exist and were used to produce the
tables in §2) plus the existing `tests/test_pol_correctness.py`. None of them
require Moose or Ikarus, though Ikarus (`pip install ikarus-rcwa`) is useful as
a second opinion since it is the reference this fix ports from.

1. **Regression baseline, before touching anything.** Run
   `python benchmark/rasterization_study.py pol --case C1_Si_pillars --q 21`
   and `fill1d --case B3_Au_slits_TM --q 201`, save the output. These should
   reproduce the tables in §2 above (they're from real runs, not
   projections) — if they don't, something about the environment/grcwa
   version differs from what this handoff was written against, and that has
   to be resolved before the fix is judged by these numbers.

2. **`tests/test_pol_correctness.py` must still pass, unmodified.** Both
   invariants — TE reproduces Laurent *exactly* (to whatever tolerance is
   already in that file), and TM converges to the same limit as Laurent, just
   faster — are the two cheapest, sharpest regression guards available and
   were written for exactly this kind of change. If TE stops matching Laurent,
   stop and re-check the swap in §2 step 5 before doing anything else.

3. **Bug A alone (sanity-check the pixel→physical conversion in isolation).**
   `pol --case C1_Si_pillars --q 21`, N = 260/520/1040/2080: the spread across
   those four values should drop from the current 1.8e-2 to roughly the 6e-5
   already demonstrated by the ad hoc `pol_sigma = 3·N/260` patch in §2 — if
   the built-in fix doesn't reach that, the physical-length conversion has a
   mistake (wrong axis, wrong factor of 2, etc.).

4. **Bug B, isolated on the worst case.** `fill1d --case B3_Au_slits_TM
   --q 201`, N = 8000/10240/16000/32000: currently climbs 0.792535 → 0.793305
   (Δ = 7.7e-4) still visibly unconverged at the top of that range. After the
   fix, Pol should land within **~1e-4 of 0.79330** (the value Li/NV/Moose all
   agree on, `RASTERIZATION.md` §9) at a *much* smaller N than 32000 — ideally
   already flat by N ≈ 8000–10240 — since a correctly-encoded orientation
   field has nothing left to converge *in N* on an exactly-aligned 1D
   interface (only in the truncation order `q`, which this test holds fixed).

5. **Both fixes together, on the hard case.** `pol --case C1_Si_pillars
   --q 21`, N = 260/520/1040/2080, compared against `ikarus[li]`
   (0.396256166 at this q, from `RASTERIZATION.md` §1) and `ikarus[normal]`
   (0.399130762 → 0.402848294 over the same N range — itself not perfectly
   flat, because of the real corner effect described in `RASTERIZATION.md`
   §4, not a bug). A fixed Pol should land somewhere in the neighbourhood of
   those two — closer to `li` than to the pre-fix Pol numbers — with a spread
   across N at least an order of magnitude tighter than the current 1.8e-2.
   Do not expect *better* than `ikarus[normal]`'s own corner-effect floor; that
   floor is physical (a genuine discontinuity in the true normal direction at
   a rectangle's corners), not a defect either implementation can remove by
   construction.

6. **`D2_ikarus_cylinder_TE` (the curved case) as the final check**, since
   that is the whole point of having a normal-vector method at all — compare
   fixed-Pol against `ikarus[normal]` (0.943083334 at q = 21 on the
   cell-centred mask, `RASTERIZATION.md` §5/§9) and expect close agreement, in
   contrast to the pre-fix Pol/Laurent numbers which sit nowhere near it.

## 5. Backward compatibility

This is a *quality* fix, not a new feature — every existing `fork[Pol]` number
on record was already flagged as grid-dependent and not currently trustworth
as an independent cross-check (`RASTERIZATION.md` §4, §7); nothing here needs
to preserve bit-compatibility with the current output. `pol_sigma`'s default
value and units are changing (§2a) — that is an intentional, documented
break, not an oversight; bump whatever changelog/version marker this repo
uses. `tests/test_pol_correctness.py` is the one thing that must not need its
*expected values* changed (its invariants — TE = Laurent exactly, TM converges
faster than Laurent to the same limit — are rule-level statements that hold
regardless of how the tangent field is built); if a fix requires loosening a
tolerance there, that is a signal to look harder at the implementation, not a
green light to loosen the test.
