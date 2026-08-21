# grcwa benchmark / cross-validation

A small, self-contained suite that builds a battery of physically-motivated
gratings, runs them through several grcwa installations, factorization modes and
one **independent codebase**, **cross-checks the results** and **compares
timing**, and exports the data to `results.json` / `results.csv`.

It is *not* a pytest module — run it directly. (The cross-code assertions that
*are* pytest live in `tests/test_ikarus_crosscheck.py`.)

## What it compares (the "suites")

Suites are **auto-discovered**: any `benchmark/grcwa*` directory that is an
importable package becomes a column, plus the current branch (`fork`) at the repo
root. Each suite is run with Laurent's rule and, if its `obj` implements
`fmm_method='pol'`, also with Pol — so `[Pol]` columns appear only for the
versions that actually support it. The column label is the directory name with
the leading `grcwa` stripped (`grcwa-weiliang-013` → `weiliang-013`).

Set `GRCWA_VARIANTS` to a comma-separated list of those labels to restrict the
grcwa variants without moving or deleting package directories. For example,
`GRCWA_VARIANTS=fork` runs only `fork[Laurent]` and, when supported,
`fork[Pol]`. Ikarus remains an independent suite and still runs all three of its
factorization modes.

On top of that family, **Ikarus** — a separate code with a separate API — adds
`ikarus[Laurent]`, `ikarus[Li]` and `ikarus[NV]` when it is installed
(`pip install ikarus-rcwa`); see [Ikarus, the independent
code](#ikarus-the-independent-code) below.

Drop the versions you want to compare into `benchmark/`, e.g.:

| directory                     | column label              | Pol? |
|-------------------------------|---------------------------|------|
| `grcwa-weiliang-013`          | `weiliang-013`            | yes  |
| `grcwa-codex`                 | `codex`                   | no   |
| `grcwa-grcwaProjects`         | `grcwaProjects`           | no   |
| `grcwa_original-grcwaProjects`| `original-grcwaProjects`  | no   |
| *(repo root, this branch)*    | `fork`                    | yes  |

Including a known-Pol version (e.g. `weiliang-013`) lets you compare Pol
implementations. Note: `fork[Pol]` was originally a faithful port of the upstream
Pol method and reproduced `weiliang-013[Pol]` bit-for-bit -- but that upstream Pol
did **not** converge for TM (it oscillated over nG). The fork has since **fixed**
two bugs in it (the ``epsinv`` convention and the tangent-field normalization; see
`tests/test_pol_correctness.py`), so `fork[Pol]` now converges to the Laurent
limit faster than Laurent and **intentionally differs** from `weiliang-013[Pol]`
on TM/metal cases. The remaining hard case is a 2D metal grating with sharp
corners (field singularities), where *neither* rule converges at practical nG.

Versions without dimensionality inference have no native 1D/0D; for the 1D cases
they fall back to the historical *degenerate-2D* setup (a tiny second period so
only `Gy=0` survives) and cannot do the 0D case natively. This makes their 1D
order count differ from the fork's native `2M+1`, which is the only reason the 1D
columns are not bit-identical.

## The battery

`lambda = 1 micron` (`freq = 1`), lengths in microns; materials as `(n, k)`.
Cases span 0D/1D/2D, real and complex permittivity, sub- vs supra-wavelength
periods, TE/TM. Defined once in `structures.py`, grouped by what they test:

**Group A — analytic anchors.** `A1_slab_air`, `A1b_slab_glass`: planar Si slabs
(RCWA reduces to TMM), checked against the exact Airy result.
`A2_formbiref_TE`, `A2_formbiref_TM`: a deep-subwavelength grating, i.e. a
form-birefringent film, against effective-medium theory.

**Group B — 1D gratings.** `B1_Si_grating_TE` / `B1_Si_grating_TM` (lossless Si,
the TM one slow under Laurent), `B2_HCG_TM` (high-contrast subwavelength
grating), `B3_Au_slits_TM` (metal slit array, plasmonic; the hardest 1D).

**Group C — 2D rectangular pillars.** `C1_Si_pillars`,
`C1b_Si_pillars_diffract`, `C2_Au_holes` (2D metal hole array, the hardest 2D:
sharp corners plus loss, where *no* rule here converges at practical `nG`).

**Group D — the Ikarus whitepaper's cross-code cases.** `D1_ikarus_hcg_TM`, a
free-standing `n = 3.5` lamellar grating in TM: the factorization stress test of
that paper's Fig. 1 and Table 1. `D2_ikarus_cylinder_TE`, a free-standing
circular pillar, whose boundary is *curved* and oblique to both lattice axes —
the case that separates the normal-vector method from Li's separable inverse
rule. Both are specified in the paper in SI units at `lambda = 700 nm` and
re-expressed here at `lambda = 1` (RCWA is scale-invariant); the group-D
geometry, unlike groups A–C, is free-standing in air on both sides.

Each case prints R, T, A=1-R-T, R+T, the actual order count `nG`, and the
minimum wall time over a few repeats. Physical expectations: lossless cases
satisfy R+T=1; absorbers have A>0; more diffraction orders open as the period
grows past the wavelength.

Every patterned layer is rasterized **once**, by `structures.layer_mask`, and
that same integer mask is handed to grcwa (as a flattened `eps` vector) and to
Ikarus (as a topology plus one material per index). Neither backend draws its
own geometry, so a disagreement *between those two* can never be a pixel-grid
artifact.

### The mask is not the structure — and on 2D it was not close

> **Status: fixed. This section is the record of the investigation, kept in the
> past tense it earned.** Everything below describes the *legacy* grid
> (`NX_1D = 8192`, `NX_2D = 256`, rect sampled at the left cell edge). The
> current default puts every rect case on an exact grid — `NX_1D = 10240`,
> `NX_2D = 260`, cell centres — and the error it diagnosed is gone:
>
> | case | legacy grid | current default |
> |---|---|---|
> | `C1_Si_pillars` | 153 / 256 → **−0.391 %** | 156 / 260 → **0.000 %** |
> | `C1b_Si_pillars_diffract` | 103 / 256 → **+0.586 %** | 104 / 260 → **0.000 %** |
> | `C2_Au_holes` | 127 / 256 → **−0.781 %** | 130 / 260 → **0.000 %** |
> | `B3_Au_slits_TM` | 6554 / 8192 → +0.006 % | 8192 / 10240 → **0.000 %** |
> | `D2_ikarus_cylinder_TE` | +0.076 % | **+0.007 %** (a circle has no exact raster) |
>
> `python benchmark/geometry_fidelity.py` prints the legacy table below;
> `report(legacy=False)` prints the current one, where 0 of 11 patterned layers
> exceed 0.10 %. Moose's side of the same mismatch was closed independently, by
> rebuilding its 2D cells on an explicit permittivity grid (`CaModel`) instead
> of the outward-rounding `Atom` rasterizer.

Sharing one mask also means sharing its errors, and any code that builds the
geometry from the parameters instead (Moose, S4, a fab process) solves a
different structure. That is what `geometry_fidelity.py` measured on the legacy
grid:

| case | nominal | mask | rasterized | error |
|---|---|---|---|---|
| 1D, `ff = 0.5` | 0.500 | 4096 / 8192 | 0.500000 | 0.000 % |
| `B3_Au_slits_TM` | 0.800 | 6554 / 8192 | 0.800049 | +0.006 % |
| `C1_Si_pillars` | 0.600 | **153** / 256 | 0.597656 | **−0.391 %** |
| `C1b_Si_pillars_diffract` | 0.400 | **103** / 256 | 0.402344 | **+0.586 %** |
| `C2_Au_holes` | 0.500 | **127** / 256 | 0.496094 | **−0.781 %** |
| `D2_ikarus_cylinder_TE` | area 0.282743 | 256², centred | 0.282959 | +0.076 % |

`0.6 · 256 = 153.6` and `0.4 · 256 = 102.4` are not integers, and the rect
branch samples the **left cell edge** with a strict `<` — so on `C2`, where the
pillar edge lands exactly on a sample, a pixel is dropped on *each* side. The 1D
masks have no such error, and the circle branch already samples cell centres,
which is why group D's area error is an order of magnitude smaller. (Cell
centres with `<=` is also what `ikarus.shapes.rectangle` uses, so the battery's
rect convention does not even match Ikarus's own.)

Half a pixel is worth far more here than the truncation error at the top of the
sweep. Ikarus, Li's rule, nothing varied but the rasterization
(`geometry_fidelity.py --solve --q 41`):

| case | geometry | `w_eff` | R (q=31) | R (q=41) | Moose, same order |
|---|---|---|---|---|---|
| `C1_Si_pillars` | mask 256² | 0.597656 | 0.389961 | 0.390105 | 0.398436 / 0.398174 |
| | one pixel wider | 0.601562 | 0.401687 | | |
| | nominal | 0.600000 | 0.396804 | 0.396956 | |
| `C1b_Si_pillars_diffract` | mask 256² | 0.402344 | 0.146998 | 0.145824 | 0.154774 (q=41) |
| | nominal | 0.400000 | 0.157185 | 0.156324 | |
| `C2_Au_holes` | mask 256² | 0.496094 | 0.672335 | 0.666899 | 0.648546 (q=41) |
| | nominal | 0.500000 | 0.659043 | 0.649316 | |

One pixel of the 256 grid moves `C1`'s R by ~0.012 — more than the whole
disagreement with Moose. At matched order, swapping the mask for the nominal
rectangle closes **81 %** of the gap on `C1` (0.0085 → 0.0016), **82 %** on
`C1b` (0.0090 → 0.0016) and **96 %** on `C2` (0.0184 → 0.0008) — three cases,
three factorization rules, one cause. What is left over is the size of Moose's
own scatter along its 2D sweep (±0.001 on C1, ±0.002 on C1b), which is itself
consistent with `FFT_MODE = 1` re-choosing the eps grid at every order.

It is the *width*, not the grid. `C1`'s nominal rectangle rendered on 260² and
on 1280² — two FFT grids a factor of five apart, both representing the same
rectangle exactly — gives Li 0.396804 and 0.396800, agreeing to 4e-6. So neither
the sampling density nor the aliasing of the eps transform carries the effect;
what the codes disagree about is which rectangle they were handed. (The
normal-vector rule is the exception, 0.397477 vs 0.399886, because it builds its
tangent field *from* the rendered grid — one more reason not to compare rules
across rasterizations.)

### Closing it from the Moose side

`benchmark/moose/moose_geometry_probe.cs` asked Moose the same questions, and
the answers close the chain:

1. **Moose rasterizes too, and binary** — `levels = 2` at every resolution the
   probe renders. But it rounds *outward*: `C1` at 256 comes out **155** px wide
   where `structures.py` takes **153** and 153.6 would be exact. Both codes
   rasterize, both are wrong, and they are wrong in opposite directions.
2. **Moose has the same width sensitivity.** At m = 10 on `C1b`, moving the atom
   from the nominal 0.400000 to the mask's 0.402344 moves Moose's R by
   **0.010060** (0.157900 → 0.147840). The same step moves Ikarus-Li by
   **0.010187**. Same structure, same physics, same response.
3. **`FFT_MODE = 1` never worked on this build.** Moose clamps
   `rRefinementFactorEpsFT` to **[30, 100]** — the RCWA dialog refuses anything
   outside it, and the API substitutes silently: passing 13 gives a result
   bit-identical to passing 30. Since `ceil(256/q)` is 13 at q = 21 and 5 at
   q = 61, *every* 2D point of the recorded sweep ran at refinement 30, and the
   absolute grid grew with the order (630 samples at q = 21, 1830 at q = 61) —
   exactly what the mode was written to prevent. The constants now say [30, 100]
   so the CSV records the refinement Moose actually used.
4. **And that residual is the last of the difference.** From the RCWA dialog on
   `C1` at m = 10 with the nominal geometry: refinement 30 → 0.398784, 50 →
   0.397764, 100 → 0.397322. Extrapolated as `1/refinement` that is **0.39688**,
   against **0.396804** (q = 31) and **0.396956** (q = 41) from grcwa and Ikarus
   on the same nominal rectangle — agreement to ~1e-4. The 0.0012–0.0016 left
   over after the geometry correction is Moose's own eps sampling, not physics.
5. **The `(0,0)` points are confirmed broken.** In the width sweep at m = 0 the
   result does not depend on the atom size at all — `C1` returns 0.363252 for
   every width, `C1b` 0.039244, `D2` 0.021876 — so Moose ignores the geometry
   entirely at zero orders. Those points must not be plotted or merged.

The UI and the script agree exactly, incidentally: entering `C1` by hand at
10/10 orders and fft-fac 30 gives 39.8784 %, bit for bit what the script
reports. Whatever else differs between the two codes, the Moose side is one
consistent solver.

**That is what the 2D Moose disagreement is**, and it is not a factorization
difference. Different rules cannot converge to different limits on one
structure, and at `nG = 2601` they do not: on `C1b` all four columns land within
0.0008 of each other (0.14535 / 0.14544 / 0.14539, with even Laurent at
0.14606), on `C1` the three faithful rules do (0.38979 / 0.39016 / 0.38974;
Laurent is still crawling in from above at 0.42744, as it does on every
high-contrast case). They miss Moose by 0.008–0.010 *together*, because they are
all solving the mask while Moose solves the nominal pillar.

The 1D cases are the control: same conventions, same rules, exact masks — and
there the faithful columns match Moose to five or six digits
(`B1_Si_grating_TM` 0.213710, `B2_HCG_TM` 0.873329, `D1_ikarus_hcg_TM`
0.100173, `B3_Au_slits_TM` 0.79330 vs 0.79326).

`D2_ikarus_cylinder_TE` is the one 2D case where the mask is *not* the problem —
its circle branch already samples cell centres, and refining it from 256² to
512² and 1024² moves R by at most 0.0015 in either rule at either order (Li
0.9180 / 0.9166 / 0.9169 and NV 0.9421 / 0.9408 / 0.9411 at q = 31). There Moose
is simply not converged: its points follow `R(m) = R∞ − c/m` with an rms
residual of 1.8e-4 and `R∞ = 0.9401`, against the python normal-vector value
0.9428.

Everything else that could differ between the three constructs was checked and
does not: the refractive indices, the wavelength (`FREQC` detunes grcwa by 5e-8
and nothing else), the periods, depths and substrates, the polarization (all four
2D cases are C4v-symmetric, so TE = TM at normal incidence anyway), the retained
order *set* (grcwa's parallelogramic truncation returns exactly the symmetric
−M…M block that Ikarus's `(M, M)` and Moose's `(m, m)` retain), the percent
scaling and order summation on the Moose side (its energy balance closes to
2e-16), and the finite-thickness end layers, which enter grcwa's S-matrix as a
phase only. Two battery-wide controls back that up: `A1b_slab_glass` fixes the
substrate and the illumination side to seven digits — every 1D case is
free-standing, so the 0D one is what tests that — and `B3_Au_slits_TM` fixes the
absorption sign convention on gold to 6e-5.

Fixing the mask is one change in `structures.layer_mask` — sample cell centres,
as the circle branch already does, and pick `NX_2D` so that `ax/Λ · NX_2D` is an
integer (`260` suffices for all three rect cases; an axis-aligned rectangle whose
edges fall on pixel boundaries has no staircase at all). It invalidates every
recorded 2D number, so it is deliberately *not* done here — `geometry_fidelity.py`
reports the error instead, and `benchmark/moose/moose_geometry_probe.cs` is the
Moose-side script that measures the same thing from the other end.

## Ikarus, the independent code

Everything else in this benchmark is grcwa comparing against grcwa, which can
only ever catch *differences between versions* — never a mistake all of them
share. [Ikarus](https://github.com/CAVITYtechnologies/ikarus) (CAVITY
technologies GmbH, [whitepaper doi 10.5281/zenodo.21966455](
https://doi.org/10.5281/zenodo.21966455), PDF in the repo root) is a separately
implemented RCWA/FMM code, so it closes that gap.

```bash
pip install ikarus-rcwa       # optional; the columns just vanish without it
```

`benchmark/ikarus_suite.py` is the adapter. Ikarus's *construct* differs from
grcwa's on nearly every axis, so the structures are genuinely rebuilt rather than
re-parametrized:

| | grcwa | Ikarus |
|---|---|---|
| units | dimensionless, `freq = 1/lambda` | SI metres, `wavelength=` |
| stack | `Add_LayerUniform` / `Add_LayerGrid`, finite end layers | cover → substrate, ends **semi-infinite** (`height=np.inf`) |
| pattern | flat `eps` vector over the grid | integer topology + one material per index |
| truncation | `nG`, a total order count | `n_orders=(Mx, My)`, the **maximum** order per axis |
| returns | `R, T = RT_Solve()` | `T, R, result = simulate()` — **transmission first** |
| rules | Laurent, Pol | `laurent`, `li`, `normal` (default `auto` = normal-vector) |

The adapter maps the battery's per-axis order count `q` to `M = (q-1)//2`, since
Ikarus retains `2M+1` harmonics per axis — so both codes end up at the *same*
`nG` (`tests/test_ikarus_crosscheck.py` asserts this, because a slip there would
make every comparison quietly unfair). Only odd `q` is representable; an even one
is reported as skipped rather than rounded.

One physical difference remains between the columns: this battery feeds grcwa a
slightly complex frequency (`structures.FREQC`, Q = 1e7) to regularize Rayleigh
anomalies, and Ikarus has no such knob. That is a 5e-8 relative detuning, and it
is the reason the cross-code Laurent agreement lands at ~1e-6 rather than at 0.

### What the cross-check found

Run `python benchmark/ikarus_whitepaper_check.py` for the live version of this.

1. **The direct rule agrees across codebases**, on all 13 cases, to between
   1e-9 and 2.6e-6. So where the rules disagree, it is the *rule* — not a bug in
   either code. That is the single most valuable thing an independent
   implementation can tell you, and it is now checked on every run.
2. **The whitepaper's physics claim holds.** On `D1_ikarus_hcg_TM` the direct
   rule crawls in from above — 0.163 at 25 orders, 0.137 at 41, 0.107 at 201 —
   toward a faithful value of 0.100, while `R+T = 1` to better than 5e-6 at every
   single one of those points. Energy balance is not a convergence test.
3. **Its Table 1 reproduces to ≤ 3e-4**, including the value it attributes to
   grcwa (see `ikarus_reference.json` for the published-vs-measured table).
4. **But that grcwa row does not describe this fork.** It describes the
   Laurent-only upstream. This fork's *fixed* Pol factorization converges to
   0.1001 — onto the faithful value, not Laurent's wrong one. On the curved case
   `D2_ikarus_cylinder_TE` (441 orders) the fork's Pol lands at 0.9430, on top of
   Ikarus's normal-vector 0.9431 and 3.7 points away from its Li 0.9058: the
   fork's tangent field is built from the rendered `eps` grid, so it behaves like
   a normal-vector method rather than like a separable one. That case also
   confirms the paper's other claim — the normal-vector method really does beat
   Li's separable rule on a boundary oblique to both axes.
5. **Ikarus still wins the convergence *rate*, decisively** — and per-solve speed
   is the wrong axis to compare these on. At matched `nG` the fork is ~2-5×
   faster per solve, but time-to-accuracy on `D1` runs the other way:

   | column | first inside 1e-3 of the faithful answer |
   |---|---|
   | `ikarus[Li]` | 17 orders, **7 ms** |
   | `ikarus[NV]` | 17 orders, **13 ms** |
   | `fork[Pol]` | 169 orders, **437 ms** |
   | `fork[Laurent]`, `ikarus[Laurent]` | never — best 2.0e-3 at 625 orders |

   So Ikarus buys back its per-solve cost about 35-fold, and the direct rule
   never arrives at all. At 625 orders the two faithful families do agree to
   1.8e-7, which is the real cross-code result on this case.
6. **One framing caveat.** The paper's grcwa harness drives this 1D grating
   through a *square 2D lattice*, so of a nominal `nG = 400` only 23 orders lie
   on `Gy = 0` and do any work; "still badly off even at 400 orders" is measured
   at 23 effective orders (the fork's native 1D path gives 0.1752 at `q = 23`,
   matching, and 0.1037 at `q = 385`). The conclusion survives — Laurent *is*
   O(1/M) and *is* ~75% high there — but the order count in that sentence is not
   the one the physics saw.

## Running

Place each grcwa version to compare as a package directory under `benchmark/`
whose name starts with `grcwa` (the directory must contain the package files
directly, i.e. `__init__.py`, `rcwa.py`, ...). For example:

```bash
# weiliang's upstream master WITH the Pol commits
git clone https://github.com/weiliangjinca/grcwa /tmp/wl
cp -r /tmp/wl/grcwa benchmark/grcwa-weiliang-013

# the fork's master, before this work
git archive origin/master grcwa | tar -x && mv grcwa benchmark/grcwa-forkmaster

# the independent cross-code column (optional)
pip install ikarus-rcwa

python benchmark/run.py        # auto-discovers them; writes results.{json,csv}
```

The current branch (`fork`) is always included. Each variant runs in its own
subprocess; the worker's `sys.path[0]` is `benchmark/`, so the vendored packages
are importable by their (distinct) directory names without clashing. The Ikarus
columns run in their own subprocesses too, selected by `SUITE=ikarus` rather than
by a directory (`FMM` then names the factorization: `laurent`, `li`, `normal`).

## Convergence study (R vs truncation order nG)

`conv_run.py` + `conv_worker.py` sweep the truncation order over the same battery
and track how fast `R(nG)` settles, comparing every factorization rule available
in every installed codebase. It writes `conv_results.{json,csv}`.

```bash
python benchmark/conv_run.py
```

The convergence worker accepts separate order grids for the two patterned
dimensionalities:

```bash
set GRCWA_NG1D_FROM_Q2D=1
set GRCWA_Q2D=1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61
set GRCWA_MAX2D=3721
```

`GRCWA_NG1D` normally contains exact total 1D order counts. The night job instead
sets `GRCWA_NG1D_FROM_Q2D=1`, which gives 1D the sorted union of every requested
`q` and `q²`; this keeps dense low-order samples while reaching the same total
order ceiling as 2D. `GRCWA_Q2D` contains per-axis counts, so its final `61`
point retains `61² = 3721` total orders. All counts must be positive and odd so
the same points are representable in Ikarus. Before starting any worker, the
runner also verifies that each real-space cell grid has at least `2q-1` samples
per active axis, which is required to represent every Fourier difference order
without aliasing. The final 1D point needs 7441 samples and uses the shared
8192-sample grid.

Every result-producing solve is timed. If its first measurement is faster than
`GRCWA_FAST_THRESHOLD_MS` (1000 ms in the night job), it receives
`GRCWA_FAST_REPEAT=3` total measurements and keeps the minimum; slower solves
run exactly once. The raw timing is retained, while a monotonic timing curve is
formed separately for each suite, factorization and dimensionality.

Set `GRCWA_CACHE=1` to checkpoint every successful point under
`GRCWA_CACHE_DIR`. Numerical results and machine-specific timing samples are
stored separately, allowing interrupted sweeps and extended order lists to
resume without recalculating existing points. `GRCWA_REFRESH_TIMING=1` refreshes
timings while retaining cached numerical results. Errors are never cached. The
runner prints each structure and order before solving it, so a long high-order
solve is visible in the live console and log.

The convergence report uses `GRCWA_CONV_TOL` (default `1e-4`) and requires at
least two consecutive measured points within tolerance. A lone crossing is
reported as provisional.

Reference per case:

* **exact 0D anchors** — the Airy result.
* **patterned cases covered by Moose** — the external Moose reference, retaining
  its provisional marker for 2D cases that have not fully settled.
* **remaining cases** — Ikarus's normal-vector value at the highest order run.
  Highest-order Laurent is used only as a fallback when Ikarus is unavailable.

## Plotting

After a run, the plotters read the exported files and write PNGs next to
themselves (all are git-ignored):

```bash
python benchmark/plot_benchmark.py   # reads results.csv      -> bench_*.png
python benchmark/plot_conv.py        # reads conv_results.json -> conv_*.png
python benchmark/plot_moose.py       # reads conv_results.json + moose_reference.json
```

`plot_benchmark.py` shows R/T/A per suite, timing, the Pol-port-faithfulness and
Laurent-agreement cross-checks, and direct-vs-faithful reflectance. `plot_conv.py`
shows the error-decay (log-log), the raw `R(nG)` settling, raw R-vs-walltime for
every case, the raw measurements plus grouped timing model, accuracy-vs-walltime
for the hardest cases, and the 0D analytic anchors. The raw-R convergence and
Moose figures also have a `_tight.png` copy whose per-case vertical range is
`R_ref +/- 0.01`; error/log figures do not. `conv_run.py` additionally writes
`conv_convergence.csv` with the first sustained `1e-4` convergence point and its
raw and estimated solve time.

In both, colour is the **codebase** and linestyle the **factorization rule**:
solid for the direct (Laurent) rule, and a distinct broken style for each
faithful one — `--` Pol (grcwa), `-.` Li and `:` NV (Ikarus).

### The plate book

One command turns a finished (or in-progress) run into a single self-contained
HTML page — every structure drawn, every convergence plate, and an interactive
explorer over the whole run:

```bash
python benchmark/make_plate_book.py    # -> benchmark/plate_book.html
```

It reads exactly three inputs, held as editable strings at the top of that file:
`conv_results.json`, `moose_reference.json` and the optional `moose_timing.json`.
There is no fallback to an older run — a missing input stops the build and names
the command that produces it.

The book's own palette differs from the plotters above on purpose: there
**colour is the factorization rule** and dash plus marker the codebase, so both
Laurent columns collapse onto one grey and the direct-versus-faithful story is
what the eye picks up first. Details in `benchmark/viz_proposals/README.md`.

Built while `run_overnight.bat` is still running, the page is a snapshot of the
last finished stage and says so in its header, along with the q reached and the
solve count — all measured at build time, never typed into the template.

## Night job

`run_overnight.bat` is the convergence-first job. It runs only fork Laurent/Pol
and Ikarus Laurent/Li/NV and skips the redundant single-order benchmark. The
first snapshot contains every odd `q` through 15 (225 total 2D orders). Each
following snapshot appends one odd order, through `q = 61` (3721 total 2D
orders), reruns only the missing solves through the persistent cache, and then
overwrites every `conv_*.png` with the enlarged dataset. Moose plots are produced
once from the final successful snapshot. The derived 1D grid reaches the same
225- and 3721-order ceilings at those two endpoints.

```bat
benchmark\run_overnight.bat quick
benchmark\run_overnight.bat
benchmark\run_overnight.bat refresh-timing
```

The quick profile exercises the same growth path with two snapshots (`1,3`, then
`1,3,5`). The normal night job has 24 snapshots and reuses all compatible cached
points. `refresh-timing` deliberately uses one full-grid pass so cached timings
are refreshed once rather than at every snapshot. The job requires all five
selected columns and every requested order; an error, skipped high-order point,
or missing timing makes it exit nonzero while retaining successful cache
checkpoints for the next run. Any other command-line argument is rejected instead
of accidentally starting the full night profile.
