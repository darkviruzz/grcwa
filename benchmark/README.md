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
Ikarus (as a topology plus one material per index). No backend draws its own
geometry, so a cross-code disagreement can never be a pixel-grid artifact.

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
   `D2_ikarus_cylinder_TE` the fork's Pol lands at 0.9286, next to Ikarus's
   normal-vector 0.9265 and far from its Li 0.8732: the fork's tangent field is
   built from the rendered `eps` grid, so it behaves like a normal-vector method
   rather than like a separable one.
5. **Ikarus still wins the convergence *rate*, decisively.** Its normal-vector
   default is settled by `q ≈ 15`; the fork's Pol oscillates around the limit
   until `q ≈ 100`. Per solve at matched `nG` the fork is ~2-5× faster, which the
   order count gives straight back on the high-contrast TM cases.
6. **One framing caveat.** The paper's grcwa harness drives this 1D grating
   through a *square 2D lattice*, so of a nominal `nG = 400` only 23 orders lie
   on `Gy = 0` and do any work; "still badly off even at 400 orders" is measured
   at 23 effective orders (the fork's native 1D path gives 0.1752 at `q = 23`,
   matching, and 0.1035 at `q = 385`). The conclusion survives — Laurent *is*
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

Reference per case:

* **groups A/A2** — the exact Airy result, and the asymptotic effective-medium
  film.
* **group D** — Ikarus's faithful normal-vector column at the highest order run.
  Laurent is still percent-level wrong at *every* order this sweep reaches on
  these cases, so its high-order value would be a useless reference; the
  normal-vector one has settled by `q ≈ 15` and is the whitepaper's own published
  number, which keeps the comparison falsifiable.
* **groups B/C** — the highest-`nG` Laurent result (the rule that provably
  converges in the limit, if slowly). To use your own external RCWA instead, edit
  the `ref` field of a case in `conv_results.json` before plotting; the two
  external references already committed here are `moose_reference.json` and
  `ikarus_reference.json`.

## Plotting

After a run, the two plotters read the exported files and write PNGs next to
themselves (both are git-ignored):

```bash
python benchmark/plot_benchmark.py   # reads results.csv      -> bench_*.png
python benchmark/plot_conv.py        # reads conv_results.json -> conv_*.png
python benchmark/plot_moose.py       # reads conv_results.json + moose_reference.json
```

`plot_benchmark.py` shows R/T/A per suite, timing, the Pol-port-faithfulness and
Laurent-agreement cross-checks, and direct-vs-faithful reflectance. `plot_conv.py`
shows the error-decay (log-log), the raw `R(nG)` settling, accuracy-vs-walltime
for the hardest cases, and the 0D analytic anchors.

In both, colour is the **codebase** and linestyle the **factorization rule**:
solid for the direct (Laurent) rule, and a distinct broken style for each
faithful one — `--` Pol (grcwa), `-.` Li and `:` NV (Ikarus).
