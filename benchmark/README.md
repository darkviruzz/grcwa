# grcwa benchmark / cross-validation

A small, self-contained suite that builds a battery of physically-motivated
gratings, runs them through several grcwa installations and factorization
modes, **cross-checks the results** and **compares timing**, and exports the
data to `results.json` / `results.csv`.

It is *not* a pytest module — run it directly.

## What it compares (the "suites")

Suites are **auto-discovered**: any `benchmark/grcwa*` directory that is an
importable package becomes a column, plus the current branch (`fork`) at the repo
root. Each suite is run with Laurent's rule and, if its `obj` implements
`fmm_method='pol'`, also with Pol — so `[Pol]` columns appear only for the
versions that actually support it. The column label is the directory name with
the leading `grcwa` stripped (`grcwa-weiliang-013` → `weiliang-013`).

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

`lambda = 1 micron` (`freq = 1`), lengths in microns. Cases span 1D/2D/0D, real
and complex permittivity, sub- vs supra-wavelength periods, TE/TM:

- `2D_Si_hole_subwave`, `2D_Si_hole_diffract` — Si hole array, lossless.
- `2D_metal_hole_absorb` — lossy metal (complex eps), absorption A = 1-R-T > 0.
- `1D_Si_TE_subwave`, `1D_Si_TM_diffract` — lamellar Si grating, lossless.
- `1D_metal_TM_absorb` — metal lamellar grating, TM, absorption.
- `0D_slab` — planar slab (TMM); cross-checked against the analytic Airy result.

Each case prints R, T, A=1-R-T, R+T, the actual order count `nG`, and the
minimum wall time over a few repeats. Physical expectations: lossless cases
satisfy R+T=1; absorbers have A>0; more diffraction orders open as the period
grows past the wavelength.

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

python benchmark/run.py        # auto-discovers them; writes results.{json,csv}
```

The current branch (`fork`) is always included. Each variant runs in its own
subprocess; the worker's `sys.path[0]` is `benchmark/`, so the vendored packages
are importable by their (distinct) directory names without clashing.

## Convergence study (R vs truncation order nG)

`conv_run.py` + `conv_worker.py` sweep the truncation order over a battery of
physically-motivated structures (planar slab and form-birefringent film with
*analytic* references; 1D Si/metal gratings TE/TM; 2D rectangular Si/metal
pillars) and track how fast `R(nG)` settles, comparing the Laurent and Pol
factorizations in two independent codebases. Materials are given as `(n, k)` at
`lambda = 1 um`. It uses `GRCWA_WEILIANG_PATH` (same setup as above) and writes
`conv_results.{json,csv}`.

```bash
export GRCWA_WEILIANG_PATH=/tmp/gwl       # optional second codebase
python benchmark/conv_run.py
```

For the gratings the reference is the highest-nG Laurent result (the rule that
provably converges in the limit); to use your own external RCWA instead, edit
the `ref` field of a case in `conv_results.json` before plotting.

## Plotting

After a run, the two plotters read the exported files and write PNGs next to
themselves (both are git-ignored):

```bash
python benchmark/plot_benchmark.py   # reads results.csv      -> bench_*.png
python benchmark/plot_conv.py        # reads conv_results.json -> conv_*.png
```

`plot_benchmark.py` shows R/T/A per suite, timing, the Pol-port-faithfulness and
Laurent-agreement cross-checks, and Laurent-vs-Pol reflectance. `plot_conv.py`
shows the error-decay (log-log), the raw `R(nG)` settling, accuracy-vs-walltime
for the hardest cases, and the 0D analytic anchors.
