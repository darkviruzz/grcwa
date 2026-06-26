# grcwa benchmark / cross-validation

A small, self-contained suite that builds a battery of physically-motivated
gratings, runs them through several grcwa installations and factorization
modes, **cross-checks the results** and **compares timing**, and exports the
data to `results.json` / `results.csv`.

It is *not* a pytest module — run it directly.

## What it compares (the four "suites")

| column            | what it is                                                        |
|-------------------|-------------------------------------------------------------------|
| `orig-0.1.2`      | weiliang's original PyPI release, **before** the Pol update       |
| `forkmaster`      | the darkviruzz fork **before** this work (Laurent only)           |
| `fork[Laurent]`   | this branch, default factorization                                |
| `fork[Pol]`       | this branch with `fmm_method='pol'` (the upstream Pol algorithm)   |

The old versions have no dimensionality inference, so for the 1D cases they
fall back to the historical *degenerate-2D* setup (a tiny second period so only
`Gy=0` survives); they cannot do the 0D case natively.

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

The current branch is auto-detected. To include the two external baselines,
point the harness at local copies of those packages:

```bash
# 1) original pre-Pol release
pip download grcwa==0.1.2 --no-deps --no-binary :all: -d /tmp/g
tar -C /tmp/g -xf /tmp/g/grcwa-0.1.2.tar.gz
mv /tmp/g/grcwa-0.1.2/grcwa /tmp/g/orig_grcwa        # import name = orig_grcwa
export GRCWA_ORIG_PATH=/tmp/g

# 2) the fork's master, before this work
mkdir -p /tmp/gfm && git archive origin/master grcwa | tar -x -C /tmp/gfm
export GRCWA_FORKMASTER_PATH=/tmp/gfm

python benchmark/run.py
```

If the env vars are not set, only the `fork` columns are produced. Each variant
runs in its own subprocess (isolated `PYTHONPATH`) so multiple grcwa versions
never clash on import.
