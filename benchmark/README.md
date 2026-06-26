# grcwa benchmark / cross-validation

A small, self-contained suite that builds a battery of physically-motivated
gratings, runs them through several grcwa installations and factorization
modes, **cross-checks the results** and **compares timing**, and exports the
data to `results.json` / `results.csv`.

It is *not* a pytest module — run it directly.

## What it compares (the "suites")

| column                    | what it is                                                           |
|---------------------------|---------------------------------------------------------------------|
| `orig-0.1.2[Laurent]`     | weiliang's original PyPI release, **before** the Pol update         |
| `weiliang-0.1.3[Laurent]` | weiliang's upstream master **with** his Pol commits, Laurent mode   |
| `weiliang-0.1.3[Pol]`     | the same upstream, `fmm_method='pol'` — the **reference** Pol result |
| `forkmaster[Laurent]`     | the darkviruzz fork **before** this work (Laurent only)             |
| `fork[Laurent]`           | this branch, default factorization                                  |
| `fork[Pol]`               | this branch with `fmm_method='pol'` (the Pol algorithm ported here)  |

The point of including `weiliang-0.1.3[Pol]` is to validate that the Pol code in
this fork is a faithful port: on the 2D cases (identical geometry and order
count) `fork[Pol]` reproduces `weiliang-0.1.3[Pol]` bit-for-bit. The Pol/Laurent
*difference* is therefore intrinsic to weiliang's algorithm, not a port bug.

The pre-dim versions have no dimensionality inference, so for the 1D cases they
fall back to the historical *degenerate-2D* setup (a tiny second period so only
`Gy=0` survives); they cannot do the 0D case natively. This makes their 1D order
count differ from the fork's native `2M+1`, which is the only reason the 1D
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

The current branch is auto-detected. To include the external baselines, point
the harness at local copies of those packages:

```bash
# 1) original pre-Pol release
pip download grcwa==0.1.2 --no-deps --no-binary :all: -d /tmp/g
tar -C /tmp/g -xf /tmp/g/grcwa-0.1.2.tar.gz
mv /tmp/g/grcwa-0.1.2/grcwa /tmp/g/orig_grcwa        # import name = orig_grcwa
export GRCWA_ORIG_PATH=/tmp/g

# 2) weiliang's upstream master WITH the Pol commits (import name = wl_grcwa)
mkdir -p /tmp/gwl && git clone https://github.com/weiliangjinca/grcwa /tmp/gwl/src
mv /tmp/gwl/src/grcwa /tmp/gwl/wl_grcwa
export GRCWA_WEILIANG_PATH=/tmp/gwl

# 3) the fork's master, before this work
mkdir -p /tmp/gfm && git archive origin/master grcwa | tar -x -C /tmp/gfm
export GRCWA_FORKMASTER_PATH=/tmp/gfm

python benchmark/run.py
```

If an env var is not set, that suite is simply skipped (the `fork` columns are
always produced). Each variant runs in its own subprocess (isolated
`PYTHONPATH`) so multiple grcwa versions never clash on import — that is why the
external packages are imported under distinct names (`orig_grcwa`, `wl_grcwa`).
