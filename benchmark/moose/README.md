# Moose side of the benchmark

`moose_convergence_bench.cs` is a Moose script (Mono/C#) that rebuilds the whole
structure battery of [`benchmark/structures.py`](../structures.py) inside Moose,
sweeps the RCWA truncation order, and writes the results **and** the cost of
every run to disk. It replaces clicking the structures together in the RCWA
dialog by hand.

```
Scripts/  ->  moose_convergence_bench.cs  ->  run
                     |
                     +-- moose_conv.csv    one row per (case, order): R/T/A + timings
                     +-- moose_sweep.json  drop-in block for ../moose_reference.json
                     +-- moose_bench_<stamp>.log
```

## Running it

1. Open the script in Moose's script editor (or drop it into the `Scripts/`
   directory) and run it. Nothing else is needed — the structures, the
   incidence and the order sweep are all in the file.
2. **First do a dry run.** Set `SHOW_STRUCTURES = true` and `DRY_RUN = true` at
   the top; the script then builds all 13 structures, shows a side view of each
   via `ConvertToCaModel`, and stops without solving. Ten seconds of looking
   beats a night of solving the wrong geometry.
3. Set both back to `false` and start the real run.

Output goes to `OUTPUT_DIR`, or to `<temp>/moose_bench` when that is left empty.
The resolved path is printed in the header of the console output.

## The knobs (all at the top of the file)

| constant | meaning |
|---|---|
| `SWEEP_1D`, `SWEEP_2D` | the truncation sweep, as Moose **max orders** `m` |
| `ONLY_CASES`, `SKIP_CASES` | comma separated case names *or* group letters (`"B"`, `"C1_Si_pillars,C2_Au_holes"`) |
| `MAX_SECONDS_PER_SOLVE` | after a solve exceeds this, the higher orders **of that case** are skipped; the rest of the battery keeps going. `0` = no limit |
| `FFT_MODE` | `1` = keep the absolute unit-cell sampling at `FFT_TARGET_SAMPLES` (what grcwa does), `0` = fixed `FFT_REFINEMENT` |
| `RESUME` | skip `(case, order)` pairs already in the CSV |
| `SHOW_STRUCTURES`, `DRY_RUN` | visual check of the geometry, see above |
| `CIRCLE_ARG_IS_RADIUS` | see the circular-atom note below |

The sweep runs **cheap orders first, across all cases**, before it moves to the
next order. Aborting half way therefore leaves a complete low-order picture of
the whole battery rather than one finished case. Every finished run is appended
to the CSV and flushed immediately, and `RESUME` picks the sweep back up where
it stopped — resumed rows are folded back into the summary and the JSON, so a
sweep that took three sessions still produces one complete result file.

Budget warning: 1D at `m = 500` keeps 1001 orders; 2D at `m = 30` keeps
`61 x 61 = 3721` orders, i.e. a ~7400 x 7400 eigenproblem. That last point is
hours and many GB, not minutes. `MAX_SECONDS_PER_SOLVE` is what keeps an
overnight run from turning into a week.

## What is measured

Per run the CSV records `R`, `T`, `A`, the specular `R0`/`T0`, an energy check,
`t_setup_s` / `t_solve_s` / `t_harvest_s` / `t_total_s`, and the process working
set before/after/peak. At the end the console prints a per-case cost table with
the empirical scaling exponent `p` in `t_solve ~ nG^p`.

`R` and `T` are **summed over all propagating diffraction orders**, because that
is what grcwa's `RT_Solve` returns. Evanescent orders carry no flux, so the sum
is restricted to the propagating window `|m| <= n * period / lambda` — much
cheaper than asking for all 1001 orders. The `energy` column
(`R + T + GetAbsorption()`, which is `1` exactly iff the sums reproduce Moose's
own internal totals) is the check that nothing was missed; the console prints it
as `|1-E|` and it should sit at round-off.

## Checking a script without Moose

`moose_api_stubs.cs` holds do-nothing stubs of the Moose scripting API,
transcribed from the class signatures in `moose.qch`. Linking a script against
them type-checks it with a plain C# compiler, which catches typos, wrong
argument counts and wrong overloads in seconds instead of after a failed run:

```bash
mcs -target:library -out:moose_api_stubs.dll benchmark/moose/moose_api_stubs.cs
mcs -out:check.exe -r:moose_api_stubs.dll benchmark/moose/moose_convergence_bench.cs
```

It is not a simulator — every stub returns zero — so this proves the script
*compiles*, not that it computes anything. If a stub signature disagrees with
your Moose build, the build wins; fix the stub.

## Conventions — read before comparing numbers

**Units.** Moose works in **microns**. The doxygen help says `[m]` for
`GratingStructure`, `Layer` and `Atom`, but that is stale: `Rcwa::Calc` and
`Rcwa::CalculateFields` say `[µm]` and every shipped sample script uses µm
(period `0.8`, wavelength `0.532`). The battery is defined at `lambda = 1 µm`
with all lengths in µm, so the numbers transfer 1:1. RCWA is scale invariant
anyway.

**Order counting.** Moose takes the *maximum* order `m` per axis and keeps
`-m..+m`, so `q = 2m+1` retained orders per axis. `structures.py` is
parametrized by the retained-order *count* `q` directly:

| | Moose input | retained per axis | total (`nG`) |
|---|---|---|---|
| 1D | `m` | `q = 2m+1` | `q` |
| 2D | `(m, m)` | `q = 2m+1` | `q * q` |

The CSV writes `m_moose`, `q` **and** `nG` so this can never be ambiguous again.
[`plot_moose.py`](../plot_moose.py) currently *assumes* the keys of
`moose_reference.json` are max orders and converts them with `2m+1`; this script
produces keys under exactly that reading, which settles the open question in its
`parse_key` comment.

**Polarization.** Moose polarization angle `0` = TM (grcwa `pol="p"`), `90` = TE
(grcwa `pol="s"`). Normal incidence, no conical angle — matching
`grcwa.obj(..., theta=0, phi=0)`.

**Fill factor.** `structures.py` fills the first fraction `ff` of the cell with
the `hi` material. Moose's `Layer(thickness, material, dutyCycle, trenchMaterial)`
keeps a fraction `dutyCycle` of `material` and cuts a trench of width
`1 - dutyCycle` — verified by `unit_tests_structures.cs`, `TestBinary`:
`dutyCycle 0.8` gives an atom of width `0.2`. So `dutyCycle == ff` with the bar
made of the `hi` material. The bar sits at a different position inside the cell
than in grcwa, which cannot matter at normal incidence: shifting the whole
grating sideways does not change diffraction efficiencies.

**Circular atom (case `D2`).** The help calls the third argument of
`Atom(posX, posY, r, material)` a *radius*, but `unit_tests_structures.cs`
(`TestAtom.TestCircular`) asserts for `Atom(0.2, 0.3, 0.2, mat)`:

```
GetStartX() == 0.1     GetStopX() == 0.3     GetWidthX() == 0.2
```

i.e. `start = pos - arg/2`: the argument is the **diameter**. The shipped unit
test wins over the help, so `D2`'s radius `0.30` (in units of the period) is
passed as `0.60`. If your Moose build disagrees, flip `CIRCLE_ARG_IS_RADIUS` —
the `SHOW_STRUCTURES` dry run makes that a ten-second check.

**FFT refinement.** `Rcwa`'s `rRefinementFactorEpsFT` multiplies the *order
count*, so the absolute sampling of the unit cell would grow with `m` and the
permittivity would be resolved differently at every point of a convergence
sweep. grcwa rasterizes on a fixed `256 x 256` grid instead. `FFT_MODE = 1`
reproduces that by choosing the refinement per run so the absolute grid stays
near `FFT_TARGET_SAMPLES`; the effective value is recorded per row in the
`fft_refinement` column.

**Group A 0D cases.** `A1_slab_air` / `A1b_slab_glass` are plain film stacks
with no lateral structure. They still need *a* period, which is set to `0.5 µm`:
subwavelength, so every order but `0` is evanescent, and clear of a Rayleigh
anomaly. `R` is then independent of the truncation, which is a free
self-consistency check of the whole sweep.

## Folding the results back into the repo

```bash
python benchmark/moose/moose_csv_to_json.py <output_dir>/moose_conv.csv
```

merges the runs into `benchmark/moose_reference.json` (existing cases are
updated point by point, untouched cases are left alone) and writes the timings
into a sibling `benchmark/moose_timing.json`. `--dry-run` shows the diff without
writing; `--json <path>` targets a different reference file. After that,
`python benchmark/plot_moose.py` picks the new points up.
