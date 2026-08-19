# Moose side of the benchmark

Two scripts live here:

* `moose_convergence_bench.cs` — the battery and the order sweep (below);
* `moose_geometry_probe.cs` — a short diagnostic for the **2D geometry gap**,
  see [its own section](#the-2d-geometry-probe) at the end.

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
| `PARALLEL_TASKS` | how many structures to solve at once (`1` = sequential, `0` = one per core) |
| `PARALLEL_NG_LIMIT` | a run above this `nG` gets the machine to itself; `0` disables |
| `PARALLEL_SELFTEST` | prove parallel == sequential on your build, then stop |
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

## Running several structures at once

A Moose solve is single-threaded — the eigenproblem is not parallelized — so on
a 20-core box one solve leaves 19 cores idle, no matter what you set. Different
structures are completely independent, though, so the sweep runs them on a pool
of worker threads, each with its own `Rcwa` instance:

```csharp
static int PARALLEL_TASKS = 8;   // 0 = Environment.ProcessorCount
```

The stage structure is unchanged — cheap orders of every case still finish
before the expensive ones start — but a stage's runs now go out together,
biggest `nG` first so the pool does not end up waiting on a job it picked up
last. Wall time drops by roughly `min(PARALLEL_TASKS, cases in the stage)`.

**Two things to know before turning it up.**

*Memory scales with it.* Each concurrent solve holds its own matrices, and 2D at
`m = 30` is several GB on its own. `PARALLEL_NG_LIMIT` is the guard: a run whose
`nG` exceeds it takes an exclusive lock, so only one memory-hungry solve is ever
in flight while cheap ones still run alongside.

*Per-solve timings get noisier.* Concurrent solves share memory bandwidth and
cache, so `t_solve_s` measured with a full pool runs longer than the same solve
alone. For a timing run — anything feeding the scaling exponent — use
`PARALLEL_TASKS = 1`. For collecting R values, turn it up.

### Prove it is safe first

Concurrent `Rcwa` instances *should* be independent; the shipped `ParallelRcwa`
does the same thing internally. But "should" is not worth a night of compute,
and something like a shared FFT plan cache is exactly the kind of thing that
corrupts results quietly rather than crashing. So before the first long parallel
run:

```csharp
static bool PARALLEL_SELFTEST = true;
```

That solves a handful of points sequentially, then the same points on the pool
`SELFTEST_REPEATS` times (default 3), compares every value bit-for-bit, and
stops. Repeating matters: interference is timing dependent, so a single clean
pass proves very little — a deliberately race-y build under test showed
mismatches in only 1 of 3 passes. **Any** mismatch, in any pass, means
`PARALLEL_TASKS = 1`.

## Matching the Python sweep exactly

`benchmark/run_overnight.bat` drives the Python side with

```bat
set "GRCWA_NG1D_FROM_Q2D=1"
set "FULL_Q_LIST=1,3,5,...,61"
```

where `q` is the per-axis retained order **count**. `conv_worker.py` expands
that into `2D: (q,q)` for every `q`, and `1D: sorted(set(q) | set(q*q))` — the
union of the list with its own squares. Moose takes the max order `m` with
`q = 2m+1`, so:

| | Python | Moose `m` |
|---|---|---|
| 2D | `q = 1 … 61` | `(q-1)/2` → `0, 1, 2, … 30` |
| 1D | `nG = {q} ∪ {q²}` | `(nG-1)/2` → `0 … 30, 40, 60, 84, 112, 144, 180, 220, 264, 312, 364, 420, 480, 544, 612, 684, 760, 840, 924, 1012, 1104, 1200, 1300, 1404, 1512, 1624, 1740, 1860` |

Both lists are in the script as commented-out alternatives, ready to swap in.
Both top out at `nG = 3721`. Watch what that means for 1D: `m = 1860` keeps 3721
orders in *one* axis, so the eigenproblem is about `7400 x 7400` — the same size
as 2D at `(30,30)`, hours and many GB per point. The 1D list looks long because
it is a union: its first 31 entries are cheap, its last 27 are not.

## What is measured

Per run the CSV records `R`, `T`, `A`, the specular `R0`/`T0`, an energy check,
`t_setup_s` / `t_solve_s` / `t_harvest_s` / `t_total_s`, and the process working
set before/after/peak. At the end the console prints a per-case cost table with
the empirical scaling exponent `p` in `t_solve ~ nG^p`.

`R` and `T` are **summed over all propagating diffraction orders**, because that
is what grcwa's `RT_Solve` returns. Evanescent orders carry no flux, so the sum
is restricted to the propagating window `|m| <= n * period / lambda` — much
cheaper than asking for all 1001 orders.

The `energy` column (`R + T + GetAbsorption()`, which is `1` exactly iff the
sums reproduce Moose's own internal totals) is what polices all of this. The
console prints it as `|1-E|`; it should sit at round-off, and a row where it
does not is printed in red, marked `energy` instead of `ok`, and refused by the
merge script. Both bugs above were caught by this column rather than by reading
the numbers — treat it as the primary result, not a diagnostic.

The `harvest` column records which polarization reading was kept, and
`R_te_tm` / `R_both` / `R_in` (with their own energy columns) keep all three
visible so the choice stays auditable.

**Validation.** Every 1D case reproduces the values that were originally entered
into Moose by hand, to all six digits those were recorded with — `B1_Si_grating_TE`
at `m = 5/10/20` gives `0.397683 / 0.380807 / 0.379884`, and so on across
groups A, B and D1.

On 2D the battery is its own hardest test, and Moose lands next to the
well-factorized codes rather than next to Laurent's rule — which is what a
mature RCWA implementation should do:

| case | Moose | Pol | Li | NV | Laurent |
|---|---|---|---|---|---|
| `C1_Si_pillars` | 0.39871 | 0.38896 | 0.38996 | 0.39065 | 0.45327 |
| `C2_Au_holes` | 0.67136 | 0.69010 | 0.67233 | 0.67995 | 0.80877 |
| `D2_ikarus_cylinder_TE` | 0.90485 | 0.94258 | 0.91802 | 0.94214 | 0.95585 |

Do not read those last two rows as converged: `C1b` and `D2` are still rising at
`(10,10)` (`nG = 441`), the highest order that run reached. Comparing Moose
against grcwa with Laurent's rule at low order is not a fair yardstick either —
that column is the slowest to converge on exactly these cases.

The small offsets against the manual entries (`C1_Si_pillars`: `0.39871`
against `0.39748`) are the FFT-refinement difference described below: the manual
runs used Moose's default refinement, this script holds the absolute unit-cell
grid constant across the sweep.

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

**Circular atom (case `D2`).** The third argument of
`Atom(posX, posY, r, material)` is the **radius**, relative to the period. The
help says so; the shipped unit test misleads. `unit_tests_structures.cs`
(`TestAtom.TestCircular`) asserts for `Atom(0.2, 0.3, 0.2, mat)`:

```
GetStartX() == 0.1     GetStopX() == 0.3     GetWidthX() == 0.2
```

i.e. `start = pos - arg/2`, which reads like a diameter — but those getters
describe a bounding box that does **not** agree with what the solver
rasterizes. Trusting them cost a run: passing `0.60` for `D2`'s radius `0.30`
built a circle of radius `0.60 * period`, which overfills the unit cell (95 %
silicon, only the corners left as air) and collapsed `R` from `0.95` to
`0.027`. Cross-checked against grcwa, an overfilled `r = 0.60` circle gives
`R ~ 0.024` converging toward Moose's `0.0267`, while every other reading
(diameter `0.6`, square, uniform film) is off by orders of magnitude. So the
radius goes in unscaled. `CIRCLE_ARG_IS_RADIUS = false` restores the old, wrong
reading.

Note that this is invisible in the `SHOW_STRUCTURES` dry run:
`ConvertToCaModel` draws a **side view**, and a side view cannot tell a pillar
of the right diameter from one of the wrong diameter once it spans most of the
cell. Judge 2D geometry by the numbers, not the picture.

**Efficiencies come back in percent.** `GetEfficiencyForGivenOrder` and
`GetAbsorption` return percent, not fractions — undocumented, and
`R + T + A = 100` is how it announces itself. Everything harvested is scaled by
`SCALE = 0.01`, so the CSV and `moose_reference.json` hold fractions. If a build
ever returns fractions, the energy balance says so immediately: it lands on `1`
with the right `SCALE` and on `100` with the wrong one.

**Output polarization.** The default `rOutputPolarization = "in"` returns only
the **co-polarized** output. On a 2D lattice the off-axis orders (both indices
non-zero) convert polarization, so their cross-polarized half is silently
dropped and up to a third of the flux disappears. It cannot be seen on 1D at
normal incidence, nor on any 2D case where only order `(0,0)` propagates — in
this battery `C1b_Si_pillars_diffract` is the single case that exposes it. Every
sum is therefore formed three ways (`"TE"+"TM"`, `"both"`, `"in"`) and the
reading that conserves energy is kept; all three are written to the CSV, and a
row where none of them conserves energy gets status `energy` instead of `ok`, so
it can never be merged as if it were sound.

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
into a sibling `benchmark/moose_timing.json`. After that,
`python benchmark/plot_moose.py` picks the new points up.

Every row is checked before it is allowed in — `status` must be `ok`, `R` must
lie in `[0, 1]`, and `R + T + A` must equal `1` within `--energy-tol`. Rejected
rows are listed with the reason rather than dropped quietly.

| flag | what it does |
|---|---|
| `--dry-run` | show the diff, write nothing |
| `--json <path>` | target a different reference file |
| `--energy-tol` | how far `R + T + A` may sit from `1` (default `1e-6`) |
| `--skip-case NAME` | drop a case entirely; repeatable. For a run whose *geometry* is known wrong, which no energy check can catch |
| `--legacy-percent` | read a CSV from before the percent fix: divide by 100 and check the balance against 100 |
| `--create` | start a fresh reference file instead of merging into an existing one |

**Starting over from nothing.** If `moose_reference.json` does not exist — a
clean checkout, or you simply want to recompute everything — pass `--create`:

```bash
python benchmark/moose/moose_csv_to_json.py <output_dir>/moose_conv.csv --create
```

That writes the whole file, wrapper and all, from the CSV alone. It is required
rather than automatic so that a mistyped `--json` path fails loudly instead of
quietly creating a stray file.

The `moose_sweep.json` the Moose script itself writes (and echoes at the end of
its console output) is the same data and is already filtered by the energy
check, but it is only the inner `cases` block — no wrapper, no `note`, and no
timing file. `--create` is the tidier route.

## The 2D geometry probe

`moose_geometry_probe.cs` exists because of one asymmetry in the results: on
every **1D** case Moose and the well-factorized python columns agree to five or
six digits, and on every **2D** case they do not (`C1_Si_pillars` 0.39817 vs
0.3898, `C1b_Si_pillars_diffract` 0.15477 vs 0.1454). Three python columns using
three different factorization rules agree with *each other* and miss Moose
together, and different rules cannot converge to different limits on one
structure — so what survives at high order is a difference in the structure, not
in the physics.

The python side has since been measured, and it is the rasterization: the shared
`256 × 256` mask of `structures.py` renders the square pillars at 153/256 instead
of 153.6 (`C1`, −0.39 %), 103/256 instead of 102.4 (`C1b`, +0.59 %) and 127/256
instead of 128 (`C2`, −0.78 %), while the 1D masks are exact. One pixel of that
grid is worth ~0.012 in R on `C1` — more than the whole gap. See
[`benchmark/geometry_fidelity.py`](../geometry_fidelity.py) and the "The mask is
not the structure" section of [`benchmark/README.md`](../README.md).

This script asks the same question from Moose's side. Three probes, all
configurable at the top of the file:

| probe | what it does |
|---|---|
| **A — geometry dump** | renders each patterned layer with `GetEpsilonDistributionsAsCaModel` at 64/100/256/300/512 and reports fill fraction, pillar width in pixels and the number of distinct permittivity values. Binary rasterization shows up as `levels = 2`; anything more means Moose area-weights its boundary pixels, which is what would let it hit the nominal geometry on a coarse grid. Seconds, no solving. |
| **B — width sweep** | solves each 2D case with the atom set to the nominal size, to **the size the python mask actually rasterizes**, and to ±1 %/±2 %. If Moose at the python width reproduces the python numbers (printed in the console header for comparison), the 2D disagreement is the rasterization and nothing else. |
| **C — refinement sweep** | `rRefinementFactorEpsFT` = 2…40 at fixed geometry, i.e. how much of Moose's own value depends on its eps sampling — and therefore whether the ±0.001 wobble along the 2D sweep comes from `FFT_MODE = 1` picking a different refinement at every order. |

Probe B uses the same refinement rule as the main sweep (`ceil(256/q)`), so its
`nominal` row is directly comparable to `moose_reference.json`.

`ORDERS` includes `m = 0` deliberately, and it costs nothing. With a single
retained order RCWA can only be a transfer-matrix calculation on the
cell-averaged permittivity, which makes R at `m = 0` a direct read-out of the
fill fraction. **The `(0,0)` points currently in `moose_reference.json` fail that
test**: `C1_Si_pillars` reports 0.363252, which is R of a *solid silicon film* to
seven digits, where the averaged medium gives 0.151138 (grcwa at `nG = 1`
reproduces that exactly); `C2_Au_holes` reports 0.040000, R of a *bare air/glass
interface*, where a gold-dominated average gives 0.973. Until the probe explains
what Moose does with zero orders, those two points should not be plotted or
merged.
