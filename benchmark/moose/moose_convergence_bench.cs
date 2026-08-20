// script:  moose_convergence_bench.cs
// purpose: rebuild the whole grcwa benchmark battery (benchmark/structures.py)
//          inside Moose, sweep the RCWA truncation order, and log both the
//          results (R/T/A) and the cost (wall time, memory) of every single run.
//
//          Written so that nothing has to be clicked together in the RCWA
//          dialog any more: press "run" once and walk away.
//
// ---------------------------------------------------------------------------
// WHAT IT DOES
// ---------------------------------------------------------------------------
//   * builds the 13 structures of benchmark/structures.py (groups A/B/C/D),
//   * solves each of them at every truncation order of the sweep below,
//   * sums the efficiencies over all *propagating* diffraction orders to get
//     the total R and T (that is what grcwa's RT_Solve returns, so the numbers
//     are directly comparable), as fractions and over both output
//     polarizations -- see the two notes on percent and polarization below,
//   * times setup / solve / harvest separately and records the process memory,
//   * appends every finished run to a CSV *immediately* (so an aborted run is
//     not a lost run), and can resume a previously aborted sweep,
//   * writes a ready-to-paste JSON block in the format of
//     benchmark/moose_reference.json,
//   * prints a per-case cost summary with the fitted scaling exponent
//     t ~ nG^p at the end.
//
// ---------------------------------------------------------------------------
// CONVENTIONS -- read this before comparing numbers to the Python side
// ---------------------------------------------------------------------------
// UNITS.  Moose works in microns.  The doxygen help claims [m] in a few places
//   (GratingStructure, Layer, Atom); that is stale -- Rcwa::Calc and
//   Rcwa::CalculateFields say [um] and every shipped sample script uses um
//   (period 0.8, wavelength 0.532, ...).  The battery is defined at
//   lambda = 1 um with all lengths in um, so the numbers go in 1:1.
//   (RCWA is scale invariant anyway.)
//
// ORDER COUNTING.  Moose takes the *maximum* order m per axis and keeps
//   -m..+m, i.e. q = 2m+1 retained orders per axis.  grcwa/structures.py is
//   parametrized by the retained-order *count* q.  So:
//
//       1D:   m  ->  q = 2m+1        retained orders  (nG = q)
//       2D:  (m,m) -> q = 2m+1 per axis                (nG = q*q)
//
//   The CSV writes m_moose, q AND nG so this can never be ambiguous again.
//   (benchmark/plot_moose.py currently *assumes* the keys of
//   moose_reference.json are max orders m and converts them with 2m+1 -- this
//   script confirms that reading by construction.)
//
// POLARIZATION.  Moose polarization angle: 0 = TM (= grcwa pol "p"),
//   90 = TE (= grcwa pol "s").  Normal incidence, no conical angle, matching
//   grcwa's obj(..., theta=0, phi=0).
//
// FILL FACTOR.  structures.py fills the first fraction ff of the unit cell
//   with the "hi" material.  Moose's Layer(thickness, material, dutyCycle,
//   trenchMaterial) leaves a fraction dutyCycle of `material` and cuts a trench
//   of width (1-dutyCycle) -- verified by unit_tests_structures.cs, TestBinary:
//   dutyCycle 0.8 -> atom width 0.2.  So dutyCycle == ff, with the bar made of
//   the "hi" material.  The bar sits at a different position inside the cell
//   than in grcwa, which is irrelevant at normal incidence (a lateral shift of
//   the whole grating cannot change the diffraction efficiencies).
//
// CIRCULAR ATOM (case D2) -- HISTORICAL, no longer applicable.  D2 used to be
//   built from an Atom(posX, posY, r, mat), whose third argument turned out to
//   be the RADIUS relative to the period, not the diameter the shipped unit
//   test (TestAtom.TestCircular) misleadingly suggested -- passing the wrong
//   one overfilled the unit cell and collapsed R from 0.95 to 0.027, caught by
//   cross-checking against grcwa.  D2 (like every other 2D case) is now built
//   from an explicit CaModel grid instead (see CAMODEL GEOMETRY below), which
//   sidesteps Atom's constructor semantics entirely -- this note is kept only
//   because the debugging story is worth not repeating with the next Atom
//   argument that turns out to mean something other than its name suggests.
//
// EFFICIENCIES ARE IN PERCENT.  GetEfficiencyForGivenOrder and GetAbsorption
//   return percent, not fractions -- undocumented, and R + T + A = 100 is how
//   it shows up.  Everything harvested is scaled by SCALE = 0.01 so the CSV
//   holds fractions like the rest of the benchmark.
//
// OUTPUT POLARIZATION.  The default rOutputPolarization = "in" returns only
//   the co-polarized output.  On a 2D lattice the off-axis orders (both
//   indices non-zero) convert polarization, so their cross-polarized half is
//   silently dropped and up to a third of the flux goes missing.  It is
//   invisible on 1D at normal incidence and on any 2D case where only order
//   (0,0) propagates -- in this battery C1b_Si_pillars_diffract is the single
//   case that exposes it.  Every sum is therefore formed three ways ("TE"+"TM",
//   "both", "in") and the one that conserves energy is kept; all three land in
//   the CSV, and a row where none of them conserves energy is marked "energy"
//   rather than "ok" so it cannot be merged as if it were sound.
//
// CAMODEL GEOMETRY.  Every 2D layer (C1, C1b, C2, D2) is now built from an
//   explicit CaModel -- Layer(double thickness, CaModel epsilonDistribution)
//   -- instead of an Atom.  The mask is rendered at NX_2D = 260 (see below),
//   cell-centred, "<=" -- bit for bit the same rule benchmark/structures.py's
//   layer_mask() uses by default and moose_raster_probe.cs's CellInside
//   verified against it: KEEP THE THREE IN STEP if any of them changes.
//
//   This closes what was the WHOLE 2D disagreement with grcwa/Ikarus:
//   moose_raster_probe.cs found Moose's own Atom rasterizer is one cell too
//   wide per axis on EVERY grid, aligned or not (a fencepost bug, not a
//   rounding one -- see RASTERIZATION.md Sec.9); CaModel sidesteps it
//   entirely by handing Moose the exact pixels instead of asking it to
//   rasterize the parameters itself.  On the rect cases (C1/C1b/C2) that mask
//   is EXACT -- NX_2D = 260 is a multiple of 20, which is exactly divisible
//   enough for C1 (w=0.6), C1b (w=0.4) and C2 (w=0.5) all at once, so channel
//   1 (shape error) is zero for those three. D2 (a circle) has no exact grid
//   at any resolution -- pi is irrational -- so it keeps a small residual
//   from that alone, same as the python side.
//
// FFT REFINEMENT -- what is left after the geometry fix, and it is NOT what
//   this script used to think it was.  rRefinementFactorEpsFT still resamples
//   an EXPLICIT CaModel grid (P1 confirmed this directly: 8.5e-5 to 4.5e-4
//   between refinement 40 and 100 on an exact mask), through a mechanism that
//   is NOT simple shape/pixel-count resampling -- the old "grcwa uses a fixed
//   256x256 grid, so target that" reasoning (FFT_MODE = 1 below, now removed)
//   was wrong on its own terms even before CaModel existed: Moose clamps
//   refinement to [30, 100], the old target-256 formula always clamped to 30
//   regardless of q, and a follow-up self-test (moose_camodel_selftest.cs)
//   found the SAME (case, order, refinement) reproducibly gives a different
//   R on two different machines/builds at refinement 40 but identical R at
//   refinement 100 -- so 100, the maximum Moose accepts, is both the least
//   refinement-dependent point measured and the only one available higher.
//
//   FFT_REFINEMENT_2D below is fixed at 100 for every 2D point, full stop --
//   there is nothing left to "target" once the geometry is exact.  A real,
//   currently unexplained residual remains even there (~4.5e-4 measured on
//   C1 at m = 10 against ikarus[li], cross-machine-verified) -- see
//   RASTERIZATION.md Sec.9 for the full account and what is still open.  1D
//   cases are unaffected by any of this: Layer(thickness, hi, dutyCycle, lo)
//   takes the fill fraction directly, no discretized grid and no refinement
//   dependence has ever been observed there (5-6 digit agreement with
//   grcwa/Ikarus since before this investigation started).
//
// RESUME WARNING.  Every 2D row this script wrote before this CaModel switch
//   used the Atom path and its +1-cell shape error -- those rows are WRONG,
//   not just old.  RESUME's dedup key is (case, m) alone, so resuming an old
//   CSV will SILENTLY KEEP the stale Atom-based 2D rows instead of
//   recomputing them.  Start a fresh OUTPUT_DIR (or delete the old CSV) the
//   first time you run this version -- do not resume across the switch.
// ---------------------------------------------------------------------------

using System;
using System.IO;
using System.Threading;
using System.Collections.Generic;
using System.Globalization;


// ===========================================================================
//  one entry of the battery -- mirrors a dict of structures.py STRUCTURES
// ===========================================================================
public class BenchCase
{
    public string Name;
    public string Group;
    public int    Dim;          // 0 = plain film stack, 1 = lamellar, 2 = 2D
    public string Pol;          // "TE" or "TM"
    public string Shape;        // "rect" or "circle" (Dim == 2 only)
    public string Desc;

    public double Period;       // um  (both axes for Dim == 2)
    public double Depth;        // um  thickness of the patterned / film layer
    public double Ff;           // Dim == 1: fill factor of the "hi" material
    public double Ax, Ay;       // Dim == 2 rect: pillar size in um
    public double Radius;       // Dim == 2 circle: radius / period

    // materials as (n, k); eps = (n + i k)^2
    public double HiN,  HiK;    // bar / pillar / film
    public double LoN,  LoK;    // trench / background
    public double SubN, SubK;   // substrate half space (superstrate = air)

    public BenchCase(string name, string group, int dim, string pol, string desc)
    {
        Name = name; Group = group; Dim = dim; Pol = pol; Desc = desc;
        Shape = "rect";
        LoN = 1.0; LoK = 0.0;
        SubN = 1.0; SubK = 0.0;
        Period = 1.0; Depth = 0.0; Ff = 0.5; Ax = 0.0; Ay = 0.0; Radius = 0.0;
    }

    public Materials Hi()  { return new Materials(HiN,  HiK ); }
    public Materials Lo()  { return new Materials(LoN,  LoK ); }
    public Materials Sub() { return new Materials(SubN, SubK); }
}


// ===========================================================================
//  result of a single (case, order) run
// ===========================================================================
public class BenchResult
{
    public string Name;
    public int    M;            // Moose max order
    public int    Q;            // retained orders per axis = 2m+1
    public double NG;           // total retained orders (q or q*q)
    public double R, T, A, R0, T0, Energy;
    // the three polarization readings of the order sums, see SumEfficiency
    public double RTeTm, TTeTm, ETeTm;
    public double RBoth, TBoth, EBoth;
    public double RIn,   TIn,   EIn;
    public string Harvest = "";
    public double TSetup, TSolve, TReap, TTotal;
    public double MemBefore, MemAfter, MemPeak;
    public int    FftRefinement;
    public string Status;
    public string Note;
}


public class MooseScript
{
    // =======================================================================
    //  CONFIGURATION -- everything you would normally want to touch
    // =======================================================================

    // Where CSV / JSON / log end up.  Empty -> <temp>/moose_bench.
    static string OUTPUT_DIR   = "C:\\Users\\mwalther\\PycharmProjects\\grcwa\\benchmark\\moose";

    // Truncation sweep.  These are Moose MAX ORDERS m (retained: 2m+1 per
    // axis).  The defaults are exactly the keys already present in
    // benchmark/moose_reference.json, so new runs line up with the old ones.
    //static readonly int[] SWEEP_1D = { 1, 3, 5, 10, 20, 50, 100, 200, 500, 1000 };
    //static readonly int[] SWEEP_2D = { 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 35, 40, 45, 50 };

    // To land on exactly the points the Python sweep uses, swap the two lines
    // above for these.  benchmark/run_overnight.bat sets
    //     FULL_Q_LIST = 1,3,5,...,61        (per-axis retained orders q)
    //     GRCWA_NG1D_FROM_Q2D = 1
    // and benchmark/conv_worker.py turns that into
    //     2D:  (q,q) for every q          -> nG = q*q
    //     1D:  sorted(set(q) | set(q*q))  -> nG = that union
    // Moose takes the max order m with q = 2m+1, so m = (q-1)/2 for 2D and
    // m = (nG-1)/2 for 1D:
    //
    static readonly int[] SWEEP_2D = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17, 20, 25, 30, 35 };
    static readonly int[] SWEEP_1D = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
        29, 30, 40, 60, 84, 112, 144, 180, 220, 264, 312, 364, 420, 480,
        544, 612, 684, 760, 840, 924, 1012, 1104, 1200, 1300, 1404, 1512,
        1624, 1740, 1860 };
    
    // Both top out at nG = 3721.  Note what that means for 1D: m = 1860 keeps
    // 3721 orders in ONE axis, so the eigenproblem is about 7400 x 7400 -- the
    // same size as 2D at (30,30), and hours plus many GB per point.  The 1D
    // list is long because it is the union of q and q*q: its first 31 entries
    // are cheap, the last 27 are not.

    // Case filter.  Empty ONLY_CASES = run everything.  Comma separated,
    // matched against the case name, e.g. "B1_Si_grating_TM,C2_Au_holes".
    // A group letter also works: "A", "B", "C", "D".
    static string ONLY_CASES   = "";
    static string SKIP_CASES   = "";

    // Cost guard.  After a solve of a case exceeds this many seconds, the
    // larger orders of THAT case are skipped (the rest of the battery keeps
    // running).  0 = no limit.  2D at m = 30 means 61*61 = 3721 orders, i.e. a
    // ~7400 x 7400 eigenproblem -- that one is hours, not minutes.
    static double MAX_SECONDS_PER_SOLVE = 600.0;

    // FFT refinement for 2D points, see "FFT REFINEMENT" in the header.  Fixed
    // at Moose's own maximum -- there is nothing left to "target" once the
    // CaModel geometry is exact; a real, currently unexplained residual
    // remains at 100 regardless (RASTERIZATION.md Sec.9).  1D/0D points do not
    // use this at all (see RefinementFor).
    const  int    FFT_REFINEMENT_2D   = 100;
    // Moose's real range, measured in the RCWA dialog: it accepts nothing below
    // 30 or above 100.  These used to say 2 and 200, which let this script hand
    // over values Moose silently replaced -- the CSV then recorded a refinement
    // that was never used.  Clamping here instead keeps the record honest.
    const  int    FFT_REFINEMENT_MIN  = 30;
    const  int    FFT_REFINEMENT_MAX  = 100;

    // The 2D CaModel grid, see "CAMODEL GEOMETRY" in the header.  Must match
    // benchmark/structures.py's NX_2D exactly -- that is the whole point of
    // building an explicit grid instead of an Atom.
    const  int    NX_2D = 260;

    // Show a side view of every structure before solving (visual check that
    // the geometry really is what you meant).  Costs a few clicks, saves hours.
    static bool   SHOW_STRUCTURES = false;
    // ... and stop right after showing them, without solving anything.
    static bool   DRY_RUN         = false;

    // Skip (case, order) pairs already present in the CSV, so an aborted sweep
    // can simply be started again.
    static bool   RESUME          = true;

    // How many structures to solve at the same time.  A single Moose solve is
    // single-threaded -- the eigenproblem is not parallelized -- so on a 20
    // core box one solve leaves 19 cores idle.  Different structures are
    // completely independent, though, so the sweep runs them on a pool of
    // worker threads, each with its own Rcwa instance.
    //   1 = the old sequential behaviour: one solve at a time, exactly timed.
    //   N = N solves at once.  Wall time drops by roughly min(N, cases per
    //       stage); per-solve times get noisier because the cores share memory
    //       bandwidth, so a timing run wants 1.
    //   0 = use Environment.ProcessorCount.
    // MEMORY SCALES WITH THIS.  Each concurrent solve holds its own matrices;
    // 2D at m = 30 is several GB on its own, so N of those at once will not
    // fit. PARALLEL_NG_LIMIT below is the guard for that.
    static int    PARALLEL_TASKS  = 10;

    // A run whose nG exceeds this is given the whole machine: only one such
    // run at a time, though cheap runs may still go alongside it.  0 disables
    // the guard.  Set it to whatever nG your RAM tolerates in duplicate.
    static long   PARALLEL_NG_LIMIT = 0;

    // Run a handful of points sequentially AND on the pool, compare them
    // bit-for-bit, then stop.  Worth one minute before trusting a long
    // parallel run: it is what proves that concurrent Rcwa instances do not
    // interfere on YOUR Moose build.  Any mismatch means PARALLEL_TASKS = 1.
    static bool   PARALLEL_SELFTEST = false;
    // Which entry of the sweep the self test uses (index, not order count).
    static int    SELFTEST_STAGE  = 2;
    // How many times the parallel pass is repeated.  Interference is timing
    // dependent, so one clean pass proves very little; every repeat must match
    // the sequential reference.
    static int    SELFTEST_REPEATS = 3;

    // Incidence.  The battery is a normal incidence battery.
    const  double WAVELENGTH   = 1.0;    // um
    const  double AOI          = 0.0;    // deg
    const  double CONICAL      = 0.0;    // deg

    // Cache handed to Rcwa (bytes).  0 is fine, the geometry changes every run.
    const  long   RCWA_CACHE   = 0;

    // Moose reports efficiencies and GetAbsorption() in PERCENT.  The rest of
    // the benchmark (and benchmark/moose_reference.json) works in fractions,
    // so everything harvested is multiplied by this.  Set it to 1.0 if a build
    // ever returns fractions -- the energy balance says which one you have:
    // it lands on 1 with the right SCALE and on 100 with the wrong one.
    const  double SCALE = 0.01;

    // How far R + T + A may sit from 1 before a row is rejected.  Round-off on
    // these sums is ~1e-9; anything above 1e-6 means orders or a polarization
    // component went missing, which is a broken row, not a noisy one.
    const  double ENERGY_TOL = 1.0e-6;

    static readonly CultureInfo INV = CultureInfo.InvariantCulture;

    // Superstrate is air for every case in the battery.
    const  double SUPER_N = 1.0;


    // =======================================================================
    //  the battery -- 1:1 with benchmark/structures.py
    // =======================================================================
    // materials at lambda = 1 um (n, k)
    const double AIR_N = 1.0,  AIR_K = 0.0;
    const double SIO2_N= 1.5,  SIO2_K= 0.0;
    const double SI_N  = 3.5,  SI_K  = 0.0;
    const double AU_N  = 0.3,  AU_K  = 7.0;   // eps ~ -48.9 + 4.2i

    // group D comes from the Ikarus whitepaper in nm at lambda = 700 nm;
    // RCWA is scale invariant, so every length is divided by 700 nm.
    static double D(double x_nm) { return x_nm / 700.0; }

    static List<BenchCase> BuildCases()
    {
        List<BenchCase> cases = new List<BenchCase>();
        BenchCase c;

        // ---- group A: analytic anchors ------------------------------------
        // 0D: a plain film stack.  Any subwavelength period works (the layer is
        // laterally uniform); 0.5 keeps every order but 0 evanescent and stays
        // clear of a Rayleigh anomaly.
        c = new BenchCase("A1_slab_air", "A", 0, "TE",
                          "planar Si slab in air (exact Airy)");
        c.Period = 0.5; c.Depth = 0.20;
        c.HiN = SI_N; c.HiK = SI_K; c.SubN = AIR_N; c.SubK = AIR_K;
        cases.Add(c);

        c = new BenchCase("A1b_slab_glass", "A", 0, "TE",
                          "Si slab on glass (exact Airy)");
        c.Period = 0.5; c.Depth = 0.20;
        c.HiN = SI_N; c.HiK = SI_K; c.SubN = SIO2_N; c.SubK = SIO2_K;
        cases.Add(c);

        c = new BenchCase("A2_formbiref_TE", "A", 1, "TE",
                          "deep-subwave 1D grating = birefringent film, TE (EMT)");
        c.Period = 0.20; c.Ff = 0.5; c.Depth = 0.30;
        c.HiN = SI_N; c.HiK = SI_K; c.LoN = AIR_N; c.LoK = AIR_K;
        c.SubN = AIR_N; c.SubK = AIR_K;
        cases.Add(c);

        c = new BenchCase("A2_formbiref_TM", "A", 1, "TM",
                          "deep-subwave 1D grating = birefringent film, TM (EMT)");
        c.Period = 0.20; c.Ff = 0.5; c.Depth = 0.30;
        c.HiN = SI_N; c.HiK = SI_K; c.LoN = AIR_N; c.LoK = AIR_K;
        c.SubN = AIR_N; c.SubK = AIR_K;
        cases.Add(c);

        // ---- group B: 1D gratings -----------------------------------------
        c = new BenchCase("B1_Si_grating_TE", "B", 1, "TE",
                          "Si transmission grating, TE (fast baseline)");
        c.Period = 1.5; c.Ff = 0.5; c.Depth = 0.50;
        c.HiN = SI_N; c.HiK = SI_K;
        cases.Add(c);

        c = new BenchCase("B1_Si_grating_TM", "B", 1, "TM",
                          "Si transmission grating, TM (slow under Laurent)");
        c.Period = 1.5; c.Ff = 0.5; c.Depth = 0.50;
        c.HiN = SI_N; c.HiK = SI_K;
        cases.Add(c);

        c = new BenchCase("B2_HCG_TM", "B", 1, "TM",
                          "high-contrast subwavelength grating, TM (Li showcase)");
        c.Period = 0.80; c.Ff = 0.5; c.Depth = 0.30;
        c.HiN = SI_N; c.HiK = SI_K;
        cases.Add(c);

        c = new BenchCase("B3_Au_slits_TM", "B", 1, "TM",
                          "metal slit array, TM (plasmonic/EOT; hardest 1D)");
        c.Period = 0.50; c.Ff = 0.8; c.Depth = 0.20;
        c.HiN = AU_N; c.HiK = AU_K;
        cases.Add(c);

        // ---- group C: 2D rectangular pillars ------------------------------
        c = new BenchCase("C1_Si_pillars", "C", 2, "TE",
                          "Si square-pillar metasurface (subwavelength)");
        c.Period = 0.50; c.Ax = 0.30; c.Ay = 0.30; c.Depth = 0.40;
        c.HiN = SI_N; c.HiK = SI_K;                  // pillar
        c.LoN = AIR_N; c.LoK = AIR_K;                // background
        c.SubN = SIO2_N; c.SubK = SIO2_K;
        cases.Add(c);

        c = new BenchCase("C1b_Si_pillars_diffract", "C", 2, "TE",
                          "Si pillars, supra-wavelength (diffraction)");
        c.Period = 1.50; c.Ax = 0.60; c.Ay = 0.60; c.Depth = 0.40;
        c.HiN = SI_N; c.HiK = SI_K;
        c.LoN = AIR_N; c.LoK = AIR_K;
        c.SubN = SIO2_N; c.SubK = SIO2_K;
        cases.Add(c);

        c = new BenchCase("C2_Au_holes", "C", 2, "TE",
                          "metal hole array, 2D EOT (hardest 2D)");
        c.Period = 0.60; c.Ax = 0.30; c.Ay = 0.30; c.Depth = 0.20;
        c.HiN = AIR_N; c.HiK = AIR_K;                // the hole is the atom
        c.LoN = AU_N;  c.LoK = AU_K;                 // background is gold
        c.SubN = SIO2_N; c.SubK = SIO2_K;
        cases.Add(c);

        // ---- group D: the Ikarus whitepaper's cross-code cases -------------
        c = new BenchCase("D1_ikarus_hcg_TM", "D", 1, "TM",
                          "Ikarus whitepaper Fig.1/Tab.1: free-standing n=3.5 "
                          + "lamellar grating, TM (factorization stress test)");
        c.Period = D(400); c.Ff = 0.5; c.Depth = D(300);
        c.HiN = SI_N; c.HiK = SI_K;
        cases.Add(c);

        c = new BenchCase("D2_ikarus_cylinder_TE", "D", 2, "TE",
                          "Ikarus whitepaper: free-standing n=3.5 circular "
                          + "pillar, TE (curved boundary oblique to both axes)");
        c.Shape = "circle";
        c.Period = D(400); c.Radius = 0.30; c.Depth = D(200);
        c.HiN = SI_N; c.HiK = SI_K;
        c.LoN = AIR_N; c.LoK = AIR_K;
        c.SubN = AIR_N; c.SubK = AIR_K;
        cases.Add(c);

        return cases;
    }


    // =======================================================================
    //  geometry
    // =======================================================================
    // Cell-centre test for the CaModel mask: is sample (i,j) of an n x n grid
    // inside c's inclusion?  BIT-FOR-BIT the rule benchmark/structures.py's
    // layer_mask() uses by default (rect: cell centres, "<="; circle: cell
    // centres, distance test) and moose_raster_probe.cs's CellInside already
    // verified matches it -- keep all three in step if this changes.
    static bool CellInside(BenchCase c, int n, int i, int j)
    {
        double x = (i + 0.5) / n - 0.5;
        double y = (j + 0.5) / n - 0.5;
        if (c.Shape == "circle")
            return x * x + y * y <= c.Radius * c.Radius + 1.0e-12;
        double wx = c.Ax / c.Period, wy = c.Ay / c.Period;
        return Math.Abs(x) <= wx / 2.0 + 1.0e-12 && Math.Abs(y) <= wy / 2.0 + 1.0e-12;
    }

    // The explicit permittivity grid for c's patterned layer, at n x n --
    // Hi() is the inclusion (pillar, or the hole's own material for C2),
    // Lo() the background, matching structures.py's pillar/bg convention.
    static CaModel BuildMask(BenchCase c, int n)
    {
        Complex hi = c.Hi().GetEpsilon(WAVELENGTH);
        Complex lo = c.Lo().GetEpsilon(WAVELENGTH);
        CaModel model = new CaModel(n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                model.SetValue(i, j, CellInside(c, n, i, j) ? hi : lo);
        return model;
    }

    static GratingStructure BuildGrating(BenchCase c)
    {
        Materials superstrate = new Materials(SUPER_N, 0.0);
        // periodY <= 0 marks a 1D grating
        double period_y = (c.Dim == 2) ? c.Period : -1.0;

        GratingStructure grating = new GratingStructure(
            c.Period, period_y, superstrate, c.Sub());

        if (c.Dim == 0)
        {
            // plain film, no lateral structure at all
            grating.AddLayerOnBottom(new Layer(c.Depth, c.Hi()));
        }
        else if (c.Dim == 1)
        {
            // Layer(thickness, barMaterial, dutyCycle, trenchMaterial):
            // dutyCycle is the remaining fraction of barMaterial == ff.  Takes
            // the fill fraction directly -- no discretized grid, see the
            // header note on FFT REFINEMENT.
            Layer layer = new Layer(c.Depth, c.Hi(), c.Ff, c.Lo());
            grating.AddLayerOnBottom(layer);
        }
        else
        {
            // Explicit CaModel grid -- see "CAMODEL GEOMETRY" in the header.
            // Replaces the old Atom-based construction, which rasterized one
            // cell too wide per axis on every grid tested, aligned or not.
            CaModel mask = BuildMask(c, NX_2D);
            Layer layer = new Layer(c.Depth, mask);
            try { layer.Declare2D(); } catch (Exception) { }
            grating.AddLayerOnBottom(layer);
        }
        return grating;
    }


    // =======================================================================
    //  harvesting
    // =======================================================================

    // Number of propagating diffraction orders per axis at normal incidence:
    // |sin(theta_m)| = |m| * lambda / (n * period) <= 1.
    static int PropagatingOrders(double n_medium, double period)
    {
        if (period <= 0.0) return 0;
        double v = n_medium * period / WAVELENGTH;
        int m = (int)Math.Floor(v + 1.0e-9);
        return m;
    }

    // One order's efficiency, in the requested output polarization.  Returns 0
    // for anything Moose does not know about, so an unsupported order or an
    // unsupported polarization string can never abort a run -- the energy
    // balance below is what notices that something went missing.
    static double Eff(Rcwa solver, char tr, int ox, int oy, string pol)
    {
        try { return solver.GetEfficiencyForGivenOrder(tr, ox, oy, pol); }
        catch (Exception) { return 0.0; }
    }

    // Total efficiency of a half space: sum over every propagating order.
    // Evanescent orders carry no flux, so restricting the sum to the
    // propagating window is exact -- and much cheaper than asking Moose for
    // all 2m+1 orders when m is 500.
    //
    // The polarization argument is the trap here.  Moose's default is
    // rOutputPolarization = "in", and that returns only the co-polarized
    // output: on a 2D lattice the off-axis orders (both indices non-zero)
    // convert polarization, so their cross-polarized half is silently dropped
    // and the energy balance comes out short.  It only shows up on a 2D case
    // that actually has propagating off-axis orders -- in this battery that is
    // C1b_Si_pillars_diffract alone, which is why it went unnoticed at first.
    //
    // So every sum is formed three ways -- explicit "TE"+"TM", the "anything
    // else sums both outputs" reading of the help, and the plain default --
    // and RunOne keeps whichever one conserves energy.  HARVEST_TE_TM etc.
    // name the three in the CSV.
    static void SumEfficiency(Rcwa solver, char tr, int mx, int my,
                              out double te_tm, out double both, out double def)
    {
        te_tm = 0.0; both = 0.0; def = 0.0;
        for (int ox = -mx; ox <= mx; ox++)
        {
            for (int oy = -my; oy <= my; oy++)
            {
                te_tm += Eff(solver, tr, ox, oy, "TE") + Eff(solver, tr, ox, oy, "TM");
                both  += Eff(solver, tr, ox, oy, "both");
                def   += Eff(solver, tr, ox, oy, "in");
            }
        }
    }

    // Every return goes through the clamp, 1D included: Moose accepts nothing
    // outside [30, 100] and silently substitutes, so a value returned here that
    // Moose would not take is a value the CSV records and the solver never used.
    // (The 1D rows of earlier runs say fft_refinement = 5 and were computed at
    // 30.  Harmless for the results -- the 1D masks are exact and those columns
    // match grcwa to six digits -- but the column was fiction.)
    static int Clamp(int refinement)
    {
        if (refinement < FFT_REFINEMENT_MIN) return FFT_REFINEMENT_MIN;
        if (refinement > FFT_REFINEMENT_MAX) return FFT_REFINEMENT_MAX;
        return refinement;
    }

    static int RefinementFor(BenchCase c, int m)
    {
        // 1D/0D: the duty-cycle Layer constructor takes the fill fraction
        // directly, no discretized grid -- refinement has never been observed
        // to matter there.  2D: fixed at the maximum, see FFT_REFINEMENT_2D.
        if (c.Dim != 2) return Clamp(FFT_REFINEMENT_MIN);
        return Clamp(FFT_REFINEMENT_2D);
    }

    // Process.WorkingSet64 comes back as 0 on some Mono builds (it did on the
    // Windows host this was first run on), so fall back down a chain until
    // something reports a real number.  GC.GetTotalMemory is the last resort
    // and only sees managed memory -- Moose's matrices are unmanaged C++, so
    // that number is a floor, not the truth.
    static double MemoryMb()
    {
        try
        {
            System.Diagnostics.Process p =
                System.Diagnostics.Process.GetCurrentProcess();
            p.Refresh();
            double ws = (double)p.WorkingSet64 / (1024.0 * 1024.0);
            if (ws > 0.0) return ws;
            double pm = (double)p.PrivateMemorySize64 / (1024.0 * 1024.0);
            if (pm > 0.0) return pm;
        }
        catch (Exception) { }
        try
        {
            double ews = (double)Environment.WorkingSet / (1024.0 * 1024.0);
            if (ews > 0.0) return ews;
        }
        catch (Exception) { }
        try { return (double)GC.GetTotalMemory(false) / (1024.0 * 1024.0); }
        catch (Exception) { return -1.0; }
    }

    static double PeakMemoryMb()
    {
        try
        {
            System.Diagnostics.Process p =
                System.Diagnostics.Process.GetCurrentProcess();
            p.Refresh();
            double pk = (double)p.PeakWorkingSet64 / (1024.0 * 1024.0);
            if (pk > 0.0) return pk;
        }
        catch (Exception) { }
        return MemoryMb();
    }


    // =======================================================================
    //  one (case, order) run
    // =======================================================================
    static BenchResult RunOne(BenchCase c, int m)
    {
        BenchResult r = new BenchResult();
        r.Name = c.Name;
        r.M    = m;
        r.Q    = 2 * m + 1;
        r.NG   = (c.Dim == 2) ? (double)r.Q * (double)r.Q : (double)r.Q;
        r.FftRefinement = RefinementFor(c, m);
        r.Status = "ok";
        r.Note   = "";
        r.MemBefore = MemoryMb();

        int my = (c.Dim == 2) ? m : 0;
        double pol_angle = (c.Pol == "TM") ? 0.0 : 90.0;

        GratingStructure grating = null;
        Rcwa solver = null;

        System.Diagnostics.Stopwatch sw_total = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
            grating = BuildGrating(c);
            solver  = new Rcwa(grating, m, my, r.FftRefinement, RCWA_CACHE);
            sw.Stop();
            r.TSetup = sw.Elapsed.TotalSeconds;

            sw = System.Diagnostics.Stopwatch.StartNew();
            solver.Calc(WAVELENGTH, AOI, CONICAL, pol_angle, c.Dim == 2);
            sw.Stop();
            r.TSolve = sw.Elapsed.TotalSeconds;

            sw = System.Diagnostics.Stopwatch.StartNew();
            int mx_r = Math.Min(m,  PropagatingOrders(SUPER_N, c.Period));
            int my_r = Math.Min(my, (c.Dim == 2)
                                    ? PropagatingOrders(SUPER_N, c.Period) : 0);
            int mx_t = Math.Min(m,  PropagatingOrders(c.SubN, c.Period));
            int my_t = Math.Min(my, (c.Dim == 2)
                                    ? PropagatingOrders(c.SubN, c.Period) : 0);

            double r_te, r_both, r_def, t_te, t_both, t_def;
            SumEfficiency(solver, 'r', mx_r, my_r, out r_te, out r_both, out r_def);
            SumEfficiency(solver, 't', mx_t, my_t, out t_te, out t_both, out t_def);
            r.A  = solver.GetAbsorption() * SCALE;
            r.R0 = Eff(solver, 'r', 0, 0, "TE") + Eff(solver, 'r', 0, 0, "TM");
            r.T0 = Eff(solver, 't', 0, 0, "TE") + Eff(solver, 't', 0, 0, "TM");
            r.R0 *= SCALE; r.T0 *= SCALE;

            // GetAbsorption() is 1 - R_moose - T_moose, so R + T + A is 1
            // exactly iff our order sums reproduce Moose's internal totals.
            // Keep the polarization convention that actually conserves energy;
            // all three are written to the CSV so the choice stays auditable.
            r.RTeTm = r_te * SCALE;   r.TTeTm = t_te * SCALE;
            r.RBoth = r_both * SCALE; r.TBoth = t_both * SCALE;
            r.RIn   = r_def * SCALE;  r.TIn   = t_def * SCALE;
            r.ETeTm = r.RTeTm + r.TTeTm + r.A;
            r.EBoth = r.RBoth + r.TBoth + r.A;
            r.EIn   = r.RIn   + r.TIn   + r.A;

            if (Math.Abs(1.0 - r.ETeTm) <= ENERGY_TOL)
            { r.R = r.RTeTm; r.T = r.TTeTm; r.Energy = r.ETeTm; r.Harvest = "TE+TM"; }
            else if (Math.Abs(1.0 - r.EBoth) <= ENERGY_TOL)
            { r.R = r.RBoth; r.T = r.TBoth; r.Energy = r.EBoth; r.Harvest = "both"; }
            else if (Math.Abs(1.0 - r.EIn) <= ENERGY_TOL)
            { r.R = r.RIn; r.T = r.TIn; r.Energy = r.EIn; r.Harvest = "in"; }
            else
            {
                // Nothing conserves energy -- report the best candidate but
                // mark the row, so it can never be merged as if it were sound.
                r.R = r.RTeTm; r.T = r.TTeTm; r.Energy = r.ETeTm;
                r.Harvest = "none";
                r.Status  = "energy";
                r.Note    = "no polarization convention conserved energy";
            }
            sw.Stop();
            r.TReap = sw.Elapsed.TotalSeconds;
        }
        catch (Exception e)
        {
            r.Status = "failed";
            r.Note   = e.Message.Replace("\n", " ").Replace(",", ";");
        }
        sw_total.Stop();
        r.TTotal   = sw_total.Elapsed.TotalSeconds;
        r.MemAfter = MemoryMb();
        r.MemPeak  = PeakMemoryMb();

        // Moose wraps C++ objects, the GC cannot free them for us.
        try { if (solver  != null) solver.Delete();  } catch (Exception) { }
        try { if (grating != null) grating.Delete(); } catch (Exception) { }

        return r;
    }


    // =======================================================================
    //  worker pool -- one solve per thread, one Rcwa instance per solve
    // =======================================================================
    // A Moose solve is single-threaded, so the only way to use a many-core box
    // is to solve several structures at once.  Everything a solve touches is
    // created and deleted inside RunOne, so the threads share nothing but the
    // bookkeeping below, which is why one lock covers all of it.
    static readonly object sLock    = new object();
    // Held for the duration of a run whose nG exceeds PARALLEL_NG_LIMIT, so
    // only one memory-hungry solve is ever in flight.
    static readonly object sBigLock = new object();

    static List<BenchCase> sQueueCase = new List<BenchCase>();
    static List<int>       sQueueM    = new List<int>();
    static int             sQueueNext;
    static StreamWriter    sCsv, sLog;
    static Dictionary<string, List<BenchResult>> sPerCase;
    static Dictionary<string, BenchResult>       sBatch;
    static int             sTotalRuns, sOkRuns;
    static bool            sQuiet;          // self test: do not touch the CSV

    static string FormatLine(BenchCase c, BenchResult r)
    {
        if (r.Status == "failed")
            return "  " + Pad(c.Name, 26) + " m=" + Pad(r.M.ToString(INV), 4)
                 + " FAILED: " + r.Note;
        return "  " + Pad(c.Name, 26)
             + " m=" + Pad(r.M.ToString(INV), 4)
             + " nG=" + Pad(((long)r.NG).ToString(INV), 7)
             + " R=" + Pad(F(r.R, 6), 10)
             + " T=" + Pad(F(r.T, 6), 10)
             + " A=" + Pad(F(r.A, 6), 10)
             + " |1-E|=" + Pad(E(Math.Abs(1.0 - r.Energy)), 10)
             + " " + Pad(r.Harvest, 6)
             + " solve=" + Pad(F(r.TSolve, 3) + "s", 10)
             + " mem=" + F(r.MemAfter, 0) + "MB";
    }

    static void Record(BenchCase c, BenchResult r)
    {
        string line = FormatLine(c, r);
        lock (sLock)
        {
            sBatch[c.Name + "@" + r.M.ToString(INV)] = r;
            if (sQuiet) return;
            sTotalRuns++;
            if (r.Status == "ok") sOkRuns++;
            sPerCase[c.Name].Add(r);
            // A row whose energy balance is broken is still printed with its
            // numbers -- they are the evidence -- but in red, so it is never
            // mistaken for a sound one.
            if (r.Status == "ok") Io.output(line);
            else if (r.Status == "failed") Io.error(line);
            else Io.error(line + "   <-- ENERGY CHECK FAILED");
            if (sCsv != null)
            {
                try { sCsv.WriteLine(CsvRow(c, r)); sCsv.Flush(); }
                catch (Exception e) { Io.error("csv write failed: " + e.Message); }
            }
            if (sLog != null)
            {
                try { sLog.WriteLine(line); sLog.Flush(); }
                catch (Exception) { }
            }
        }
    }

    static void Worker()
    {
        while (true)
        {
            BenchCase c; int m;
            lock (sLock)
            {
                if (sQueueNext >= sQueueCase.Count) return;
                c = sQueueCase[sQueueNext];
                m = sQueueM[sQueueNext];
                sQueueNext++;
            }
            long ng = NgOf(c, m);
            BenchResult r;
            if (PARALLEL_NG_LIMIT > 0 && ng > PARALLEL_NG_LIMIT)
                lock (sBigLock) { r = RunOne(c, m); }
            else
                r = RunOne(c, m);
            Record(c, r);
        }
    }

    static long NgOf(BenchCase c, int m)
    {
        long q = 2L * m + 1L;
        return (c.Dim == 2) ? q * q : q;
    }

    static int WorkerCount()
    {
        int n = PARALLEL_TASKS;
        if (n == 0) n = Environment.ProcessorCount;
        if (n < 1) n = 1;
        return n;
    }

    // Runs the queue to completion.  With one worker nothing is spawned at all,
    // so the sequential path stays exactly what it was.
    static void RunQueue(List<BenchCase> qc, List<int> qm)
    {
        sQueueCase = qc; sQueueM = qm; sQueueNext = 0;
        sBatch = new Dictionary<string, BenchResult>();
        int workers = Math.Min(WorkerCount(), qc.Count);
        if (workers <= 1) { Worker(); return; }
        Thread[] pool = new Thread[workers];
        for (int i = 0; i < workers; i++)
        {
            pool[i] = new Thread(new ThreadStart(Worker));
            pool[i].IsBackground = false;
            pool[i].Start();
        }
        for (int i = 0; i < workers; i++) pool[i].Join();
    }

    // Solve the same points twice, once alone and once on the pool, and compare
    // them bit for bit.  Concurrent Rcwa instances SHOULD be independent -- the
    // shipped ParallelRcwa does the same thing internally -- but "should" is
    // not something to bet a night of compute on, and a shared FFT plan cache
    // would be exactly the kind of thing that quietly corrupts results.
    static bool SelfTest(List<BenchCase> cases)
    {
        List<BenchCase> qc = new List<BenchCase>();
        List<int> qm = new List<int>();
        for (int i = 0; i < cases.Count; i++)
        {
            int[] sweep = (cases[i].Dim == 2) ? SWEEP_2D : SWEEP_1D;
            if (sweep.Length == 0) continue;
            int idx = Math.Min(SELFTEST_STAGE, sweep.Length - 1);
            qc.Add(cases[i]); qm.Add(sweep[idx]);
        }
        Io.output("");
        Io.output(" self test: " + qc.Count.ToString(INV)
                  + " points, sequential vs " + WorkerCount().ToString(INV)
                  + " workers, "
                  + Math.Max(1, SELFTEST_REPEATS).ToString(INV)
                  + " parallel passes");

        sQuiet = true;
        int saved = PARALLEL_TASKS;
        PARALLEL_TASKS = 1;
        RunQueue(qc, qm);
        Dictionary<string, BenchResult> seq = sBatch;
        PARALLEL_TASKS = saved;

        int repeats = Math.Max(1, SELFTEST_REPEATS);
        List<Dictionary<string, BenchResult>> runs =
            new List<Dictionary<string, BenchResult>>();
        for (int rep = 0; rep < repeats; rep++)
        {
            RunQueue(qc, qm);
            runs.Add(sBatch);
        }
        sQuiet = false;

        bool ok = true;
        Io.output(" " + Pad("point", 32) + Pad("sequential", 14)
                  + Pad("parallel", 14) + "verdict");
        for (int i = 0; i < qc.Count; i++)
        {
            string key = qc[i].Name + "@" + qm[i].ToString(INV);
            BenchResult a = seq.ContainsKey(key) ? seq[key] : null;
            BenchResult shown = null;
            int mismatches = 0;
            for (int rep = 0; rep < repeats; rep++)
            {
                BenchResult b = runs[rep].ContainsKey(key) ? runs[rep][key] : null;
                if (shown == null) shown = b;
                bool same = a != null && b != null
                            && a.R == b.R && a.T == b.T && a.A == b.A
                            && a.R0 == b.R0 && a.T0 == b.T0
                            && a.Status == b.Status && a.Harvest == b.Harvest;
                if (!same) { mismatches++; if (shown == null || b != null) shown = b; }
            }
            if (mismatches > 0) ok = false;
            Io.output(" " + Pad(key, 32)
                      + Pad(a == null ? "-" : F(a.R, 9), 14)
                      + Pad(shown == null ? "-" : F(shown.R, 9), 14)
                      + (mismatches == 0
                         ? "identical"
                         : "DIFFERENT in " + mismatches.ToString(INV) + " of "
                           + repeats.ToString(INV) + " passes"));
        }
        Io.output("");
        if (ok)
            Io.success(" self test passed: parallel results are bit-identical to "
                       + "sequential. PARALLEL_TASKS = "
                       + WorkerCount().ToString(INV) + " is safe on this build.");
        else
            Io.error(" SELF TEST FAILED: parallel results differ from "
                     + "sequential. Concurrent Rcwa instances interfere on "
                     + "this build -- set PARALLEL_TASKS = 1. A failure in "
                     + "only some passes is still a failure: interference is "
                     + "timing dependent, so a long run would corrupt "
                     + "different points every time.");
        return ok;
    }


    // =======================================================================
    //  output helpers
    // =======================================================================
    static string F(double v, int digits)
    {
        return v.ToString("F" + digits.ToString(INV), INV);
    }

    static string G(double v)
    {
        return v.ToString("R", INV);
    }

    static string E(double v)
    {
        return v.ToString("0.0e+00", INV);
    }

    static string ResolveOutputDir()
    {
        string dir = OUTPUT_DIR;
        if (dir == null || dir.Length == 0)
            dir = Path.Combine(Path.GetTempPath(), "moose_bench");
        try
        {
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
        }
        catch (Exception e)
        {
            Io.error("cannot create output directory " + dir + ": " + e.Message);
            return null;
        }
        return dir;
    }

    // R/T/A are FRACTIONS (0..1), not percent -- see SCALE.
    const string CSV_HEADER =
        "case,group,dim,pol,column,m_moose,q,nG,fft_refinement,"
        + "R,T,A,R0,T0,energy,harvest,"
        + "R_te_tm,T_te_tm,energy_te_tm,R_both,T_both,energy_both,"
        + "R_in,T_in,energy_in,"
        + "t_setup_s,t_solve_s,t_harvest_s,t_total_s,"
        + "mem_before_mb,mem_after_mb,mem_peak_mb,status,note";

    static string CsvRow(BenchCase c, BenchResult r)
    {
        return c.Name + "," + c.Group + "," + c.Dim.ToString(INV) + "," + c.Pol
            + ",moose,"
            + r.M.ToString(INV) + "," + r.Q.ToString(INV) + ","
            + ((long)r.NG).ToString(INV) + ","
            + r.FftRefinement.ToString(INV) + ","
            + G(r.R) + "," + G(r.T) + "," + G(r.A) + ","
            + G(r.R0) + "," + G(r.T0) + "," + G(r.Energy) + ","
            + r.Harvest + ","
            + G(r.RTeTm) + "," + G(r.TTeTm) + "," + G(r.ETeTm) + ","
            + G(r.RBoth) + "," + G(r.TBoth) + "," + G(r.EBoth) + ","
            + G(r.RIn) + "," + G(r.TIn) + "," + G(r.EIn) + ","
            + F(r.TSetup, 6) + "," + F(r.TSolve, 6) + ","
            + F(r.TReap, 6) + "," + F(r.TTotal, 6) + ","
            + F(r.MemBefore, 1) + "," + F(r.MemAfter, 1) + ","
            + F(r.MemPeak, 1) + "," + r.Status + "," + r.Note;
    }

    static double ParseD(string s)
    {
        double v;
        if (Double.TryParse(s, NumberStyles.Float, INV, out v)) return v;
        return 0.0;
    }

    // Set when the CSV on disk was written by a different version of this
    // script; the run then starts a fresh, timestamped CSV instead of
    // appending rows of a second format to it.
    static bool sCsvHeaderMismatch = false;

    // Runs already in the CSV, keyed "case@m" -- for RESUME.  They are fed back
    // into per_case so that the summary and the JSON fragment of a resumed run
    // describe the WHOLE sweep, not just the part computed in this session.
    static Dictionary<string, BenchResult> ReadPrevious(string csv_path)
    {
        Dictionary<string, BenchResult> done = new Dictionary<string, BenchResult>();
        if (!File.Exists(csv_path)) return done;
        try
        {
            StreamReader sr = new StreamReader(csv_path);
            string header = sr.ReadLine();
            if (header == null || header != CSV_HEADER)
            {
                sr.Close();
                sCsvHeaderMismatch = true;
                Io.error("CSV header of " + csv_path + " does not match this "
                         + "script's format -- not resuming from it, and "
                         + "writing to a new file instead");
                return done;
            }
            string line;
            while ((line = sr.ReadLine()) != null)
            {
                string[] f = line.Split(',');
                if (f.Length < 33) continue;
                if (f[32] != "ok") continue;            // retry anything failed
                BenchResult r = new BenchResult();
                r.Name   = f[0];
                r.M      = (int)ParseD(f[5]);
                r.Q      = (int)ParseD(f[6]);
                r.NG     = ParseD(f[7]);
                r.FftRefinement = (int)ParseD(f[8]);
                r.R      = ParseD(f[9]);
                r.T      = ParseD(f[10]);
                r.A      = ParseD(f[11]);
                r.R0     = ParseD(f[12]);
                r.T0     = ParseD(f[13]);
                r.Energy = ParseD(f[14]);
                r.Harvest = f[15];
                r.RTeTm  = ParseD(f[16]); r.TTeTm = ParseD(f[17]); r.ETeTm = ParseD(f[18]);
                r.RBoth  = ParseD(f[19]); r.TBoth = ParseD(f[20]); r.EBoth = ParseD(f[21]);
                r.RIn    = ParseD(f[22]); r.TIn   = ParseD(f[23]); r.EIn   = ParseD(f[24]);
                r.TSetup = ParseD(f[25]);
                r.TSolve = ParseD(f[26]);
                r.TReap  = ParseD(f[27]);
                r.TTotal = ParseD(f[28]);
                r.MemBefore = ParseD(f[29]);
                r.MemAfter  = ParseD(f[30]);
                r.MemPeak   = ParseD(f[31]);
                r.Status = "ok";
                r.Note   = "from csv";
                done[r.Name + "@" + r.M.ToString(INV)] = r;
            }
            sr.Close();
        }
        catch (Exception e)
        {
            Io.error("could not read " + csv_path + " for resume: " + e.Message);
        }
        return done;
    }

    static bool Selected(BenchCase c)
    {
        if (ONLY_CASES.Length > 0)
        {
            bool hit = false;
            string[] want = ONLY_CASES.Split(',');
            for (int i = 0; i < want.Length; i++)
            {
                string w = want[i].Trim();
                if (w.Length == 0) continue;
                if (w == c.Name || w == c.Group) hit = true;
            }
            if (!hit) return false;
        }
        if (SKIP_CASES.Length > 0)
        {
            string[] skip = SKIP_CASES.Split(',');
            for (int i = 0; i < skip.Length; i++)
            {
                string w = skip[i].Trim();
                if (w.Length == 0) continue;
                if (w == c.Name || w == c.Group) return false;
            }
        }
        return true;
    }

    static string Pad(string s, int n)
    {
        while (s.Length < n) s = s + " ";
        return s;
    }

    // log-log least squares slope of t over nG -> the empirical scaling
    // exponent p in t ~ nG^p.  Needs at least three points to mean anything.
    static double ScalingExponent(List<BenchResult> runs)
    {
        int n = 0;
        double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
        for (int i = 0; i < runs.Count; i++)
        {
            BenchResult r = runs[i];
            if (r.Status != "ok" || r.TSolve <= 0.0 || r.NG <= 1.0) continue;
            double x = Math.Log(r.NG);
            double y = Math.Log(r.TSolve);
            sx += x; sy += y; sxx += x * x; sxy += x * y; n++;
        }
        if (n < 3) return Double.NaN;
        double den = n * sxx - sx * sx;
        if (Math.Abs(den) < 1.0e-12) return Double.NaN;
        return (n * sxy - sx * sy) / den;
    }


    // =======================================================================
    //  main
    // =======================================================================
    static void Main()
    {
        List<BenchCase> all = BuildCases();
        List<BenchCase> cases = new List<BenchCase>();
        for (int i = 0; i < all.Count; i++)
            if (Selected(all[i])) cases.Add(all[i]);

        string dir = ResolveOutputDir();
        string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss", INV);
        string csv_path  = (dir == null) ? null : Path.Combine(dir, "moose_conv.csv");
        string json_path = (dir == null) ? null : Path.Combine(dir, "moose_sweep.json");
        string log_path  = (dir == null) ? null : Path.Combine(dir, "moose_bench_" + stamp + ".log");

        Io.output("=================================================================");
        Io.output(" Moose convergence + timing benchmark");
        Io.output(" started      : " + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss", INV));
        Io.output(" cases        : " + cases.Count.ToString(INV) + " of "
                  + all.Count.ToString(INV));
        Io.output(" wavelength   : " + F(WAVELENGTH, 4) + " um, aoi " + F(AOI, 2)
                  + " deg, conical " + F(CONICAL, 2) + " deg");
        Io.output(" sweep 1D (m) : " + Join(SWEEP_1D));
        Io.output(" sweep 2D (m) : " + Join(SWEEP_2D));
        Io.output(" 2D geometry  : explicit CaModel, " + NX_2D.ToString(INV) + "x"
                  + NX_2D.ToString(INV) + " cell-centred (exact for C1/C1b/C2; "
                  + "D2's circle has no exact grid)");
        Io.output(" fft refinement: " + FFT_REFINEMENT_2D.ToString(INV)
                  + " for every 2D point (Moose's own max; a real residual "
                  + "remains there through a mechanism not yet identified, "
                  + "see RASTERIZATION.md Sec.9), " + FFT_REFINEMENT_MIN.ToString(INV)
                  + " for 1D/0D (no discretized grid, refinement does not apply)");
        if (RESUME)
            Io.output(" !! RESUME=true: a CSV from before the CaModel switch has WRONG "
                      + "2D rows (Atom's +1-cell shape error) that RESUME will silently "
                      + "keep -- start a fresh OUTPUT_DIR if this is the first run since "
                      + "the switch.");
        Io.output(" cpu cores    : " + Environment.ProcessorCount.ToString(INV)
                  + ",  parallel solves: " + WorkerCount().ToString(INV)
                  + (WorkerCount() > 1 ? "  (per-solve times are noisier)" : "")
                  + (PARALLEL_NG_LIMIT > 0
                     ? ",  nG > " + PARALLEL_NG_LIMIT.ToString(INV) + " runs alone"
                     : ""));
        Io.output(" output       : " + (dir == null ? "<console only>" : dir));
        Io.output("=================================================================");

        if (SHOW_STRUCTURES)
        {
            for (int i = 0; i < cases.Count; i++)
            {
                BenchCase c = cases[i];
                GratingStructure g = BuildGrating(c);
                CaModel view = g.ConvertToCaModel(WAVELENGTH);
                view.AddToLog(g.GetLog());
                view.Show(c.Name + "  (" + c.Desc + ")");
                g.Delete();
            }
        }
        if (DRY_RUN)
        {
            Io.success("dry run: structures built" + (SHOW_STRUCTURES ? " and shown" : "")
                       + ", nothing solved.");
            return;
        }

        if (PARALLEL_SELFTEST)
        {
            SelfTest(cases);
            Io.output("self test only -- set PARALLEL_SELFTEST = false to sweep.");
            return;
        }

        StreamWriter csv = null;
        StreamWriter log = null;
        Dictionary<string, BenchResult> done = new Dictionary<string, BenchResult>();
        if (csv_path != null)
        {
            try
            {
                bool fresh = !File.Exists(csv_path);
                if (RESUME) done = ReadPrevious(csv_path);
                if (sCsvHeaderMismatch)
                {
                    csv_path = Path.Combine(dir, "moose_conv_" + stamp + ".csv");
                    fresh = true;
                }
                csv = new StreamWriter(csv_path, true);
                if (fresh) { csv.WriteLine(CSV_HEADER); csv.Flush(); }
                log = new StreamWriter(log_path, false);
            }
            catch (Exception e)
            {
                Io.error("cannot open output files (" + e.Message
                         + ") -- continuing with console output only");
                csv = null; log = null;
            }
        }
        if (done.Count > 0)
            Io.output(" resume       : " + done.Count.ToString(INV)
                      + " runs already in the CSV will be skipped");

        // one result list per case, for the summary at the end
        Dictionary<string, List<BenchResult>> per_case =
            new Dictionary<string, List<BenchResult>>();
        sPerCase = per_case;
        sCsv = csv; sLog = log;
        Dictionary<string, bool> exhausted = new Dictionary<string, bool>();
        for (int i = 0; i < cases.Count; i++)
        {
            per_case[cases[i].Name] = new List<BenchResult>();
            exhausted[cases[i].Name] = false;
        }
        // Fold the resumed rows back in, ordered by the sweep, so summary and
        // JSON describe the complete sweep and not just this session's part.
        for (int i = 0; i < cases.Count; i++)
        {
            BenchCase c0 = cases[i];
            int[] sw0 = (c0.Dim == 2) ? SWEEP_2D : SWEEP_1D;
            for (int k = 0; k < sw0.Length; k++)
            {
                string k0 = c0.Name + "@" + sw0[k].ToString(INV);
                if (done.ContainsKey(k0)) per_case[c0.Name].Add(done[k0]);
            }
        }

        // Outer loop over the sweep, inner over the cases: the cheap orders of
        // EVERY case are finished before the expensive ones start, so aborting
        // half way still leaves a complete low-order picture.
        int stages = Math.Max(SWEEP_1D.Length, SWEEP_2D.Length);
        sTotalRuns = 0; sOkRuns = 0;
        double t_wall = 0.0;
        System.Diagnostics.Stopwatch sw_all = System.Diagnostics.Stopwatch.StartNew();

        for (int stage = 0; stage < stages; stage++)
        {
            // Collect this stage's work first, then hand it to the pool. The
            // expensive runs go in first so the last worker to finish is not
            // one that only just picked up the biggest job.
            List<BenchCase> qc = new List<BenchCase>();
            List<int> qm = new List<int>();
            for (int i = 0; i < cases.Count; i++)
            {
                BenchCase c = cases[i];
                int[] sweep = (c.Dim == 2) ? SWEEP_2D : SWEEP_1D;
                if (stage >= sweep.Length) continue;
                if (exhausted[c.Name]) continue;
                int m = sweep[stage];
                if (done.ContainsKey(c.Name + "@" + m.ToString(INV)))
                {
                    Io.output("  skip (done)  " + Pad(c.Name, 26)
                              + " m=" + m.ToString(INV));
                    continue;
                }
                int at = qc.Count;
                while (at > 0 && NgOf(qc[at - 1], qm[at - 1]) < NgOf(c, m)) at--;
                qc.Insert(at, c); qm.Insert(at, m);
            }
            if (qc.Count == 0) continue;

            RunQueue(qc, qm);

            // The cost guard runs once the stage is done rather than the
            // instant a solve overruns: with several solves in flight there is
            // no sensible way to stop the ones already started.
            if (MAX_SECONDS_PER_SOLVE > 0.0)
            {
                for (int i = 0; i < qc.Count; i++)
                {
                    string k = qc[i].Name + "@" + qm[i].ToString(INV);
                    if (!sBatch.ContainsKey(k)) continue;
                    BenchResult r = sBatch[k];
                    if (r.TSolve <= MAX_SECONDS_PER_SOLVE) continue;
                    exhausted[qc[i].Name] = true;
                    Io.output("  -> " + qc[i].Name + ": solve took "
                              + F(r.TSolve, 1) + "s > budget "
                              + F(MAX_SECONDS_PER_SOLVE, 0)
                              + "s, skipping the higher orders of this case");
                }
            }
        }
        sw_all.Stop();
        t_wall = sw_all.Elapsed.TotalSeconds;

        // ---- summary -------------------------------------------------------
        Io.output("");
        Io.output("=================================================================");
        Io.output(" cost summary  (t_solve, seconds; p = exponent of t ~ nG^p;"
                  + " resumed rows included)");
        Io.output("=================================================================");
        Io.output(" " + Pad("case", 26) + Pad("runs", 6) + Pad("nG_max", 9)
                  + Pad("t_max[s]", 11) + Pad("sum t[s]", 11) + "p");
        for (int i = 0; i < cases.Count; i++)
        {
            BenchCase c = cases[i];
            List<BenchResult> runs = per_case[c.Name];
            double tmax = 0.0, tsum = 0.0, ngmax = 0.0;
            int nok = 0;
            for (int k = 0; k < runs.Count; k++)
            {
                if (runs[k].Status != "ok") continue;
                nok++;
                tsum += runs[k].TSolve;
                if (runs[k].TSolve > tmax) tmax = runs[k].TSolve;
                if (runs[k].NG > ngmax) ngmax = runs[k].NG;
            }
            double p = ScalingExponent(runs);
            Io.output(" " + Pad(c.Name, 26) + Pad(nok.ToString(INV), 6)
                      + Pad(((long)ngmax).ToString(INV), 9)
                      + Pad(F(tmax, 3), 11) + Pad(F(tsum, 3), 11)
                      + (Double.IsNaN(p) ? "n/a" : F(p, 2)));
        }
        Io.output("");
        Io.output(" this session: " + sOkRuns.ToString(INV) + " ok / "
                  + sTotalRuns.ToString(INV) + " attempted, wall time "
                  + F(t_wall, 1) + " s"
                  + (done.Count > 0
                     ? "  (+ " + done.Count.ToString(INV) + " resumed from CSV)"
                     : ""));

        // ---- moose_reference.json fragment ---------------------------------
        string json = BuildJson(cases, per_case);
        if (json_path != null)
        {
            try
            {
                StreamWriter jw = new StreamWriter(json_path, false);
                jw.Write(json);
                jw.Close();
                Io.success(" wrote " + json_path);
            }
            catch (Exception e) { Io.error("json write failed: " + e.Message); }
        }
        Io.output("");
        Io.output("---- moose_reference.json fragment (paste into \"cases\") ----");
        Io.output(json);

        if (csv != null) { try { csv.Close(); } catch (Exception) { } }
        if (log != null) { try { log.Close(); } catch (Exception) { } }
        if (csv_path != null) Io.success(" wrote " + csv_path);
        Io.success("done.");
    }

    static string Join(int[] a)
    {
        string s = "";
        for (int i = 0; i < a.Length; i++)
        {
            if (i > 0) s += ", ";
            s += a[i].ToString(INV);
        }
        return s;
    }

    // Emits exactly the shape of benchmark/moose_reference.json's "cases":
    // 1D keys are the max order m, 2D keys are "(m,m)"; "ref" is the value at
    // the highest order that actually ran.
    static string BuildJson(List<BenchCase> cases,
                            Dictionary<string, List<BenchResult>> per_case)
    {
        string s = "{\n";
        bool first_case = true;
        for (int i = 0; i < cases.Count; i++)
        {
            BenchCase c = cases[i];
            List<BenchResult> runs = per_case[c.Name];
            int nok = 0;
            for (int k = 0; k < runs.Count; k++) if (runs[k].Status == "ok") nok++;
            if (nok == 0) continue;

            if (!first_case) s += ",\n";
            first_case = false;

            double best_ng = -1.0, best_r = 0.0;
            string sweep = "";
            bool first_pt = true;
            for (int k = 0; k < runs.Count; k++)
            {
                BenchResult r = runs[k];
                if (r.Status != "ok") continue;
                string key = (c.Dim == 2)
                    ? "(" + r.M.ToString(INV) + "," + r.M.ToString(INV) + ")"
                    : r.M.ToString(INV);
                if (!first_pt) sweep += ", ";
                first_pt = false;
                sweep += "\"" + key + "\": " + G(r.R);
                if (r.NG > best_ng) { best_ng = r.NG; best_r = r.R; }
            }
            s += "    \"" + c.Name + "\": {\n";
            if (c.Dim == 2) s += "      \"dim\": 2,\n";
            s += "      \"pol\": \"" + (c.Dim == 0 ? "TMM" : c.Pol) + "\",\n";
            s += "      \"ref\": " + G(best_r) + ", \"ref_provisional\": true,\n";
            s += "      \"sweep\": {" + sweep + "}\n";
            s += "    }";
        }
        s += "\n}\n";
        return s;
    }
}
