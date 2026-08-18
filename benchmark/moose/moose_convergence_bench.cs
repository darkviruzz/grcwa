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
//     are directly comparable),
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
// CIRCULAR ATOM (case D2).  The help calls the third argument of
//   Atom(posX, posY, r, material) a "radius", but unit_tests_structures.cs
//   (TestAtom.TestCircular) asserts for Atom(0.2, 0.3, 0.2, mat):
//       GetStartX() == 0.1, GetStopX() == 0.3, GetWidthX() == 0.2
//   i.e. start = pos - arg/2:  the argument is the *diameter*, not the radius.
//   The shipped unit test wins over the help, so D2's radius 0.30 (in units of
//   the period) is passed as 0.60.  Flip CIRCLE_ARG_IS_RADIUS below if your
//   Moose build disagrees -- ConvertToCaModel + SHOW_STRUCTURES makes that a
//   ten second check.
//
// FFT REFINEMENT.  Rcwa's rRefinementFactorEpsFT multiplies the *order count*,
//   so the absolute sampling of the unit cell would grow with m and the
//   permittivity would be resolved differently at every point of a convergence
//   sweep.  grcwa rasterizes on a fixed 256x256 grid instead.  FFT_MODE = 1
//   reproduces that: the refinement is chosen per run so the absolute grid
//   stays ~FFT_TARGET_SAMPLES.  FFT_MODE = 0 uses a fixed refinement factor.
// ---------------------------------------------------------------------------

using System;
using System.IO;
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
    static string OUTPUT_DIR   = "";

    // Truncation sweep.  These are Moose MAX ORDERS m (retained: 2m+1 per
    // axis).  The defaults are exactly the keys already present in
    // benchmark/moose_reference.json, so new runs line up with the old ones.
    static readonly int[] SWEEP_1D = { 1, 3, 5, 10, 20, 50, 100, 200, 500 };
    static readonly int[] SWEEP_2D = { 1, 2, 3, 4, 5, 7, 10, 15, 20, 30 };

    // Case filter.  Empty ONLY_CASES = run everything.  Comma separated,
    // matched against the case name, e.g. "B1_Si_grating_TM,C2_Au_holes".
    // A group letter also works: "A", "B", "C", "D".
    static string ONLY_CASES   = "";
    static string SKIP_CASES   = "";

    // Cost guard.  After a solve of a case exceeds this many seconds, the
    // larger orders of THAT case are skipped (the rest of the battery keeps
    // running).  0 = no limit.  2D at m = 30 means 61*61 = 3721 orders, i.e. a
    // ~7400 x 7400 eigenproblem -- that one is hours, not minutes.
    static double MAX_SECONDS_PER_SOLVE = 1800.0;

    // FFT sampling of the unit cell (2D only, see header).
    //   0 = fixed refinement factor FFT_REFINEMENT
    //   1 = pick the refinement per run so the absolute grid stays about
    //       FFT_TARGET_SAMPLES (this is what grcwa does with NX_2D = 256)
    static int    FFT_MODE            = 1;
    const  int    FFT_REFINEMENT      = 5;
    const  int    FFT_TARGET_SAMPLES  = 256;
    const  int    FFT_REFINEMENT_MIN  = 2;
    const  int    FFT_REFINEMENT_MAX  = 200;

    // See the header note on the circular atom constructor.
    static bool   CIRCLE_ARG_IS_RADIUS = false;

    // Show a side view of every structure before solving (visual check that
    // the geometry really is what you meant).  Costs a few clicks, saves hours.
    static bool   SHOW_STRUCTURES = false;
    // ... and stop right after showing them, without solving anything.
    static bool   DRY_RUN         = false;

    // Skip (case, order) pairs already present in the CSV, so an aborted sweep
    // can simply be started again.
    static bool   RESUME          = true;

    // Incidence.  The battery is a normal incidence battery.
    const  double WAVELENGTH   = 1.0;    // um
    const  double AOI          = 0.0;    // deg
    const  double CONICAL      = 0.0;    // deg

    // Cache handed to Rcwa (bytes).  0 is fine, the geometry changes every run.
    const  long   RCWA_CACHE   = 0;

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
            // dutyCycle is the remaining fraction of barMaterial == ff.
            Layer layer = new Layer(c.Depth, c.Hi(), c.Ff, c.Lo());
            grating.AddLayerOnBottom(layer);
        }
        else
        {
            // background layer carrying one centred atom
            Atom[] atoms = new Atom[1];
            if (c.Shape == "circle")
            {
                // see header: the third argument is the full width, not r
                double arg = CIRCLE_ARG_IS_RADIUS ? c.Radius : 2.0 * c.Radius;
                atoms[0] = new Atom(0.5, 0.5, arg, c.Hi());
            }
            else
            {
                // Atom widths/positions are relative to the period (0..1)
                atoms[0] = new Atom(0.5, 0.5,
                                    c.Ax / c.Period, c.Ay / c.Period, c.Hi());
            }
            Layer layer = new Layer(c.Depth, c.Lo(), 1, atoms);
            layer.Declare2D();          // force the 2D treatment of this layer
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

    // Total efficiency of a half space: sum over every propagating order.
    // Evanescent orders carry no flux, so restricting the sum to the
    // propagating window is exact -- and much cheaper than asking Moose for
    // all 2m+1 orders when m is 500.  The energy balance R+T+A printed for
    // every run is the check that nothing was missed.
    static double SumEfficiency(Rcwa solver, char tr, int mx, int my)
    {
        double sum = 0.0;
        for (int ox = -mx; ox <= mx; ox++)
        {
            for (int oy = -my; oy <= my; oy++)
            {
                try
                {
                    sum += solver.GetEfficiencyForGivenOrder(tr, ox, oy);
                }
                catch (Exception)
                {
                    // an order Moose does not know about contributes nothing
                }
            }
        }
        return sum;
    }

    static int RefinementFor(BenchCase c, int m)
    {
        if (c.Dim != 2) return FFT_REFINEMENT;
        if (FFT_MODE == 0) return FFT_REFINEMENT;
        int q = 2 * m + 1;
        int refinement = (int)Math.Ceiling((double)FFT_TARGET_SAMPLES / (double)q);
        if (refinement < FFT_REFINEMENT_MIN) refinement = FFT_REFINEMENT_MIN;
        if (refinement > FFT_REFINEMENT_MAX) refinement = FFT_REFINEMENT_MAX;
        return refinement;
    }

    static double MemoryMb()
    {
        try
        {
            System.Diagnostics.Process p =
                System.Diagnostics.Process.GetCurrentProcess();
            p.Refresh();
            return (double)p.WorkingSet64 / (1024.0 * 1024.0);
        }
        catch (Exception) { return -1.0; }
    }

    static double PeakMemoryMb()
    {
        try
        {
            System.Diagnostics.Process p =
                System.Diagnostics.Process.GetCurrentProcess();
            p.Refresh();
            return (double)p.PeakWorkingSet64 / (1024.0 * 1024.0);
        }
        catch (Exception) { return -1.0; }
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

            r.R  = SumEfficiency(solver, 'r', mx_r, my_r);
            r.T  = SumEfficiency(solver, 't', mx_t, my_t);
            r.R0 = solver.GetEfficiencyForGivenOrder('r', 0, 0);
            r.T0 = solver.GetEfficiencyForGivenOrder('t', 0, 0);
            r.A  = solver.GetAbsorption();
            sw.Stop();
            r.TReap = sw.Elapsed.TotalSeconds;

            // GetAbsorption() is 1 - R_moose - T_moose, so this is 1 exactly
            // iff our order sums reproduce Moose's internal totals.
            r.Energy = r.R + r.T + r.A;
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

    const string CSV_HEADER =
        "case,group,dim,pol,column,m_moose,q,nG,fft_refinement,"
        + "R,T,A,R0,T0,energy,"
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
                if (f.Length < 23) continue;
                if (f[22] != "ok") continue;            // retry anything failed
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
                r.TSetup = ParseD(f[15]);
                r.TSolve = ParseD(f[16]);
                r.TReap  = ParseD(f[17]);
                r.TTotal = ParseD(f[18]);
                r.MemBefore = ParseD(f[19]);
                r.MemAfter  = ParseD(f[20]);
                r.MemPeak   = ParseD(f[21]);
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
        Io.output(" fft mode     : " + (FFT_MODE == 1
                  ? "constant grid ~" + FFT_TARGET_SAMPLES.ToString(INV) + " samples"
                  : "fixed refinement " + FFT_REFINEMENT.ToString(INV)));
        Io.output(" cpu cores    : " + Environment.ProcessorCount.ToString(INV));
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
        int total_runs = 0, ok_runs = 0;
        double t_wall = 0.0;
        System.Diagnostics.Stopwatch sw_all = System.Diagnostics.Stopwatch.StartNew();

        for (int stage = 0; stage < stages; stage++)
        {
            for (int i = 0; i < cases.Count; i++)
            {
                BenchCase c = cases[i];
                int[] sweep = (c.Dim == 2) ? SWEEP_2D : SWEEP_1D;
                if (stage >= sweep.Length) continue;
                if (exhausted[c.Name]) continue;
                int m = sweep[stage];

                string key = c.Name + "@" + m.ToString(INV);
                if (done.ContainsKey(key))
                {
                    Io.output("  skip (done)  " + Pad(c.Name, 26) + " m=" + m.ToString(INV));
                    continue;
                }

                BenchResult r = RunOne(c, m);
                total_runs++;
                if (r.Status == "ok") ok_runs++;
                per_case[c.Name].Add(r);

                string line;
                if (r.Status == "ok")
                {
                    line = "  " + Pad(c.Name, 26)
                        + " m=" + Pad(m.ToString(INV), 4)
                        + " nG=" + Pad(((long)r.NG).ToString(INV), 7)
                        + " R=" + Pad(F(r.R, 6), 10)
                        + " T=" + Pad(F(r.T, 6), 10)
                        + " A=" + Pad(F(r.A, 6), 10)
                        + " |1-E|=" + Pad(E(Math.Abs(1.0 - r.Energy)), 10)
                        + " solve=" + Pad(F(r.TSolve, 3) + "s", 10)
                        + " mem=" + F(r.MemAfter, 0) + "MB";
                    Io.output(line);
                }
                else
                {
                    line = "  " + Pad(c.Name, 26) + " m=" + Pad(m.ToString(INV), 4)
                        + " FAILED: " + r.Note;
                    Io.error(line);
                }

                if (csv != null)
                {
                    try { csv.WriteLine(CsvRow(c, r)); csv.Flush(); }
                    catch (Exception e) { Io.error("csv write failed: " + e.Message); }
                }
                if (log != null)
                {
                    try { log.WriteLine(line); log.Flush(); }
                    catch (Exception) { }
                }

                if (MAX_SECONDS_PER_SOLVE > 0.0 && r.TSolve > MAX_SECONDS_PER_SOLVE)
                {
                    exhausted[c.Name] = true;
                    Io.output("  -> " + c.Name + ": solve took "
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
        Io.output(" this session: " + ok_runs.ToString(INV) + " ok / "
                  + total_runs.ToString(INV) + " attempted, wall time "
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
