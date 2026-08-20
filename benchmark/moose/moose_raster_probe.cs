// script:  moose_raster_probe.cs
// purpose: answer the three questions that gate the rasterization fix
//          (benchmark/RASTERIZATION.md, "Step 0").  Everything the python side
//          can answer about the geometry has been answered; these three need
//          Moose itself.
//
// ---------------------------------------------------------------------------
// WHY THIS SCRIPT EXISTS
// ---------------------------------------------------------------------------
// benchmark/RASTERIZATION.md separates the geometry error into two channels:
//
//   1. SHAPE     the pixel image is not the nominal shape.  O(1/N), and exactly
//                zero when every boundary falls on a cell edge.
//   2. SAMPLING  the DFT of the samples is not the Fourier integral of the
//                pixel image.  For a cell image the two differ by exactly
//                sinc(m/N), i.e. O(1/N^2), on ANY grid, aligned or not.
//
// On the python side both are now measured and channel 2 is removable exactly.
// Moose is the one code whose grid we cannot reach into, so what it does has to
// be measured from the outside.  Three questions, one probe each:
//
// P1  CAN MOOSE BE HANDED AN EXPLICIT PERMITTIVITY GRID?
//     The API has  Layer(double rThickness, CaModel pEpsilonDistribution).  If
//     Moose consumes such a grid as given, then grcwa, Ikarus and Moose can all
//     be handed the SAME pixel image, the geometry drops out of the cross-code
//     comparison entirely -- including for the circle, which no grid renders
//     exactly -- and the pixel-image-to-nominal-shape gap becomes one common
//     quantity measured once instead of three that never cancel.
//     This is the highest-value unknown in the whole study.
//
// P2  IS THE REFINEMENT RESIDUAL 1/r^2 OR 1/r?
//     Refitting the three points already on record (C1, m = 10, nominal
//     geometry: 30 -> 0.398784, 50 -> 0.397764, 100 -> 0.397322) says 1/r^2,
//     clearly: max residual 5.2e-6 against 1.2e-4 for 1/r, and a two-point
//     Richardson with p = 2 from 30 and 100 predicts the measured point at 50
//     to 8e-6.  That exponent IS channel 2, so it also says Moose does the same
//     plain point-sample DFT as everyone else.  Three points and one fit is
//     thin; this measures four points per case and fits both exponents.
//     R(inf) matters directly: with p = 2 it is 0.397181 on C1, where the
//     1/refinement extrapolation in benchmark/README.md says 0.396618 -- a
//     3e-4 difference, well above the tolerance the study is aiming at.
//
// P3  IS THE GEOMETRY EXACT ON THE GRID MOOSE ACTUALLY USES?
//     A centred rectangle of relative width w is rendered exactly on N cells
//     iff w*N and (1-w)*N/2 are both integral.  Moose's grid is
//     refinement * (2m+1), and 2m+1 is always odd, so the condition is on the
//     refinement alone: a multiple of 5 for C1 (w = 0.6), of 10 for C1b
//     (w = 0.4), of 4 for C2 (w = 0.5) -- so a multiple of 20 serves all three,
//     i.e. 40, 60, 80, 100.  Note what that says about the sweep on record:
//     refinement 30 and 50 are fine for C1 and C1b but NOT for C2, whose
//     recorded values therefore carry a channel-1 error that C1's do not.
//     P3 checks that claim by counting pixels in the rendered layer.
//
// ---------------------------------------------------------------------------
// WHAT COMES BACK
// ---------------------------------------------------------------------------
// One CSV row per solve plus one per rendered grid, a log, and three verdict
// blocks on the console.  The interesting lines are:
//
//   P1  "roundtrip-eps" must reproduce "atom" to ~1e-9.  If it does, Moose
//       takes an explicit eps grid and the value convention is epsilon.
//       If "roundtrip-index" reproduces it instead, the convention is n+ik.
//       If NEITHER does, the constructor is not usable for this and P1 is
//       answered negatively -- which is also a result, and the fallback
//       (aligned grids + Richardson in the refinement) is what P2/P3 set up.
//   P1  "mask-narrow"/"mask-wide" must NOT reproduce "atom".  If they do, the
//       CaModel is being ignored and every other P1 row is meaningless.
//   P1  the two refinements per CaModel row say whether rRefinementFactorEpsFT
//       still resamples a grid that was handed in explicitly.
//   P1  "aniso" (wx != wy) catches a transposed or shifted grid convention,
//       which a C4v-symmetric pillar cannot.
//   P2  R_inf(p=2) vs R_inf(p=1) with their residuals, per case.
//   P3  cells vs w*N per (case, refinement, order).
//
// ---------------------------------------------------------------------------
// CONVENTIONS
// ---------------------------------------------------------------------------
// Identical to moose_convergence_bench.cs and moose_geometry_probe.cs --
// microns, max order m per axis (q = 2m+1 retained), polarization angle
// 0 = TM / 90 = TE, efficiencies in PERCENT (scaled by 0.01 here), R summed
// over the propagating orders in both output polarizations, and R + T + A
// checked against 1.
//
// The mask this script builds is bit-for-bit the rule proposed for
// structures.py v2 and implemented in benchmark/rasterization_study.py as
// rect_fill(..., rule="centre") / circle_fill(..., rule="centre"): sample the
// CELL CENTRE (i + 0.5)/N and take the pixel when the centre is inside the
// closed shape.  Any change here has to be made there too, or P1 stops
// comparing the same thing.
//
// IF IT DOES NOT COMPILE.  The one construction this script needs that no other
// script in benchmark/moose uses is
//     new Layer(double thickness, CaModel epsilonDistribution)
// which is transcribed from moose.qch into moose_api_stubs.cs but has never been
// exercised on a real build.  If Moose rejects that line, set RUN_CAMODEL =
// false and run the rest: P1 is then answered negatively, which is itself the
// answer, and P2 + P3 are exactly the fallback that answer needs.  Please paste
// the compiler message back either way -- an overload that exists with a
// different signature is a different outcome from one that does not exist.
//
// Runtime: P1 is ~60 solves at nG = 441, P2 is 24 solves at nG = 441...961,
// P3 does not solve at all.  Minutes, not hours, on a pool.
// ---------------------------------------------------------------------------
//
// P4  ADDED AFTER THE FIRST RUN.  P2 (the atom-path refinement fit) came back
//     inconclusive: R(refinement) is NON-MONOTONE in three of four cases, which
//     no single power law can produce.  P3 explains why -- Moose's Atom
//     rasterizer is one cell too wide on EVERY grid tested, aligned or not (see
//     RASTERIZATION.md Sec.9: e.g. N=840 renders 505 cells where the exact
//     width is 504, on every one of the 8 combinations checked, no exception).
//     P2 was therefore fitting a 1/N shape error and a 1/N^2 sampling error at
//     once.
//
//     P4 reruns the refinement sweep on the MASK path (P1's CaModel
//     construction, already confirmed exact and grid-resolution-independent),
//     in two variants:
//       "fixed"    the mask is rendered once per case at a FIXED resolution
//                  (c.CaRes[0]) and only the SOLVE's refinement changes -- a
//                  clean test of whether refinement resamples an explicit grid
//                  that never changes size.
//       "matched"  the mask is rendered AT N = refinement * (2m+1), i.e. the
//                  solve's own internal grid size -- so there is nothing left
//                  to resample.  If R is flat across refinement here, Moose's
//                  refinement dependence is entirely a resampling artifact of a
//                  MISMATCHED grid size, not a property of the solve itself --
//                  which would mean the CaModel path has NO remaining grid
//                  dependence at all, for axis-aligned geometry.
//
//     Runtime: up to 48 more solves, similar order of magnitude to P1+P2.
// ---------------------------------------------------------------------------

using System;
using System.IO;
using System.Threading;
using System.Collections.Generic;
using System.Globalization;


// ===========================================================================
//  one case of the battery
// ===========================================================================
public class ProbeCase
{
    public string Name;
    public string Pol;          // "TE" or "TM"
    public string Shape;        // "rect" or "circle"
    public double Period;       // um
    public double Depth;        // um
    public double WNom;         // rect: atom width / period;  circle: radius / period
    public double HNom;         // rect: atom height / period (= WNom except for the
                                // deliberately anisotropic control case)
    public double PillarN, PillarK;   // the atom
    public double BgN, BgK;           // the background of the patterned layer
    public double SubN, SubK;         // substrate half space
    public int[]  CaRes;        // CaModel resolutions for P1 (exact grids where
                                // the shape has any)

    public Materials Pillar() { return new Materials(PillarN, PillarK); }
    public Materials Bg()     { return new Materials(BgN, BgK); }
    public Materials Sub()    { return new Materials(SubN, SubK); }
}


// one queued solve
public class Job
{
    public ProbeCase C;
    public int    M;
    public int    Refinement;
    public string Probe;     // "camodel" | "refinement"
    public string Source;    // see BuildFor()
    public int    Res;       // CaModel resolution, 0 = built from an Atom
    public double W, H;      // the atom size the geometry is built at
    public string Tag;
}


public class ProbeResult
{
    public string Case;
    public string Probe;
    public string Source;
    public string Tag;
    public int    M;
    public int    Q;
    public long   NG;
    public int    Res;
    public double W, H;
    public int    Refinement;
    public double R, T, A, Energy;
    public double Fill;       // fill fraction of the CaModel that was handed in
    public int    RowPx;      // atom cells along its centre row
    public string Status = "ok";
    public string Note = "";
    public double TSolve;
}


public class MooseScript
{
    // =======================================================================
    //  CONFIGURATION
    // =======================================================================

    // Where the CSV and the log end up.  Empty -> <temp>/moose_bench.
    static string OUTPUT_DIR = "C:\\Users\\mwalther\\PycharmProjects\\grcwa\\benchmark\\moose";

    // Which probes to run.
    static bool RUN_CAMODEL    = true;    // P1
    static bool RUN_REFINEMENT = true;    // P2
    static bool RUN_ALIGNMENT  = true;    // P3  (renders only, no solving)
    static bool RUN_MASK_REFINEMENT = true;  // P4  (added after the first run)

    // Max orders m for the solving probes (q = 2m+1 retained per axis).
    // m = 10 -> q = 21 -> nG = 441 is what the python tables in
    // benchmark/RASTERIZATION.md are taken at, so keep it in the list.
    static readonly int[] ORDERS = { 10, 15 };

    // P2: the refinements to fit over.  MULTIPLES OF 20 on purpose -- see the
    // header, P3.  Moose clamps rRefinementFactorEpsFT to [30, 100]; 20 itself
    // would be silently raised to 30 and is therefore not in the list.
    static readonly int[] REFINEMENTS = { 40, 60, 80, 100 };

    // P1: the refinement pair each CaModel row is solved at, to see whether
    // refinement still resamples a grid that was handed in explicitly.
    static readonly int[] CAMODEL_REFINEMENTS = { 40, 100 };

    // P3: how big a rendered grid we are willing to walk.  A full scan is
    // nx*ny GetValue calls; the centre row alone is nx.  Above this the script
    // reads the centre row only and reports levels as -1.
    const long FULL_SCAN_MAX = 1000000L;

    // P3: refusing to allocate the very biggest grids (100 * 61 = 6100 per axis
    // is 37 M complex values).  Rows above this are reported as skipped.
    const int RENDER_MAX_N = 4200;

    // Write every rendered grid to its own CSV (real part, plus imaginary when
    // it is not zero).  260^2 is ~1.5 MB per file; the 2000+ grids of P3 are
    // not something to switch on casually.  Off by default.
    static bool DUMP_GRIDS = false;

    // Case filter: empty = all.  Comma separated names.
    static string ONLY_CASES = "";

    // Concurrent solves.  A Moose solve is single threaded; different solves are
    // independent (moose_convergence_bench.cs's PARALLEL_SELFTEST proved that on
    // this build).  1 = sequential.  0 = one per core.
    static int PARALLEL_TASKS = 6;

    // Incidence -- the battery is a normal incidence battery at lambda = 1 um.
    const double WAVELENGTH = 1.0;
    const double AOI        = 0.0;
    const double CONICAL    = 0.0;
    const long   RCWA_CACHE = 0;
    const double SUPER_N    = 1.0;

    // Efficiencies come back in percent; the rest of the benchmark is fractions.
    const double SCALE = 0.01;
    const double ENERGY_TOL = 1.0e-6;

    // How close two solves have to be before this script calls them the same
    // structure.  A CaModel round trip that works should be bit-identical; 1e-9
    // leaves room for a different code path summing the same orders.
    const double SAME_TOL = 1.0e-9;

    static readonly CultureInfo INV = CultureInfo.InvariantCulture;

    // materials at lambda = 1 um (n, k)
    const double AIR_N = 1.0, AIR_K = 0.0;
    const double SIO2_N = 1.5, SIO2_K = 0.0;
    const double SI_N = 3.5, SI_K = 0.0;
    const double AU_N = 0.3, AU_K = 7.0;


    // =======================================================================
    //  the cases, 1:1 with benchmark/structures.py
    // =======================================================================
    // CaRes: resolutions at which the CaModel is built for P1.  For a centred
    // rectangle of relative width w these must satisfy "w*N and (1-w)*N/2
    // integral" or the grid itself introduces a shape error and P1 stops being
    // a clean test of the plumbing.  Multiples of 20 satisfy all three rect
    // cases at once (260 = 20*13 is the smallest at least 256).  A circle has
    // no exact grid at all, so D2 just gets two powers of two.
    static List<ProbeCase> BuildCases()
    {
        List<ProbeCase> cases = new List<ProbeCase>();
        ProbeCase c;

        // C1: Si square pillars, period 0.5, ax = ay = 0.3 -> w = 0.6, on glass.
        c = new ProbeCase();
        c.Name = "C1_Si_pillars"; c.Pol = "TE"; c.Shape = "rect";
        c.Period = 0.50; c.Depth = 0.40; c.WNom = 0.600000; c.HNom = 0.600000;
        c.PillarN = SI_N; c.PillarK = SI_K; c.BgN = AIR_N; c.BgK = AIR_K;
        c.SubN = SIO2_N; c.SubK = SIO2_K;
        c.CaRes = new int[] { 260, 520 };
        cases.Add(c);

        // C1b: the same pillars at period 1.5 (diffractive), w = 0.4.
        c = new ProbeCase();
        c.Name = "C1b_Si_pillars_diffract"; c.Pol = "TE"; c.Shape = "rect";
        c.Period = 1.50; c.Depth = 0.40; c.WNom = 0.400000; c.HNom = 0.400000;
        c.PillarN = SI_N; c.PillarK = SI_K; c.BgN = AIR_N; c.BgK = AIR_K;
        c.SubN = SIO2_N; c.SubK = SIO2_K;
        c.CaRes = new int[] { 260, 520 };
        cases.Add(c);

        // C2: air holes in gold, period 0.6, w = 0.5, on glass.  The hole is
        // the atom, the background is gold.
        c = new ProbeCase();
        c.Name = "C2_Au_holes"; c.Pol = "TE"; c.Shape = "rect";
        c.Period = 0.60; c.Depth = 0.20; c.WNom = 0.500000; c.HNom = 0.500000;
        c.PillarN = AIR_N; c.PillarK = AIR_K;
        c.BgN = AU_N; c.BgK = AU_K;
        c.SubN = SIO2_N; c.SubK = SIO2_K;
        c.CaRes = new int[] { 260, 520 };
        cases.Add(c);

        // D2: free-standing Si cylinder, radius 0.30 of the period.
        c = new ProbeCase();
        c.Name = "D2_ikarus_cylinder_TE"; c.Pol = "TE"; c.Shape = "circle";
        c.Period = 400.0 / 700.0; c.Depth = 200.0 / 700.0;
        c.WNom = 0.300000; c.HNom = 0.300000;
        c.PillarN = SI_N; c.PillarK = SI_K; c.BgN = AIR_N; c.BgK = AIR_K;
        c.SubN = AIR_N; c.SubK = AIR_K;
        c.CaRes = new int[] { 256, 512 };
        cases.Add(c);

        // AN: NOT part of the battery.  A deliberately anisotropic pillar
        // (0.6 x 0.4) whose only job is to catch a transposed or shifted
        // CaModel index convention -- something none of the four C4v-symmetric
        // cases above can do, because for them a transpose is invisible.
        c = new ProbeCase();
        c.Name = "AN_aniso_control"; c.Pol = "TE"; c.Shape = "rect";
        c.Period = 0.50; c.Depth = 0.40; c.WNom = 0.600000; c.HNom = 0.400000;
        c.PillarN = SI_N; c.PillarK = SI_K; c.BgN = AIR_N; c.BgK = AIR_K;
        c.SubN = SIO2_N; c.SubK = SIO2_K;
        c.CaRes = new int[] { 260 };
        cases.Add(c);

        return cases;
    }


    // =======================================================================
    //  geometry -- from an Atom, and from an explicit grid
    // =======================================================================
    static GratingStructure NewGrating(ProbeCase c)
    {
        Materials superstrate = new Materials(SUPER_N, 0.0);
        return new GratingStructure(c.Period, c.Period, superstrate, c.Sub());
    }

    // The reference construction: Moose draws the shape itself, as
    // moose_convergence_bench.cs does.
    static GratingStructure AtomGrating(ProbeCase c, double w, double h,
                                        out Layer patterned)
    {
        GratingStructure grating = NewGrating(c);
        Atom[] atoms = new Atom[1];
        if (c.Shape == "circle")
            atoms[0] = new Atom(0.5, 0.5, w, c.Pillar());   // 3rd arg is the RADIUS
        else
            atoms[0] = new Atom(0.5, 0.5, w, h, c.Pillar());

        Layer layer = new Layer(c.Depth, c.Bg(), 1, atoms);
        layer.Declare2D();
        grating.AddLayerOnBottom(layer);
        patterned = layer;
        return grating;
    }

    // The construction under test: the patterned layer IS an explicit grid.
    static GratingStructure CaGrating(ProbeCase c, CaModel model, out Layer patterned)
    {
        GratingStructure grating = NewGrating(c);
        Layer layer = new Layer(c.Depth, model);
        // Declare2D is what an Atom layer needs to be treated as crossed rather
        // than lamellar.  Whether a layer that IS a 2D grid needs it, or even
        // accepts it, is undocumented -- so try, and carry on if it refuses.
        try { layer.Declare2D(); } catch (Exception) { }
        grating.AddLayerOnBottom(layer);
        patterned = layer;
        return grating;
    }

    // Principal square root with a NON-NEGATIVE imaginary part, i.e. the
    // exp(-i w t) convention the whole battery uses (absorbers have k > 0).
    static void CSqrt(double re, double im, out double sre, out double sim)
    {
        double r = Math.Sqrt(re * re + im * im);
        double a = Math.Sqrt(0.5 * (r + re));
        double b = Math.Sqrt(0.5 * (r - re));
        if (im < 0.0) b = -b;
        sre = a; sim = b;
        if (sim < 0.0) { sre = -sre; sim = -sim; }
    }

    // Is the cell centre (i+0.5)/n inside the shape?  This is bit-for-bit
    // rect_fill(rule="centre") / circle_fill(rule="centre") of
    // benchmark/rasterization_study.py -- keep the two in step.
    static bool CellInside(ProbeCase c, int n, int i, int j, double w, double h)
    {
        double x = (i + 0.5) / n - 0.5;
        double y = (j + 0.5) / n - 0.5;
        if (c.Shape == "circle")
            return x * x + y * y <= w * w + 1.0e-12;
        return Math.Abs(x) <= w / 2.0 + 1.0e-12 && Math.Abs(y) <= h / 2.0 + 1.0e-12;
    }

    // The mask as an explicit permittivity (or index) grid.
    static CaModel MaskCaModel(ProbeCase c, int n, double w, double h, bool asIndex,
                               out double fill, out int rowpx)
    {
        Complex ea = c.Pillar().GetEpsilon(WAVELENGTH);
        Complex eb = c.Bg().GetEpsilon(WAVELENGTH);
        double are = ea.Re(), aim = ea.Im(), bre = eb.Re(), bim = eb.Im();
        if (asIndex)
        {
            double t1, t2;
            CSqrt(are, aim, out t1, out t2); are = t1; aim = t2;
            CSqrt(bre, bim, out t1, out t2); bre = t1; bim = t2;
        }

        CaModel model = new CaModel(n, n);
        long inside = 0;
        rowpx = 0;
        int jc = n / 2;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                bool hit = CellInside(c, n, i, j, w, h);
                if (hit) inside++;
                if (hit && j == jc) rowpx++;
                model.SetValue(i, j, hit ? new Complex(are, aim)
                                         : new Complex(bre, bim));
            }
        }
        fill = (double)inside / ((double)n * (double)n);
        return model;
    }

    // A copy of a rendered grid, optionally converted from eps to n+ik.
    static CaModel CopyCaModel(CaModel src, bool toIndex, out double fill,
                               out int rowpx, double target_re)
    {
        int nx = src.GetDimX(), ny = src.GetDimY();
        CaModel model = new CaModel(nx, ny);
        long inside = 0;
        rowpx = 0;
        int jc = ny / 2;
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                Complex v = src.GetValue(i, j);
                double re = v.Re(), im = v.Im();
                if (Math.Abs(re - target_re) < 1.0e-9)
                {
                    inside++;
                    if (j == jc) rowpx++;
                }
                if (toIndex)
                {
                    double t1, t2;
                    CSqrt(re, im, out t1, out t2);
                    re = t1; im = t2;
                }
                model.SetValue(i, j, new Complex(re, im));
            }
        }
        fill = (double)inside / ((double)nx * (double)ny);
        return model;
    }


    // =======================================================================
    //  solving
    // =======================================================================
    static int PropagatingOrders(double n_medium, double period)
    {
        if (period <= 0.0) return 0;
        double v = n_medium * period / WAVELENGTH;
        return (int)Math.Floor(v + 1.0e-9);
    }

    static double Eff(Rcwa solver, char tr, int ox, int oy, string pol)
    {
        try { return solver.GetEfficiencyForGivenOrder(tr, ox, oy, pol); }
        catch (Exception) { return 0.0; }
    }

    // Sum over the propagating window, in both output polarizations.  On a 2D
    // lattice the default "in" reading drops the cross-polarized half of the
    // off-axis orders -- see the header of moose_convergence_bench.cs.
    static double SumEfficiency(Rcwa solver, char tr, int mx, int my)
    {
        double s = 0.0;
        for (int ox = -mx; ox <= mx; ox++)
            for (int oy = -my; oy <= my; oy++)
                s += Eff(solver, tr, ox, oy, "TE") + Eff(solver, tr, ox, oy, "TM");
        return s;
    }

    // Build the grating this job asks for.  The CaModel handed to a Layer is
    // deliberately NOT deleted afterwards: whether Layer takes ownership of it
    // is undocumented, and a double delete would take the process down in the
    // middle of a run.  The probe is short and the grids are small.
    static GratingStructure BuildFor(Job jb, out double fill, out int rowpx,
                                     out string note)
    {
        ProbeCase c = jb.C;
        Layer patterned;
        fill = Double.NaN; rowpx = -1; note = "";

        if (jb.Source == "atom")
            return AtomGrating(c, jb.W, jb.H, out patterned);

        if (jb.Source == "mask-eps" || jb.Source == "mask-index")
        {
            CaModel m = MaskCaModel(c, jb.Res, jb.W, jb.H,
                                    jb.Source == "mask-index", out fill, out rowpx);
            return CaGrating(c, m, out patterned);
        }

        if (jb.Source == "roundtrip-eps" || jb.Source == "roundtrip-index")
        {
            GratingStructure src = AtomGrating(c, jb.W, jb.H, out patterned);
            CaModel dump = patterned.GetEpsilonDistributionsAsCaModel(
                WAVELENGTH, jb.Res, jb.Res, false);
            if (dump == null)
            {
                note = "GetEpsilonDistributionsAsCaModel returned null";
                try { src.Delete(); } catch (Exception) { }
                return null;
            }
            double target = c.Pillar().GetEpsilon(WAVELENGTH).Re();
            CaModel m = CopyCaModel(dump, jb.Source == "roundtrip-index",
                                    out fill, out rowpx, target);
            try { dump.Delete(); } catch (Exception) { }
            try { src.Delete(); } catch (Exception) { }
            return CaGrating(c, m, out patterned);
        }

        note = "unknown source " + jb.Source;
        return null;
    }

    static ProbeResult RunOne(Job jb)
    {
        ProbeCase c = jb.C;
        ProbeResult r = new ProbeResult();
        r.Case = c.Name; r.Probe = jb.Probe; r.Source = jb.Source; r.Tag = jb.Tag;
        r.M = jb.M; r.Q = 2 * jb.M + 1; r.NG = (long)r.Q * (long)r.Q;
        r.Res = jb.Res; r.W = jb.W; r.H = jb.H; r.Refinement = jb.Refinement;
        r.Fill = Double.NaN; r.RowPx = -1;

        GratingStructure grating = null;
        Rcwa solver = null;
        double pol_angle = (c.Pol == "TM") ? 0.0 : 90.0;

        try
        {
            double fill; int rowpx; string note;
            grating = BuildFor(jb, out fill, out rowpx, out note);
            r.Fill = fill; r.RowPx = rowpx;
            if (grating == null)
            {
                r.Status = "failed";
                r.Note = (note.Length > 0) ? note : "grating build returned null";
                return r;
            }

            solver = new Rcwa(grating, jb.M, jb.M, jb.Refinement, RCWA_CACHE);
            System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
            solver.Calc(WAVELENGTH, AOI, CONICAL, pol_angle, true);
            sw.Stop();
            r.TSolve = sw.Elapsed.TotalSeconds;

            int mr = Math.Min(jb.M, PropagatingOrders(SUPER_N, c.Period));
            int mt = Math.Min(jb.M, PropagatingOrders(c.SubN, c.Period));
            r.R = SumEfficiency(solver, 'r', mr, mr) * SCALE;
            r.T = SumEfficiency(solver, 't', mt, mt) * SCALE;
            r.A = solver.GetAbsorption() * SCALE;
            r.Energy = r.R + r.T + r.A;
            if (Math.Abs(1.0 - r.Energy) > ENERGY_TOL)
            {
                r.Status = "energy";
                r.Note = "R+T+A off by " + E(Math.Abs(1.0 - r.Energy));
            }
        }
        catch (Exception e)
        {
            r.Status = "failed";
            r.Note = e.Message.Replace("\n", " ").Replace(",", ";");
        }

        try { if (solver != null) solver.Delete(); } catch (Exception) { }
        try { if (grating != null) grating.Delete(); } catch (Exception) { }
        return r;
    }


    // =======================================================================
    //  worker pool (same pattern as moose_geometry_probe.cs)
    // =======================================================================
    static readonly object sLock = new object();
    static List<Job> sQ = new List<Job>();
    static int sNext;
    static List<ProbeResult> sOut;
    static StreamWriter sCsv;
    static StreamWriter sLog;

    static void Worker()
    {
        while (true)
        {
            Job jb;
            lock (sLock)
            {
                if (sNext >= sQ.Count) return;
                jb = sQ[sNext];
                sNext++;
            }
            ProbeResult r = RunOne(jb);
            lock (sLock)
            {
                sOut.Add(r);
                string line = Line(r);
                if (r.Status == "ok") Io.output(line);
                else Io.error(line + "   <-- " + r.Status + " " + r.Note);
                if (sCsv != null)
                { try { sCsv.WriteLine(CsvRow(r)); sCsv.Flush(); } catch (Exception) { } }
                if (sLog != null)
                { try { sLog.WriteLine(line); sLog.Flush(); } catch (Exception) { } }
            }
        }
    }

    static void Enqueue(ProbeCase c, int m, int refi, string probe, string source,
                        int res, double w, double h, string tag)
    {
        Job jb = new Job();
        jb.C = c; jb.M = m; jb.Refinement = refi; jb.Probe = probe;
        jb.Source = source; jb.Res = res; jb.W = w; jb.H = h; jb.Tag = tag;
        sQ.Add(jb);
    }

    static void RunQueue()
    {
        sNext = 0;
        int workers = PARALLEL_TASKS;
        if (workers == 0) workers = Environment.ProcessorCount;
        if (workers < 1) workers = 1;
        if (workers > sQ.Count) workers = sQ.Count;
        if (workers <= 1) { Worker(); }
        else
        {
            Thread[] pool = new Thread[workers];
            for (int i = 0; i < workers; i++)
            {
                pool[i] = new Thread(new ThreadStart(Worker));
                pool[i].IsBackground = false;
                pool[i].Start();
            }
            for (int i = 0; i < workers; i++) pool[i].Join();
        }
        sQ = new List<Job>();
    }


    // =======================================================================
    //  P3 -- what does Moose render on the grid it actually uses?
    // =======================================================================
    // The grid of a solve is rRefinementFactorEpsFT * (2m+1) samples per axis.
    // This renders the patterned layer at exactly that size and counts the atom
    // cells along the centre row.  For a centred rectangle the exact count is
    // w*N, an integer precisely when the refinement is a multiple of 5 (C1),
    // 10 (C1b) or 4 (C2) -- 2m+1 being odd never helps.
    //
    // CAVEAT, the same one moose_geometry_probe.cs carries: this renders the
    // layer for INSPECTION.  It is not proof of what the solver's own FFT grid
    // does.  A renderer and a solver that disagreed about the geometry would
    // itself be worth knowing, and P2's residual is the independent check.
    static void Alignment(List<ProbeCase> cases, StreamWriter csv, string dir)
    {
        Io.output("");
        Io.output("=================================================================");
        Io.output(" P3 -- geometry on the grid a solve actually uses");
        Io.output("      N = refinement * (2m+1);  exact needs w*N integral");
        Io.output("=================================================================");
        Io.output(" " + Pad("case", 26) + Pad("m", 5) + Pad("fft", 6) + Pad("N", 7)
                  + Pad("cells", 8) + Pad("w*N exact", 12) + Pad("w_eff", 11)
                  + Pad("levels", 8) + "verdict");

        for (int i = 0; i < cases.Count; i++)
        {
            ProbeCase c = cases[i];
            for (int k = 0; k < ORDERS.Length; k++)
            {
                for (int j = 0; j < REFINEMENTS.Length; j++)
                {
                    int m = ORDERS[k];
                    int refi = REFINEMENTS[j];
                    int n = refi * (2 * m + 1);
                    if (n > RENDER_MAX_N)
                    {
                        Io.output(" " + Pad(c.Name, 26) + Pad(m.ToString(INV), 5)
                                  + Pad(refi.ToString(INV), 6) + Pad(n.ToString(INV), 7)
                                  + "skipped (over RENDER_MAX_N = "
                                  + RENDER_MAX_N.ToString(INV) + ")");
                        continue;
                    }

                    Layer layer = null;
                    GratingStructure g = null;
                    try
                    {
                        g = AtomGrating(c, c.WNom, c.HNom, out layer);
                        CaModel model = layer.GetEpsilonDistributionsAsCaModel(
                            WAVELENGTH, n, n, false);
                        if (model == null)
                        {
                            Io.error("  " + Pad(c.Name, 26) + " m=" + m.ToString(INV)
                                     + " fft=" + refi.ToString(INV) + " N=" + n.ToString(INV)
                                     + ": GetEpsilonDistributionsAsCaModel returned null");
                            continue;
                        }
                        int nx = model.GetDimX(), ny = model.GetDimY();
                        double target = c.Pillar().GetEpsilon(WAVELENGTH).Re();

                        int rowpx = 0;
                        int jc = ny / 2;
                        for (int x = 0; x < nx; x++)
                            if (Math.Abs(model.GetValue(x, jc).Re() - target) < 1.0e-9)
                                rowpx++;

                        // The distinct values, not just how many: they say whether
                        // the grid carries eps or n+ik and which sign the loss has,
                        // without solving anything.  Si eps = 12.25, air 1, gold
                        // -48.91 + 4.2i; as an index they would read 3.5, 1, 0.3+7i.
                        int levels = -1;
                        List<double> lvRe = new List<double>();
                        List<double> lvIm = new List<double>();
                        if ((long)nx * (long)ny <= FULL_SCAN_MAX)
                        {
                            for (int x = 0; x < nx; x++)
                                for (int y = 0; y < ny; y++)
                                {
                                    Complex v = model.GetValue(x, y);
                                    double re = v.Re(), im = v.Im();
                                    bool known = false;
                                    for (int L = 0; L < lvRe.Count; L++)
                                        if (Math.Abs(lvRe[L] - re) < 1.0e-9
                                            && Math.Abs(lvIm[L] - im) < 1.0e-9) known = true;
                                    if (!known && lvRe.Count < 16)
                                    { lvRe.Add(re); lvIm.Add(im); }
                                }
                            levels = lvRe.Count;
                        }

                        double want = c.WNom * nx;
                        bool integral = Math.Abs(want - Math.Round(want)) < 1.0e-9;
                        double w_eff = (nx > 0) ? (double)rowpx / nx : 0.0;
                        string verdict;
                        if (c.Shape == "circle")
                            verdict = "circle -- no exact grid exists";
                        else if (!integral)
                            verdict = "w*N = " + F(want, 3) + " NOT integral -> shape error";
                        else if (Math.Abs(rowpx - want) < 0.5)
                            verdict = "exact";
                        else
                            verdict = "off by " + (rowpx - (int)Math.Round(want)).ToString(INV)
                                      + " cells";

                        Io.output(" " + Pad(c.Name, 26) + Pad(m.ToString(INV), 5)
                                  + Pad(refi.ToString(INV), 6) + Pad(nx.ToString(INV), 7)
                                  + Pad(rowpx.ToString(INV), 8) + Pad(F(want, 2), 12)
                                  + Pad(F(w_eff, 6), 11)
                                  + Pad(levels < 0 ? "-" : levels.ToString(INV), 8)
                                  + verdict);
                        if (csv != null)
                        {
                            // columns: case,probe,source,tag,m,q,nG,ca_res,
                            // fft_refinement,w,h, R,T,A,energy (empty -- nothing
                            // was solved), fill,row_px, t_solve (empty), status,note
                            csv.WriteLine(c.Name + ",alignment,render,,"
                                + m.ToString(INV) + "," + (2 * m + 1).ToString(INV) + ",,"
                                + nx.ToString(INV) + "," + refi.ToString(INV) + ","
                                + G(c.WNom) + "," + G(c.HNom) + ",,,,,"
                                + G(w_eff) + "," + rowpx.ToString(INV) + ",,ok,"
                                + verdict.Replace(",", ";"));
                            csv.Flush();
                        }
                        if (levels > 0)
                        {
                            string vals = "";
                            for (int L = 0; L < lvRe.Count; L++)
                            {
                                if (L > 0) vals += " | ";
                                vals += F(lvRe[L], 6);
                                if (Math.Abs(lvIm[L]) > 1.0e-12)
                                    vals += (lvIm[L] < 0 ? " - " : " + ")
                                            + F(Math.Abs(lvIm[L]), 6) + "i";
                            }
                            Io.output("      values: " + vals);
                        }
                        if (DUMP_GRIDS) DumpGrid(model, dir, c.Name, nx);
                        model.Delete();
                    }
                    catch (Exception e)
                    {
                        Io.error("  " + Pad(c.Name, 26) + " m=" + m.ToString(INV)
                                 + " fft=" + refi.ToString(INV) + " N=" + n.ToString(INV)
                                 + " FAILED: " + e.Message);
                    }
                    try { if (g != null) g.Delete(); } catch (Exception) { }
                }
            }
        }
    }

    static void DumpGrid(CaModel model, string dir, string name, int n)
    {
        try
        {
            string path = Path.Combine(dir, "moose_grid_" + name + "_"
                                       + n.ToString(INV) + ".csv");
            StreamWriter w = new StreamWriter(path, false);
            int nx = model.GetDimX(), ny = model.GetDimY();
            for (int i = 0; i < nx; i++)
            {
                string line = "";
                for (int j = 0; j < ny; j++)
                {
                    if (j > 0) line += ",";
                    line += G(model.GetValue(i, j).Re());
                }
                w.WriteLine(line);
            }
            w.Close();
        }
        catch (Exception e) { Io.error("grid dump failed: " + e.Message); }
    }


    // =======================================================================
    //  least squares on  R(r) = a + b * r^-p
    // =======================================================================
    static bool FitPower(List<double> rs, List<double> Rs, double p,
                         out double a, out double b, out double maxres)
    {
        a = Double.NaN; b = Double.NaN; maxres = Double.NaN;
        int n = rs.Count;
        if (n < 2) return false;
        double s0 = n, s1 = 0.0, s2 = 0.0, t0 = 0.0, t1 = 0.0;
        for (int i = 0; i < n; i++)
        {
            double x = Math.Pow(rs[i], -p);
            s1 += x; s2 += x * x; t0 += Rs[i]; t1 += Rs[i] * x;
        }
        double det = s0 * s2 - s1 * s1;
        if (Math.Abs(det) < 1.0e-300) return false;
        a = (t0 * s2 - t1 * s1) / det;
        b = (s0 * t1 - s1 * t0) / det;
        maxres = 0.0;
        for (int i = 0; i < n; i++)
        {
            double d = Math.Abs(a + b * Math.Pow(rs[i], -p) - Rs[i]);
            if (d > maxres) maxres = d;
        }
        return true;
    }


    // =======================================================================
    //  output helpers
    // =======================================================================
    static string F(double v, int digits)
    { return v.ToString("F" + digits.ToString(INV), INV); }

    static string G(double v) { return v.ToString("R", INV); }

    static string E(double v) { return v.ToString("0.0e+00", INV); }

    static string Pad(string s, int n)
    {
        while (s.Length < n) s = s + " ";
        return s;
    }

    static string Line(ProbeResult r)
    {
        return " " + Pad(r.Case, 26) + Pad(r.Source, 17) + Pad("(" + r.Tag + ")", 11)
             + " m=" + Pad(r.M.ToString(INV), 4)
             + " res=" + Pad(r.Res > 0 ? r.Res.ToString(INV) : "-", 6)
             + " fft=" + Pad(r.Refinement.ToString(INV), 5)
             + " R=" + Pad(F(r.R, 9), 13)
             + " T=" + Pad(F(r.T, 6), 10)
             + " |1-E|=" + Pad(E(Math.Abs(1.0 - r.Energy)), 10)
             + " t=" + F(r.TSolve, 2) + "s";
    }

    const string CSV_HEADER =
        "case,probe,source,tag,m,q,nG,ca_res,fft_refinement,w,h,"
        + "R,T,A,energy,fill,row_px,t_solve_s,status,note";

    static string CsvRow(ProbeResult r)
    {
        return r.Case + "," + r.Probe + "," + r.Source + "," + r.Tag + ","
            + r.M.ToString(INV) + "," + r.Q.ToString(INV) + "," + r.NG.ToString(INV) + ","
            + r.Res.ToString(INV) + "," + r.Refinement.ToString(INV) + ","
            + G(r.W) + "," + G(r.H) + ","
            + G(r.R) + "," + G(r.T) + "," + G(r.A) + "," + G(r.Energy) + ","
            + (Double.IsNaN(r.Fill) ? "" : G(r.Fill)) + ","
            + (r.RowPx < 0 ? "" : r.RowPx.ToString(INV)) + ","
            + F(r.TSolve, 3) + "," + r.Status + "," + r.Note;
    }

    static ProbeResult Find(string name, string source, string tag, int res,
                            int m, int refi)
    {
        for (int i = 0; i < sOut.Count; i++)
        {
            ProbeResult r = sOut[i];
            if (r.Case != name || r.Source != source || r.Tag != tag) continue;
            if (r.Res != res || r.M != m || r.Refinement != refi) continue;
            return r;
        }
        return null;
    }

    static string Cmp(ProbeResult a, ProbeResult b, string same, string diff)
    {
        if (a == null || b == null) return "missing";
        if (a.Status == "failed" || b.Status == "failed")
            return "FAILED: " + (a.Status == "failed" ? a.Note : b.Note);
        double d = Math.Abs(a.R - b.R);
        string s = (d <= SAME_TOL ? same : diff) + "  (|dR| = " + E(d) + ")";
        // A CaModel layer that solves but does not conserve energy is not a
        // working construction, however close its R happens to land.
        if (a.Status != "ok" || b.Status != "ok")
            s += "   [!! " + (a.Status != "ok" ? a.Status : b.Status) + "]";
        return s;
    }


    // =======================================================================
    //  main
    // =======================================================================
    static void Main()
    {
        List<ProbeCase> all = BuildCases();
        List<ProbeCase> cases = new List<ProbeCase>();
        for (int i = 0; i < all.Count; i++)
        {
            if (ONLY_CASES.Length > 0)
            {
                bool hit = false;
                string[] want = ONLY_CASES.Split(',');
                for (int k = 0; k < want.Length; k++)
                    if (want[k].Trim() == all[i].Name) hit = true;
                if (!hit) continue;
            }
            cases.Add(all[i]);
        }

        string dir = OUTPUT_DIR;
        if (dir == null || dir.Length == 0)
            dir = Path.Combine(Path.GetTempPath(), "moose_bench");
        string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss", INV);
        StreamWriter csv = null;
        StreamWriter log = null;
        try
        {
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
            string csv_path = Path.Combine(dir, "moose_raster_probe_" + stamp + ".csv");
            csv = new StreamWriter(csv_path, false);
            csv.WriteLine(CSV_HEADER);
            csv.Flush();
            log = new StreamWriter(
                Path.Combine(dir, "moose_raster_probe_" + stamp + ".log"), false);
            Io.output(" output       : " + csv_path);
        }
        catch (Exception e)
        {
            Io.error("cannot open output files (" + e.Message
                     + ") -- console output only");
            csv = null; log = null;
        }
        sCsv = csv; sLog = log;
        sOut = new List<ProbeResult>();

        Io.output("=================================================================");
        Io.output(" Moose rasterization probe  (RASTERIZATION.md, step 0)");
        Io.output(" started      : " + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss", INV));
        Io.output(" cases        : " + cases.Count.ToString(INV));
        Io.output(" orders (m)   : " + Join(ORDERS));
        Io.output(" refinements  : " + Join(REFINEMENTS) + "   (multiples of 20)");
        Io.output(" wavelength   : " + F(WAVELENGTH, 4) + " um, normal incidence");
        Io.output(" parallel     : " + PARALLEL_TASKS.ToString(INV)
                  + " (cores: " + Environment.ProcessorCount.ToString(INV) + ")");
        Io.output("=================================================================");

        int m1 = ORDERS[0];
        int r1 = CAMODEL_REFINEMENTS[0];

        // ---- P1 -------------------------------------------------------------
        if (RUN_CAMODEL)
        {
            Io.output("");
            Io.output("=================================================================");
            Io.output(" P1 -- can Moose be handed an explicit permittivity grid?");
            Io.output("      Layer(double thickness, CaModel epsilonDistribution)");
            Io.output("");
            Io.output(" What the numbers should look like if it works: the mask on an");
            Io.output(" exact grid and Moose's own Atom render are two exact renderings");
            Io.output(" of the SAME rectangle, so R-R(atom) should be small but need not");
            Io.output(" be zero -- what is left is the sampling channel, i.e. the two");
            Io.output(" grids' different N.  The round trip, by contrast, hands back the");
            Io.output(" very grid Moose rendered, so THAT one should be bit-identical.");
            Io.output("=================================================================");
            for (int i = 0; i < cases.Count; i++)
            {
                ProbeCase c = cases[i];
                for (int j = 0; j < CAMODEL_REFINEMENTS.Length; j++)
                {
                    int refi = CAMODEL_REFINEMENTS[j];
                    Enqueue(c, m1, refi, "camodel", "atom", 0, c.WNom, c.HNom, "nominal");
                    for (int k = 0; k < c.CaRes.Length; k++)
                    {
                        Enqueue(c, m1, refi, "camodel", "roundtrip-eps", c.CaRes[k],
                                c.WNom, c.HNom, "nominal");
                        Enqueue(c, m1, refi, "camodel", "mask-eps", c.CaRes[k],
                                c.WNom, c.HNom, "nominal");
                    }
                }
                // value convention and the "is the grid used at all" control:
                // one resolution, one refinement is enough to answer both.
                int res0 = c.CaRes[0];
                Enqueue(c, m1, r1, "camodel", "roundtrip-index", res0,
                        c.WNom, c.HNom, "nominal");
                Enqueue(c, m1, r1, "camodel", "mask-index", res0,
                        c.WNom, c.HNom, "nominal");
                Enqueue(c, m1, r1, "camodel", "mask-eps", res0,
                        c.WNom * 0.96, c.HNom * 0.96, "narrow");
                Enqueue(c, m1, r1, "camodel", "mask-eps", res0,
                        c.WNom * 1.04, c.HNom * 1.04, "wide");
            }
            RunQueue();
        }

        // ---- P2 -------------------------------------------------------------
        if (RUN_REFINEMENT)
        {
            Io.output("");
            Io.output("=================================================================");
            Io.output(" P2 -- R against rRefinementFactorEpsFT, nominal geometry");
            Io.output("=================================================================");
            for (int k = 0; k < ORDERS.Length; k++)
                for (int i = 0; i < cases.Count; i++)
                {
                    if (cases[i].Name == "AN_aniso_control") continue;  // not battery
                    for (int j = 0; j < REFINEMENTS.Length; j++)
                    {
                        // P1 already solved the Atom reference at (m1, its two
                        // refinements); Find() picks those up, so re-queueing
                        // them would only burn time.
                        if (RUN_CAMODEL && ORDERS[k] == m1
                            && Contains(CAMODEL_REFINEMENTS, REFINEMENTS[j])) continue;
                        Enqueue(cases[i], ORDERS[k], REFINEMENTS[j], "refinement",
                                "atom", 0, cases[i].WNom, cases[i].HNom, "nominal");
                    }
                }
            RunQueue();
        }

        // ---- P3 -------------------------------------------------------------
        if (RUN_ALIGNMENT) Alignment(cases, csv, dir);

        // ---- P4 -------------------------------------------------------------
        if (RUN_MASK_REFINEMENT)
        {
            Io.output("");
            Io.output("=================================================================");
            Io.output(" P4 -- refinement sweep on the MASK path (exact geometry)");
            Io.output("      'fixed'   = mask rendered once at ca_res, refinement varies");
            Io.output("      'matched' = mask rendered AT refinement*(2m+1) -- nothing to");
            Io.output("                  resample, if this is flat the grid dependence is");
            Io.output("                  purely a mismatched-resolution artifact");
            Io.output("=================================================================");
            for (int k = 0; k < ORDERS.Length; k++)
                for (int i = 0; i < cases.Count; i++)
                {
                    ProbeCase c = cases[i];
                    if (c.Name == "AN_aniso_control" || c.Shape == "circle") continue;
                    int m = ORDERS[k];
                    int res0 = c.CaRes[0];
                    for (int j = 0; j < REFINEMENTS.Length; j++)
                    {
                        int refi = REFINEMENTS[j];
                        Enqueue(c, m, refi, "mask-refinement", "mask-eps", res0,
                                c.WNom, c.HNom, "fixed");
                        int matched = refi * (2 * m + 1);
                        if (matched > RENDER_MAX_N)
                        {
                            Io.output(" " + c.Name + " m=" + m.ToString(INV) + " fft="
                                     + refi.ToString(INV) + " matched N=" + matched.ToString(INV)
                                     + " skipped (over RENDER_MAX_N)");
                            continue;
                        }
                        Enqueue(c, m, refi, "mask-refinement", "mask-eps", matched,
                                c.WNom, c.HNom, "matched");
                    }
                }
            RunQueue();
        }

        // ---- verdicts -------------------------------------------------------
        if (RUN_CAMODEL)
        {
            Io.output("");
            Io.output("=================================================================");
            Io.output(" P1 VERDICT");
            Io.output("=================================================================");
            for (int i = 0; i < cases.Count; i++)
            {
                ProbeCase c = cases[i];
                Io.output("");
                Io.output(" " + c.Name);
                ProbeResult atom1 = Find(c.Name, "atom", "nominal", 0, m1, r1);
                if (atom1 == null || atom1.Status == "failed")
                {
                    Io.error("   the Atom reference itself failed -- nothing to compare");
                    continue;
                }
                Io.output("   reference (Atom, fft=" + r1.ToString(INV) + "): R = "
                          + F(atom1.R, 9));

                int res0 = c.CaRes[0];
                Io.output("   value convention");
                Io.output("     roundtrip as eps    : "
                    + Cmp(Find(c.Name, "roundtrip-eps", "nominal", res0, m1, r1), atom1,
                          "REPRODUCES the Atom layer -> CaModel takes EPSILON",
                          "differs"));
                Io.output("     roundtrip as n+ik   : "
                    + Cmp(Find(c.Name, "roundtrip-index", "nominal", res0, m1, r1), atom1,
                          "REPRODUCES the Atom layer -> CaModel takes the INDEX",
                          "differs"));

                // Compared against the NOMINAL MASK, not against the Atom: both
                // sides are then CaModel layers and the width is the only thing
                // that changed, so "identical" can only mean the grid is unused.
                ProbeResult mask0 = Find(c.Name, "mask-eps", "nominal", res0, m1, r1);
                Io.output("   is the grid used at all?  (these MUST differ)");
                Io.output("     mask -4% width      : "
                    + Cmp(Find(c.Name, "mask-eps", "narrow", res0, m1, r1), mask0,
                          "SAME as the nominal mask -> the CaModel is being IGNORED",
                          "differs, good"));
                Io.output("     mask +4% width      : "
                    + Cmp(Find(c.Name, "mask-eps", "wide", res0, m1, r1), mask0,
                          "SAME as the nominal mask -> the CaModel is being IGNORED",
                          "differs, good"));
                Io.output("     eps grid vs n+ik grid: "
                    + Cmp(Find(c.Name, "mask-index", "nominal", res0, m1, r1), mask0,
                          "SAME -> the VALUES are not being read either",
                          "differs, good"));

                Io.output("   the mask itself (python's cell-centred rule)");
                for (int k = 0; k < c.CaRes.Length; k++)
                {
                    ProbeResult mk = Find(c.Name, "mask-eps", "nominal", c.CaRes[k], m1, r1);
                    if (mk == null) { Io.output("     N=" + c.CaRes[k] + ": missing"); continue; }
                    Io.output("     N=" + Pad(c.CaRes[k].ToString(INV), 6)
                              + " R=" + Pad(F(mk.R, 9), 13)
                              + " R-R(atom)=" + Pad(E(mk.R - atom1.R), 11)
                              + " fill=" + (Double.IsNaN(mk.Fill) ? "-" : F(mk.Fill, 6))
                              + " row=" + (mk.RowPx < 0 ? "-" : mk.RowPx.ToString(INV))
                              + "/" + c.CaRes[k].ToString(INV));
                }

                Io.output("   does the refinement still resample an explicit grid?");
                for (int k = 0; k < c.CaRes.Length; k++)
                {
                    ProbeResult a = Find(c.Name, "mask-eps", "nominal", c.CaRes[k], m1,
                                         CAMODEL_REFINEMENTS[0]);
                    ProbeResult b = (CAMODEL_REFINEMENTS.Length > 1)
                        ? Find(c.Name, "mask-eps", "nominal", c.CaRes[k], m1,
                               CAMODEL_REFINEMENTS[1]) : null;
                    Io.output("     N=" + Pad(c.CaRes[k].ToString(INV), 6) + " fft "
                        + CAMODEL_REFINEMENTS[0].ToString(INV) + " vs "
                        + (CAMODEL_REFINEMENTS.Length > 1
                           ? CAMODEL_REFINEMENTS[1].ToString(INV) : "-") + ": "
                        + Cmp(a, b, "identical -> refinement does NOT touch it",
                                    "differs -> refinement still resamples"));
                }
            }
        }

        if (RUN_REFINEMENT)
        {
            Io.output("");
            Io.output("=================================================================");
            Io.output(" P2 VERDICT -- R(r) = a + b/r^p over refinements " + Join(REFINEMENTS));
            Io.output(" The claim under test: p = 2 (the sinc/sampling channel),");
            Io.output(" not p = 1.  On C1 at m = 10 the three points already on");
            Io.output(" record give R_inf = 0.397181 (p=2, max resid 5.2e-06) against");
            Io.output(" 0.396618 (p=1, max resid 1.2e-04).");
            Io.output("=================================================================");
            Io.output(" " + Pad("case", 26) + Pad("m", 5) + Pad("R_inf(p=2)", 14)
                      + Pad("resid", 11) + Pad("R_inf(p=1)", 14) + Pad("resid", 11)
                      + "verdict");
            for (int k = 0; k < ORDERS.Length; k++)
                for (int i = 0; i < cases.Count; i++)
                {
                    ProbeCase c = cases[i];
                    if (c.Name == "AN_aniso_control") continue;
                    List<double> rs = new List<double>();
                    List<double> Rs = new List<double>();
                    for (int j = 0; j < REFINEMENTS.Length; j++)
                    {
                        // source/tag/res identify the geometry; the probe label
                        // does not, and the P1 rows are the same solve.
                        ProbeResult r = Find(c.Name, "atom", "nominal", 0,
                                             ORDERS[k], REFINEMENTS[j]);
                        if (r == null || r.Status == "failed") continue;
                        rs.Add((double)REFINEMENTS[j]); Rs.Add(r.R);
                    }
                    double a2, b2, e2, a1, b1, e1;
                    bool ok2 = FitPower(rs, Rs, 2.0, out a2, out b2, out e2);
                    bool ok1 = FitPower(rs, Rs, 1.0, out a1, out b1, out e1);
                    double spread = 0.0;
                    for (int j = 0; j < Rs.Count; j++)
                        for (int L = 0; L < Rs.Count; L++)
                            if (Math.Abs(Rs[j] - Rs[L]) > spread)
                                spread = Math.Abs(Rs[j] - Rs[L]);
                    string verdict;
                    if (!ok2 || !ok1 || rs.Count < 3)
                        verdict = "too few points (" + rs.Count.ToString(INV) + ")";
                    else if (spread < 1.0e-12)
                        verdict = "R does not move with the refinement at all";
                    else if (e2 < e1)
                    {
                        double gain = e1 / Math.Max(e2, 1.0e-15);
                        verdict = "p = 2 fits better (by "
                                + (gain > 1000.0 ? ">1000" : F(gain, 1)) + "x), spread "
                                + E(spread);
                    }
                    else if (e1 < e2)
                        verdict = "p = 1 fits better -- NOT the sinc channel, spread "
                                + E(spread);
                    else
                        verdict = "tie -- inconclusive, spread " + E(spread);
                    Io.output(" " + Pad(c.Name, 26) + Pad(ORDERS[k].ToString(INV), 5)
                        + Pad(ok2 ? F(a2, 9) : "-", 14) + Pad(ok2 ? E(e2) : "-", 11)
                        + Pad(ok1 ? F(a1, 9) : "-", 14) + Pad(ok1 ? E(e1) : "-", 11)
                        + verdict);
                }
        }

        if (RUN_MASK_REFINEMENT)
        {
            Io.output("");
            Io.output("=================================================================");
            Io.output(" P4 VERDICT -- same fit, MASK path, exact geometry");
            Io.output(" 'fixed': one CaModel per case, only the solve's refinement varies.");
            Io.output(" 'matched': the CaModel is rebuilt at N = refinement*(2m+1) every");
            Io.output(" time, so there is nothing left for refinement to resample -- flat");
            Io.output(" here (spread ~ 0) would mean the CaModel path has no remaining grid");
            Io.output(" dependence at all on axis-aligned geometry.");
            Io.output("=================================================================");
            string[] tags = new string[] { "fixed", "matched" };
            for (int ti = 0; ti < tags.Length; ti++)
            {
                Io.output("");
                Io.output(" -- " + tags[ti] + " --");
                Io.output(" " + Pad("case", 26) + Pad("m", 5) + Pad("R_inf(p=2)", 14)
                          + Pad("resid", 11) + Pad("R_inf(p=1)", 14) + Pad("resid", 11)
                          + "verdict");
                for (int k = 0; k < ORDERS.Length; k++)
                    for (int i = 0; i < cases.Count; i++)
                    {
                        ProbeCase c = cases[i];
                        if (c.Name == "AN_aniso_control" || c.Shape == "circle") continue;
                        int m = ORDERS[k];
                        List<double> rs = new List<double>();
                        List<double> Rs = new List<double>();
                        for (int j = 0; j < REFINEMENTS.Length; j++)
                        {
                            int refi = REFINEMENTS[j];
                            int res = (tags[ti] == "fixed") ? c.CaRes[0] : refi * (2 * m + 1);
                            ProbeResult r = Find(c.Name, "mask-eps", tags[ti], res, m, refi);
                            if (r == null || r.Status == "failed") continue;
                            rs.Add((double)refi); Rs.Add(r.R);
                        }
                        double a2, b2, e2, a1, b1, e1;
                        bool ok2 = FitPower(rs, Rs, 2.0, out a2, out b2, out e2);
                        bool ok1 = FitPower(rs, Rs, 1.0, out a1, out b1, out e1);
                        double spread = 0.0;
                        for (int j = 0; j < Rs.Count; j++)
                            for (int L = 0; L < Rs.Count; L++)
                                if (Math.Abs(Rs[j] - Rs[L]) > spread)
                                    spread = Math.Abs(Rs[j] - Rs[L]);
                        string verdict;
                        if (rs.Count < 3)
                            verdict = "too few points (" + rs.Count.ToString(INV) + ")";
                        else if (spread < 1.0e-9)
                            verdict = "FLAT -- no remaining grid dependence, spread "
                                    + E(spread);
                        else if (!ok2 || !ok1)
                            verdict = "fit failed, spread " + E(spread);
                        else if (e2 < e1)
                        {
                            double gain = e1 / Math.Max(e2, 1.0e-15);
                            verdict = "p = 2 fits better (by "
                                    + (gain > 1000.0 ? ">1000" : F(gain, 1)) + "x), spread "
                                    + E(spread);
                        }
                        else if (e1 < e2)
                            verdict = "p = 1 fits better, spread " + E(spread);
                        else
                            verdict = "tie -- inconclusive, spread " + E(spread);
                        Io.output(" " + Pad(c.Name, 26) + Pad(m.ToString(INV), 5)
                            + Pad(ok2 ? F(a2, 9) : "-", 14) + Pad(ok2 ? E(e2) : "-", 11)
                            + Pad(ok1 ? F(a1, 9) : "-", 14) + Pad(ok1 ? E(e1) : "-", 11)
                            + verdict);
                    }
            }
        }

        if (csv != null) { try { csv.Close(); } catch (Exception) { } }
        if (log != null) { try { log.Close(); } catch (Exception) { } }
        Io.success("done.");
    }

    static bool Contains(int[] a, int v)
    {
        for (int i = 0; i < a.Length; i++) if (a[i] == v) return true;
        return false;
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
}
