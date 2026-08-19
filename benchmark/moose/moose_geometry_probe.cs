// script:  moose_geometry_probe.cs
// purpose: find out WHICH GEOMETRY Moose actually solves for the 2D cases of
//          benchmark/structures.py -- because that, and not the physics, is
//          what makes the 2D Moose values sit next to a different limit than
//          grcwa and Ikarus.
//
// ---------------------------------------------------------------------------
// WHY THIS SCRIPT EXISTS
// ---------------------------------------------------------------------------
// On every 1D case of the battery Moose agrees with the well-factorized python
// columns to five or six digits:
//
//     A2_formbiref_TE   0.545760 vs 0.545760      B2_HCG_TM     0.873329 vs 0.873329
//     B1_Si_grating_TM  0.213710 vs 0.213710      B3_Au_slits   0.793256 vs 0.793300
//     D1_ikarus_hcg_TM  0.100173 vs 0.100173
//
// so wavelength, units, materials, polarization, the percent scaling and the
// order-summed R are all right.  On every 2D case it does not:
//
//     case                     Moose      fork[Pol]  ikarus[Li]  ikarus[NV]
//     C1_Si_pillars            0.39817    0.38979    0.39016     0.38974
//     C1b_Si_pillars_diffract  0.15477    0.14535    0.14544     0.14539
//     C2_Au_holes              0.64834    0.69554    0.66719     0.67929
//
// Three python columns that use three different factorization rules agree with
// each other to <0.001 and disagree with Moose by 0.008 ... 0.03.  Different
// factorization rules converge to the SAME limit if they are fed the same
// structure, so a gap that survives at high order is a geometry gap, not a
// physics gap.
//
// The suspect is the rasterization.  structures.py hands both python codes ONE
// shared 256 x 256 binary mask.  Sampled on that grid with a strict "<" and
// left-edge samples, the square pillars do not come out at their nominal size:
//
//     case   nominal w/period   mask pixels    w_eff        error
//     C1     0.600              153 / 256      0.597656     -0.391 %
//     C1b    0.400              103 / 256      0.402344     +0.586 %
//     C2     0.500              127 / 256      0.496094     -0.781 %
//
// (0.6*256 = 153.6 and 0.4*256 = 102.4 are not integers, and for C2 the edge
// falls exactly on a sample so the strict "<" drops a pixel on each side.)
// The 1D masks have no such error: NX_1D = 8192 and ff = 0.5 give exactly 4096
// pixels, which is why 1D agrees and 2D does not.
//
// Measured on the python side, R is roughly ten times more sensitive to that
// half-pixel than to the truncation order (ikarus, Li rule, q = 31):
//
//     C1   w_eff 0.597656 -> R 0.38996      C1b  w_eff 0.402344 -> R 0.14700
//          w_eff 0.599219 -> R 0.39446           w_eff 0.400000 -> R 0.15719
//          w_eff 0.601562 -> R 0.40169           w_eff 0.398438 -> R 0.16025
//
// i.e. ONE pixel of the 256 grid is worth ~0.012 in R, and the whole
// Moose-to-python gap is smaller than one pixel.  Feed the python codes the
// nominal (unrasterized) width and they move onto Moose.
//
// This script asks Moose the same question from its side.
//
// ---------------------------------------------------------------------------
// THE THREE PROBES
// ---------------------------------------------------------------------------
// A  GEOMETRY DUMP (no solving, seconds).  Renders each patterned layer with
//    Layer::GetEpsilonDistributionsAsCaModel at several resolutions and reports
//    the fill fraction, the pillar width along the centre row, and whether any
//    intermediate permittivity appears.  That answers directly whether Moose
//    rasterizes binary (like the python battery) or area-weighted/analytic, and
//    how big its atoms really are.  If the fill fraction comes out at the
//    nominal 0.36 / 0.16 / 0.25 / 0.2827 independently of the resolution, Moose
//    is solving the nominal structure and the python side is not.
//
// B  WIDTH SWEEP (the decisive one).  Solves each 2D case at a fixed order with
//    the atom width set to
//        * the nominal value,
//        * the value the python 256-mask actually rasterizes (W_PY below),
//        * +-1 % and +-2 % around nominal,
//    so the local slope dR/dw is measured IN MOOSE.  Two things fall out:
//      - if Moose at W_PY reproduces the python numbers quoted above, the whole
//        2D disagreement is the rasterization and nothing else;
//      - if Moose's R does not move at all between those widths, then Moose's
//        own eps sampling cannot resolve them, which is worth knowing too.
//
// C  REFINEMENT SWEEP.  Same case and order, nominal width, rRefinementFactorEpsFT
//    = 2, 3, 5, 10, 20, 40.  Shows how much of Moose's own 2D value depends on
//    its eps sampling -- i.e. whether the +-0.001 wobble along the Moose 2D
//    sweep in moose_reference.json is the refinement changing with the order
//    (FFT_MODE = 1 in moose_convergence_bench.cs picks refinement =
//    ceil(256 / q), so the absolute grid jumps around between ~256 and ~305).
//
// ---------------------------------------------------------------------------
// CONVENTIONS
// ---------------------------------------------------------------------------
// Identical to moose_convergence_bench.cs -- microns, max order m per axis
// (q = 2m+1 retained), polarization angle 0 = TM / 90 = TE, efficiencies in
// percent (scaled by 0.01 here), R summed over the propagating orders in both
// output polarizations, and R + T + A checked against 1.
//
// Runtime: probe A is seconds.  With ORDERS = {10, 15} probes B and C are about
// 60 solves at nG = 441 and 961; on the pool that is minutes, not hours.
// ---------------------------------------------------------------------------

using System;
using System.IO;
using System.Threading;
using System.Collections.Generic;
using System.Globalization;


// ===========================================================================
//  one 2D case of the battery
// ===========================================================================
public class ProbeCase
{
    public string Name;
    public string Pol;          // "TE" or "TM"
    public string Shape;        // "rect" or "circle"
    public double Period;       // um
    public double Depth;        // um
    public double WNom;         // rect: atom width / period;  circle: radius / period
    public double WPy;          // what the shared 256x256 python mask rasterizes
    public double PillarN, PillarK;   // the atom
    public double BgN, BgK;           // the background of the patterned layer
    public double SubN, SubK;         // substrate half space

    public Materials Pillar() { return new Materials(PillarN, PillarK); }
    public Materials Bg()     { return new Materials(BgN, BgK); }
    public Materials Sub()    { return new Materials(SubN, SubK); }
}


public class ProbeResult
{
    public string Case;
    public string Probe;        // "width" or "refinement"
    public int    M;
    public int    Q;
    public long   NG;
    public double W;            // the atom size that was used
    public string WTag;         // "nominal", "python-mask", "-1%", ...
    public int    Refinement;
    public double R, T, A, Energy;
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
    static bool RUN_GEOMETRY_DUMP = true;
    static bool RUN_WIDTH_SWEEP   = true;
    static bool RUN_REFINEMENT    = true;

    // Max orders m for the solving probes (q = 2m+1 retained per axis).
    // m = 10 -> q = 21 -> nG = 441;  m = 15 -> q = 31 -> nG = 961.  Those are
    // the two order counts the python numbers in the header were taken at.
    //
    // m = 0 is in the list for a different reason and costs nothing: with a
    // single retained order RCWA can only be a transfer-matrix calculation on
    // the CELL-AVERAGED permittivity, so R at m = 0 is a direct read-out of the
    // fill fraction Moose used.  The (0,0) points already in
    // benchmark/moose_reference.json do not survive that test -- C1 reports
    // 0.363252, which is exactly R of a SOLID silicon film (the averaged medium
    // gives 0.151138, and grcwa at nG = 1 reproduces that to eight digits), and
    // C2 reports 0.040000, exactly R of a bare air/glass interface rather than
    // the 0.973 of a gold-dominated average.  Whatever Moose does with zero
    // orders, it is not the average medium; the width sweep at m = 0 says
    // whether it looks at the geometry at all.
    static readonly int[] ORDERS = { 0, 10, 15 };

    // Refinement factors for probe C.
    static readonly int[] REFINEMENTS = { 2, 3, 5, 10, 20, 40 };

    // Resolutions at which probe A renders the permittivity distribution.
    static readonly int[] DUMP_RES = { 64, 100, 256, 300, 512 };

    // Refinement used by probes A and B.  0 = use exactly the rule
    // moose_convergence_bench.cs uses with FFT_MODE = 1, i.e.
    // ceil(FFT_TARGET_SAMPLES / q) clamped to [2, 200], so probe B's "nominal"
    // row is directly comparable to the numbers already in
    // benchmark/moose_reference.json.  A positive value pins the refinement
    // instead (5 is Moose's own default).
    static int FIXED_REFINEMENT = 0;
    const int FFT_TARGET_SAMPLES = 256;
    const int FFT_REFINEMENT_MIN = 2;
    const int FFT_REFINEMENT_MAX = 200;

    static int RefinementFor(int m)
    {
        if (FIXED_REFINEMENT > 0) return FIXED_REFINEMENT;
        int q = 2 * m + 1;
        int refinement = (int)Math.Ceiling((double)FFT_TARGET_SAMPLES / (double)q);
        if (refinement < FFT_REFINEMENT_MIN) refinement = FFT_REFINEMENT_MIN;
        if (refinement > FFT_REFINEMENT_MAX) refinement = FFT_REFINEMENT_MAX;
        return refinement;
    }

    // Case filter: empty = all four 2D cases.  Comma separated names.
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

    static readonly CultureInfo INV = CultureInfo.InvariantCulture;

    // materials at lambda = 1 um (n, k)
    const double AIR_N = 1.0, AIR_K = 0.0;
    const double SIO2_N = 1.5, SIO2_K = 0.0;
    const double SI_N = 3.5, SI_K = 0.0;
    const double AU_N = 0.3, AU_K = 7.0;


    // =======================================================================
    //  the four 2D cases, 1:1 with benchmark/structures.py
    // =======================================================================
    static List<ProbeCase> BuildCases()
    {
        List<ProbeCase> cases = new List<ProbeCase>();
        ProbeCase c;

        // C1: Si square pillars, period 0.5, ax = ay = 0.3 -> w = 0.6, on glass.
        // python mask: 153 of 256 pixels -> 0.59765625
        c = new ProbeCase();
        c.Name = "C1_Si_pillars"; c.Pol = "TE"; c.Shape = "rect";
        c.Period = 0.50; c.Depth = 0.40; c.WNom = 0.600000; c.WPy = 153.0 / 256.0;
        c.PillarN = SI_N; c.PillarK = SI_K; c.BgN = AIR_N; c.BgK = AIR_K;
        c.SubN = SIO2_N; c.SubK = SIO2_K;
        cases.Add(c);

        // C1b: the same pillars at period 1.5 (diffractive), ax = ay = 0.6 -> w = 0.4.
        // python mask: 103 of 256 pixels -> 0.40234375
        c = new ProbeCase();
        c.Name = "C1b_Si_pillars_diffract"; c.Pol = "TE"; c.Shape = "rect";
        c.Period = 1.50; c.Depth = 0.40; c.WNom = 0.400000; c.WPy = 103.0 / 256.0;
        c.PillarN = SI_N; c.PillarK = SI_K; c.BgN = AIR_N; c.BgK = AIR_K;
        c.SubN = SIO2_N; c.SubK = SIO2_K;
        cases.Add(c);

        // C2: air holes in gold, period 0.6, ax = ay = 0.3 -> w = 0.5, on glass.
        // python mask: 127 of 256 pixels -> 0.49609375
        c = new ProbeCase();
        c.Name = "C2_Au_holes"; c.Pol = "TE"; c.Shape = "rect";
        c.Period = 0.60; c.Depth = 0.20; c.WNom = 0.500000; c.WPy = 127.0 / 256.0;
        c.PillarN = AIR_N; c.PillarK = AIR_K;      // the hole is the atom
        c.BgN = AU_N; c.BgK = AU_K;                // background is gold
        c.SubN = SIO2_N; c.SubK = SIO2_K;
        cases.Add(c);

        // D2: free-standing Si cylinder, period 400/700, radius 0.30 of the
        // period, depth 200/700.  The python mask is cell-centred here, so its
        // AREA error is only +0.076 % -> r_eff = 0.3001139 (not 0.30 exactly).
        c = new ProbeCase();
        c.Name = "D2_ikarus_cylinder_TE"; c.Pol = "TE"; c.Shape = "circle";
        c.Period = 400.0 / 700.0; c.Depth = 200.0 / 700.0;
        c.WNom = 0.300000; c.WPy = 0.3001139;
        c.PillarN = SI_N; c.PillarK = SI_K; c.BgN = AIR_N; c.BgK = AIR_K;
        c.SubN = AIR_N; c.SubK = AIR_K;
        cases.Add(c);

        return cases;
    }


    // =======================================================================
    //  geometry
    // =======================================================================
    // Builds the grating and hands back the patterned layer too, so probe A can
    // render it.  `w` is the atom width (rect) or radius (circle), relative to
    // the period, exactly as moose_convergence_bench.cs passes it.
    static GratingStructure BuildGrating(ProbeCase c, double w, out Layer patterned)
    {
        Materials superstrate = new Materials(SUPER_N, 0.0);
        GratingStructure grating = new GratingStructure(
            c.Period, c.Period, superstrate, c.Sub());

        Atom[] atoms = new Atom[1];
        if (c.Shape == "circle")
            atoms[0] = new Atom(0.5, 0.5, w, c.Pillar());
        else
            atoms[0] = new Atom(0.5, 0.5, w, w, c.Pillar());

        Layer layer = new Layer(c.Depth, c.Bg(), 1, atoms);
        layer.Declare2D();
        grating.AddLayerOnBottom(layer);
        patterned = layer;
        return grating;
    }


    // =======================================================================
    //  probe A -- what does Moose's permittivity distribution look like?
    // =======================================================================
    // Renders the patterned layer at DUMP_RES and reports
    //   fill      = fraction of the cell that carries the ATOM permittivity,
    //   px        = pixels of atom along the centre row (rect: the pillar width),
    //   levels    = how many distinct permittivity values appear.
    // levels == 2 everywhere means Moose rasterizes binary like the python
    // battery; more than 2 means it area-weights the boundary pixels, which is
    // exactly what would let it hit the nominal geometry on a coarse grid.
    //
    // NOTE this renders the layer for INSPECTION.  It is not proof of what the
    // solver's own FFT grid does -- probe C is the test for that -- but a
    // renderer and a solver that disagree about the geometry would itself be
    // the answer to the question.
    static void GeometryDump(List<ProbeCase> cases, StreamWriter csv)
    {
        Io.output("");
        Io.output("=================================================================");
        Io.output(" PROBE A -- permittivity distribution of the patterned layer");
        Io.output("=================================================================");
        Io.output(" " + Pad("case", 26) + Pad("res", 7) + Pad("fill", 11)
                  + Pad("fill_nom", 11) + Pad("px_row", 9) + Pad("w_eff", 11)
                  + Pad("levels", 8) + "note");

        for (int i = 0; i < cases.Count; i++)
        {
            ProbeCase c = cases[i];
            double fill_nom = (c.Shape == "circle")
                ? Math.PI * c.WNom * c.WNom
                : c.WNom * c.WNom;

            for (int k = 0; k < DUMP_RES.Length; k++)
            {
                int res = DUMP_RES[k];
                Layer layer = null;
                GratingStructure g = null;
                try
                {
                    g = BuildGrating(c, c.WNom, out layer);
                    CaModel model = layer.GetEpsilonDistributionsAsCaModel(
                        WAVELENGTH, res, res, false);
                    if (model == null)
                    {
                        Io.error("  " + Pad(c.Name, 26) + Pad(res.ToString(INV), 7)
                                 + "GetEpsilonDistributionsAsCaModel returned null");
                        continue;
                    }

                    int nx = model.GetDimX();
                    int ny = model.GetDimY();
                    // The atom's permittivity and the background's, as this build
                    // reports them -- read off the model rather than recomputed,
                    // so a dispersion or convention surprise shows up here.
                    Complex e_atom = c.Pillar().GetEpsilon(WAVELENGTH);
                    double target = (e_atom == null) ? 0.0 : e_atom.Re();

                    int atom_px = 0;
                    List<double> levels = new List<double>();
                    for (int x = 0; x < nx; x++)
                    {
                        for (int y = 0; y < ny; y++)
                        {
                            Complex v = model.GetValue(x, y);
                            double re = (v == null) ? 0.0 : v.Re();
                            if (Math.Abs(re - target) < 1.0e-9) atom_px++;
                            bool known = false;
                            for (int L = 0; L < levels.Count; L++)
                                if (Math.Abs(levels[L] - re) < 1.0e-9) known = true;
                            if (!known && levels.Count < 16) levels.Add(re);
                        }
                    }
                    int row_px = 0;
                    int yc = ny / 2;
                    for (int x = 0; x < nx; x++)
                    {
                        Complex v = model.GetValue(x, yc);
                        double re = (v == null) ? 0.0 : v.Re();
                        if (Math.Abs(re - target) < 1.0e-9) row_px++;
                    }

                    double fill = (nx * ny > 0) ? (double)atom_px / (nx * ny) : 0.0;
                    double w_eff = (nx > 0) ? (double)row_px / nx : 0.0;
                    string note = (nx != res || ny != res)
                        ? "model is " + nx.ToString(INV) + "x" + ny.ToString(INV)
                        : "";
                    Io.output(" " + Pad(c.Name, 26) + Pad(res.ToString(INV), 7)
                              + Pad(F(fill, 6), 11) + Pad(F(fill_nom, 6), 11)
                              + Pad(row_px.ToString(INV), 9) + Pad(F(w_eff, 6), 11)
                              + Pad(levels.Count.ToString(INV), 8) + note);
                    if (csv != null)
                    {
                        // fields 9..14 are solve-only (nG, w, w_tag, refinement,
                        // t_solve, status), so skip them and land `note` in the
                        // same column the solve rows use.
                        csv.WriteLine(c.Name + ",geometry," + res.ToString(INV) + ","
                            + G(fill) + "," + G(fill_nom) + "," + row_px.ToString(INV)
                            + "," + G(w_eff) + "," + levels.Count.ToString(INV)
                            + ",,,,,," + note + ",,");
                        csv.Flush();
                    }
                    model.Delete();
                }
                catch (Exception e)
                {
                    Io.error("  " + Pad(c.Name, 26) + Pad(res.ToString(INV), 7)
                             + "FAILED: " + e.Message);
                }
                try { if (g != null) g.Delete(); } catch (Exception) { }
            }
        }

        // The bounding-box getters, for the record: they are what misled the
        // circular-atom reading in moose_convergence_bench.cs (README.md).
        Io.output("");
        Io.output(" atom bounding boxes as this build reports them:");
        for (int i = 0; i < cases.Count; i++)
        {
            ProbeCase c = cases[i];
            try
            {
                Atom a = (c.Shape == "circle")
                    ? new Atom(0.5, 0.5, c.WNom, c.Pillar())
                    : new Atom(0.5, 0.5, c.WNom, c.WNom, c.Pillar());
                Io.output("  " + Pad(c.Name, 26) + " arg=" + F(c.WNom, 6)
                          + "  startX=" + F(a.GetStartX(), 6)
                          + "  stopX=" + F(a.GetStopX(), 6)
                          + "  widthX=" + F(a.GetWidthX(), 6));
                a.Delete();
            }
            catch (Exception e)
            {
                Io.error("  " + Pad(c.Name, 26) + " bounding box failed: " + e.Message);
            }
        }
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
    // off-axis orders, so "TE"+"TM" is the one that conserves energy -- see the
    // header of moose_convergence_bench.cs.
    static double SumEfficiency(Rcwa solver, char tr, int mx, int my)
    {
        double s = 0.0;
        for (int ox = -mx; ox <= mx; ox++)
            for (int oy = -my; oy <= my; oy++)
                s += Eff(solver, tr, ox, oy, "TE") + Eff(solver, tr, ox, oy, "TM");
        return s;
    }

    static ProbeResult RunOne(ProbeCase c, int m, double w, string wtag,
                              int refinement, string probe)
    {
        ProbeResult r = new ProbeResult();
        r.Case = c.Name; r.Probe = probe; r.M = m; r.Q = 2 * m + 1;
        r.NG = (long)r.Q * (long)r.Q;
        r.W = w; r.WTag = wtag; r.Refinement = refinement;

        GratingStructure grating = null;
        Rcwa solver = null;
        Layer patterned = null;
        double pol_angle = (c.Pol == "TM") ? 0.0 : 90.0;

        try
        {
            grating = BuildGrating(c, w, out patterned);
            solver = new Rcwa(grating, m, m, refinement, RCWA_CACHE);
            System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
            solver.Calc(WAVELENGTH, AOI, CONICAL, pol_angle, true);
            sw.Stop();
            r.TSolve = sw.Elapsed.TotalSeconds;

            int mr = Math.Min(m, PropagatingOrders(SUPER_N, c.Period));
            int mt = Math.Min(m, PropagatingOrders(c.SubN, c.Period));
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
    //  worker pool (same pattern as moose_convergence_bench.cs)
    // =======================================================================
    static readonly object sLock = new object();
    static List<ProbeCase> sQc = new List<ProbeCase>();
    static List<int>       sQm = new List<int>();
    static List<double>    sQw = new List<double>();
    static List<string>    sQtag = new List<string>();
    static List<int>       sQref = new List<int>();
    static List<string>    sQprobe = new List<string>();
    static int             sNext;
    static List<ProbeResult> sOut;
    static StreamWriter    sCsv;
    static StreamWriter    sLog;

    static void Worker()
    {
        while (true)
        {
            ProbeCase c; int m; double w; string tag; int refi; string probe;
            lock (sLock)
            {
                if (sNext >= sQc.Count) return;
                c = sQc[sNext]; m = sQm[sNext]; w = sQw[sNext];
                tag = sQtag[sNext]; refi = sQref[sNext]; probe = sQprobe[sNext];
                sNext++;
            }
            ProbeResult r = RunOne(c, m, w, tag, refi, probe);
            lock (sLock)
            {
                sOut.Add(r);
                string line = Line(r);
                if (r.Status == "ok") Io.output(line);
                else Io.error(line + "   <-- " + r.Status + " " + r.Note);
                if (sCsv != null)
                {
                    try { sCsv.WriteLine(CsvRow(r)); sCsv.Flush(); }
                    catch (Exception) { }
                }
                if (sLog != null)
                {
                    try { sLog.WriteLine(line); sLog.Flush(); }
                    catch (Exception) { }
                }
            }
        }
    }

    static void Enqueue(ProbeCase c, int m, double w, string tag, int refi, string probe)
    {
        sQc.Add(c); sQm.Add(m); sQw.Add(w); sQtag.Add(tag);
        sQref.Add(refi); sQprobe.Add(probe);
    }

    static void RunQueue()
    {
        sNext = 0;
        int workers = PARALLEL_TASKS;
        if (workers == 0) workers = Environment.ProcessorCount;
        if (workers < 1) workers = 1;
        if (workers > sQc.Count) workers = sQc.Count;
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
        sQc = new List<ProbeCase>(); sQm = new List<int>(); sQw = new List<double>();
        sQtag = new List<string>(); sQref = new List<int>(); sQprobe = new List<string>();
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
        return " " + Pad(r.Case, 26) + Pad(r.Probe, 11)
             + " m=" + Pad(r.M.ToString(INV), 4)
             + " w=" + Pad(F(r.W, 7), 11)
             + Pad("(" + r.WTag + ")", 15)
             + " fft=" + Pad(r.Refinement.ToString(INV), 5)
             + " R=" + Pad(F(r.R, 6), 10)
             + " T=" + Pad(F(r.T, 6), 10)
             + " |1-E|=" + Pad(E(Math.Abs(1.0 - r.Energy)), 10)
             + " t=" + F(r.TSolve, 2) + "s";
    }

    const string CSV_HEADER =
        "case,probe,res_or_m,fill_or_R,fill_nom_or_T,px_or_A,w_eff_or_energy,"
        + "levels_or_q,nG,w,w_tag,fft_refinement,t_solve_s,status,note,,";

    static string CsvRow(ProbeResult r)
    {
        return r.Case + "," + r.Probe + "," + r.M.ToString(INV) + ","
            + G(r.R) + "," + G(r.T) + "," + G(r.A) + "," + G(r.Energy) + ","
            + r.Q.ToString(INV) + "," + r.NG.ToString(INV) + ","
            + G(r.W) + "," + r.WTag + "," + r.Refinement.ToString(INV) + ","
            + F(r.TSolve, 3) + "," + r.Status + "," + r.Note + ",,";
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
            string csv_path = Path.Combine(dir, "moose_geometry_probe_" + stamp + ".csv");
            csv = new StreamWriter(csv_path, false);
            csv.WriteLine(CSV_HEADER);
            csv.Flush();
            log = new StreamWriter(
                Path.Combine(dir, "moose_geometry_probe_" + stamp + ".log"), false);
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
        Io.output(" Moose 2D geometry probe");
        Io.output(" started      : " + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss", INV));
        Io.output(" cases        : " + cases.Count.ToString(INV));
        Io.output(" orders (m)   : " + Join(ORDERS));
        Io.output(" wavelength   : " + F(WAVELENGTH, 4) + " um, normal incidence");
        Io.output(" parallel     : " + PARALLEL_TASKS.ToString(INV)
                  + " (cores: " + Environment.ProcessorCount.ToString(INV) + ")");
        Io.output("=================================================================");

        if (RUN_GEOMETRY_DUMP) GeometryDump(cases, csv);

        if (RUN_WIDTH_SWEEP)
        {
            Io.output("");
            Io.output("=================================================================");
            Io.output(" PROBE B -- R against the atom size (fft refinement: "
                      + (FIXED_REFINEMENT > 0
                         ? "pinned at " + FIXED_REFINEMENT.ToString(INV)
                         : "same rule as the main sweep, ceil(256/q))"));
            Io.output(" python (ikarus, Li rule) at the same widths, for comparison:");
            Io.output("");
            Io.output("   case                q    w=python-mask        w=nominal      moose");
            Io.output("   C1_Si_pillars      31    0.597656 0.389961   0.600000 0.396804   0.398436");
            Io.output("   C1b_..._diffract   31    0.402344 0.146998   0.400000 0.157185   0.155264");
            Io.output("   C1b_..._diffract   41    0.402344 0.145824   0.400000 0.156324   0.154774");
            Io.output("   C2_Au_holes        41    0.496094 0.666899   0.500000 0.649316   0.648546");
            Io.output("");
            Io.output(" The python columns move onto Moose when they are handed the");
            Io.output(" nominal rectangle instead of the mask: 81 / 82 / 96 % of the gap.");
            Io.output(" So if Moose at w=python-mask comes down onto the python numbers,");
            Io.output(" the 2D disagreement is the rasterization and nothing else.");
            Io.output("=================================================================");
            for (int k = 0; k < ORDERS.Length; k++)
            {
                int m = ORDERS[k];
                for (int i = 0; i < cases.Count; i++)
                {
                    ProbeCase c = cases[i];
                    int refi = RefinementFor(m);
                    Enqueue(c, m, c.WNom, "nominal", refi, "width");
                    Enqueue(c, m, c.WPy, "python-mask", refi, "width");
                    Enqueue(c, m, c.WNom * 0.98, "-2%", refi, "width");
                    Enqueue(c, m, c.WNom * 0.99, "-1%", refi, "width");
                    Enqueue(c, m, c.WNom * 1.01, "+1%", refi, "width");
                    Enqueue(c, m, c.WNom * 1.02, "+2%", refi, "width");
                }
            }
            RunQueue();
        }

        if (RUN_REFINEMENT)
        {
            Io.output("");
            Io.output("=================================================================");
            Io.output(" PROBE C -- R against rRefinementFactorEpsFT (nominal geometry)");
            Io.output("=================================================================");
            for (int k = 0; k < ORDERS.Length; k++)
                for (int i = 0; i < cases.Count; i++)
                    for (int j = 0; j < REFINEMENTS.Length; j++)
                        Enqueue(cases[i], ORDERS[k], cases[i].WNom, "nominal",
                                REFINEMENTS[j], "refinement");
            RunQueue();
        }

        // ---- summary -------------------------------------------------------
        Io.output("");
        Io.output("=================================================================");
        Io.output(" SUMMARY -- R(nominal) - R(python-mask) per case and order");
        Io.output(" A gap of the size of the Moose-to-python offset (C1 ~0.009,");
        Io.output(" C1b ~0.010) means the rasterization explains the whole thing.");
        Io.output("=================================================================");
        Io.output(" " + Pad("case", 26) + Pad("m", 5) + Pad("R(nominal)", 13)
                  + Pad("R(python-mask)", 16) + "difference");
        for (int k = 0; k < ORDERS.Length; k++)
        {
            for (int i = 0; i < cases.Count; i++)
            {
                double rn = Double.NaN, rp = Double.NaN;
                for (int j = 0; j < sOut.Count; j++)
                {
                    ProbeResult r = sOut[j];
                    if (r.Probe != "width" || r.Case != cases[i].Name
                        || r.M != ORDERS[k] || r.Status != "ok") continue;
                    if (r.WTag == "nominal") rn = r.R;
                    if (r.WTag == "python-mask") rp = r.R;
                }
                Io.output(" " + Pad(cases[i].Name, 26)
                          + Pad(ORDERS[k].ToString(INV), 5)
                          + Pad(Double.IsNaN(rn) ? "-" : F(rn, 6), 13)
                          + Pad(Double.IsNaN(rp) ? "-" : F(rp, 6), 16)
                          + ((Double.IsNaN(rn) || Double.IsNaN(rp))
                             ? "-" : F(rn - rp, 6)));
            }
        }

        if (csv != null) { try { csv.Close(); } catch (Exception) { } }
        if (log != null) { try { log.Close(); } catch (Exception) { } }
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
}
