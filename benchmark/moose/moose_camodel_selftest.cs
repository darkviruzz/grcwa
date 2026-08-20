// script:  moose_camodel_selftest.cs
// purpose: is the CaModel/Layer(double, CaModel) construction path safe under
//          the concurrency moose_raster_probe.cs and moose_convergence_bench.cs
//          use (PARALLEL_TASKS worker threads pulling independent jobs)?
//
// ---------------------------------------------------------------------------
// WHY THIS SCRIPT EXISTS
// ---------------------------------------------------------------------------
// Two runs of moose_raster_probe.cs on two different machines, same nominal
// parameters (C1_Si_pillars, m = 10, the exact 260x260 mask, PARALLEL_TASKS=6),
// gave DIFFERENT R at fft = 40 (0.398145 vs 0.396608, atom path; 0.396257 vs
// 0.396608, mask path -- both differ between runs) but IDENTICAL R at fft = 100
// (0.397322457 both times, to 10 digits; 0.395804217 both times, to 10 digits
// -- both paths).
//
// That is not random noise -- it is a specific, structured pattern: fft = 40
// solves take ~25-30s, fft = 100 solves take ~180s.  Under PARALLEL_TASKS = 6
// many fft = 40 jobs are in flight AT THE SAME TIME (a long queue of short
// jobs), while fft = 100 jobs, being 6-7x slower, have far fewer instances
// overlapping at any moment.  If ANY of GratingStructure / Layer / CaModel /
// Rcwa holds shared or thread-affine state that the CaModel construction path
// touches during BUILD (not just during Calc), concurrent construction is
// exactly where it would show up -- and it would show up MORE at fft = 40
// (much more overlap) than at fft = 100 (much less), which is exactly what was
// observed.
//
// moose_convergence_bench.cs already has a PARALLEL_SELFTEST for the ATOM
// construction path (sequential vs parallel, repeated, bitwise compared) and
// found it safe.  moose_raster_probe.cs's newer CaModel/Layer(double, CaModel)
// path has never been tested that way.  This script is that test, scoped
// narrowly: same case, same order, same refinement, repeated many times, once
// sequential and once at the SAME parallelism the two runs above used.
//
// ---------------------------------------------------------------------------
// WHAT TO DO WITH THE RESULT
// ---------------------------------------------------------------------------
//   sequential spread ~ 0, parallel spread ~ 0            -> Moose is
//       deterministic and thread-safe here; the run1/run2 difference is a
//       real machine/build difference, not a bug in how this repo drives it.
//       Compare Moose version/build strings between the two machines next.
//   sequential spread ~ 0, parallel spread >> 0            -> confirmed race
//       condition in the CaModel/Layer construction path under concurrency.
//       Every P1/P4 number collected with PARALLEL_TASKS > 1 needs a rerun
//       with PARALLEL_TASKS = 1 before it can be trusted, and
//       moose_raster_probe.cs's default should probably become 1 until this
//       is root-caused (or fixed by serializing GratingStructure/Layer/CaModel
//       construction, e.g. with a lock, while leaving Calc() itself parallel).
//   sequential spread >> 0                                  -> Moose itself is
//       non-deterministic on this build even single-threaded (a converged-but-
//       not-exactly-reproducible iterative solve, or true randomness
//       somewhere) -- a bigger finding, worth its own investigation before
//       trusting ANY single recorded value without averaging repeats.
//
// Runtime: 2 cases x 2 refinements x 2 modes x N_REPEAT solves.  At
// N_REPEAT = 8 and ~25-30s/solve for fft=40, ~180s/solve for fft=100, this is
// roughly (8*30 + 8*180) * 2 cases * 2 modes(sequential adds wall time,
// parallel does not) -- call it 15-25 minutes wall clock with the parallel
// runs overlapped, longer if run entirely sequentially.  Turn N_REPEAT down
// for a quicker first look.
// ---------------------------------------------------------------------------

using System;
using System.IO;
using System.Threading;
using System.Collections.Generic;
using System.Globalization;


public class SelfTestScript
{
    // =======================================================================
    //  CONFIGURATION
    // =======================================================================
    static string OUTPUT_DIR = "C:\\Users\\mwalther\\PycharmProjects\\grcwa\\benchmark\\moose";

    static int N_REPEAT = 8;

    // The refinements from the run1/run2 discrepancy: 40 (differed between
    // runs) and 100 (matched to 10 digits between runs) -- so this script
    // reproduces both the suspect case and the control in one go.
    static readonly int[] REFINEMENTS = { 40, 100 };

    // Matches what moose_raster_probe.cs used when the discrepancy showed up.
    static int PARALLEL_TASKS_TO_TEST = 6;

    static readonly int M = 10;                 // q = 21, nG = 441
    static readonly int CA_RES = 260;            // exact grid for w = 0.6/0.5

    const double WAVELENGTH = 1.0;
    const double AOI = 0.0, CONICAL = 0.0;
    const long RCWA_CACHE = 0;
    const double SUPER_N = 1.0;
    const double SCALE = 0.01;
    const double SI_N = 3.5, SI_K = 0.0, AIR_N = 1.0, AIR_K = 0.0, SIO2_N = 1.5, SIO2_K = 0.0;

    static readonly CultureInfo INV = CultureInfo.InvariantCulture;

    // Two cases, both from the battery, both C1-like (square pillar, w = 0.6)
    // so the same CA_RES = 260 is exact for both.
    public class Case { public string Name; public double Period, Depth; }
    static List<Case> Cases()
    {
        List<Case> c = new List<Case>();
        c.Add(new Case { Name = "C1_Si_pillars", Period = 0.50, Depth = 0.40 });
        return c;
    }

    static bool CellInside(int n, int i, int j, double w, double h)
    {
        double x = (i + 0.5) / n - 0.5;
        double y = (j + 0.5) / n - 0.5;
        return Math.Abs(x) <= w / 2.0 + 1.0e-12 && Math.Abs(y) <= h / 2.0 + 1.0e-12;
    }

    static CaModel Mask(int n, double w, double h)
    {
        Materials si = new Materials(SI_N, SI_K);
        Materials air = new Materials(AIR_N, AIR_K);
        CaModel model = new CaModel(n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                model.SetValue(i, j, CellInside(n, i, j, w, h)
                    ? si.GetEpsilon(WAVELENGTH) : air.GetEpsilon(WAVELENGTH));
        return model;
    }

    static double SolveOnce(Case c, int refi)
    {
        Materials superstrate = new Materials(SUPER_N, 0.0);
        Materials sub = new Materials(SIO2_N, SIO2_K);
        GratingStructure grating = new GratingStructure(c.Period, c.Period, superstrate, sub);
        CaModel mask = Mask(CA_RES, 0.6, 0.6);
        Layer layer = new Layer(c.Depth, mask);
        try { layer.Declare2D(); } catch (Exception) { }
        grating.AddLayerOnBottom(layer);

        Rcwa solver = new Rcwa(grating, M, M, refi, RCWA_CACHE);
        solver.Calc(WAVELENGTH, AOI, CONICAL, 90.0, true);   // TE

        double r = 0.0;
        for (int ox = -M; ox <= M; ox++)
            for (int oy = -M; oy <= M; oy++)
            {
                try { r += solver.GetEfficiencyForGivenOrder('r', ox, oy, "TE"); }
                catch (Exception) { }
                try { r += solver.GetEfficiencyForGivenOrder('r', ox, oy, "TM"); }
                catch (Exception) { }
            }
        r *= SCALE;

        try { solver.Delete(); } catch (Exception) { }
        try { grating.Delete(); } catch (Exception) { }
        return r;
    }

    // -- sequential ----------------------------------------------------------
    static double[] RunSequential(Case c, int refi, int n)
    {
        double[] outp = new double[n];
        for (int i = 0; i < n; i++) outp[i] = SolveOnce(c, refi);
        return outp;
    }

    // -- parallel, same pattern as moose_raster_probe.cs's worker pool -------
    static readonly object sLock = new object();
    static int sNext;
    static int sTotal;
    static Case sCase;
    static int sRefi;
    static double[] sOut;

    static void Worker()
    {
        while (true)
        {
            int idx;
            lock (sLock)
            {
                if (sNext >= sTotal) return;
                idx = sNext; sNext++;
            }
            double r = SolveOnce(sCase, sRefi);
            lock (sLock) { sOut[idx] = r; }
        }
    }

    static double[] RunParallel(Case c, int refi, int n, int workers)
    {
        sCase = c; sRefi = refi; sTotal = n; sNext = 0; sOut = new double[n];
        Thread[] pool = new Thread[workers];
        for (int i = 0; i < workers; i++)
        {
            pool[i] = new Thread(new ThreadStart(Worker));
            pool[i].IsBackground = false;
            pool[i].Start();
        }
        for (int i = 0; i < workers; i++) pool[i].Join();
        return sOut;
    }

    static double Spread(double[] a)
    {
        double lo = a[0], hi = a[0];
        for (int i = 1; i < a.Length; i++) { if (a[i] < lo) lo = a[i]; if (a[i] > hi) hi = a[i]; }
        return hi - lo;
    }

    static string Fmt(double[] a)
    {
        string s = "";
        for (int i = 0; i < a.Length; i++) s += (i > 0 ? " " : "") + a[i].ToString("F9", INV);
        return s;
    }

    static void Main()
    {
        string dir = OUTPUT_DIR;
        if (dir == null || dir.Length == 0) dir = Path.Combine(Path.GetTempPath(), "moose_bench");
        if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
        string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss", INV);
        StreamWriter log = new StreamWriter(
            Path.Combine(dir, "moose_camodel_selftest_" + stamp + ".log"), false);
        Action<string> outp = s => { Console.WriteLine(s); log.WriteLine(s); log.Flush(); };

        outp("=================================================================");
        outp(" CaModel construction path -- sequential vs parallel self-test");
        outp(" N_REPEAT = " + N_REPEAT.ToString(INV) + "   parallel workers = "
             + PARALLEL_TASKS_TO_TEST.ToString(INV) + "   cores = "
             + Environment.ProcessorCount.ToString(INV));
        outp("=================================================================");

        foreach (Case c in Cases())
        {
            foreach (int refi in REFINEMENTS)
            {
                outp("");
                outp(" " + c.Name + "  m=" + M.ToString(INV) + "  fft=" + refi.ToString(INV));

                System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
                double[] seq = RunSequential(c, refi, N_REPEAT);
                double seqTime = sw.Elapsed.TotalSeconds;
                double seqSpread = Spread(seq);
                outp("   sequential (" + seqTime.ToString("F1", INV) + "s): "
                     + Fmt(seq));
                outp("   sequential spread: " + seqSpread.ToString("E2", INV)
                     + (seqSpread > 1.0e-9 ? "   <-- Moose is NOT deterministic, even single-threaded"
                                           : "   -- deterministic"));

                sw.Restart();
                double[] par = RunParallel(c, refi, N_REPEAT, PARALLEL_TASKS_TO_TEST);
                double parTime = sw.Elapsed.TotalSeconds;
                double parSpread = Spread(par);
                outp("   parallel   (" + parTime.ToString("F1", INV) + "s, "
                     + PARALLEL_TASKS_TO_TEST.ToString(INV) + " workers): " + Fmt(par));
                outp("   parallel spread:   " + parSpread.ToString("E2", INV)
                     + (parSpread > 1.0e-9 ? "   <-- CONCURRENCY BUG in the CaModel construction path"
                                           : "   -- safe under this concurrency"));

                double seqMean = 0.0; foreach (double v in seq) seqMean += v; seqMean /= seq.Length;
                double parMean = 0.0; foreach (double v in par) parMean += v; parMean /= par.Length;
                outp("   sequential mean vs parallel mean: " + (parMean - seqMean).ToString("E2", INV));
            }
        }

        outp("");
        outp("=================================================================");
        outp(" Compare fft=40 (many short jobs overlap under 6 workers) against");
        outp(" fft=100 (few long jobs overlap) above.  If the parallel spread is");
        outp(" large at 40 and small at 100, that is the race-condition signature");
        outp(" already seen between the two independent moose_raster_probe.cs");
        outp(" runs (fft=40 differed by up to 1.5e-3 between machines, fft=100");
        outp(" matched to 1e-10).");
        outp("=================================================================");
        log.Close();
    }
}
