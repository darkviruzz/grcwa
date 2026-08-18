// Compile-time stubs of the Moose scripting API, transcribed from the class
// signatures in moose.qch (the doxygen help shipped with Moose).
//
// These do nothing at all -- their only purpose is to let a Moose script be
// type-checked with a plain C# compiler *before* it is run inside Moose:
//
//     mcs -target:library -out:moose_api_stubs.dll moose_api_stubs.cs
//     mcs -out:check.exe -r:moose_api_stubs.dll my_script.cs
//
// which catches typos, wrong argument counts and wrong overloads in seconds
// instead of after a failed run.  It is NOT a simulator: every method returns
// zero/null, so a script linked against these stubs computes nothing.
//
// If a signature here disagrees with your Moose build, the build wins -- fix
// the stub.  (Known case: the help and the shipped unit tests disagree about
// the circular Atom constructor, see README.md.)
using System;

public class Complex
{
    public double re, im;
    public Complex(double rRe = 0.0, double rIm = 0.0) { re = rRe; im = rIm; }
    public double Re() { return re; }
    public double Im() { return im; }
    public double Abs() { return 0; }
    public double AbsSquare() { return 0; }
    public double Arg() { return 0; }
    public static double Abs(Complex v) { return 0; }
}

public class Materials
{
    public Materials(string rModel, string rName) { }
    public Materials(Materials pMaterial) { }
    public Materials(double rIndexN, double rIndexK) { }
    public void Delete() { }
    public Complex GetIndex(double rWavelength) { return null; }
    public Complex GetEpsilon(double rWavelength) { return null; }
    public double GetIndexN(double rWavelength) { return 0; }
    public double GetIndexK(double rWavelength) { return 0; }
    public string GetName() { return null; }
    public bool IsAir() { return false; }
    public bool Equals(Materials pMaterial) { return false; }
    public void AddSmallImag(bool rAddSmallImag) { }
}

public class Atom
{
    public Atom(double rPosX, double rWidthX, Materials pMaterial) { }
    public Atom(double rPosX, double rPosY, double rWidthX, double rWidthY, Materials pMaterial) { }
    public Atom(double rPosX, double rPosY, double rRadius, Materials pMaterial) { }
    public Atom(double rPosX, double rWidthX, double rAngleX, int rNSlices, Materials pMaterial) { }
    public Atom(double rPosX, double rPosY, double rWidthX, double rWidthY,
                double rAngleLowX, double rAngleHighX, double rAngleLowY,
                double rAngleHighY, int rNSlices, Materials pMaterial) { }
    public Atom(Atom pAtom) { }
    public void Delete() { }
    public double GetWidthX() { return 0; }
    public double GetWidthY() { return 0; }
    public void SetWidthX(double w) { }
    public void SetWidthY(double w) { }
    public double GetPositionX() { return 0; }
    public double GetPositionY() { return 0; }
    public double GetStartX() { return 0; }
    public double GetStopX() { return 0; }
    public double GetStartY() { return 0; }
    public double GetStopY() { return 0; }
    public Complex GetIndex(double rWaveLength) { return null; }
    public Materials GetMaterial() { return null; }
    public void SetMaterial(Materials pMaterial) { }
    public void SetAutoDelete(bool rAutoDelete) { }
}

public class CaModel
{
    public CaModel(int nx, int ny) { }
    public CaModel(int nx, int ny, double dx, double dy) { }
    public void Delete() { }
    public int Show() { return 0; }
    public int Show(string rTitle) { return 0; }
    public void AddToLog(string rComment, bool rAddTimeStamp = true) { }
    public void SetTitle(string t) { }
    public void SetValue(int x, int y, Complex v) { }
    public Complex GetValue(int x, int y) { return null; }
    public void UpdateView(int w, bool a, bool b) { }
    public int GetDimX() { return 0; }
    public int GetDimY() { return 0; }
}

public class Layer
{
    public Layer(double rThickness, Materials pMaterial, string rAddInfo = "", bool rStructurable = true) { }
    public Layer(double rThickness, Materials pMaterial, double rDutyCycle, Materials pMaterialAtom) { }
    public Layer(double rThickness, Materials pMaterial, int rNAtoms, Atom[] pAtoms) { }
    public Layer(double rThickness, CaModel pEpsilonDistribution) { }
    public Layer(Layer pLayer) { }
    public void Delete() { }
    public string GetStructure() { return null; }
    public bool IsStructured() { return false; }
    public int GetNAtoms() { return 0; }
    public Atom GetAtom(int rAtom) { return null; }
    public double GetThickness() { return 0; }
    public Complex GetIndex(double rWaveLength, int rAtom = -1) { return null; }
    public void SetMaterial(Materials pMaterial, int rAtom = -1) { }
    public Materials GetMaterial(int rAtom = -1) { return null; }
    public void Declare2D() { }
    public void StructureBinary(double rDutyCycle, Materials rMaterialAtom) { }
    public void SetThickness(double rThickness) { }
    public void SetDutyCycle(double rDutyCycle) { }
    public CaModel GetEpsilonDistributionsAsCaModel(double wl, int nx, int ny, bool inv) { return null; }
    public void SetAutoDelete(bool rAutoDelete) { }
}

public class GratingStructure
{
    public GratingStructure(double rPeriodX, double rPeriodY, Materials rSuperstrate, Materials rSubstrate) { }
    public GratingStructure(GratingStructure pGratingStruture) { }
    public GratingStructure(string rXmlString) { }
    public void Delete() { }
    public string GetXmlLog() { return null; }
    public void SetPeriodX(double p) { }
    public void SetPeriodY(double p) { }
    public double GetPeriodX() { return 0; }
    public double GetPeriodY() { return 0; }
    public Materials GetSuperstrate() { return null; }
    public Materials GetSubstrate() { return null; }
    public void AddSimpleBinaryGrating(double rDepth, double rDutyCycle, double rThickness, Materials rMaterialTrench, Materials rMaterialBar) { }
    public void AddDoubleLayerStack(int n, double ta, double tb, Materials a, Materials b) { }
    public void AddLayerOnBottom(double rThickness, Materials rMaterial) { }
    public void AddLayerOnTop(Layer pLayer) { }
    public void AddLayerOnBottom(Layer pLayer) { }
    public void AddGratingStructureOnBottom(GratingStructure g) { }
    public void InsertLayerBelow(int rNLayer, Layer pLayer) { }
    public void StructureBinaryGrating(double rDepth, double rDutyCycle, Materials rMaterialTrench) { }
    public void StructureFromMaskLayer(Atom[] pAtoms, double[] pDepths) { }
    public int GetNLayers() { return 0; }
    public Layer GetLayer(int rLayer) { return null; }
    public void ResolveAllLayers() { }
    public CaModel ConvertToCaModel(double rWavelength, int rWidth = 500, int rHeight = 600) { return null; }
    public void FlipUpsideDown() { }
    public string GetLog() { return null; }
    public static GratingStructure GetGratingFromRcwaDialog() { return null; }
    public void SetAutoDelete(bool rAutoDelete) { }
}

public class Rcwa
{
    public Rcwa(GratingStructure pGrating, int rOrdersX, int rOrdersY,
                int rRefinementFactorEpsFT = 5, long rRcwaCacheSize = 0) { }
    public void Delete() { }
    public void Calc(double rWavelength, double rTheta, double rConicalAngle,
                     double rPolarAngle, bool rEnforce2D = false) { }
    public double GetEfficiencyForGivenOrder(char rTransReflect, int rOrderX, int rOrderY,
                                             string rOutputPolarization = "in") { return 0; }
    public Complex GetPropagatingRayleighCoefficient(char tr, int ox, int oy, char comp) { return null; }
    public double GetAbsorption() { return 0; }
    public CaModel CalculateFields(double wl, double th, string pol, string field, int w, int h) { return null; }
}

public class RcwaParameters
{
    public RcwaParameters(int rNOrdersX, int rNOrdersY, int rFFTRefinement, long rCacheSize, bool rIncidenceFromSuperstrate) { }
    public void Delete() { }
}

public class RcwaStorage
{
    public RcwaStorage(string rTRA, bool rIncludeCompPol, int minx, int maxx, int miny, int maxy) { }
    public RcwaStorage(RcwaStorage pStorage) { }
    public void Delete() { }
    public Complex GetResult(string rQuantity, int ox, int oy, bool rIncludePhase) { return null; }
    public void SetAutoDelete(bool b) { }
}

public class RcwaIncidence
{
    public RcwaIncidence(double wl, double aoi, double conical, double pol) { }
    public void Delete() { }
}

public class RcwaTask
{
    public RcwaTask(RcwaIncidence i, GratingStructure g, RcwaStorage s, int n) { }
    public void Delete() { }
    public RcwaStorage GetStorage() { return null; }
}

public class ParallelRcwa
{
    public ParallelRcwa(int rNThreadsLocal) { }
    public void Delete() { }
    public void Calc(RcwaParameters p, RcwaTask[] tasks) { }
}

public class Io
{
    public static void output(string s) { Console.WriteLine(s); }
    public static void output(double v) { Console.WriteLine(v); }
    public static void output(Complex v) { }
    public static void error(string s) { Console.WriteLine("ERR " + s); }
    public static void error(double v) { }
    public static void success(string s) { Console.WriteLine("OK  " + s); }
    public static void success(double v) { }
    public static bool ShowParameterDialog(string t, int n, string[] l, string[] v, bool d = false) { return true; }
    public static bool ShowParameterDialog(string t, int n, string[] l, double[] v, bool d = false) { return true; }
    public static void ShowWarning(string t, string x) { }
    public static void ShowWarning(string x) { }
}
