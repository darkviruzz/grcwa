"""Example: a 0D planar multilayer (thin-film stack).

With no in-plane structuring there is no lattice: pass L1=None, L2=None. grcwa
then keeps a single order (0,0), nG=1, and reduces exactly to the transfer-
matrix method (TMM). Only uniform layers are used.

Physical setup: a single-layer anti-reflection (AR) coating on glass at
lambda = 1 micron (freq = 1.0), normal incidence. A bare air/glass interface
(n_glass = 1.5) reflects R0 = ((1-1.5)/(1+1.5))^2 = 0.04. A quarter-wave
coating with index n_c = sqrt(n_glass) = 1.225 and thickness lambda/(4 n_c)
ideally cancels the reflection (R -> 0).
"""
import grcwa
import numpy as np

freq = 1.0                      # lambda = 1 micron
n_glass = 1.5
n_c = np.sqrt(n_glass)          # ideal single-layer AR index
d_c = 1.0 / (4 * n_c)          # quarter-wave optical thickness at lambda=1

def reflectance(with_coating, theta=0.0, pol='s'):
    obj = grcwa.obj(1, None, None, freq, theta, 0., verbose=0)
    obj.Add_LayerUniform(1.0, 1.0)                 # incident: air
    if with_coating:
        obj.Add_LayerUniform(d_c, n_c**2)          # AR coating
    obj.Add_LayerUniform(1.0, n_glass**2)          # substrate: glass
    obj.Init_Setup()
    pa, sa = (1., 0.) if pol == 'p' else (0., 1.)
    obj.MakeExcitationPlanewave(pa, 0., sa, 0., order=0)
    return obj.RT_Solve(normalize=1)

R_bare, T_bare = reflectance(False)
R_ar, T_ar = reflectance(True)
print(f"bare air/glass : R={np.real(R_bare):.4f}  (analytic 0.0400)")
print(f"with AR coating: R={np.real(R_ar):.6f}  (should be ~0)")
print(f"energy check   : bare R+T={np.real(R_bare+T_bare):.4f}, "
      f"AR R+T={np.real(R_ar+T_ar):.4f}")

print("\nAngle dependence of the AR coating (still much better than bare):")
for deg in [0, 20, 40, 60]:
    Rs, _ = reflectance(True, theta=np.deg2rad(deg), pol='s')
    Rp, _ = reflectance(True, theta=np.deg2rad(deg), pol='p')
    print(f"  theta={deg:>2} deg:  R_s={np.real(Rs):.4f}  R_p={np.real(Rp):.4f}")
