"""Example: a 1D binary grating (lamellar grating).

grcwa is a general 2D solver, but a 1D grating (periodic along x, invariant
along y) is requested simply by passing L2=None. Only orders (m,0) are kept,
so the truncation `nG` becomes exactly 2M+1, and a grid layer is given as a
1D array of length Nx (Ny defaults to 1).

Physical setup (units where the vacuum speed of light = 1, so freq = 1/lambda):
we work at lambda = 1 micron (freq = 1.0) and sweep the grating period across
the wavelength. A sub-wavelength period supports only the 0th order, so the
grating behaves like an effective-medium thin film (R+T = 1, no diffraction);
once the period exceeds ~lambda, higher diffraction orders open up.
"""
import grcwa
import numpy as np

freq = 1.0          # lambda = 1 micron
# a tiny fictitious loss regularizes the Rayleigh/Wood anomaly that occurs at
# normal incidence when an open diffraction order goes grazing (e.g. Lambda
# exactly = lambda); at Qabs -> inf this approaches the lossless result.
Qabs = 1e6
freqc = freq * (1 + 1j / 2 / Qabs)
theta = 0.0         # normal incidence
phi = 0.0
nG = 51             # -> 51 retained orders (M = 25)

Nx = 512
xs = np.linspace(0, 1, Nx, endpoint=False)

# Silicon-like ridges (n = 3.5 -> eps = 12.25) in air, 50% duty cycle,
# free-standing slab of thickness 0.3 micron.
eps_ridge = 3.5**2
eps_grid = np.where(xs < 0.5, eps_ridge, 1.0)
thick = 0.3

lam = 1.0 / freq    # wavelength in the surrounding air
for period in [0.4, 1.5, 2.5]:   # sub-, then supra-wavelength
    obj = grcwa.obj(nG, [period, 0], None, freqc, theta, phi, verbose=0)
    obj.Add_LayerUniform(1.0, 1.0)         # incident: air
    obj.Add_LayerGrid(thick, Nx)           # 1D grating (Ny defaults to 1)
    obj.Add_LayerUniform(1.0, 1.0)         # transmission: air
    obj.Init_Setup()
    # s-polarized (TE: E along the grooves)
    obj.MakeExcitationPlanewave(0., 0., 1., 0., order=0)
    obj.GridLayer_geteps(eps_grid)

    R, T = obj.RT_Solve(normalize=1)
    # number of propagating orders from the grating equation (normal incidence,
    # air on both sides): |m| < Lambda/lambda
    mmax = int(np.floor(period / lam - 1e-9))
    n_prop = 2 * mmax + 1
    print(f"period={period:.2f} um (Lambda/lambda={period/lam:.2f}):  "
          f"R={np.real(R):.4f}  T={np.real(T):.4f}  R+T={np.real(R+T):.4f}  "
          f"propagating orders = {n_prop}")

print("\nExpected: lossless so R+T=1; for Lambda<lambda only the 0th order "
      "propagates (effective medium), and more diffraction orders open up as "
      "the period grows past the wavelength.")
