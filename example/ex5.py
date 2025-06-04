"""Binary grating example with Ny=1 to demonstrate 1-D support."""
import grcwa
import numpy as np

nG = 51
L1 = [1.0, 0]
L2 = [0, 1.0]
freq = 1.0
theta = 0.0
phi = 0.0

Nx = 200
Ny = 1

ep0 = 1.
epp = 4.
epbkg = 1.

thick0 = 1.
thickp = 0.2
thickN = 1.

# binary grating
x = np.linspace(0, 1.0, Nx, endpoint=False)
epgrid = np.ones((Nx, Ny)) * epp
epgrid[x < 0.5] = epbkg

obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=1)
obj.Add_LayerUniform(thick0, ep0)
obj.Add_LayerGrid(thickp, Nx, Ny)
obj.Add_LayerUniform(thickN, ep0)
obj.Init_Setup()

planewave = {'p_amp': 1, 's_amp': 0, 'p_phase': 0, 's_phase': 0}
obj.MakeExcitationPlanewave(planewave['p_amp'], planewave['p_phase'],
                            planewave['s_amp'], planewave['s_phase'])
obj.GridLayer_geteps(epgrid.flatten())
R, T = obj.RT_Solve(normalize=1)
print('R=', R, 'T=', T, 'R+T=', R + T)
