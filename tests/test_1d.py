import numpy as np
import grcwa

nG = 51
L1 = [1.0, 0]
L2 = [0, 1.0]
Nx = 101
Ny = 1
freq = 1.0
theta = 0.0
phi = 0.0

thick0 = 1.0
thickp = 0.2
thickN = 1.0

ep0 = 1.0
epp = 4.0
epbkg = 1.0

x = np.linspace(0, 1.0, Nx, endpoint=False)
epgrid = np.ones((Nx, Ny)) * epp
epgrid[x < 0.5] = epbkg

planewave = {'p_amp': 1, 's_amp': 0, 'p_phase': 0, 's_phase': 0}

def test_binary_grating_1d():
    obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)
    obj.Add_LayerUniform(thick0, ep0)
    obj.Add_LayerGrid(thickp, Nx, Ny)
    obj.Add_LayerUniform(thickN, ep0)
    obj.Init_Setup()
    obj.MakeExcitationPlanewave(planewave['p_amp'], planewave['p_phase'],
                                planewave['s_amp'], planewave['s_phase'])
    obj.GridLayer_geteps(epgrid.flatten())
    assert obj.Patterned_epinv_list[0].shape[0] == nG
