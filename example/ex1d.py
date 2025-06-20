"""Demonstrate 1-D binary gratings with Ny=1 and Nx=1."""
import grcwa
import numpy as np

nG = 21
freq = 1.0
theta = 0.0
phi = 0.0

planewave = {'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}

# Variation along x (Ny=1)
L1 = [0.5, 0]
L2 = [0, 0]
Nx = 40
Ny = 1

obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=1, mode='1D')
obj.Add_LayerUniform(1., 1.)
obj.Add_LayerGrid(0.2, Nx, Ny)
obj.Add_LayerUniform(1., 1.)
obj.Init_Setup()
obj.MakeExcitationPlanewave(**planewave, order=0)

ep = np.ones((Nx, Ny)) * 4.
mask = np.arange(Nx) < Nx//2
ep[mask,0] = 1.
obj.GridLayer_geteps(ep.flatten())
R, T = obj.RT_Solve(normalize=1)
print('Ny=1: R=', R, ' T=', T)

# Variation along y (Nx=1)
L1 = [0, 0]
L2 = [0, 0.5]
Nx = 1
Ny = 40

obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=1, mode='1D')
obj.Add_LayerUniform(1., 1.)
obj.Add_LayerGrid(0.2, Nx, Ny)
obj.Add_LayerUniform(1., 1.)
obj.Init_Setup()
obj.MakeExcitationPlanewave(**planewave, order=0)

ep = np.ones((Nx, Ny)) * 4.
mask = np.arange(Ny) < Ny//2
ep[0,mask] = 1.
obj.GridLayer_geteps(ep.flatten())
R, T = obj.RT_Solve(normalize=1)
print('Nx=1: R=', R, ' T=', T)
