import numpy as np
import grcwa
from grcwa.fft_funs import get_conv
from .utils import t_grad

try:
    import autograd.numpy as npa
    from autograd import grad
    AG_AVAILABLE = True
except ImportError:
    AG_AVAILABLE = False
    
L1 = [0.5,0]
L2 = [0,0.2]
nG = 100
method = 0
kx0 = 0.1
ky0 = 0.2

Lk1,Lk2 = grcwa.Lattice_Reciprocate(L1,L2)
G,nGout = grcwa.Lattice_getG(nG,Lk1,Lk2,method=method)
kx, ky = grcwa.Lattice_SetKs(G, kx0, ky0, Lk1, Lk2)
    
def test_bloch():
    assert nGout>0,'negative nG'
    assert nGout<=nG,'wrong nG'

def test_getG_numpy_int():
    nG_np = np.int32(50)
    G2, nGout2 = grcwa.Lattice_getG(nG_np, Lk1, Lk2, method=method)
    assert nGout2 > 0

if AG_AVAILABLE:
    grcwa.set_backend('autograd')
    Nx = 51
    Ny = 71
    dN = 1./Nx/Ny
    tol = 1e-2    
    
    def test_fft():
        def fun(ep):
            epout = npa.reshape(ep,(Nx,Ny))
            epsinv, eps2 = grcwa.Epsilon_fft(dN,epout,G)
            return npa.real(npa.sum(epsinv))

        grad_fun = grad(fun)

        x = 1.+10.*np.random.random(Nx*Ny)
        dx = 1e-3
        ind = np.random.randint(Nx*Ny,size=1)[0]        
        FD, AD = t_grad(fun,grad_fun,x,dx,ind)
        assert abs(FD-AD)<abs(FD)*tol,'wrong fft gradient'

    def test_fft_aniso():
        def fun(ep):
            epout = [npa.reshape(ep[x*Nx*Ny:(x+1)*Nx*Ny],(Nx,Ny)) for x in range(3)]
            epsinv, eps2 = grcwa.Epsilon_fft(dN,epout,G)
            return npa.real(npa.sum(eps2))

        grad_fun = grad(fun)

        x = 1.+10.*np.random.random(3*Nx*Ny)
        dx = 1e-3
        ind = np.random.randint(Nx*Ny*2,size=1)[0]        
        FD, AD = t_grad(fun,grad_fun,x,dx,ind)
        assert abs(FD-AD)<abs(FD)*tol,'wrong fft gradient'        

    def test_ifft():
        ix = np.random.randint(Nx,size=1)[0]
        iy = np.random.randint(Ny,size=1)[0]
        def fun(x):
            out = grcwa.get_ifft(Nx,Ny,x,G)
            return npa.real(out[ix,iy])

        grad_fun = grad(fun)

        x = 10.*np.random.random(nGout)
        dx = 1e-3
        ind = np.random.randint(nGout,size=1)[0]
        FD, AD = t_grad(fun,grad_fun,x,dx,ind)
        assert abs(FD-AD)<abs(FD)*tol,'wrong ifft gradient'

# -------------------------------------------------------------
# Tests below do not require autograd

def _old_get_conv(dN, grid, G):
    """Original meshgrid-based convolution for reference."""
    nG = G.shape[0]
    sfft = np.fft.fft2(grid) * dN
    ix = range(nG)
    ii, jj = np.meshgrid(ix, ix, indexing="ij")
    return sfft[G[ii, 0] - G[jj, 0], G[ii, 1] - G[jj, 1]]


def test_get_conv_broadcast_random():
    """New broadcast implementation matches previous version."""
    np.random.seed(0)
    for _ in range(3):
        Nx = np.random.randint(8, 16)
        Ny = np.random.randint(8, 16)
        grid = np.random.random((Nx, Ny))
        dN = 1.0 / Nx / Ny
        nG_local = np.random.randint(5, 15)
        G_local, _ = grcwa.Lattice_getG(nG_local, Lk1, Lk2, method=method)
        new = get_conv(dN, grid, G_local)
        old = _old_get_conv(dN, grid, G_local)
        assert np.allclose(new, old)
