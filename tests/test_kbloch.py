import numpy as np
import grcwa
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
        # Ensure the updated get_ifft produces the same output as the
        # original loop-based implementation using the numpy backend.
        def old_get_ifft(Nx, Ny, s_in, G):
            dN = 1.0 / Nx / Ny
            s0 = np.zeros((Nx, Ny), dtype=complex)
            for i in range(G.shape[0]):
                stmp = np.zeros((Nx, Ny), dtype=complex)
                stmp[G[i, 0], G[i, 1]] = 1.0
                s0 = s0 + s_in[i] * stmp
            return np.fft.ifft2(s0) / dN

        grcwa.set_backend('numpy')
        x = 10.0 * np.random.random(nGout)
        new = grcwa.get_ifft(Nx, Ny, x, G)
        ref = old_get_ifft(Nx, Ny, x, G)
        grcwa.set_backend('autograd')
        assert np.allclose(ref, new)

    def test_get_conv():
        # Ensure broadcasting version of get_conv matches the previous
        # meshgrid implementation on random grids.
        def old_get_conv(dN, s_in, G):
            sfft = np.fft.fft2(s_in) * dN
            nG, _ = G.shape
            ix = range(nG)
            ii, jj = np.meshgrid(ix, ix, indexing='ij')
            return sfft[G[ii, 0] - G[jj, 0], G[ii, 1] - G[jj, 1]]

        grcwa.set_backend('numpy')
        s = np.random.random((Nx, Ny))
        new = grcwa.get_conv(dN, s, G)
        ref = old_get_conv(dN, s, G)
        grcwa.set_backend('autograd')
        assert np.allclose(ref, new)

    def test_getG_1d():
        Lk1 = np.array([1.0, 0.0])
        Lk2 = np.array([0.0, 0.0])
        G1, n1 = grcwa.Lattice_getG(5, Lk1, Lk2, method=0)
        assert G1.ndim == 1
        assert n1 == len(G1)
        assert G1[0] == -2 and G1[-1] == 2

    def test_fft_1d():
        N = 16
        dN1 = 1.0 / N
        G1 = np.arange(-2, 3)
        s = np.random.random((N, 1))
        grcwa.set_backend('numpy')
        new = grcwa.get_fft(dN1, s, G1)
        ref = np.fft.fft(s.flatten()) * dN1
        grcwa.set_backend('autograd')
        assert np.allclose(ref[G1], new)

    def test_ifft_1d():
        N = 8
        dN1 = 1.0 / N
        G1 = np.arange(-2, 3)
        x = np.random.random(len(G1))
        grcwa.set_backend('numpy')
        new = grcwa.get_ifft(1, N, x, G1)
        ref_full = np.zeros(N, dtype=complex)
        ref_full[G1] = x
        ref = np.fft.ifft(ref_full) / dN1
        grcwa.set_backend('autograd')
        assert np.allclose(ref.reshape(1, N), new)
