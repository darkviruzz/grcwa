from . import backend as bd

def Epsilon_fft(dN,eps_grid,G):
    '''dN = 1/Nx/Ny
    For now, assume epsilon is diagonal; if epsilon has xz,yz component, just simply add them to off-diagonal eps2
    
    eps_grid is  (1) for isotropic, a numpy 2d array in the format of (Nx,Ny),
                 (2) for anisotropic, a list of numpy 2d array [(Nx,Ny),(Nx,Ny),(Nx,Ny)]
    '''

    if isinstance(eps_grid, list):
        epsx_fft = get_conv(dN, eps_grid[0], G)
        epsy_fft = get_conv(dN, eps_grid[1], G)
        epsz_fft = get_conv(dN, eps_grid[2], G)
        epsinv = bd.inv(epsz_fft)
    
        tmp1 = bd.vstack((epsx_fft,bd.zeros_like(epsx_fft)))
        tmp2 = bd.vstack((bd.zeros_like(epsx_fft),epsy_fft))
        eps2 = bd.hstack((tmp1,tmp2))
        
    else:
        eps_fft = get_conv(dN, eps_grid, G)
        epsinv = bd.inv(eps_fft)

        tmp1 = bd.vstack((eps_fft, bd.zeros_like(eps_fft)))
        tmp2 = bd.vstack((bd.zeros_like(eps_fft), eps_fft))
        eps2 = bd.hstack((tmp1, tmp2))

    return epsinv, eps2
    
def _get_conv1d(dN, s_in, G, axis):
    nG, _ = G.shape
    sfft = bd.fft(s_in) * dN
    orders = G[:, axis]
    ix = range(nG)
    ii, jj = bd.meshgrid(ix, ix, indexing='ij')
    N = s_in.shape[0]
    return sfft[(orders[ii] - orders[jj]) % N]


def get_conv(dN, s_in, G):
    ''' Attain convolution matrix
    dN = 1/Nx/Ny
    s_in: np.array of length Nx*Ny or Nx or Ny
    G: shape (nG,2), 2 for Lk1,Lk2
    s_out: 1/N sum a_m exp(-2pi i mk/n), shape (nGx*nGy)
    '''
    nG, _ = G.shape

    if s_in.ndim == 2 and 1 in s_in.shape:
        axis = 0 if s_in.shape[1] == 1 else 1
        vec = s_in[:, 0] if axis == 0 else s_in[0, :]
        return _get_conv1d(dN, vec, G, axis)

    if s_in.ndim == 1:
        return _get_conv1d(dN, s_in, G, 0)

    sfft = bd.fft2(s_in) * dN
    ix = range(nG)
    ii, jj = bd.meshgrid(ix, ix, indexing='ij')
    s_out = sfft[G[ii, 0] - G[jj, 0], G[ii, 1] - G[jj, 1]]
    return s_out

def _get_fft1d(dN, s_in, G, axis):
    sfft = bd.fft(s_in) * dN
    return sfft[G[:, axis]]


def get_fft(dN, s_in, G):
    '''
    FFT to get Fourier components

    s_in: np.2d array of size (Nx,Ny)
    G: shape (nG,2), 2 for Gx,Gy
    s_out: 1/N sum a_m exp(-2pi i mk/n), shape (nGx*nGy)
    '''

    if s_in.ndim == 2 and 1 in s_in.shape:
        axis = 0 if s_in.shape[1] == 1 else 1
        vec = s_in[:, 0] if axis == 0 else s_in[0, :]
        return _get_fft1d(dN, vec, G, axis)

    if s_in.ndim == 1:
        return _get_fft1d(dN, s_in, G, 0)

    sfft = bd.fft2(s_in) * dN
    return sfft[G[:, 0], G[:, 1]]


def _get_ifft1d(Nx, Ny, s_in, G, axis):
    dN = 1.0 / Nx / Ny
    nG, _ = G.shape
    N = Nx if axis == 0 else Ny
    s0 = bd.zeros(N, dtype=complex)
    orders = G[:, axis]
    for i in range(nG):
        stmp = bd.zeros(N, dtype=complex)
        stmp[orders[i] % N] = 1.0
        s0 = s0 + s_in[i] * stmp
    out = bd.ifft(s0) / dN
    if axis == 0:
        return out.reshape(Nx, 1)
    else:
        return out.reshape(1, Ny)


def get_ifft(Nx, Ny, s_in, G):
    '''
    Reconstruct real-space fields
    '''
    if Ny == 1:
        return _get_ifft1d(Nx, Ny, s_in, G, 0)
    if Nx == 1:
        return _get_ifft1d(Nx, Ny, s_in, G, 1)

    dN = 1.0 / Nx / Ny
    nG, _ = G.shape

    s0 = bd.zeros((Nx, Ny), dtype=complex)
    for i in range(nG):
        x = G[i, 0]
        y = G[i, 1]

        stmp = bd.zeros((Nx, Ny), dtype=complex)
        stmp[x, y] = 1.0
        s0 = s0 + s_in[i] * stmp

    s_out = bd.ifft2(s0) / dN
    return s_out
