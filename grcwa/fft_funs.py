from . import backend as bd

def Epsilon_fft(dN,eps_grid,G):
    '''dN = 1/Nx/Ny
    For now, assume epsilon is diagonal; if epsilon has xz,yz component, just simply add them to off-diagonal eps2
    
    eps_grid is  (1) for isotropic, a numpy 2d array in the format of (Nx,Ny),
                 (2) for anisotropic, a list of numpy 2d array [(Nx,Ny),(Nx,Ny),(Nx,Ny)]
    '''

    if len(eps_grid) == 3 and eps_grid[0].ndim == 2:
        epsx_fft = get_conv(dN,eps_grid[0],G)
        epsy_fft = get_conv(dN,eps_grid[1],G)
        epsz_fft = get_conv(dN,eps_grid[2],G)
        epsinv = bd.inv(epsz_fft)
    
        tmp1 = bd.vstack((epsx_fft,bd.zeros_like(epsx_fft)))
        tmp2 = bd.vstack((bd.zeros_like(epsx_fft),epsy_fft))
        eps2 = bd.hstack((tmp1,tmp2))
        
    elif eps_grid[0].ndim == 1:
        eps_fft = get_conv(dN,eps_grid,G)
        epsinv = bd.inv(eps_fft)
    
        tmp1 = bd.vstack((eps_fft,bd.zeros_like(eps_fft)))
        tmp2 = bd.vstack((bd.zeros_like(eps_fft),eps_fft))
        eps2 = bd.hstack((tmp1,tmp2))
    else:
        raise ValueError("Wrong eps_grid type")

    return epsinv, eps2
    
def get_conv(dN,s_in,G):
    ''' Attain convolution matrix
    dN = 1/Nx/Ny
    s_in: np.array of length Nx*Ny
    G: shape (nG,2), 2 for Lk1,Lk2
    s_out: 1/N sum a_m exp(-2pi i mk/n), shape (nGx*nGy)
    '''
    sfft = bd.fft2(s_in) * dN

    gi = G[:, 0][:, None] - G[:, 0]
    gj = G[:, 1][:, None] - G[:, 1]
    s_out = sfft[gi, gj]
    return s_out

def get_fft(dN,s_in,G):
    '''
    FFT to get Fourier components
    
    s_in: np.2d array of size (Nx,Ny)
    G: shape (nG,2), 2 for Gx,Gy
    s_out: 1/N sum a_m exp(-2pi i mk/n), shape (nGx*nGy)
    '''
    
    sfft = bd.fft2(s_in)*dN
    return sfft[G[:,0],G[:,1]]


def get_ifft(Nx,Ny,s_in,G):
    '''
    Reconstruct real-space fields
    '''
    dN = 1.0 / Nx / Ny

    # Directly assign each Fourier coefficient to its corresponding
    # location in the frequency domain array.  This is equivalent to
    # the previous explicit loop but vectorised for efficiency.
    s0 = bd.zeros((Nx, Ny), dtype=complex)
    s0[G[:, 0], G[:, 1]] = s_in

    s_out = bd.ifft2(s0)/dN
    return s_out
