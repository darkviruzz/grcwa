import numpy as np

from . import backend as bd

# Threshold for detecting material interfaces in the Pol tangent field.
# Gradients below this are treated as uniform (no interface).
POL_GRAD_TOL = np.finfo(np.float32).eps  # ~1.19e-7


def Epsilon_fft(dN, eps_grid, G):
    '''dN = 1/Nx/Ny
    For now, assume epsilon is diagonal; if epsilon has xz,yz component, just simply add them to off-diagonal eps2

    eps_grid is  (1) for isotropic, a numpy 2d array in the format of (Nx,Ny),
                 (2) for anisotropic, a list of numpy 2d array [(Nx,Ny),(Nx,Ny),(Nx,Ny)]
    '''

    if len(eps_grid) == 3 and eps_grid[0].ndim == 2:
        epsx_fft = get_conv(dN, eps_grid[0], G)
        epsy_fft = get_conv(dN, eps_grid[1], G)
        epsz_fft = get_conv(dN, eps_grid[2], G)
        epsinv = bd.inv(epsz_fft)

        tmp1 = bd.vstack((epsx_fft, bd.zeros_like(epsx_fft)))
        tmp2 = bd.vstack((bd.zeros_like(epsx_fft), epsy_fft))
        eps2 = bd.hstack((tmp1, tmp2))

    elif eps_grid[0].ndim == 1:
        eps_fft = get_conv(dN, eps_grid, G)
        epsinv = bd.inv(eps_fft)

        tmp1 = bd.vstack((eps_fft, bd.zeros_like(eps_fft)))
        tmp2 = bd.vstack((bd.zeros_like(eps_fft), eps_fft))
        eps2 = bd.hstack((tmp1, tmp2))
    else:
        raise ValueError("Wrong eps_grid type")

    return epsinv, eps2


def Epsilon_fft_pol(dN, eps_grid, G, pol_sigma=1.0 / 12.0, pol_niter=0):
    """Fourier-space epsilon matrices using the Pol method (S4 paper Eq. 51).

    Implements the PolBasisVL formulation from S4 (fmm_PolBasisVL.cpp), which
    applies proper Fourier factorization rules (Li 1996/1997) using a tangent
    vector field at material interfaces.  This converges significantly faster
    than the naive FFT/Laurent formulation (Epsilon_fft) for structures with
    discontinuous permittivity.

    The two returned matrices feed into the RCWA eigensolve exactly like
    Epsilon_fft.  The method is fully compatible with HIPS/autograd for
    gradient-based topology optimization.

    Parameters
    ----------
    dN : float
        1/(Nx*Ny), normalisation constant for the Fourier transform.
    eps_grid : 2d array (Nx, Ny)
        Isotropic permittivity on the real-space grid.  May be an autograd
        ArrayBox for automatic differentiation.
    G : array (nG, 2)
        Integer G-vector indices (Lk1, Lk2 components).
    pol_sigma : float
        Gaussian blur sigma as a fraction of the grid period.  It is converted
        to pixels using the largest grid dimension.
    pol_niter : int
        Number of optional blur+reset iterations.  The default 0 performs the
        reference single blur without re-pinning rasterized interface pixels.

    Returns
    -------
    epsinv : (nG, nG) complex array
        Inverse of the Toeplitz matrix of eps (used for the kp matrix).
    eps2 : (2*nG, 2*nG) complex array
        Pol-corrected in-plane epsilon operator.  It acts on the rotated field
        vector [-Ey, Ex], rather than directly on [Ex, Ey].

    References
    ----------
    V. Liu & S. Fan, Comp. Phys. Comm. 183, 2233 (2012), Eq. 51.
    """

    # --- epsinv: for the Ez/kp term, use the SAME convention as Laurent,
    # epsinv = inv([[eps]]).  (The earlier code used the reciprocal Toeplitz
    # [[1/eps]] here, which is inconsistent with how kp couples into the
    # eigenproblem M = ep2*kp - kkT and destroyed TM convergence -- Pol oscillated
    # instead of settling.  With inv([[eps]]) the kp matrix matches the Laurent
    # baseline that the eps2 correction is built on top of.) ---
    eps_hat = get_conv(dN, eps_grid, G)     # Toeplitz(eps)
    eta_hat = get_conv(dN, 1.0 / eps_grid, G)  # Toeplitz(1/eps), used in mDelta
    epsinv = bd.inv(eps_hat)

    # Tangent-field projection operators P_ij (autograd-compatible)
    P_xx, P_xy, P_yx, P_yy = _compute_tangent_field_pol(
        eps_grid, pol_sigma=pol_sigma, pol_niter=pol_niter
    )

    # Fourier-transform the projection operators into (nG, nG) matrices
    P_xx_hat = get_conv(dN, P_xx, G)
    P_xy_hat = get_conv(dN, P_xy, G)
    P_yx_hat = get_conv(dN, P_yx, G)
    P_yy_hat = get_conv(dN, P_yy, G)

    # S4 sign convention (fmm_PolBasisVL.cpp lines 257-268):
    #   mDelta = inv(Eta) - Epsilon       (negated Delta in Eq. 51)
    #   eps2[block] += mDelta @ P[block]
    # eps2 acts on [-Ey, Ex].  Consequently these blocks represent the
    # rotated tensor [[eps_yy, -eps_yx], [-eps_xy, eps_xx]], and the tangent
    # projector returned below is the physical normal projector in this basis.
    mDelta = bd.inv(eta_hat) - eps_hat

    E_xx = eps_hat + bd.dot(mDelta, P_xx_hat)
    E_xy = bd.dot(mDelta, P_xy_hat)
    E_yx = bd.dot(mDelta, P_yx_hat)
    E_yy = eps_hat + bd.dot(mDelta, P_yy_hat)

    top = bd.hstack((E_xx, E_xy))
    bot = bd.hstack((E_yx, E_yy))
    eps2 = bd.vstack((top, bot))

    return epsinv, eps2


def _compute_tangent_field_pol(eps_grid, pol_sigma=1.0 / 12.0, pol_niter=0):
    """Tangent vector field and projection operators for the Pol method.

    Computes a smooth, undirected interface orientation and returns the
    tangent outer-product projection matrices used by Epsilon_fft_pol.  Since
    eps2 acts on [-Ey, Ex], this tangent projection applies the inverse rule
    along the physical interface normal.  The implementation is fully
    autograd-compatible so that gradients propagate correctly through the Pol
    correction during topology optimization.

    Algorithm:
      1. Detect interfaces via periodic central finite differences of the real
         and imaginary parts of eps.
      2. Encode the normal as the doubled angle z = (gx + i*gy)^2, so opposite
         interface normals have the same orientation instead of cancelling.
      3. Blur Re(z) and Im(z) once to diffuse the orientation through the cell.
         Optional positive pol_niter values retain the legacy blur+reset
         extension, but the reference/default path does not re-pin the smooth
         field to staircase pixels at a rasterized curved boundary.
      4. Recover the unit tangent projection directly from cos(2*theta) and
         sin(2*theta), with no angle/arctan2 operation.

    All operations use the ``bd`` backend so autograd can differentiate
    through them.  The only raw-numpy operations are the Gaussian kernel
    and interface mask (fixed constants) and the early-exit checks.

    Parameters
    ----------
    eps_grid : 2d array (Nx, Ny)
        Isotropic permittivity grid.  May be an autograd ArrayBox.
    pol_sigma : float
        Gaussian blur sigma as a fraction of the grid period.  It is converted
        to pixels using max(Nx, Ny), matching the reference normal-vector rule.
    pol_niter : int
        Number of optional blur+reset iterations.  0 (the default) means a
        single blur with no interface reset.

    Returns
    -------
    P_xx, P_xy, P_yx, P_yy : 2d arrays (Nx, Ny)
        Projection-operator components.  P_yx == P_xy (symmetric).
        Autograd-tracked when eps_grid is tracked.
    """
    Nx = eps_grid.shape[0]
    Ny = eps_grid.shape[1]

    eps_re = bd.real(eps_grid)
    eps_im = bd.imag(eps_grid)

    # --- 1. Interface detection: periodic central differences. ---
    # Keep real and imaginary gradients separate.  Combining their squared
    # doubled-angle fields below gives a complex-safe structure tensor: a step
    # with contrast a+i*b is weighted by a**2+b**2, so neither opposite signs
    # nor equal-magnitude phase contrasts can make a real interface disappear.
    def _central_grad(field):
        x_next = bd.concatenate([field[1:, :], field[:1, :]], axis=0)
        x_prev = bd.concatenate([field[-1:, :], field[:-1, :]], axis=0)
        y_next = bd.concatenate([field[:, 1:], field[:, :1]], axis=1)
        y_prev = bd.concatenate([field[:, -1:], field[:, :-1]], axis=1)
        return 0.5 * (x_next - x_prev), 0.5 * (y_next - y_prev)

    grad_x_re, grad_y_re = _central_grad(eps_re)
    grad_x_im, grad_y_im = _central_grad(eps_im)

    # Early exit for uniform permittivity (detached check).
    def _detach(arr):
        return np.real(np.asarray(arr._value if hasattr(arr, "_value") else arr))

    _grad_mag = np.sqrt(
        _detach(grad_x_re) ** 2 + _detach(grad_y_re) ** 2
        + _detach(grad_x_im) ** 2 + _detach(grad_y_im) ** 2)
    _max_grad = np.max(_grad_mag)
    if _max_grad < POL_GRAD_TOL:
        z = bd.zeros_like(eps_re)
        return z, z, z, z

    # --- 2. Double-angle encoding of the undirected interface normal. ---
    # Squaring maps normals n and -n to the same value, so opposite-facing
    # interfaces reinforce instead of cancelling during the extension.  The
    # projection is recovered algebraically below; no angle/arctan2 backend
    # primitive is needed.
    z_raw = ((grad_x_re + 1j * grad_y_re) ** 2
             + (grad_x_im + 1j * grad_y_im) ** 2)
    z_re_raw = bd.real(z_raw)
    z_im_raw = bd.imag(z_raw)

    # --- 3. Iterative blur+reset to extend the orientation smoothly ---
    # The blur kernel and interface mask are constant numpy arrays (not
    # differentiated).  pol_sigma is a fraction of the period; converting it
    # with the largest dimension also handles native 1D grids shaped (N, 1).
    kx_freq = np.fft.fftfreq(Nx)
    ky_freq = np.fft.fftfreq(Ny)
    KX, KY = np.meshgrid(kx_freq, ky_freq, indexing="ij")
    sigma_px = pol_sigma * max(Nx, Ny)
    blur_kernel = np.exp(-2 * np.pi**2 * sigma_px**2 * (KX**2 + KY**2))

    if pol_niter <= 0:
        # Single blur, no reset.
        z_re = bd.real(bd.ifft2(bd.fft2(z_re_raw) * blur_kernel))
        z_im = bd.real(bd.ifft2(bd.fft2(z_im_raw) * blur_kernel))
    else:
        # Interface mask: constant (detached) binary array.
        mask = (_grad_mag > POL_GRAD_TOL).astype(float)
        mask_inv = 1.0 - mask

        z_re = z_re_raw
        z_im = z_im_raw
        for _ in range(pol_niter):
            z_re = mask * z_re_raw + mask_inv * bd.real(
                bd.ifft2(bd.fft2(z_re) * blur_kernel))
            z_im = mask * z_im_raw + mask_inv * bd.real(
                bd.ifft2(bd.fft2(z_im) * blur_kernel))

    # --- 4. Recover the tangent projection from the doubled normal angle. ---
    # A detached mask gives undefined/near-zero orientations a zero projection.
    # Adding the invalid mask *inside* sqrt keeps autograd finite at z=0 while
    # retaining an exact |z| denominator on valid pixels.  A blanket additive
    # floor would bias cos(2 theta) away from +/-1 and leak the correction into
    # TE on an axis-aligned 1D interface.
    z_mag_sq = z_re * z_re + z_im * z_im
    _raw_mag_sq = np.real(np.asarray(
        z_mag_sq._value if hasattr(z_mag_sq, "_value") else z_mag_sq))
    peak_z_mag_sq = float(np.max(_raw_mag_sq))
    if peak_z_mag_sq < POL_GRAD_TOL**4:
        z = bd.zeros_like(eps_re)
        return z, z, z, z

    z_tol_sq = POL_GRAD_TOL**2 * peak_z_mag_sq
    valid = (_raw_mag_sq > z_tol_sq).astype(float)
    z_mag = bd.sqrt(z_mag_sq + (1.0 - valid))
    cos_2theta = valid * z_re / z_mag
    sin_2theta = valid * z_im / z_mag

    # eps2 uses the rotated basis [-Ey, Ex], so the physical normal projector
    # becomes the tangent projector in this basis.  For an x-normal 1D grating
    # this gives P_xx=0, P_yy=1 and leaves TE exactly on Laurent's rule.
    P_xx = 0.5 * valid * (1.0 - cos_2theta)
    P_yy = 0.5 * valid * (1.0 + cos_2theta)
    P_xy = -0.5 * valid * sin_2theta

    return P_xx, P_xy, P_xy, P_yy


def get_conv(dN, s_in, G):
    ''' Attain convolution matrix
    dN = 1/Nx/Ny
    s_in: np.array of length Nx*Ny
    G: shape (nG,2), 2 for Lk1,Lk2
    s_out: 1/N sum a_m exp(-2pi i mk/n), shape (nGx*nGy)

    Auto-upsamples via nearest-neighbor when the grid is too coarse
    for the requested G-vectors (prevents FFT index wrap-around / aliasing).
    This is what makes 1D grids (Ny=1) and coarse grids robust.
    '''
    Nx = s_in.shape[0]
    Ny = s_in.shape[1]

    # Largest G-vector difference we must index in the FFT spectrum.
    max_gdiff_x = int(np.max(G[:, 0])) - int(np.min(G[:, 0]))
    max_gdiff_y = int(np.max(G[:, 1])) - int(np.min(G[:, 1]))

    if max_gdiff_x >= Nx or max_gdiff_y >= Ny:
        # Upsample (nearest-neighbor) so every needed difference order is
        # representable without FFT index wrap-around.  Integer-division
        # indexing keeps this autograd-safe.
        scale_x = max(int(np.ceil((max_gdiff_x + 1) / Nx)), 1)
        scale_y = max(int(np.ceil((max_gdiff_y + 1) / Ny)), 1)
        idx_x = np.arange(Nx * scale_x) // scale_x
        idx_y = np.arange(Ny * scale_y) // scale_y
        s_work = s_in[idx_x][:, idx_y]
        dN_work = 1.0 / (Nx * scale_x) / (Ny * scale_y)
    else:
        s_work = s_in
        dN_work = dN

    sfft = bd.fft2(s_work) * dN_work

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

    # Scatter each Fourier coefficient to its location in the frequency
    # domain array.  This is done by gathering from a padded copy of s_in
    # rather than by assigning into a zero array: writing a traced value
    # into a concrete array (s0[idx] = s_in) is not differentiable under
    # autograd -- it raises "must be real number, not ArrayBox" -- while
    # indexing *out of* a traced array is a differentiable primitive.
    # Vectorised, so the speed of the assignment version is retained.
    nG = G.shape[0]

    # Negative orders wrap around, matching numpy's negative indexing.
    flat = (G[:, 0] % Nx) * Ny + (G[:, 1] % Ny)

    if len(np.unique(flat)) != nG:
        raise ValueError(
            'get_ifft: the %dx%d real-space grid is too coarse for the '
            'given G vectors; distinct orders alias onto the same grid '
            'point. Use a finer grid or a smaller truncation order.'
            % (Nx, Ny))

    # take[p] selects, for flat position p, the coefficient living there;
    # positions carrying no coefficient point at the trailing zero.
    take = np.full(Nx * Ny, nG, dtype=int)
    take[flat] = np.arange(nG)

    padded = bd.concatenate((s_in, bd.zeros(1, dtype=complex)))
    s0 = bd.reshape(padded[take], (Nx, Ny))

    s_out = bd.ifft2(s0)/dN
    return s_out
