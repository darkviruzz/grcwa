import numpy as np
import grcwa
import grcwa.rcwa as rcwa_core
from .utils import t_grad

try:
    import autograd.numpy as npa
    from autograd import grad
    AG_AVAILABLE = True
except ImportError:
    AG_AVAILABLE = False

tol = 1e-2  # error tolerance for autograd v.s. FD
tolS4 = 1e-3 # error tolerance for S4 v.s. this code

Nlayer = 1
nG = 101    
L1 = [0.1,0]
L2 = [0,0.1]
# all patterned layers below have the same griding structure: Nx*Ny
Nx = 100
Ny = 100

# now consider 3 layers: vacuum + patterned + vacuum
epsuniform0 = 1. # dielectric for layer 1 (uniform)
epsuniformN = 1.  # dielectric for layer N (uniform)

thick0 = 1. # thickness for vacuum layer 1
thickN = 1.

# frequency and angles
freq = 1.
theta = np.pi/18
phi = np.pi/9
Pscale = 1.

pthick = [0.2]    
# eps for patterned layer
radius = 0.4
epgrid = np.ones((Nx,Ny),dtype=float)
x0 = np.linspace(0,1.,Nx)
y0 = np.linspace(0,1.,Ny)
x, y = np.meshgrid(x0,y0,indexing='ij')
sphere = (x-.5)**2+(y-.5)**2<radius**2
epgrid[sphere] = 12.

planewave={'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}

def rcwa_assembly(epgrid,freq,theta,phi,planewave,pthick,Pscale=1.):
    '''
    planewave:{'p_amp',...}
    '''
    obj = grcwa.obj(nG,L1,L2,freq,theta,phi,verbose=1)
    obj.Add_LayerUniform(thick0,epsuniform0)
    for i in range(Nlayer):
        obj.Add_LayerGrid(pthick[i],Nx,Ny)
    obj.Add_LayerUniform(thickN,epsuniformN)
    
    obj.Init_Setup(Pscale=Pscale,Gmethod=0)
    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)
    obj.GridLayer_geteps(epgrid)
    
    return obj

   
def test_rcwa():
    ## compared to S4
    planewave={'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}
    obj=rcwa_assembly(epgrid,freq,theta,phi,planewave,pthick,Pscale=1.)
    R,T= obj.RT_Solve(normalize=0)
    assert abs(T-0.85249901083265)<tolS4 * T
    ## compared to S4
    planewave={'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
    obj=rcwa_assembly(epgrid,freq,theta,phi,planewave,pthick,Pscale=1.)
    R,T= obj.RT_Solve(normalize=0)
    assert abs(T-0.83900479939861)<tolS4 * T

    #others
    ai,bi = obj.GetAmplitudes(1,0.)
    assert len(ai) == obj.nG*2

    e,h = obj.Solve_FieldOnGrid(1,0.)
    assert e[0].shape == (Nx,Ny)

    Mx = np.real(obj.Patterned_epinv_list[0])
    val = obj.Volume_integral(1,Mx,Mx,Mx,normalize=1)
    assert np.real(val)>0

    Tx,Ty,Tz = obj.Solve_ZStressTensorIntegral(0)
    assert Tz<0


def test_smatrix_cache_unavailable_before_patterned_setup():
    planewave={'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}
    obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)
    obj.Add_LayerUniform(thick0, epsuniform0)
    obj.Add_LayerGrid(pthick[0], Nx, Ny)
    obj.Add_LayerUniform(thickN, epsuniformN)
    obj.Init_Setup(Pscale=1., Gmethod=0)
    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],
                                planewave['s_amp'],planewave['s_phase'],order=0)

    # Patterned layer eigensystem not yet available
    assert obj._get_smatrix_cache() is None


def test_getsmatrix_cached_vs_uncached_identical():
    planewave={'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}
    obj = rcwa_assembly(epgrid, freq, theta, phi, planewave, pthick, Pscale=1.)

    cache = obj._get_smatrix_cache()
    assert cache is not None
    assert len(cache['phi_inv_list']) == obj.Layer_N
    assert len(cache['kpphi_inv_list']) == obj.Layer_N

    s_cached = rcwa_core.GetSMatrix(0, obj.Layer_N-1, obj.q_list, obj.phi_list,
                                    obj.kp_list, obj.thickness_list,
                                    smatrix_cache=cache)
    s_plain = rcwa_core.GetSMatrix(0, obj.Layer_N-1, obj.q_list, obj.phi_list,
                                   obj.kp_list, obj.thickness_list,
                                   smatrix_cache=None)

    for blk_cached, blk_plain in zip(s_cached, s_plain):
        assert np.allclose(blk_cached, blk_plain)


def test_exterior_interior_cached_vs_uncached_identical():
    planewave={'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}
    obj = rcwa_assembly(epgrid, freq, theta, phi, planewave, pthick, Pscale=1.)

    cache = obj._get_smatrix_cache()
    aN_c, b0_c = rcwa_core.SolveExterior(obj.a0, obj.bN, obj.q_list, obj.phi_list,
                                         obj.kp_list, obj.thickness_list,
                                         smatrix_cache=cache)
    aN_p, b0_p = rcwa_core.SolveExterior(obj.a0, obj.bN, obj.q_list, obj.phi_list,
                                         obj.kp_list, obj.thickness_list,
                                         smatrix_cache=None)
    assert np.allclose(aN_c, aN_p)
    assert np.allclose(b0_c, b0_p)

    ai_c, bi_c = rcwa_core.SolveInterior(1, obj.a0, obj.bN, obj.q_list, obj.phi_list,
                                         obj.kp_list, obj.thickness_list,
                                         smatrix_cache=cache)
    ai_p, bi_p = rcwa_core.SolveInterior(1, obj.a0, obj.bN, obj.q_list, obj.phi_list,
                                         obj.kp_list, obj.thickness_list,
                                         smatrix_cache=None)
    assert np.allclose(ai_c, ai_p)
    assert np.allclose(bi_c, bi_p)

def test_smatrix_cache_reuse_and_invalidate():
    planewave={'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}
    obj = rcwa_assembly(epgrid,freq,theta,phi,planewave,pthick,Pscale=1.)

    # First solve builds cache
    R1, T1 = obj.RT_Solve(normalize=0)
    assert obj._smatrix_cache is not None
    old_kpphi = np.array(obj._smatrix_cache['kpphi_inv_list'][1], copy=True)

    # Repeated solve should reuse the same cache object and result
    R2, T2 = obj.RT_Solve(normalize=0)
    assert np.allclose([R1, T1], [R2, T2])

    # Updating patterned epsilon should invalidate and rebuild cache
    epgrid2 = np.array(epgrid, copy=True)
    epgrid2[0, 0] = epgrid2[0, 0] + 0.5
    obj.GridLayer_geteps(epgrid2.flatten())
    assert obj._smatrix_cache is None

    obj.RT_Solve(normalize=0)
    assert obj._smatrix_cache is not None
    assert not np.allclose(obj._smatrix_cache['kpphi_inv_list'][1], old_kpphi)


if AG_AVAILABLE:
    grcwa.set_backend('autograd')
    def test_epsgrad():
        def fun(x):
            obj=rcwa_assembly(x,freq,theta,phi,planewave,pthick,Pscale=1.)
            R,T= obj.RT_Solve(normalize=1)            
            return R

        grad_fun = grad(fun)

        x = epgrid.flatten()
        dx = 1e-3
        ind = np.random.randint(Nx*Ny*Nlayer,size=1)[0]
        FD, AD = t_grad(fun,grad_fun,x,dx,ind)
        assert abs(FD-AD)<abs(FD)*tol,'wrong epsgrid gradient'

    def test_thickgrad():        
        def fun(x):
            obj=rcwa_assembly(epgrid.flatten(),freq,theta,phi,planewave,x,Pscale=1.)
            R,T= obj.RT_Solve(normalize=1)            
            return R

        grad_fun = grad(fun)

        x = [0.1]
        dx = 1e-3
        ind = 0
        FD, AD = t_grad(fun,grad_fun,x,dx,ind)
        assert abs(FD-AD)<abs(FD)*tol,'wrong thickness gradient'

    def test_periodgrad():        
        def fun(x):
            obj=rcwa_assembly(epgrid.flatten(),freq,theta,phi,planewave,pthick,Pscale=x)
            R,T= obj.RT_Solve(normalize=1)            
            return R

        grad_fun = grad(fun)

        x = 1.0
        dx = 1e-3
        ind = 0
        FD, AD = t_grad(fun,grad_fun,x,dx,ind)
        assert abs(FD-AD)<abs(FD)*tol,'wrong thickness gradient'

    def test_freqgrad():        
        def fun(x):
            obj=rcwa_assembly(epgrid.flatten(),x,theta,phi,planewave,pthick,Pscale=1.)
            R,T= obj.RT_Solve(normalize=1)            
            return R

        grad_fun = grad(fun)

        x = 1.0
        dx = 1e-3
        ind = 0
        FD, AD = t_grad(fun,grad_fun,x,dx,ind)
        assert abs(FD-AD)<abs(FD)*tol,'wrong thickness gradient'

    def test_thetagrad():        
        def fun(x):
            obj=rcwa_assembly(epgrid.flatten(),freq,x,phi,planewave,pthick,Pscale=1.)
            R,T= obj.RT_Solve(normalize=1)            
            return R

        grad_fun = grad(fun)

        x = np.pi/10
        dx = 1e-3
        ind = 0
        FD, AD = t_grad(fun,grad_fun,x,dx,ind)
        assert abs(FD-AD)<abs(FD)*tol,'wrong thickness gradient'                                                
