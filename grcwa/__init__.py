"""Top-level package for grcwa."""
from .backend import backend, set_backend
from .fft_funs import Epsilon_fft, Epsilon_fft_pol, get_fft, get_ifft, get_conv
from .kbloch import Lattice_Reciprocate,Lattice_getG,Lattice_SetKs,Gsel_circular,Gsel_parallelogramic,Gsel_1D,Gsel_0D
from .rcwa import obj

__author__ = """Weiliang Jin"""
__email__ = 'jwlaaa@gmail.com'
__version__ = '0.1.2'
