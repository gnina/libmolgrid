from .molgrid import *
import sys, inspect
import numpy as np

# Vec3 used to be named float3 (it borrowed CUDA's vector_types.h name even
# in non-CUDA builds); keep the old name available for existing user code.
float3 = Vec3

if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata

try:
    __version__ = metadata.version('molgrid')
except metadata.PackageNotFoundError:
    __version__ = '0+unknown'

def tonumpy(g):
    '''Return a numpy array copy of grid g'''
    typ = getattr(np,g.type())
    arr = np.empty(g.shape,dtype=typ)
    g.copyTo(arr)
    return arr

#dynamically add tonumpy methods to grid classes
for name in dir(molgrid):
    C = getattr(molgrid,name)
    if (name.startswith('Grid') or name.startswith('MGrid')):
        if inspect.isclass(C) and C.__module__.startswith('molgrid'):
            setattr(C,'tonumpy',tonumpy)
            setattr(C, '__array__',tonumpy)

#extend gridmaker to generate new numpy arrays
#extend grid maker to create pytorch Tensor
def make_grid_ndarray(gridmaker, center, c):
    '''Create appropriately sized numpy array of grid densities. '''    
    dims = gridmaker.grid_dimensions(c.max_type) # this should be grid_dims or get_grid_dims
    t = np.zeros(dims, dtype=np.float32)
    gridmaker.forward(center, c, t)
    return t 

GridMaker.make_ndarray = make_grid_ndarray
    

#define pytorch specific functionality
try:
    import torch
    from .torch_bindings import *
    
except ImportError as e:
    print(e)
    sys.stderr.write("Failed to import torch.\n")
