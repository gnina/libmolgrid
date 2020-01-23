from .molgrid import *
import sys, inspect
import numpy as np

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
