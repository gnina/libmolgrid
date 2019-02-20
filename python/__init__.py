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

#define pytorch specific functionality
try:
    import torch
    from .torch_bindings import *
    
except ImportError as e:
    print(e)
    sys.stderr.write("Failed to import torch.\n")