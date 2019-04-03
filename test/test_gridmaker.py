import pytest
import molgrid
import numpy as np
import os
import torch

'''
import pytest
import molgrid
import numpy as np
import os
import torch

datadir = '../../test/data'
fname = datadir+"/small.types"
e = molgrid.ExampleProvider(data_root=datadir+"/structs")
e.populate(fname)
ex = e.next()
c = ex.coord_sets[1]
    
center = c.coord.tonumpy().mean(axis=0)
center = tuple(center.astype(float))

gmaker = molgrid.GridMaker()
dims = gmaker.grid_dimensions(c.max_type)

mgridout = molgrid.MGrid4f(*dims)
npout = np.zeros(dims, dtype=np.float32)

 
'''
                            
from pytest import approx
from numpy import around

# there's like umpteen gazillion configuration options for example provider..
datadir = os.path.dirname(__file__)+'/data'
#make sure we can read in molecular data
def test_a_grid():
    fname = datadir+"/small.types"
    e = molgrid.ExampleProvider(data_root=datadir+"/structs")
    e.populate(fname)
    ex = e.next()
    c = ex.coord_sets[1]
    
    assert np.min(c.type_index.tonumpy()) >= 0

    gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(c.max_type) # this should be grid_dims or get_grid_dims
    center = c.coord.tonumpy().mean(axis=0)
    center = tuple(center.astype(float))


    mgridout = molgrid.MGrid4f(*dims)    
    mgridgpu = molgrid.MGrid4f(*dims)    
    npout = np.zeros(dims, dtype=np.float32)
    torchout = torch.zeros(dims, dtype=torch.float32)
    cudaout = torch.zeros(dims, dtype=torch.float32, device='cuda')
    
    gmaker.forward(center, c, mgridout.cpu())
    gmaker.forward(center, c, mgridgpu.gpu())

    gmaker.forward(center, c, npout)
    gmaker.forward(center, c, torchout)
    gmaker.forward(center, c, cudaout)
    
    
    newt = gmaker.make_tensor(center, c)
    newa = gmaker.make_ndarray(center, c)
    
    assert 1.438691 == approx(mgridout.tonumpy().max())
    assert 1.438691 == approx(mgridgpu.tonumpy().max())
    assert 1.438691 == approx(npout.max())
    assert 1.438691 == approx(torchout.numpy().max())
    assert 1.438691 == approx(cudaout.cpu().numpy().max())
    assert 1.438691 == approx(newt.cpu().numpy().max())
    assert 1.438691 == approx(newa.max())

    #should overwrite by default, yes?
    gmaker.forward(center, c, mgridout.cpu())
    gmaker.forward(center, c, mgridgpu.gpu())
    assert 1.438691 == approx(mgridout.tonumpy().max())
    assert 1.438691 == approx(mgridgpu.tonumpy().max())
    
    
    dims = gmaker.grid_dimensions(e.type_size())
    mgridout = molgrid.MGrid4f(*dims)    
    mgridgpu = molgrid.MGrid4f(*dims)   
    gmaker.forward(ex, mgridout.cpu())
    gmaker.forward(ex, mgridgpu.gpu())
    
    gmaker.forward(ex, mgridout.cpu())
    gmaker.forward(ex, mgridgpu.gpu())    
    
    assert 2.094017 == approx(mgridout.tonumpy().max())
    assert 2.094017 == approx(mgridgpu.tonumpy().max())
