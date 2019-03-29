import pytest
import molgrid
import numpy as np
import os
import torch

                            
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
    dims = gmaker.grid_dims(c.max_type) # this should be grid_dims or get_grid_dims
    center = c.coord.tonumpy().mean(axis=0)
    center = tuple(center.astype(float))


    mgridout = molgrid.MGrid4f(*dims)    
    mgridgpu = molgrid.MGrid4f(*dims)    
    npout = np.zeros(dims, dtype=np.float32)
    torchout = torch.zeros(dims, dtype=torch.float32)
    cudaout = torch.zeros(dims, dtype=torch.float32, device='cuda:0')
    
    gmaker.forward(center, c, mgridout.cpu())
    gmaker.forward(center, c, mgridgpu.gpu())
    #why aren't thse autoconverting? they shouldbe autoconverting
    gmaker.forward(center, c, molgrid.Grid4f(npout))
    gmaker.forward(center, c, molgrid.Grid4f(torchout))
    gmaker.forward(center, c, molgrid.Grid4fCUDA(cudaout))
    
    assert 1.4387 == approx(mgridout.tonumpy().max())
    assert 1.4387 == approx(mgridgpu.tonumpy().max())
    assert 1.4387 == approx(npout.max())
    assert 1.4387 == approx(torchout.max())
    assert 1.4387 == approx(cudaout.cpu().max())

    #should overwrite by default, yes?
    gmaker.forward(center, c, mgridout.cpu())
    gmaker.forward(center, c, mgridgpu.gpu())
    assert 1.4387 == approx(mgridout.tonumpy().max())
    assert 1.4387 == approx(mgridgpu.tonumpy().max())
