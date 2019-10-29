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
def test_dx():
    fname = datadir+"/small.types"
    e = molgrid.ExampleProvider(data_root=datadir+"/structs")
    e.populate(fname)
    ex = e.next()
    c = ex.coord_sets[1]
    
    assert np.min(c.type_index.tonumpy()) >= 0

    gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(e.num_types()) # this should be grid_dims or get_grid_dims
    center = tuple(c.center())

    mgridout = molgrid.MGrid4f(*dims)    
    gmaker.forward(center, c, mgridout.cpu())
    
    molgrid.write_dx("tmp.dx", mgridout[0].cpu(), center, 0.5)
    
    mgridin = molgrid.read_dx("tmp.dx")
    os.remove("tmp.dx")

    g = mgridin.grid().tonumpy()
    go = mgridout[0].tonumpy()
    np.testing.assert_array_almost_equal(g,go,decimal=5)
    
    assert center == approx(list(mgridin.center()))
    assert mgridin.resolution() == 0.5
    
    #dump everything
    molgrid.write_dx_grids("/tmp/tmp", e.get_type_names(), mgridout.cpu(), center, gmaker.get_resolution(),0.5)
    checkgrid = molgrid.MGrid4f(*dims)
    molgrid.read_dx_grids("/tmp/tmp", e.get_type_names(), checkgrid.cpu())
    
    np.testing.assert_array_almost_equal(mgridout.tonumpy(), 2.0*checkgrid.tonumpy(),decimal=5)
    
    
