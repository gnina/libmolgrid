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
import numpy as np

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
    center = c.center()
    center = tuple(center)


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

def test_radius_multiples():
    g1 = molgrid.GridMaker(resolution=.1,dimension=6.0)
    c = np.array([[0,0,0]],np.float32)
    t = np.array([0],np.float32)
    r = np.array([1.0],np.float32)
    coords = molgrid.CoordinateSet(molgrid.Grid2f(c),molgrid.Grid1f(t),molgrid.Grid1f(r),1)
    shape = g1.grid_dimensions(1)
    cpugrid = molgrid.MGrid4f(*shape)
    cpugrid2 = molgrid.MGrid4f(*shape)
    gpugrid = molgrid.MGrid4f(*shape)

    g1.forward((0,0,0),coords, cpugrid.cpu())
    g1.forward((0,0,0),coords, gpugrid.gpu())
    g1.forward((0,0,0),c,t,r, cpugrid2.cpu())
    
    np.testing.assert_allclose(cpugrid.tonumpy(),gpugrid.tonumpy(),atol=1e-5)
    np.testing.assert_allclose(cpugrid.tonumpy(),cpugrid2.tonumpy(),atol=1e-6)
    g = cpugrid.tonumpy()
    
    assert g[0,30,30,30] == approx(1)
    
    #cut a line across
    line = g[0,30,30,:]
    xvals = np.abs(np.arange(-3,3.1,.1))
    gauss = np.exp(-2*xvals**2)
    for i in range(20,41):
        assert line[i] == approx(gauss[i])

    for i in list(range(0,15))+list(range(45,61)):
        assert line[i] == approx(0)
        
    quad = 4*np.exp(-2)*xvals**2 - 12 *np.exp(-2) * xvals + 9*np.exp(-2)
    for i in list(range(15,20))+list(range(41,45)):
        assert line[i] == approx(quad[i],abs=1e-5)        
        
    #funkier grid
    g2 = molgrid.GridMaker(resolution=.1,dimension=6.0,radius_scale=0.5,gassian_radius_multiple=3.0)
    cpugrid = molgrid.MGrid4f(*shape)
    gpugrid = molgrid.MGrid4f(*shape)
    g2.forward((0,0,0),coords, cpugrid.cpu())
    g2.forward((0,0,0),coords, gpugrid.gpu())
    
    np.testing.assert_allclose(cpugrid.tonumpy(),gpugrid.tonumpy(),atol=1e-5)
    g = cpugrid.tonumpy()
    
    assert g[0,30,30,30] == approx(1)
    
    #cut a line across
    line = g[0,30,:,30]
    xvals = np.abs(np.arange(-3,3.1,.1))*2.0
    gauss = np.exp(-2*xvals**2)
    #should be guassian the whole way, although quickly hits numerical zero
    for i in range(0,61):
        assert line[i] == approx(gauss[i],abs=1e-5)
        
def test_backwards():
    g1 = molgrid.GridMaker(resolution=.1,dimension=6.0)
    c = np.array([[1.0,0,0]],np.float32)
    t = np.array([0],np.float32)
    r = np.array([2.0],np.float32)
    coords = molgrid.CoordinateSet(c,t,r,1)
    shape = g1.grid_dimensions(1)
    
    #make diff with gradient in center
    diff = molgrid.MGrid4f(*shape)
    diff[0,30,30,30] = 1.0  
    
    cpuatoms = molgrid.MGrid2f(1,3)
    gpuatoms = molgrid.MGrid2f(1,3)
    
    #apply random rotation
    T = molgrid.Transform((0,0,0), 0, True)
    T.forward(coords, coords);
    
    g1.backward((0,0,0),coords,diff.cpu(), cpuatoms.cpu())
    g1.backward((0,0,0),coords,diff.gpu(), gpuatoms.gpu())

    T.backward(cpuatoms.cpu(), cpuatoms.cpu(), False)
    T.backward(gpuatoms.gpu(), gpuatoms.gpu(), False)
    
    print(cpuatoms.tonumpy(), gpuatoms.tonumpy())
    # results should be ~ -.6, 0, 0
    np.testing.assert_allclose(cpuatoms.tonumpy(), gpuatoms.tonumpy(), atol=1e-5)
    np.testing.assert_allclose(cpuatoms.tonumpy().flatten(), [-0.60653067,0,0], atol=1e-5)
    
   
def test_vector_types():
    g1 = molgrid.GridMaker(resolution=.25,dimension=6.0)
    c = np.array([[0,0,0]],np.float32)
    t = np.array([0],np.float32)
    vt = np.array([[1.0,0]],np.float32)
    vt2 = np.array([[0.5,0.5]],np.float32)
    r = np.array([1.0],np.float32)
    coords = molgrid.CoordinateSet(molgrid.Grid2f(c),molgrid.Grid1f(t),molgrid.Grid1f(r),2)
    vcoords = molgrid.CoordinateSet(molgrid.Grid2f(c),molgrid.Grid2f(vt),molgrid.Grid1f(r))
    v2coords = molgrid.CoordinateSet(molgrid.Grid2f(c),molgrid.Grid2f(vt2),molgrid.Grid1f(r))

    shape = g1.grid_dimensions(2)
    reference = molgrid.MGrid4f(*shape)
    vgrid = molgrid.MGrid4f(*shape)
    v2grid = molgrid.MGrid4f(*shape)
    v3grid = molgrid.MGrid4f(*shape)
    
    g1.forward((0,0,0),coords, reference.cpu())
    g1.forward((0,0,0),vcoords, vgrid.cpu())
    g1.forward((0,0,0),v2coords, v2grid.cpu())
    g1.forward((0,0,0),c,vt,r, v3grid.cpu())        
    np.testing.assert_allclose(reference.tonumpy(),vgrid.tonumpy(),atol=1e-5)
    np.testing.assert_allclose(vgrid.tonumpy(),v3grid.tonumpy(),atol=1e-6)
    
    v2g = v2grid.tonumpy()
    g = reference.tonumpy()

    np.testing.assert_allclose(g[0,:],v2g[0,:]*2.0,atol=1e-5)
    np.testing.assert_allclose(g[0,:],v2g[1,:]*2.0,atol=1e-5)
    
    vgridgpu = molgrid.MGrid4f(*shape)
    v2gridgpu = molgrid.MGrid4f(*shape)
    g1.forward((0,0,0),vcoords, vgridgpu.gpu())
    g1.forward((0,0,0),v2coords, v2gridgpu.gpu())
    
    np.testing.assert_allclose(reference.tonumpy(),vgridgpu.tonumpy(),atol=1e-5)
    v2gpu = v2gridgpu.tonumpy()
    

    np.testing.assert_allclose(g[0,:],v2gpu[0,:]*2.0,atol=1e-5)
    np.testing.assert_allclose(g[0,:],v2gpu[1,:]*2.0,atol=1e-5)    
    
    
    
def test_backward_vec():
    g1 = molgrid.GridMaker(resolution=.1,dimension=6.0)
    c = np.array([[1.0,0,0],[-1,-1,0]],np.float32)
    t = np.array([[0,1.0,0],[1.0,0,0]],np.float32)
    r = np.array([2.0,2.0],np.float32)
    coords = molgrid.CoordinateSet(c,t,r)
    shape = g1.grid_dimensions(3)
    
    #make diff with gradient in center
    diff = molgrid.MGrid4f(*shape)
    diff[0,30,30,30] = 1.0  
    diff[1,30,30,30] = -1.0  
    
    cpuatoms = molgrid.MGrid2f(2,3)
    cputypes = molgrid.MGrid2f(2,3)
    gpuatoms = molgrid.MGrid2f(2,3)    
    gputypes = molgrid.MGrid2f(2,3)
    
    g1.backward((0,0,0),coords,diff.cpu(), cpuatoms.cpu(), cputypes.cpu())
    
    assert cputypes[0][0] > 0
    assert cputypes[0][1] < 0
    assert cputypes[0][2] == 0

    g1.backward((0,0,0),coords,diff.gpu(), gpuatoms.gpu(), gputypes.gpu())

    np.testing.assert_allclose(gpuatoms.tonumpy(),cpuatoms.tonumpy(),atol=1e-5)
    np.testing.assert_allclose(gputypes.tonumpy(),cputypes.tonumpy(),atol=1e-5)




    
