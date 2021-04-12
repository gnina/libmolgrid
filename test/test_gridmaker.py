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
    
    dims = gmaker.grid_dimensions(e.num_types())
    mgridout = molgrid.MGrid4f(*dims)    
    mgridgpu = molgrid.MGrid4f(*dims)   
    
    #pass transform
    gmaker.forward(ex, molgrid.Transform(center, 0, False), mgridout.cpu())
    gmaker.forward(ex, molgrid.Transform(center, 0, False), mgridgpu.gpu())
    assert 2.094017 == approx(mgridout.tonumpy().max())
    assert 2.094017 == approx(mgridgpu.tonumpy().max())    
        
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
    g2 = molgrid.GridMaker(resolution=.1,dimension=6.0,radius_scale=0.5,gaussian_radius_multiple=3.0)
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
    c = np.array([[0,0,0],[2,0,0]],np.float32)
    t = np.array([0,1],np.float32)
    vt = np.array([[1.0,0],[0,1.0]],np.float32)
    vt2 = np.array([[0.5,0.0],[0.0,0.5]],np.float32)
    r = np.array([1.0,1.0],np.float32)
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
    np.testing.assert_allclose(g[1,:],v2g[1,:]*2.0,atol=1e-5)
    
    vgridgpu = molgrid.MGrid4f(*shape)
    v2gridgpu = molgrid.MGrid4f(*shape)
    g1.forward((0,0,0),vcoords, vgridgpu.gpu())
    g1.forward((0,0,0),v2coords, v2gridgpu.gpu())
    
    np.testing.assert_allclose(reference.tonumpy(),vgridgpu.tonumpy(),atol=1e-5)
    v2gpu = v2gridgpu.tonumpy()
    
    np.testing.assert_allclose(g[0,:],v2gpu[0,:]*2.0,atol=1e-5)
    np.testing.assert_allclose(g[1,:],v2gpu[1,:]*2.0,atol=1e-5)    
    
    #create target grid with equal type density at 1,0,0
    tc = molgrid.Grid2f(np.array([[1,0,0]],np.float32))
    tv = molgrid.Grid2f(np.array([[0.5,0.5]],np.float32))
    tr = molgrid.Grid1f(np.array([1.0],np.float32))
    targetc = molgrid.CoordinateSet(tc,tv,tr)
    tgrid = molgrid.MGrid4f(*shape)
    g1.forward((0,0,0),targetc,tgrid.cpu())
    
    gradc = molgrid.MGrid2f(2,3)
    gradt = molgrid.MGrid2f(2,2)
    g1.backward((0,0,0),vcoords,tgrid.cpu(),gradc.cpu(),gradt.cpu())
    assert gradc[0,0] == approx(-gradc[1,0],abs=1e-4)
    assert gradc[0,0] > 0
    
    gradc.fill_zero()
    gradt.fill_zero()
    g1.backward((0,0,0),vcoords,tgrid.gpu(),gradc.gpu(),gradt.gpu())

    assert gradc[0,0] == approx(-gradc[1,0],abs=1e-4)
    assert gradc[0,0] > 0
    

    
def test_vector_types_mol():
    '''Test vector types with a real molecule'''
    fname = datadir+"/small.types"
    e = molgrid.ExampleProvider(data_root=datadir+"/structs")    
    e.populate(fname)
    ex = e.next()
        
    ev = molgrid.ExampleProvider(data_root=datadir+"/structs",make_vector_types=True)
    ev.populate(fname)
    exv = ev.next()
    
    assert exv.has_vector_types()
    assert not ex.has_vector_types()

    gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(ex.num_types()) # this should be grid_dims or get_grid_dims    
    
    mgridout = molgrid.MGrid4f(*dims)    
    mgridgpu = molgrid.MGrid4f(*dims)
        
    mgridoutv = molgrid.MGrid4f(*dims)    
    mgridgpuv = molgrid.MGrid4f(*dims)
    
    d = np.ones(dims,np.float32)
    diff = molgrid.MGrid4f(*dims)
    diff.copyFrom(d)       
    
    gmaker.forward(ex, mgridout.cpu())
    gmaker.forward(ex, mgridgpu.gpu())
    center = ex.coord_sets[-1].center()
    c = ex.merge_coordinates()
    backcoordscpu = molgrid.MGrid2f(c.size(),3)
    backcoordsgpu = molgrid.MGrid2f(c.size(),3)
    
    gmaker.backward(center, c, diff.cpu(), backcoordscpu.cpu())
    gmaker.backward(center, c, diff.gpu(), backcoordsgpu.gpu())

    #vector types
    gmaker.set_radii_type_indexed(True)
    
    gmaker.forward(exv, mgridoutv.cpu())
    gmaker.forward(exv, mgridgpuv.gpu())
    
    cv = exv.merge_coordinates()
    vbackcoordscpu = molgrid.MGrid2f(cv.size(),3)
    vbackcoordsgpu = molgrid.MGrid2f(cv.size(),3)
    vbacktypescpu = molgrid.MGrid2f(cv.size(),cv.num_types())
    vbacktypesgpu = molgrid.MGrid2f(cv.size(),cv.num_types())
        
    gmaker.backward(center, cv, diff.cpu(), vbackcoordscpu.cpu(),vbacktypescpu.cpu())
    gmaker.backward(center, cv, diff.gpu(), vbackcoordsgpu.gpu(),vbacktypesgpu.gpu())
    
    np.testing.assert_allclose(mgridout.tonumpy(),mgridoutv.tonumpy(),atol=1e-5)
    np.testing.assert_allclose(mgridgpu.tonumpy(),mgridgpuv.tonumpy(),atol=1e-5)
    np.testing.assert_allclose(mgridoutv.tonumpy(),mgridgpuv.tonumpy(),atol=1e-5)

    np.testing.assert_allclose(vbackcoordscpu.tonumpy(),backcoordscpu.tonumpy(),atol=1e-5)
    np.testing.assert_allclose(vbackcoordsgpu.tonumpy(),backcoordsgpu.tonumpy(),atol=1e-5)
    np.testing.assert_allclose(vbackcoordscpu.tonumpy(),vbackcoordsgpu.tonumpy(),atol=1e-4)
    np.testing.assert_allclose(vbacktypescpu.tonumpy(),vbacktypesgpu.tonumpy(),atol=1e-4)

def test_vector_types_duplicate():
    fname = datadir+"/smalldup.types"

    teste = molgrid.ExampleProvider(molgrid.GninaVectorTyper(),shuffle=False, duplicate_first=True,data_root=datadir+"/structs")
    teste.populate(fname)
    batch_size = 1
    gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(molgrid.GninaVectorTyper().num_types()*4)
    
    tensor_shape = (batch_size,)+dims
    input_tensor_1 = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
    
    batch_1 = teste.next_batch(batch_size)
    gmaker.forward(batch_1, input_tensor_1,random_translation=0.0, random_rotation=False)
    
    input_tensor_2 = torch.zeros(tensor_shape, dtype=torch.float32, device='cpu')
    
    gmaker.forward(batch_1, input_tensor_2,random_translation=0.0, random_rotation=False)   
    
    np.testing.assert_allclose(input_tensor_1.cpu().detach().numpy(),input_tensor_2.detach().numpy(),atol=1e-4)
    assert input_tensor_1.cpu().detach().numpy().max() < 75
        
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

def test_type_radii():
    g1 = molgrid.GridMaker(resolution=.25,dimension=6.0,radius_type_indexed=True)
    c = np.array([[0,0,0]],np.float32)
    t = np.array([0],np.float32)
    r = np.array([1.0],np.float32)
    coords = molgrid.CoordinateSet(molgrid.Grid2f(c),molgrid.Grid1f(t),molgrid.Grid1f(r),2)
    coords.make_vector_types(True, [3.0,1.0])

    shape = g1.grid_dimensions(3) #includes dummy type
    reference = molgrid.MGrid4f(*shape)
    gpudata = molgrid.MGrid4f(*shape)

    assert g1.get_radii_type_indexed()
    
    g1.forward((0,0,0),coords, reference.cpu())
    g1.forward((0,0,0),coords, gpudata.gpu())
    
    np.testing.assert_allclose(reference.tonumpy(),gpudata.tonumpy(),atol=1e-5)

    assert reference.tonumpy().sum() > 2980 #radius of 1 would be 116
    
    reference.fill_zero()
    reference[0][20][12][12] = -1
    reference[1][20][12][12] = 1
    reference[2][20][12][12] = 2

    cpuatoms = molgrid.MGrid2f(1,3)
    cputypes = molgrid.MGrid2f(1,3)
    gpuatoms = molgrid.MGrid2f(1,3)    
    gputypes = molgrid.MGrid2f(1,3)
    
    g1.backward((0,0,0),coords,reference.cpu(), cpuatoms.cpu(), cputypes.cpu())

    assert cpuatoms[0][0] < 0
    assert cpuatoms[0][1] == 0
    assert cpuatoms[0][2] == 0
    
    assert cputypes[0][0] < 0
    assert cputypes[0][1] == 0
    assert cputypes[0][2] == 0

    g1.backward((0,0,0),coords,reference.gpu(), gpuatoms.gpu(), gputypes.gpu())
    
    np.testing.assert_allclose(gpuatoms.tonumpy(),cpuatoms.tonumpy(),atol=1e-5)
    np.testing.assert_allclose(gputypes.tonumpy(),cputypes.tonumpy(),atol=1e-5)

def test_backward_gradients():
    #test that we have the right value along a single dimension
    gmaker = molgrid.GridMaker(resolution=0.5,dimension=6.0,gaussian_radius_multiple=-2.0) #use full truncated gradient    
    xvals = np.arange(-0.9,3,.1)
    
    for device in ('cuda','cpu'):
        types = torch.ones(1,1,dtype=torch.float32,device=device)
        radii = torch.ones(1,dtype=torch.float32,device=device)
        for i in range(3): #test along each axis                        
            for x in xvals:
                coords = torch.zeros(1,3,dtype=torch.float32,device=device)
                coords[0][i] = x
                coords.requires_grad=True
                outgrid = molgrid.Coords2GridFunction.apply(gmaker, (0,0,0), coords, types, radii)
                if i == 0:
                    gp = outgrid[0][8][6][6]
                elif i == 1:
                    gp = outgrid[0][6][8][6]
                else:
                    gp = outgrid[0][6][6][8]                    
                Lg = torch.autograd.grad(gp,coords,create_graph=True)[0]
                fancyL = torch.sum(Lg**2) 
                val = float(torch.autograd.grad(fancyL,coords)[0][0][i])
                d = x-1
                correct = -128*d**3*np.exp(-4*d**2) + 32*d*np.exp(-4*d**2)  #formulate based on distance
                assert val == approx(correct,abs=1e-4)
        
    #check that diagonal is symmetric and decreases at this range
    for device in ('cuda','cpu'):
        types = torch.ones(1,1,dtype=torch.float32,device=device)
        radii = torch.ones(1,dtype=torch.float32,device=device)
        coords = torch.zeros(1,3,dtype=torch.float32,requires_grad=True,device=device)    
        
        outgrid = molgrid.Coords2GridFunction.apply(gmaker, (0,0,0), coords, types, radii)
        gp = outgrid[0][7][7][7]                               
        Lg = torch.autograd.grad(gp,coords,create_graph=True)[0]
        fancyL = torch.sum(Lg**2) 
        fL1 = torch.autograd.grad(fancyL,coords)[0][0]
        
        gp2 = outgrid[0][8][8][8]                               
        Lg = torch.autograd.grad(gp2,coords,create_graph=True)[0]
        fancyL = torch.sum(Lg**2) 
        fL2 = torch.autograd.grad(fancyL,coords)[0][0]
        
        assert fL1[0] == fL1[1]
        assert fL1[2] == fL1[1]
        assert fL2[0] == fL2[1]
        assert fL2[2] == fL2[1]    
        assert fL2[0] < fL1[0]    