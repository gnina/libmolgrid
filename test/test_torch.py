import pytest
import molgrid
import torch
import os
import numpy as np
from molgrid import tensor_as_grid, Coords2GridFunction, BatchedCoords2GridFunction
from pytest import approx
import torch.nn as nn
import torch.nn.functional as F

'''Test ability to manipulate data in torch tensors'''

def test_mgrid_copyto_tensor():
    mg2 = molgrid.MGrid2f(3,4)
    mg3 = molgrid.MGrid3d(3,4,2)
    for i in range(3):
        for j in range(4):
            mg2[i,j] = i*j+1
            for k in range(2):
                mg3[i,j,k] = i*j+1+k

    t2 = torch.FloatTensor(3,4)
    t3 = torch.DoubleTensor(3,4,2)

    mg2.copyTo(t2)
    mg3.copyTo(t3)

    for i in range(3):
        for j in range(4):
            assert t2[i,j] == i*j+1
            for k in range(2):
                assert t3[i,j,k] == i*j+1+k


def test_mgrid_copyto_tensor_cuda():
    mg2 = molgrid.MGrid2f(3,4)
    mg3 = molgrid.MGrid3d(3,4,2)
    for i in range(3):
        for j in range(4):
            mg2[i,j] = i*j+1
            for k in range(2):
                mg3[i,j,k] = i*j+1+k

    t2 = torch.cuda.FloatTensor(3,4)
    t3 = torch.cuda.DoubleTensor(3,4,2)

    mg2.copyTo(t2)
    mg3.copyTo(t3)

    for i in range(3):
        for j in range(4):
            assert t2[i,j] == i*j+1
            for k in range(2):
                assert t3[i,j,k] == i*j+1+k

def test_mgrid_copyfrom_tensor():
    mg2 = molgrid.MGrid2f(3,4)
    mg3 = molgrid.MGrid3d(3,4,2)
    t2 = torch.FloatTensor(3,4)
    t3 = torch.DoubleTensor(3,4,2)

    for i in range(3):
        for j in range(4):
            t2[i,j] = i*j+1
            for k in range(2):
                t3[i,j,k] = i*j+1+k



    mg2.copyFrom(t2)
    mg3.copyFrom(t3)

    for i in range(3):
        for j in range(4):
            assert mg2[i,j] == i*j+1
            for k in range(2):
                assert mg3[i,j,k] == i*j+1+k


def test_mgrid_copyfrom_tensor_cuda():
    mg2 = molgrid.MGrid2f(3,4)
    mg3 = molgrid.MGrid3d(3,4,2)
    t2 = torch.cuda.FloatTensor(3,4)
    t3 = torch.cuda.DoubleTensor(3,4,2)

    for i in range(3):
        for j in range(4):
            t2[i,j] = i*j+1
            for k in range(2):
                t3[i,j,k] = i*j+1+k

    mg2.copyFrom(t2)
    mg3.copyFrom(t3)

    for i in range(3):
        for j in range(4):
            assert mg2[i,j] == i*j+1
            for k in range(2):
                assert mg3[i,j,k] == i*j+1+k


def test_torch_gnina_example_provider():
    datadir = os.path.dirname(__file__)+'/data'
    fname = datadir+"/small.types"
    e = molgrid.ExampleProvider(data_root=datadir+"/structs")
    e.populate(fname)

    batch_size = 100
    batch = e.next_batch(batch_size)
    #extract labels
    nlabels = e.num_labels()
    assert nlabels == 3
    labels = labels = torch.zeros((batch_size,nlabels), dtype=torch.float32)

    batch.extract_labels(labels)
    label0  = torch.zeros(batch_size, dtype=torch.float32)

    batch.extract_label(0, label0)

    assert label0[0] == 1
    assert labels[0,0] == 1
    assert float(labels[0][1]) == approx(6.05)
    assert float(labels[0][2]) == approx(0.162643)

    for i in range(nlabels):
        assert label0[i] == labels[i][0]

def test_function():
    for dev in ('cuda','cpu'):
        gmaker = molgrid.GridMaker(resolution=.1,dimension=6.0)
        c = torch.tensor([[1.0,0,0],[1,0,0]],device=dev,dtype=torch.float32,requires_grad=True)
        vt = torch.tensor([[0,1.0,0],[1.0,0,0]],device=dev,dtype=torch.float32,requires_grad=True)
        r = torch.tensor([2.0,2.0],device=dev,dtype=torch.float32)
        
        grid = Coords2GridFunction.apply(gmaker, (0,0,0), c, vt, r)
    
        shape = gmaker.grid_dimensions(3)    
        #make diff with gradient in center
        diff = torch.zeros(*shape,dtype=torch.float32,device=dev)
        diff[0,30,30,30] = 1.0  
        diff[1,30,30,30] = -1.0  
            
        grid.backward(diff)
        assert c.grad[0].cpu().numpy() == approx([0.60653,0,0],abs=1e-4)
        assert c.grad[1].cpu().numpy() == approx([-0.60653,0,0],abs=1e-4)
        
        assert vt.grad[0].cpu().numpy() == approx([0.60653,-0.60653,0],abs=1e-4)
        assert vt.grad[1].cpu().numpy() == approx([0.60653,-0.60653,0],abs=1e-4)
        
def test_batched_function():
    for dev in ('cuda','cpu'):
        gmaker = molgrid.GridMaker(resolution=.1,dimension=6.0)
        c = torch.tensor([[[1.0,0,0],[1,0,0]],[[0,1,0],[0,1,0]]],device=dev,dtype=torch.float32,requires_grad=True)
        vt = torch.tensor([[[0,1.0,0],[1.0,0,0]],[[0,1.0,0],[1.0,0,0]]],device=dev,dtype=torch.float32,requires_grad=True)
        r = torch.tensor([[2.0,2.0],[2.0,2.0]],device=dev,dtype=torch.float32)
        
        grid = BatchedCoords2GridFunction.apply(gmaker, (0,0,0), c, vt, r)
    
        shape = gmaker.grid_dimensions(3)    
        #make diff with gradient in center
        diff = torch.zeros(2,*shape,dtype=torch.float32,device=dev)
        diff[0,0,30,30,30] = 1.0  
        diff[0,1,30,30,30] = -1.0  
        diff[1,0,30,30,30] = 1.0  
        diff[1,1,30,30,30] = -1.0              
        grid.backward(diff)
        assert c.grad[0][0].cpu().numpy() == approx([0.60653,0,0],abs=1e-4)
        assert c.grad[0][1].cpu().numpy() == approx([-0.60653,0,0],abs=1e-4)
        
        assert vt.grad[0][0].cpu().numpy() == approx([0.60653,-0.60653,0],abs=1e-4)
        assert vt.grad[0][1].cpu().numpy() == approx([0.60653,-0.60653,0],abs=1e-4)    
        
        assert c.grad[1][0].cpu().numpy() == approx([0,0.60653,0],abs=1e-4)
        assert c.grad[1][1].cpu().numpy() == approx([0,-0.60653,0],abs=1e-4)
        
        assert vt.grad[1][0].cpu().numpy() == approx([0.60653,-0.60653,0],abs=1e-4)
        assert vt.grad[1][1].cpu().numpy() == approx([0.60653,-0.60653,0],abs=1e-4)    

def test_coords2grid():
    gmaker = molgrid.GridMaker(resolution=0.5,
                           dimension=23.5,
                           radius_scale=1,
                           radius_type_indexed=True)
    n_types = molgrid.defaultGninaLigandTyper.num_types()
    radii = np.array(list(molgrid.defaultGninaLigandTyper.get_type_radii()),np.float32)
    dims = gmaker.grid_dimensions(n_types)
    grid_size = dims[0] * dims[1] * dims[2] * dims[3]

    c2grid = molgrid.Coords2Grid(gmaker, center=(0,0,0))
    n_atoms = 2
    batch_size = 1
    coords = nn.Parameter(torch.randn(n_atoms, 3,device='cuda'))
    types = nn.Parameter(torch.randn(n_atoms, n_types+1,device='cuda'))
    
    coords.data[0,:] = torch.tensor([ 1,0,0])
    coords.data[1,:] = torch.tensor([-1,0,0])
    types.data[...] = 0
    types.data[:,10] = 1

    batch_radii = torch.tensor(np.tile(radii, (batch_size, 1)), dtype=torch.float32,  device='cuda')    

    grid_gen = c2grid(coords.unsqueeze(0), types.unsqueeze(0)[:,:,:-1], batch_radii)
    
    assert float(grid_gen[0][10].sum()) == approx(float(grid_gen.sum()))
    assert grid_gen.sum() > 0
    
    target = torch.zeros_like(grid_gen)
    target[0,:,24,24,24] = 1000.0
        
    grad_coords = molgrid.MGrid2f(n_atoms,3)
    grad_types = molgrid.MGrid2f(n_atoms,n_types)
    r = molgrid.MGrid1f(len(radii))
    r.copyFrom(radii)
    
    grid_loss = F.mse_loss(target, grid_gen)
    grid_loss.backward()
    print(grid_loss)
    print(coords.grad.detach().cpu().numpy())


