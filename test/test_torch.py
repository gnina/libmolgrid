import pytest
import molgrid
import torch
import os
from molgrid import tensor_as_grid, Coords2GridFunction
from pytest import approx

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
