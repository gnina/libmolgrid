import pytest
import molgrid
import torch
from molgrid import tensor_as_grid
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
    
    mg2.copyTo(tensor_as_grid(t2))
    mg3.copyTo(tensor_as_grid(t3))

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
    
    mg2.copyTo(tensor_as_grid(t2))
    mg3.copyTo(tensor_as_grid(t3))

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
                

    
    mg2.copyFrom(tensor_as_grid(t2))
    mg3.copyFrom(tensor_as_grid(t3))

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
    
    mg2.copyFrom(tensor_as_grid(t2))
    mg3.copyFrom(tensor_as_grid(t3))

    for i in range(3):
        for j in range(4):
            assert mg2[i,j] == i*j+1
            for k in range(2):
                assert mg3[i,j,k] == i*j+1+k                                             