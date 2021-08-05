import pytest
import molgrid
import numpy as np
import os
import torch

            
from pytest import approx
from numpy import around
import numpy as np

# there's like umpteen gazillion configuration options for example provider..
datadir = os.path.dirname(__file__)+'/data'

def test_downsampling():
    'convert to coarser resolution grid of same size'
    src = molgrid.MGrid4f(1,5,5,5)
    arr = np.arange(5**3).astype(np.float32).reshape(1,5,5,5)
    src.copyFrom(arr)
    
    dst = molgrid.MGrid4f(1,3,3,3)
    
    gi = molgrid.GridInterpolater(0.5, 2.0, 1.0, 2.0)
    t = molgrid.Transform()
    gi.forward(src.cpu(), t, dst.cpu())
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                assert dst[0,i,j,k] == src[0,i*2,j*2,k*2]
    
from scipy.interpolate import RegularGridInterpolator    
def scipy_interp(transform, ingrid, incenter, inres, indim, outcenter, outres, outdim, pad=0):
    '''Use scipy to interpolate between grids.  Take Grid3 for data'''
    insteps = int(indim/inres)+3
    outsteps = int(outdim/outres)+1
    
    inpts = [None,None,None]
    outpts = [None,None,None]
    for i in range(3):
        inpts[i] = np.linspace(incenter[i]-indim/2.0-inres,incenter[i]+indim/2.0+inres,insteps)
        outpts[i] = np.linspace(outcenter[i]-outdim/2.0,outcenter[i]+outdim/2.0,outsteps)

    indata = np.zeros((insteps,insteps,insteps))
    indata[1:-1,1:-1,1:-1] = ingrid.tonumpy()
    invalues = RegularGridInterpolator(inpts,indata,fill_value=pad,bounds_error=False)
   
    out = np.zeros((outsteps,outsteps,outsteps))
    #backtransform all output grid points and get interped value
    coords = molgrid.MGrid2f(1,3)
    x,y,z = np.meshgrid(*outpts,indexing='ij')
    for i in range(outsteps):
        for j in range(outsteps):
            for k in range(outsteps):                
                coords.copyFrom(np.array([x[i,j,k],y[i,j,k],z[i,j,k]],dtype=np.float32).reshape(1,3))
                transform.backward(coords,coords)                
                out[i,j,k] = invalues([coords[0][0],coords[0][1],coords[0][2]])
                
    return out


def test_upsampling():
    'convert to finer resolution grid'
    src = molgrid.MGrid4f(1,3,3,3)
    arr = np.arange(3**3).astype(np.float32).reshape(1,3,3,3)
    src.copyFrom(arr)
    
    dst = molgrid.MGrid4f(1,5,5,5)
    
    gi = molgrid.GridInterpolater(1.0, 2.0, 0.5, 2.0)
    t = molgrid.Transform()
    gi.forward(src.cpu(), t, dst.cpu())
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                assert src[0,i,j,k] == dst[0,i*2,j*2,k*2]    

def test_translation():
    'extract smaller grid from bigger'
    src = molgrid.MGrid4f(1,5,5,5)
    arr = np.arange(5**3).astype(np.float32).reshape(1,5,5,5)
    src.copyFrom(arr)
    
    dst = molgrid.MGrid4f(1,3,3,3)    
    t = molgrid.Transform()
    t.set_translation((0.5,0.5,0.5))

    gi = molgrid.GridInterpolater(0.5, 2.0, 0.5, 1.0)
    gi.forward(src.cpu(), t, dst.cpu())
    
    np.testing.assert_allclose(dst.tonumpy(),arr[:,:3,:3,:3],atol=1e-5)
        


def test_rotations():
    'test rotations without translation'
    src = molgrid.MGrid4f(1,5,5,5)
    arr = np.arange(5**3).astype(np.float32).reshape(1,5,5,5)
    src.copyFrom(arr)
    
    dst = molgrid.MGrid4f(1,3,3,3)    
    
    for i in range(10): #10 random samples
        t = molgrid.Transform((0,0,0),0,True)
    
        gi = molgrid.GridInterpolater(0.5, 2.0, 0.5, 1.0)
        gi.forward(src.cpu(), t, dst.cpu())
        
        out = scipy_interp(t, src[0], (0,0,0), 0.5, 2.0, (0,0,0), 0.5, 1.0, pad=0)
        np.testing.assert_allclose(out, dst[0].tonumpy(),atol=1e-4)
        
def test_transforms():
    'test rotations with translation and different centers'
    src = molgrid.MGrid4f(1,5,5,5)
    arr = np.arange(5**3).astype(np.float32).reshape(1,5,5,5)
    src.copyFrom(arr)
    
    dst = molgrid.MGrid4f(1,3,3,3)    
    
    for i in range(10): #10 random samples
        t = molgrid.Transform((0,0,0),0.5,True)
        gi = molgrid.GridInterpolater(0.5, 2.0, 0.5, 1.0)
        gi.forward(src.cpu(), t, dst.cpu())
        t.set_rotation_center((0.1,0,0))
        out = scipy_interp(t, src[0], (0.1,0.0,0.0), 0.5, 2.0, (0.6,0.5,0.5), 0.5, 1.0, pad=0)
        np.testing.assert_allclose(out[:-1,:-1,:-1], dst[0].tonumpy()[1:,1:,1:],atol=1e-4)      

def test_mol_transforms():
    '''compare interpolated transformed grid to grid generated from transformed molecule'''
    pass