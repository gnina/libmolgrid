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
    

def test_upsampling():
    'convert to finer resolution grid'
    pass

def test_translation():
    'extract smaller grid from bigger'
    pass

def test_rotations():
    'test rotations without translation'
    

def test_mol_transforms():
    '''compare interpolated transformed grid to grid generated from transformed molecule'''
    pass