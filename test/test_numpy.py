import pytest
import molgrid
import numpy as np
from pytest import approx

        
def test_numpy():
  mg3d = molgrid.MGrid3d(3,3,3)
  mg2f = molgrid.MGrid2f(2,4)
  
  a2f = np.arange(27).astype(np.float32).reshape(3,-1)
  a3d = np.arange(27).astype(np.float).reshape(3,3,-1)

  g2f = molgrid.Grid2f(a2f)
  g3d = molgrid.Grid3d(a3d)
  
  g2f[0,0] = 100
  g3d[0,0,0] = 101
        
  assert a2f[0,0] == 100
  assert a3d[0,0,0] == 101
  
  mg2f.copyFrom(g2f)
  mg3d.copyFrom(g3d)
  
  assert mg2f[1,3] == 7
  
  mg2f[1,3] = 200
  assert g2f[1,3] == 12
  assert a2f[1,3] == 12
  mg2f.copyTo(g2f)
  assert g2f[0,7] == 200
  assert a2f[0,7] == 200

def test_numpy_conv():
  mg3d = molgrid.MGrid3d(3,3,3)
  mg2f = molgrid.MGrid2f(2,4)
  
  a2f = np.arange(27).astype(np.float32).reshape(3,-1)
  a3d = np.arange(27).astype(np.float).reshape(3,3,-1)

  a2f[0,0] = 100
  a3d[0,0,0] = 101        
  
  mg2f.copyFrom(a2f)
  mg3d.copyFrom(a3d)
  
  assert mg2f[1,3] == 7
  
  mg2f[1,3] = 200
  assert a2f[1,3] == 12
  mg2f.copyTo(a2f)
  assert a2f[0,7] == 200
  
def test_tonumpy():
    '''tonumpy copies'''
    mg = molgrid.MGrid1d(10)
    for i in range(10):
        mg[i] = i
        
    a1 = mg.tonumpy()
    mg[0] = 100
    a2 = molgrid.tonumpy(mg)
    for i in range(10):
        mg[i] = 0
        
    a3 = mg.tonumpy()
    assert a1.sum() == 45
    assert a2.sum() == 145
    assert a3.sum() == 0
    
def test_clear():
     g1 = molgrid.Grid1d(np.random.rand(20))
     mg = molgrid.MGrid3d(4,5,6)
     mg.copyFrom(np.random.rand(4,5,6))
     
     mg.fill_zero()
     g1.fill_zero()
     
     assert np.all(g1.tonumpy() == 0)
     assert np.all(mg.tonumpy() == 0)
     
