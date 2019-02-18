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
