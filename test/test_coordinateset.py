import pytest
import molgrid
try:
    import pybel #2.0
except ImportError:
    from openbabel import pybel  #3.0

import numpy as np

from pytest import approx

#create a coordinateset from a molecule and transform it
def test_coordset_from_mol():
    m = pybel.readstring('smi','c1ccccc1CO')
    m.addh()
    m.make3D()
    
    c = molgrid.CoordinateSet(m,molgrid.ElementIndexTyper())
    oldcoord = c.coord_radius.tonumpy()[:,:3]
    #simple translate
    t = molgrid.Transform(molgrid.Quaternion(), (0,0,0), (1,1,1))
    t.forward(c,c)
    newcoord = c.coord_radius.tonumpy()[:,:3]
    assert np.sum(newcoord-oldcoord) == approx(48)

#create a coordinateset from numpy and transform it
def test_coordset_from_array():
    
    coords = np.array([[1,0,-1],[1,3,-1],[1,0,-1]],np.float32)
    types = np.array([3,2,1],np.float32)
    radii = np.array([1.5,1.5,1.0],np.float32)
    c = molgrid.CoordinateSet(coords, types, radii, 4)

    oldcoord = c.coord.tonumpy()
    #simple translate
    t = molgrid.Transform(molgrid.Quaternion(), (0,0,0), (-1,0,1))
    t.forward(c,c)
    newcoord = c.coord.tonumpy()
    
    assert c.coord[1,1] == 3.0
    assert np.sum(newcoord) == approx(3.0)
    
    c2 = c.clone()
    c2.coord[1,1] = 0
    assert c.coord[1,1] == 3.0
