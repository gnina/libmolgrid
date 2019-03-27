import pytest
import molgrid
import pybel
import numpy as np

from pytest import approx

#create a coordinateset from a molecule and transform it
def test_coordset_from_mol():
    m = pybel.readstring('smi','c1ccccc1CO')
    m.addh()
    m.make3D()
    
    c = molgrid.CoordinateSet(m,molgrid.ElementIndexTyper())
    oldcoord = c.coord.tonumpy()
    #simple translate
    t = molgrid.Transform(molgrid.Quaternion(), (0,0,0), (1,1,1))
    t.forward(c,c)
    newcoord = c.coord.tonumpy()
    assert np.sum(newcoord-oldcoord) == approx(48)

#create a coordinateset from numpy and transform it
def test_coordset_from_array():
    pass