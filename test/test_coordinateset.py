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
    oldcoord = c.coords.tonumpy()
    #simple translate
    t = molgrid.Transform(molgrid.Quaternion(), (0,0,0), (1,1,1))
    t.forward(c,c)
    newcoord = c.coords.tonumpy()
    assert np.sum(newcoord-oldcoord) == approx(48)
    

#create a coordinateset from a molecule and convert to vector
#types with a dummy atom and typed radii
def test_coordset_from_mol_vec():
    m = pybel.readstring('smi','c1ccccc1CO')
    m.addh()
    m.make3D()
    
    c = molgrid.CoordinateSet(m) #default gnina ligand types
    c.make_vector_types(True, molgrid.defaultGninaLigandTyper.get_type_radii())
    
    assert c.type_vector.dimension(1) == 15
    assert c.radii.dimension(0) == 15
    assert c.has_vector_types()

    
    
#create a coordinateset from numpy and transform it
def test_coordset_from_array():
    
    coords = np.array([[1,0,-1],[1,3,-1],[1,0,-1]],np.float32)
    types = np.array([3,2,1],np.float32)
    radii = np.array([1.5,1.5,1.0],np.float32)
    c = molgrid.CoordinateSet(coords, types, radii, 4)

    oldcoordr = c.coords.tonumpy()
    #simple translate
    t = molgrid.Transform(molgrid.Quaternion(), (0,0,0), (-1,0,1))
    t.forward(c,c)
    newcoord = c.coords.tonumpy()
    
    assert c.coords[1,1] == 3.0
    assert np.sum(newcoord) == approx(3.0)
    
    c2 = c.clone()
    c2.coords[1,1] = 0
    assert c.coords[1,1] == 3.0

def test_coordset_merge():
    m = pybel.readstring('smi','c1ccccc1CO')
    m.addh()
    m.make3D()
    
    c = molgrid.CoordinateSet(m,molgrid.ElementIndexTyper())
    c2 = molgrid.CoordinateSet(m)

    c3 = molgrid.CoordinateSet(c,c2)
    c4 = molgrid.CoordinateSet(c,c2,False)

    assert c3.max_type == (c.max_type + c2.max_type)
    assert c3.coords.dimension(0) == (c.coords.dimension(0)+c2.type_index.size())

    assert c4.max_type == max(c.max_type,c2.max_type)
    assert c4.coords.dimension(0) == (c.coords.dimension(0)+c2.type_index.size())
    
    t = np.concatenate([c.type_index.tonumpy(),c2.type_index.tonumpy()+c.max_type])
    assert np.array_equal(t, c3.type_index.tonumpy())
    
    #test merging without unique types, which makes no sense
    assert c4.coords.tonumpy().shape == (24,3)
    t = np.concatenate([c.type_index.tonumpy(),c2.type_index.tonumpy()])
    assert np.array_equal(t, c4.type_index.tonumpy())
    
   
def test_coordset_merge():
    m = pybel.readstring('smi','c1ccccc1CO')
    m.addh()
    m.make3D()
    
    c = molgrid.CoordinateSet(m)
    c.make_vector_types()
    
    coords = np.zeros([10,3],np.float32)
    types = np.zeros([10,15],np.float32)
    radii = np.zeros(10,np.float32)
    
    n = c.copyTo(coords,types,radii)
    assert n == 8
    
    assert np.sum(coords) != 0
    #types should be padded out
    assert types[:,11].sum() == 0
    #coords too 
    assert coords[8:].sum() == 0
    assert radii[8:].sum() == 0
    
    #check truncation
    coordsm = np.zeros([5,3],np.float32)
    typesm = np.zeros([5,8],np.float32)
    radiim = np.zeros(5,np.float32)
    n = c.copyTo(coordsm,typesm,radiim)
    assert n == 5
    
    assert np.all(coordsm == coords[:5])
    assert np.all(typesm == types[:5,:8])
    assert np.all(radiim == radii[:5])
    