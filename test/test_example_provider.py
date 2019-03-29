import pytest
import molgrid
import numpy as np
import os

from pytest import approx
from numpy import around

# there's like umpteen gazillion configuration options for example provider..
datadir = os.path.dirname(__file__)+'/data'
#make sure we can read in molecular data
def test_mol_example_provider(capsys):
    fname = datadir+"/smallmol.types"
    e = molgrid.ExampleProvider(data_root=datadir+"/structs")
    e.populate(fname)
    with capsys.disabled(): #bunch openbabel garbage
        ex = e.next()
        b = e.next_batch(10) #should wrap around
    
    #with defaults, file should be read in order
    assert ex.labels[0] == 1 
    assert ex.labels[1] == approx(3.3747)
    assert ex.coord_sets[0].size() == 1289
    assert ex.coord_sets[1].size() == 8
    
    coords = ex.coord_sets[1].coord.tonumpy()
    assert tuple(coords[0]) == approx((26.6450,6.1410,4.6680))
    assert len(ex.coord_sets) == 2
    l0 = [ex.labels[0] for ex in b]
    l1 = [ex.labels[1] for ex in b]    
    
    #labels should be in order
    assert (1,1,0,0,0,1,1,1,0,0) == tuple(l0)
    
    assert (6.0000, 3.8697, -6.6990, -4.3010, -9.0000, 3.3747, 6.0000, 3.8697, -6.6990, -4.3010) == approx(tuple(l1))