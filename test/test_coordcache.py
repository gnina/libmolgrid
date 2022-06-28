import pytest
import molgrid
import numpy as np
import os
import torch

from pytest import approx
from numpy import around

# there's like umpteen gazillion configuration options for example provider..
datadir = os.path.dirname(__file__) + '/data/structs/'


#create a coordinateset from a molecule with coordcache
def test_coordset_from_mol():
    rname = datadir+'187l/187l_rec.pdb'
    lname = datadir+'187l/187l_ligand.sdf'
    
    cache1 = molgrid.CoordCache()
    set1 = molgrid.CoordinateSet()
    cache1.set_coords(lname,set1)
    
    set2 = cache1.make_coords(lname)
    
    assert set1.size() == set2.size()
    assert set1.num_types() == set2.num_types()
    
    t = molgrid.ElementIndexTyper()
    cache2 = molgrid.CoordCache(t)
    set3 = cache2.make_coords(lname)

    assert set1.num_types() != set3.num_types()
    
    s = molgrid.ExampleProviderSettings(data_root=datadir)
    cache3 = molgrid.CoordCache(settings=s)
    
    set4 = cache3.make_coords('187l/187l_ligand.sdf')
    
    assert set4.num_types() != set3.num_types()
    assert set4.size() == set1.size() 
    
    rset = cache1.make_coords(rname)
    assert set1.size() != rset.size()