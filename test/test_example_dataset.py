import pytest
import molgrid
import numpy as np
import os
import torch

from pytest import approx
from numpy import around

datadir = os.path.dirname(__file__) + '/data'

#make sure we can map and iterate    
def test_example_dataset():
    fname = datadir + "/small.types"
    e = molgrid.ExampleDataset(data_root=datadir + "/structs")
    e.populate(fname)

    assert len(e) == 1000
    assert e[-1].labels[1] == approx(-10.3)
    assert e[3].labels[1] == approx(-6.05)
    
    for ex in e:
        pass
    assert ex.labels[1] == approx(-10.3)
    
    for ex in e:
        break
        
    assert ex.labels[1] == approx(6.05)
