import pytest
import molgrid
import pybel
import numpy as np
import os

from pytest import approx

# there's like umpteen gazillion configuration options for example provider..
datadir = os.path.dirname(__file__)+'/data'
#make sure we can read in molecular data
def test_mol_example_provider():
    fname = datadir+"/smallmol.types"