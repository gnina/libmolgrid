import pytest
import molgrid

def test_grid():
    g = molgrid.MGrid2f(10,2)
    assert g.size() == 20