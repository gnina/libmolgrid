import pytest
import molgrid

def test_grid():
    g = molgrid.MGrid2f(10,2)
    assert g.size() == 20
    g2 = molgrid.Grid2f(g.cpu())
    g2[5][1] = 3.0
    assert g[5,1] == g2[5,1]
    g1 = g2[5]
    g1[1] = 4.0
    assert g[5,1] == 4.0
    
    gclone = g.clone()
    gclone[5,1] = 5.5
    assert g[5,1] == 4.0
    assert gclone[5][1] == 5.5
    
