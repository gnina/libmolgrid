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

def test_gnina_example_provider():
    fname = datadir+"/small.types"
    e = molgrid.ExampleProvider(data_root=datadir+"/structs")
    e.populate(fname)

    batch_size = 100
    batch = e.next_batch(batch_size)
    #extract labels
    nlabels = e.num_labels()
    assert nlabels == 3
    labels = molgrid.MGrid2f(batch_size,nlabels)
    gpulabels = molgrid.MGrid2f(batch_size,nlabels)

    batch.extract_labels(labels.cpu())
    batch.extract_labels(gpulabels.gpu())
    assert np.array_equal(labels.tonumpy(), gpulabels.tonumpy())
    label0 = molgrid.MGrid1f(batch_size)
    label1 = molgrid.MGrid1f(batch_size)
    label2 = molgrid.MGrid1f(batch_size)
    batch.extract_label(0, label0.cpu())
    batch.extract_label(1, label1.cpu())
    batch.extract_label(2, label2.gpu())

    assert label0[0] == 1
    assert label1[0] == approx(6.05)
    assert label2[0] == approx(0.162643)
    assert labels[0,0] == 1
    assert labels[0][1] == approx(6.05)
    assert labels[0][2] == approx(0.162643)

    for i in range(nlabels):
        assert label0[i] == labels[i][0]
        assert label1[i] == labels[i][1]
        assert label2[i] == labels[i][2]
        
    ex = batch[0]
    crec = ex.coord_sets[0]
    assert crec.size() == 1781
    assert list(crec.coord[0]) == approx([45.042, 12.872, 13.001])
    assert list(crec.type_index)[:10] == [6.0, 1.0, 1.0, 7.0, 0.0, 6.0, 1.0, 1.0, 7.0, 1.0]
    
    clig = ex.coord_sets[1]
    assert clig.size() == 10
    assert list(clig.coord[9]) == approx([27.0536, 3.2453, 32.4511])
    assert list(clig.type_index) == [8.0, 1.0, 1.0, 9.0, 10.0, 0.0, 0.0, 1.0, 9.0, 8.0]


def test_cached_example_provider():
    fname = datadir+"/small.types"
    e = molgrid.ExampleProvider(ligmolcache=datadir+'/lig.molcache2',recmolcache=datadir+'/rec.molcache2')
    e.populate(fname)

    batch_size = 100
    batch = e.next_batch(batch_size)
    #extract labels
    nlabels = e.num_labels()
    assert nlabels == 3
    labels = molgrid.MGrid2f(batch_size,nlabels)
    gpulabels = molgrid.MGrid2f(batch_size,nlabels)

    batch.extract_labels(labels.cpu())
    batch.extract_labels(gpulabels.gpu())
    assert np.array_equal(labels.tonumpy(), gpulabels.tonumpy())
    label0 = molgrid.MGrid1f(batch_size)
    label1 = molgrid.MGrid1f(batch_size)
    label2 = molgrid.MGrid1f(batch_size)
    batch.extract_label(0, label0.cpu())
    batch.extract_label(1, label1.cpu())
    batch.extract_label(2, label2.gpu())

    assert label0[0] == 1
    assert label1[0] == approx(6.05)
    assert label2[0] == approx(0.162643)
    assert labels[0,0] == 1
    assert labels[0][1] == approx(6.05)
    assert labels[0][2] == approx(0.162643)

    for i in range(nlabels):
        assert label0[i] == labels[i][0]
        assert label1[i] == labels[i][1]
        assert label2[i] == labels[i][2]
        
    ex = batch[0]
    crec = ex.coord_sets[0]
    assert crec.size() == 1781
    assert list(crec.coord[0]) == approx([45.042, 12.872, 13.001])
    assert list(crec.type_index)[:10] == [6.0, 1.0, 1.0, 7.0, 0.0, 6.0, 1.0, 1.0, 7.0, 1.0]
    
    clig = ex.coord_sets[1]
    assert clig.size() == 10
    assert list(clig.coord[9]) == approx([27.0536, 3.2453, 32.4511])        
    assert list(clig.type_index) == [8.0, 1.0, 1.0, 9.0, 10.0, 0.0, 0.0, 1.0, 9.0, 8.0]

