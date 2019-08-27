import pytest
import molgrid
import numpy as np
import os
import torch

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

    coords = ex.coord_sets[1].coords.tonumpy()
    assert tuple(coords[0,:]) == approx((26.6450,6.1410,4.6680))
    assert len(ex.coord_sets) == 2
    l0 = [ex.labels[0] for ex in b]
    l1 = [ex.labels[1] for ex in b]

    #labels should be in order
    assert (1,1,0,0,0,1,1,1,0,0) == tuple(l0)

    assert (6.0000, 3.8697, -6.6990, -4.3010, -9.0000, 3.3747, 6.0000, 3.8697, -6.6990, -4.3010) == approx(tuple(l1))

def test_custom_typer_example_provider():
    fname = datadir+"/small.types"
    t = molgrid.ElementIndexTyper(80)
    e = molgrid.ExampleProvider(t,data_root=datadir+"/structs")
    e.populate(fname)
    batch = e.next_batch(10)
    c = batch[0].coord_sets[0]
    assert c.max_type == 80
    
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
    assert list(crec.coords[0]) == approx([45.042, 12.872, 13.001])
    assert crec.radii[0] == approx(1.8)
    assert list(crec.type_index)[:10] == [6.0, 1.0, 1.0, 7.0, 0.0, 6.0, 1.0, 1.0, 7.0, 1.0]
    
    clig = ex.coord_sets[1]
    assert clig.size() == 10
    assert list(clig.coords[9]) == approx([27.0536, 3.2453, 32.4511])
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
    assert list(crec.coords[0]) == approx([45.042, 12.872, 13.001])
    assert list(crec.type_index)[:10] == [6.0, 1.0, 1.0, 7.0, 0.0, 6.0, 1.0, 1.0, 7.0, 1.0]
    
    clig = ex.coord_sets[1]
    assert clig.size() == 10
    assert list(clig.coords[9]) == approx([27.0536, 3.2453, 32.4511])
    assert clig.radii[9] == approx(1.8)        
    assert list(clig.type_index) == [8.0, 1.0, 1.0, 9.0, 10.0, 0.0, 0.0, 1.0, 9.0, 8.0]

def test_grouped_example_provider():
    fname = datadir+"/grouped.types"
    batch_size = 3
    e = molgrid.ExampleProvider(data_root=datadir+"/structs",max_group_size=5,group_batch_size=batch_size)
    e.populate(fname)

    
    def testprovider(e,gsize):
        def getrecs(b):
            return [ ex.coord_sets[0].src for ex in b]
        
        def getligs(b):
            ligs = []
            for ex in b:
                if len(ex.coord_sets) > 1:
                    ligs.append(ex.coord_sets[1].src)
                else:
                    ligs.append(None)
            return ligs
        
        for _ in range(10):
            batch = e.next_batch(batch_size)
            firstrecs = getrecs(batch)
            firstligs = getligs(batch)
            grps = [ex.group for ex in batch]
            for x in batch:
                assert not x.seqcont
            for i in range(gsize-1):
                #rest of group - should match receptor but not ligand
                batch = e.next_batch(batch_size)
                recs = getrecs(batch)
                ligs = getligs(batch)
                for (x,g) in zip(batch,grps):
                    assert x.seqcont        
                    assert x.group == g 
                           
                for (r1,r2) in zip(firstrecs,recs):
                    if r2:
                        assert r1 == r2
                for (l1,l2) in zip(firstligs, ligs):
                    assert l1 != l2
    
    e = molgrid.ExampleProvider(data_root=datadir+"/structs",max_group_size=5,group_batch_size=batch_size)
    e.populate(fname)
    testprovider(e,5)
    
    e = molgrid.ExampleProvider(data_root=datadir+"/structs",max_group_size=7,group_batch_size=batch_size,shuffle=True,balanced=True)
    e.populate(fname)
    testprovider(e,7)   
    
def test_make_vector_types_ex_provider(capsys):
    fname = datadir+"/ligonly.types"
    e = molgrid.ExampleProvider(molgrid.NullIndexTyper(),molgrid.defaultGninaLigandTyper, data_root=datadir+"/structs",make_vector_types=True)
    e.populate(fname)
    batch_size = 10
    b = e.next_batch(batch_size)

    gmaker = molgrid.GridMaker(dimension=23.5,radius_type_indexed=True)
    shape = gmaker.grid_dimensions(molgrid.defaultGninaLigandTyper.num_types())
    mgrid = molgrid.MGrid5f(batch_size,*shape)

    c = b[0].merge_coordinates()
    tv = c.type_vector.tonumpy()
    assert tv.shape == (10,14) #no dummy type
    assert tv[0].sum() == 1.0
    assert tv[0][8] == 1.0
    
    gmaker.forward(b, mgrid)
    
    assert b[0].coord_sets[0].has_vector_types()
    assert b[0].coord_sets[1].has_vector_types()
    
    assert b[0].type_size() == 14
    
def test_type_sizing():
    fname = datadir+"/ligonly.types"
    e = molgrid.ExampleProvider(data_root=datadir+"/structs",make_vector_types=True)
    e.populate(fname)
    batch_size = 10
    b = e.next_batch(batch_size)
    #provider and example should agree on number of types, even if one coordset is empty
    assert e.type_size() == b[0].type_size()
    
def test_vector_sum_types():
    fname = datadir+"/ligonly.types"
    e = molgrid.ExampleProvider(data_root=datadir+"/structs",make_vector_types=True)
    e.populate(fname)
    batch_size = 10
    b = e.next_batch(batch_size)
    sum = molgrid.MGrid2f(batch_size, e.type_size())
    b.sum_types(sum)
    sum2 = np.zeros(sum.shape,np.float32)
    b.sum_types(sum2)
    sum3 = torch.empty(sum.shape,dtype=torch.float32,device='cuda')
    b.sum_types(sum3)
    np.testing.assert_allclose(sum.tonumpy(),sum3.detach().cpu().numpy(),atol=1e-5)
    np.testing.assert_allclose(sum.tonumpy(),sum2,atol=1e-5)
    np.testing.assert_allclose(sum[0].tonumpy(), [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 3., 0.,
       0., 0., 0., 0., 0., 2., 2., 1., 0., 0., 0.], atol=1e-5)

    e = molgrid.ExampleProvider(molgrid.NullIndexTyper(), molgrid.defaultGninaLigandTyper, data_root=datadir+"/structs",make_vector_types=True)
    e.populate(fname)
    b = e.next_batch(batch_size)
    sum = molgrid.MGrid2f(batch_size, e.type_size())
    b.sum_types(sum)
    np.testing.assert_allclose(sum[0].tonumpy(), [ 2., 3., 0.,
       0., 0., 0., 0., 0., 2., 2., 1., 0., 0., 0.], atol=1e-5)

