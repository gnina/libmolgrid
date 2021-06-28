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

    batch = e.next_batch(1)
    a = np.array([0],dtype=np.float32)
    batch.extract_label(1,a)


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

def test_cached_with_typer_example_provider():
    fname = datadir+"/ligonly.types"
    t = molgrid.ElementIndexTyper(80)
    e = molgrid.ExampleProvider(t,ligmolcache=datadir+'/lig.molcache2')
    e.populate(fname)
    batch = e.next_batch(10)
    c = batch[0].coord_sets[1]
    assert c.max_type == 80
    assert c.type_index[0] == 7

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
    assert tv.shape == (10,14) 
    assert tv[0].sum() == 1.0
    assert tv[0][8] == 1.0
    
    
    e2 = molgrid.ExampleProvider(data_root=datadir+"/structs",make_vector_types=True)
    e2.populate(fname)
    b2 = e2.next_batch(batch_size)
    c2 = b2[0].merge_coordinates(unique_index_types=True)
    tv2 = c2.type_vector.tonumpy()
    assert tv2.shape == (10,28)
    
    gmaker.forward(b, mgrid)
    
    assert b[0].coord_sets[0].has_vector_types()
    assert b[0].coord_sets[1].has_vector_types()
    
    assert b[0].num_types() == 14
        
def test_type_sizing():
    fname = datadir+"/ligonly.types"
    e = molgrid.ExampleProvider(data_root=datadir+"/structs",make_vector_types=True)
    e.populate(fname)
    batch_size = 10
    b = e.next_batch(batch_size)
    #provider and example should agree on number of types, even if one coordset is empty
    assert e.num_types() == b[0].num_types()
    
def test_vector_sum_types():
    fname = datadir+"/ligonly.types"
    e = molgrid.ExampleProvider(data_root=datadir+"/structs",make_vector_types=True)
    e.populate(fname)
    batch_size = 10
    b = e.next_batch(batch_size)
    sum = molgrid.MGrid2f(batch_size, e.num_types())
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
    sum = molgrid.MGrid2f(batch_size, e.num_types())
    b.sum_types(sum)
    np.testing.assert_allclose(sum[0].tonumpy(), [ 2., 3., 0.,
       0., 0., 0., 0., 0., 2., 2., 1., 0., 0., 0.], atol=1e-5)

def test_copied_examples():
    fname = datadir+"/ligonly.types"
    e = molgrid.ExampleProvider(data_root=datadir+"/structs")
    e.populate(fname)
    batch_size = 10
    b = e.next_batch(batch_size)
    for i in range(1,batch_size):
        sqsum = np.square(b[0].coord_sets[1].coords.tonumpy() - b[i].coord_sets[1].coords.tonumpy()).sum()
        assert sqsum > 0
        
    #now with duplicates
    e = molgrid.ExampleProvider(data_root=datadir+"/structs",num_copies=batch_size)
    e.populate(fname)
    b = e.next_batch(batch_size)
    for i in range(1,batch_size):
        sqsum = np.square(b[0].coord_sets[1].coords.tonumpy() - b[i].coord_sets[1].coords.tonumpy()).sum()
        assert sqsum == 0    
        
    #transforming one of the duplicates should not effect the others
    orig = b[0].coord_sets[1].coords.tonumpy()
    lig = b[1].coord_sets[1]
    t = molgrid.Transform(lig.center(), 2,random_rotation=True)
    t.forward(lig,lig)
    new0 = b[0].coord_sets[1].coords.tonumpy()
    new1 = b[1].coord_sets[1].coords.tonumpy()
    
    np.testing.assert_allclose(orig,new0)
    sqsum = np.square(new1-orig).sum()
    assert sqsum > 0
    
def test_example_provider_iterator_interface():
    fname = datadir+"/small.types"
    BSIZE=25
    e = molgrid.ExampleProvider(data_root=datadir+"/structs",default_batch_size=BSIZE)
    e.populate(fname)
    
    e2 = molgrid.ExampleProvider(data_root=datadir+"/structs",default_batch_size=BSIZE)
    e2.populate(fname)

    nlabels = e.num_labels()
    labels = molgrid.MGrid2f(BSIZE,nlabels)
    labels2 = molgrid.MGrid2f(BSIZE,nlabels)

    for (i, b) in enumerate(e):
        b2 = e2.next_batch()
        b.extract_labels(labels.cpu())
        b2.extract_labels(labels2.cpu())
        np.testing.assert_allclose(labels,labels2)
        if i > 10:
            break
        
def test_pytorch_dataset():
    fname = datadir+"/small.types"
    
    e = molgrid.ExampleProvider(data_root=datadir+"/structs")
    e.populate(fname)
    m = molgrid.MolDataset(fname,data_root=datadir+"/structs")
    
    assert len(m) == 1000

    ex = e.next()
    coordinates = ex.merge_coordinates()

    center, coords, types, radii, labels = m[0]

    assert list(center.shape) == [3]
    np.testing.assert_allclose(coords, coordinates.coords.tonumpy())
    np.testing.assert_allclose(types, coordinates.type_index.tonumpy())
    np.testing.assert_allclose(radii, coordinates.radii.tonumpy())

    assert len(labels) == 3
    assert labels[0] == 1
    np.testing.assert_allclose(labels[1],6.05)
    np.testing.assert_allclose(labels[-1],0.162643)

    center, coords, types, radii, labels = m[-1]
    assert labels[0] == 0
    np.testing.assert_allclose(labels[1], -10.3)    

    '''Testing out the collate_fn when used with torch.utils.data.DataLoader'''
    torch_loader = torch.utils.data.DataLoader(      
        m, batch_size=8,collate_fn=molgrid.MolDataset.collateMolDataset)
    iterator = iter(torch_loader)
    next(iterator)
    lengths, center, coords, types, radii, labels = next(iterator)
    assert len(lengths) == 8
    assert center.shape[0] == 8
    assert coords.shape[0] == 8
    assert types.shape[0] == 8
    assert radii.shape[0] == 8
    assert radii.shape[0] == 8
    assert labels.shape[0] == 8

    mcenter, mcoords, mtypes, mradii, mlabels = m[10]
    np.testing.assert_allclose(center[2],mcenter) 
    np.testing.assert_allclose(coords[2][:lengths[2]],mcoords)
    np.testing.assert_allclose(types[2][:lengths[2]],mtypes)
    np.testing.assert_allclose(radii[2][:lengths[2]],mradii.unsqueeze(1))
    assert len(labels[2]) == len(mlabels)
    assert labels[2][0] == mlabels[0]
    assert labels[2][1] == mlabels[1]
        
def test_duplicated_examples():
    '''This is for files with multiple ligands'''
    fname = datadir+"/multilig.types"
    e = molgrid.ExampleProvider(data_root=datadir+"/structs")
    e.populate(fname)
    batch_size = 10
    b = e.next_batch(batch_size)
    for i in range(1,batch_size):
        assert len(b[i].coord_sets) == 3 #one rec and two ligands
        #ligands should be different
        sqsum = np.square(b[i].coord_sets[1].coords.tonumpy() - b[i].coord_sets[2].coords.tonumpy()).sum()
        assert sqsum > 0    
        
    e = molgrid.ExampleProvider(data_root=datadir+"/structs",duplicate_first=True)
    e.populate(fname)
    batch_size = 10
    b = e.next_batch(batch_size)
    for i in range(1,batch_size):
        assert len(b[i].coord_sets) == 4 #rec lig rec lig
        #ligands should be different
        sqsum = np.square(b[i].coord_sets[1].coords.tonumpy() - b[i].coord_sets[3].coords.tonumpy()).sum()
        assert sqsum > 0
        #receptors should be the same
        sqsum = np.square(b[i].coord_sets[0].coords.tonumpy() - b[i].coord_sets[2].coords.tonumpy()).sum()
        
        
def test_example_provider_epoch_iteration():
    fname = datadir+"/small.types"
    e = molgrid.ExampleProvider(data_root=datadir+"/structs",default_batch_size=10,iteration_scheme=molgrid.IterationScheme.LargeEpoch)
    e.populate(fname)
    
    assert e.small_epoch_size() == 1000
    assert e.large_epoch_size() == 1000
    
    cnt = 0
    for batch in e:
        cnt += 1
    assert cnt == 100
    
    e = molgrid.ExampleProvider(data_root=datadir+"/structs",default_batch_size=10,balanced=True,iteration_scheme=molgrid.IterationScheme.LargeEpoch)
    e.populate(fname)
    
    assert e.small_epoch_size() == 326
    assert e.large_epoch_size() == 1674
    
    cnt = 0
    for batch in e:
        cnt += 1
    assert cnt == 168
    
    
    e = molgrid.ExampleProvider(data_root=datadir+"/structs",default_batch_size=10,balanced=False,stratify_receptor=True,iteration_scheme=molgrid.IterationScheme.SmallEpoch)
    e.populate(fname)
    
    assert e.small_epoch_size() == 120
    assert e.large_epoch_size() == 1260
    
    cnt = 0
    for batch in e:
        cnt += 1
    assert cnt == 12   
    
    values = set()
    e = molgrid.ExampleProvider(data_root=datadir+"/structs",default_batch_size=8,balanced=True,stratify_receptor=True,iteration_scheme=molgrid.IterationScheme.SmallEpoch)
    e.populate(fname)
    
    assert e.small_epoch_size() == 112
    assert e.large_epoch_size() == 2240
    
    cnt = 0
    small = 0
    large = 0
    for batch in e:
        for ex in batch:
            key = ex.coord_sets[0].src+":"+ex.coord_sets[1].src
            #small epoch should see an example at _most_ once
            assert key not in values
            values.add(key)
        cnt += 1
        s = e.get_small_epoch_num()
        assert s >= small
        if s > small:
            assert s == small+1
            small = s
        l = e.get_large_epoch_num()
        assert l >= large
        if l > large:
            assert l == large+1
            large = l            
    assert cnt == 14      
    
    e = molgrid.ExampleProvider(data_root=datadir+"/structs",default_batch_size=10,balanced=True,stratify_receptor=True,iteration_scheme=molgrid.IterationScheme.LargeEpoch)
    e.populate(fname)
    
    assert e.small_epoch_size() == 112
    assert e.large_epoch_size() == 2240
    
    values = set()
    cnt = 0
    small = 0
    large = 0
    for batch in e:
        for ex in batch:
            key = ex.coord_sets[0].src+":"+ex.coord_sets[1].src
            values.add(key)
        cnt += 1
        s = e.get_small_epoch_num()
        assert s >= small
        if s > small:
            assert s == small+1
            small = s
        l = e.get_large_epoch_num()
        assert l >= large
        if l > large:
            assert l == large+1
            large = l            
    assert cnt == 224   
    assert len(values) == e.size() #large epoch should see everything at least once
