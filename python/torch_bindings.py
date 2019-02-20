import torch
import molgrid as mg

def tensor_as_grid(t):
    '''Return a Grid view of tensor t'''
    gname = 'Grid'
    gname += str(t.dim())
    g = globals()
    if isinstance(t,torch.FloatTensor):
        gname += 'f'
        return getattr(mg,gname)(mg.tofloatptr(t.data_ptr()),*t.shape)
    elif isinstance(t,torch.DoubleTensor):
        gname += 'd'
        return getattr(mg,gname)(mg.todoubleptr(t.data_ptr()),*t.shape)
    elif isinstance(t,torch.cuda.FloatTensor):
        gname += 'fCUDA'
        return getattr(mg,gname)(mg.tofloatptr(t.data_ptr()),*t.shape)
    elif isinstance(t,torch.cuda.DoubleTensor):
        gname += 'dCUDA'
        return getattr(mg,gname)(mg.todoubleptr(t.data_ptr()),*t.shape)    
    else:
        raise ValueError('Tensor base type %s not supported as grid type.'%str(t.dtype))
    
    return t