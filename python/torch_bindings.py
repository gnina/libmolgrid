import torch
import molgrid as mg
import types
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

#extend grid maker to create pytorch Tensor
def make_grid_tensor(gridmaker, center, c):
    '''Create appropriately sized pytorch tensor of grid densities.  set_gpu_enabled can be used to control if result is located on the cpu or gpu'''
    dims = gridmaker.grid_dimensions(c.max_type) # this should be grid_dims or get_grid_dims
    if mg.get_gpu_enabled():
        t = torch.zeros(dims, dtype=torch.float32, device='cuda:0')
    else:
        t = torch.zeros(dims, dtype=torch.float32)
    gridmaker.forward(center, c, t)
    return t 

mg.GridMaker.make_tensor = make_grid_tensor
    
class Coords2GridFunction(torch.Function):
    '''Layer for converting from coordinate and type tensors to a molecular grid'''
    
    @staticmethod
    def forward(ctx, gmaker, coords, types, radii):
        '''coords are Nx3, types are NxT, radii are N'''
        ctx.save_for_backward(coords, types, radii)
        ctx.gmaker = gmaker
        #gmaker.forward()
        
        #will need to create output tensor to view as grid
        
    @staticmethod
    def backward(ctx, grid_gradient):
        coords, types, radii = ctx.saved_tensors
        gmaker = ctx.gmaker