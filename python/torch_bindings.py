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
    
class Coords2GridFunction(torch.autograd.Function):
    '''Layer for converting from coordinate and type tensors to a molecular grid'''
    
    @staticmethod
    def forward(ctx, gmaker, center, coords, types, radii):
        '''coords are Nx3, types are NxT, radii are N'''
        ctx.save_for_backward(coords, types, radii)
        ctx.gmaker = gmaker
        ctx.center = center
        shape = gmaker.grid_dimensions(types.shape[1]) #ntypes == nchannels
        output = torch.empty(*shape,dtype=torch.float32,device=coords.device)
        gmaker.forward(center, coords, types, radii, output)
        return output
        
    @staticmethod
    def backward(ctx, grid_gradient):
        '''Return Nx3 coordinate gradient and NxT type gradient'''
        coords, types, radii = ctx.saved_tensors
        gmaker = ctx.gmaker
        center = ctx.center
        grad_coords = torch.empty(*coords.shape,dtype=coords.dtype,device=coords.device)
        grad_types = torch.empty(*types.shape,dtype=types.dtype,device=types.device)
        #radii are fixed
        gmaker.backward(center, coords, types, radii, grid_gradient, grad_coords, grad_types)
        return None, None, grad_coords, grad_types, None
        
        
class Coords2Grid(torch.nn.Module):
    def __init__(self, gmaker, center):
        '''Convert coordinates/types/radii to a grid using the provided
        GridMaker and grid center'''
        super(Coords2Grid, self).__init__()
        self.gmaker = gmaker
        self.center = center
        
    def forward(self, coords, types, radii):
        return Coords2GridFunction.apply(self.gmaker, self.center, coords, types, radii)
    
    def extra_repr(self):
        return 'resolution {:.2f}, dimension {}, center {:.3f},{:.3f},{:.3f}'.format(
                self.gmaker.get_resolution(), self.gmaker.get_dimension(), self.center[0], self.center[1], self.center[2])        