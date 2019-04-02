import molgrid
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import os

@pytest.mark.slow()
def test_train_torch_cnn():
    batch_size = 50
    datadir = os.path.dirname(__file__)+'/data'
    fname = datadir+"/small.types"

    molgrid.set_random_seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    class Net(nn.Module):
        def __init__(self, dims):
            super(Net, self).__init__()
            self.pool0 = nn.MaxPool3d(2)
            self.conv1 = nn.Conv3d(dims[0], 32, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool3d(2)
            self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool3d(2)
            self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)

            self.last_layer_size = dims[1]//8 * dims[2]//8 * dims[3]//8 * 128
            self.fc1 = nn.Linear(self.last_layer_size, 2)

        def forward(self, x):
            x = self.pool0(x)
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            x = x.view(-1, self.last_layer_size)
            x = self.fc1(x)
            return x

    def weights_init(m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight.data)

    batch_size = 50
    e = molgrid.ExampleProvider(data_root=datadir+"/structs",balanced=True,shuffle=True)
    e.populate(fname)

    gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(e.type_size())
    tensor_shape = (batch_size,)+dims

    model = Net(dims).to('cuda')
    model.apply(weights_init)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda:0')
    labels = torch.zeros(batch_size, dtype=torch.long)

    losses = []
    for iteration in range(100):
        #load data
        batch = e.next_batch(batch_size)
        input_tensor.zero_()
        for (i,ex) in enumerate(batch):
            gmaker.forward(ex, input_tensor[i], 0, random_rotation=False) #not rotating since convergence is faster this way
            labels[i] = ex.labels[0]
        labels = labels.to('cuda')

        optimizer.zero_grad()
        output = model(input_tensor)
        loss = F.cross_entropy(output,labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))

    avefinalloss = np.array(losses[-5:]).mean()
    assert avefinalloss < .4
