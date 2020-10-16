libmolgrid
==========

libmolgrid is under active development, but should be suitable for use by early adopters.

If you use libmolgrid in your research, please cite:

**libmolgrid: Graphics Processing Unit Accelerated Molecular Gridding for Deep Learning Applications.** J Sunseri, DR Koes. *Journal of Chemical Information and Modeling*, 2020 [arxiv](https://arxiv.org/pdf/1912.04822.pdf)

```
@article{sunseri2020libmolgrid,
  title={libmolgrid: Graphics Processing Unit Accelerated Molecular Gridding for Deep Learning Applications},
  author={Sunseri, Jocelyn and Koes, David R},
  journal={Journal of Chemical Information and Modeling},
  volume={60},
  number={3},
  pages={1079--1084},
  year={2020},
  publisher={ACS Publications}
}
```

## Documentation

[https://gnina.github.io/libmolgrid/](https://gnina.github.io/libmolgrid/)

## Installation

### PIP

```pip install molgrid```

### conda
```conda install -c jsunseri molgrid```

### Build from Source

```apt install git build-essential libboost-all-dev python3-pip rapidjson-dev
pip3 install numpy pytest pyquaternion
```

[Install cmake 3.12 or higher.](https://cmake.org/install/)

[Install CUDA.](https://developer.nvidia.com/cuda-downloads)

[Install OpenBabel 3.0.](https://github.com/openbabel/openbabel)

`apt install libeigen3-dev libboost-all-dev`

```git clone https://github.com/gnina/libmolgrid.git
cd libmolgrid
mkdir build
cd build
cmake ..
make -j8
sudo make install
```




## Example
```
import molgrid
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import os

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
    dims = gmaker.grid_dimensions(e.num_types())
    tensor_shape = (batch_size,)+dims

    model = Net(dims).to('cuda')
    model.apply(weights_init)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
    float_labels = torch.zeros(batch_size, dtype=torch.float32)

    losses = []
    for iteration in range(100):
        #load data
        batch = e.next_batch(batch_size)
        gmaker.forward(batch, input_tensor, 0, random_rotation=False) #not rotating since convergence is faster this way
        batch.extract_label(0, float_labels)
        labels = float_labels.long().to('cuda')

        optimizer.zero_grad()
        output = model(input_tensor)
        loss = F.cross_entropy(output,labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))

```

