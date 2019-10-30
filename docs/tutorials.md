---
layout: page
homepage: false
---

# Tutorials
Here are some notebooks demonstrating basic tasks implemented with libmolgrid.
We demonstrate usage with PyTorch, Keras with the Tensorflow backend, and
Caffe, and show how to implement and train a few types of models. 
1. [Train basic CNN with PyTorch](#torch_train)
2. [Train basic CNN with Keras](#keras_train)

## Train basic CNN with PyTorch<a name="torch_train"></a>
```python
import molgrid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import os

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

# use the libmolgrid ExampleProvider to obtain shuffled, balanced, and stratified batches from a file
e = molgrid.ExampleProvider(data_root=datadir+"/structs",balanced=True,shuffle=True)
e.populate(fname)

# use the libmolgrid gridmaker to generate grids from a batch or directly from molecular structures
gmaker = molgrid.GridMaker()
dims = gmaker.grid_dimensions(e.type_size())
tensor_shape = (batch_size,)+dims

model = Net(dims).to('cuda')
model.apply(weights_init)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
float_labels = torch.zeros(batch_size, dtype=torch.float32)

losses = []
for iteration in range(100):
    # load data
    batch = e.next_batch(batch_size)
    # libmolgrid can interoperate directly with Torch tensors, using views over the same memory.
    # internally, the libmolgrid GridMaker can use libmolgrid Transforms to apply random rotations and translations for data augmentation
    # the user may also use libmolgrid Transforms directly in python
    gmaker.forward(batch, input_tensor, 0, random_rotation=False) 
    batch.extract_label(0, float_labels)
    labels = float_labels.long().to('cuda')

    optimizer.zero_grad()
    output = model(input_tensor)
    loss = F.cross_entropy(output,labels)
    loss.backward()
    optimizer.step()
    losses.append(float(loss))

avefinalloss = np.array(losses[-5:]).mean()
assert avefinalloss < .4
```

## Train basic CNN with Keras<a name="keras_train"></a>
```python
import molgrid
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import keras.layers
import os

molgrid.set_random_seed(0)
np.random.seed(0)

def create_model(dims):
    """ Creates a 3D CNN by defining and applying layers simultaneously. """

    input_layer = keras.layers.Input(shape=dims) 
    pool0 = keras.layers.MaxPooling3D(data_format="channels_first")(input_layer)
    conv1 = keras.layers.Conv3D(filters=32, kernel_size=3, data_format="channels_first", activation="relu")(pool0)
    pool1 = keras.layers.MaxPooling3D(data_format="channels_first")(conv1)
    conv2 = keras.layers.Conv3D(filters=64, kernel_size=3, data_format="channels_first", activation="relu")(pool1)
    pool2 = keras.layers.MaxPooling3D(data_format="channels_first")(conv2)
    conv3 = keras.layers.Conv3D(filters=128, kernel_size=3, data_format="channels_first", activation="relu")(pool2)

    flatten = keras.layers.Flatten(data_format="channels_first")(conv3)
    
    fc1 = keras.layers.Dense(2,activation='softmax')(flatten)  

    # Define and return model
    model = keras.models.Model(inputs=input_layer, outputs=fc1)
    model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9), loss="sparse_categorical_crossentropy")

    return model

batch_size = 50
datadir = os.getcwd() +'/../data'
fname = datadir+"/small.types"

e = molgrid.ExampleProvider(data_root=datadir+"/structs",balanced=True,shuffle=True)
e.populate(fname)

gmaker = molgrid.GridMaker()
dims = gmaker.grid_dimensions(e.type_size())
tensor_shape = (batch_size,)+dims

model = create_model(dims)

labels = molgrid.MGrid1f(batch_size)
input_tensor = molgrid.MGrid5f(*tensor_shape)

losses = []
for iteration in range(40):
    # load data
    batch = e.next_batch(batch_size)
    
    gmaker.forward(batch, input_tensor, 0, random_rotation=True) 
    batch.extract_label(0, labels)

    loss = model.train_on_batch(input_tensor.tonumpy(), labels.tonumpy())
    losses.append(float(loss))
    print(float(loss))

avefinalloss = np.array(losses[-5:]).mean()
assert avefinalloss < .4
```
