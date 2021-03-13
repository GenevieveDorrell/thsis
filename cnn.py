import time
import pandas as pd
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


from geoProc_latlon import get_numpy_from_tiffs

"""
no normalization no slope
Accuracy:  27.333333333333332
"""

#geo data
lossyear = "data\\2013\Hansen_GFC2013_lossyear_50N_130W.tif"
treecover = "data\\2013\Hansen_GFC-2019-v1.7_treecover2000_50N_130W.tif"
elevation = "data\\2013\\USGS_13_n43w124.tif"
slope = "data\\2013\slope2.tif"

#file paths for bigger fire
sev1_fp = "data\\2013\or4273212351520130726\or4273212351520130726_20130703_20140706_dnbr6.tif"
rgb1_fp = "data\\2013\or4273212351520130726\or4273212351520130726_20130703_l8_refl.tif"

#file paths for smaller fire
sev2_fp = "data\\2013\or4285712358520130726\or4285712358520130726_20130703_20140706_dnbr6.tif"
rgb2_fp = "data\\2013\or4285712358520130726\or4285712358520130726_20130703_l8_refl.tif"

#mill fire
sev3_fp = "data\\2017\or4293912360020170827\or4293912360020170827_20170714_20180717_dnbr6.tif"
rgb3_fp = "data\\2017\or4293912360020170827\or4293912360020170827_20170714_l8_refl.tif"

#big windy complex
sev4_fp = "data\\2018\or4252812357120180715\or4252812357120180715_20180701_20190720_dnbr6.tif"
rgb4_fp = "data\\2018\or4252812357120180715\or4252812357120180715_20180701_L8_refl.tif"

#other fire
sev5_fp = "data\\2013\or4261412376020130726\or4261412376020130726_20130703_20140706_dnbr6.tif"
rgb5_fp = "data\\2013\or4261412376020130726\or4261412376020130726_20130703_l8_refl.tif"

# y_train, y_test, x_train, x_test = get_numpy_from_tiffs([sev1_fp, sev2_fp, sev3_fp, sev4_fp, sev5_fp],
#                                                         [rgb1_fp, rgb2_fp, rgb3_fp, rgb4_fp, rgb5_fp], 
#                                                         [treecover, lossyear, elevation])
y_train, y_test, x_train, x_test = get_numpy_from_tiffs([sev1_fp],[rgb1_fp], [treecover])#, slope])

print(y_train.shape)
print(y_test.shape)
print(x_train.shape)
print(x_test.shape)
input_length = x_train.shape[1]
num_classes = 4

mean = x_train.mean()
std = x_train.std()


train_data = TensorDataset(torch.tensor(x_train, dtype=torch.float), 
                           torch.tensor(y_train, dtype=torch.long))
test_data = TensorDataset(torch.tensor(x_test, dtype=torch.float), 
                          torch.tensor(y_test, dtype=torch.long))
                          #transform=transformer)

torch.manual_seed(42)

batch_size = 100
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# https://www.jeremyjordan.me/convnet-architectures/



# Building CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.batch_size = batch_size
        # input [batch_size x 1 x 28 x 28], 10 output nodes, 3x3 kernel
        self.conv1 = nn.Conv1d(1, batch_size, input_length)
        self.conv2 = nn.Conv1d(10, 20, 4)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_length, self.batch_size)
        self.fc2 = nn.Linear(self.batch_size, 128)
        self.fc3 = nn.Linear(128, num_classes)     # 10 output nodes
    
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = x.view(self.batch_size, 1,input_length)
        x = F.relu(self.conv1(x), (2, 2))
        #x = x.view(self.batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#batchNormalization


# Initializing model, loss function, and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training model on data
start = time.time()
for epoch in range(5):
    print("\nEpoch", epoch+1)
    running_loss = 0
    for i, (feats, labels) in enumerate(train_loader):
        if feats.shape[0] == batch_size:
            feats = feats.view([batch_size, 1, 1, input_length])

            optimizer.zero_grad()

            output = net(feats)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i * batch_size % 10000 == 0:
                print(i * batch_size, "/", len(train_loader) * batch_size, 
                        "\tLoss:", running_loss/(i*batch_size+1))

print(round((time.time() - start), 2), "s")



# Testing model with test dataset
with torch.no_grad():
    count = 0
    catagories = [[0,0],[0,0],[0,0],[0,0], [0,0]]
    for i, (test, labels) in enumerate(test_loader):
        if test.shape[0] == batch_size:
            test = test.view([batch_size, 1, 1, input_length])

            test_out = net(test)
            pred = torch.max(test_out, 1)[1]
            for j, item in enumerate(pred):

                count += int(item == labels[j])
                catagories[item][0] += int(item == labels[j])
                catagories[labels[j]][1] += 1
                


    print("Accuracy: ", (count / len(x_test)) * 100)
    print(catagories)
    for i in range(num_classes):
        print(f"Acuracy of {i}: {catagories[i][0]/catagories[i][1]}")


# Saving/loading model
PATH = './torch_conv100.nn'
torch.save(net.state_dict(), PATH)
net = Net()
net.load_state_dict(torch.load(PATH))