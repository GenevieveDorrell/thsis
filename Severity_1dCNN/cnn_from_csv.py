import time
import pandas as pd
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

"""
epochs: 10
Learning rate: 0.001
Accuracy:  40.5
Acuracy of class 0: 0.37333333333333335
Acuracy of class 1: 0.4066666666666667
Acuracy of class 2: 0.5433333333333333
Acuracy of class 3: 0.2966666666666667
"""

#get data from csvs
x_train = pd.read_csv('csvs/balanced-severity-x_training.csv',header=0)
y_train = pd.read_csv('csvs/balanced-severity-y_training.csv',header=0)
x_train = pd.DataFrame.to_numpy(x_train)
y_train = pd.DataFrame.to_numpy(y_train.iloc[:,0])


x_test = pd.read_csv('csvs/balanced-severity-x_test.csv',header=0)
y_test = pd.read_csv('csvs/balanced-severity-y_test.csv',header=0)
x_test = pd.DataFrame.to_numpy(x_test)
y_test = pd.DataFrame.to_numpy(y_test.iloc[:,0])
print(x_test.shape)

#global variables
num_classes = 4
input_length = x_test.shape[1]
train_data = TensorDataset(torch.tensor(x_train, dtype=torch.float), 
                           torch.tensor(y_train, dtype=torch.long))
test_data = TensorDataset(torch.tensor(x_test, dtype=torch.float), 
                          torch.tensor(y_test, dtype=torch.long))

torch.manual_seed(42)
batch_size = 100
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Building CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.batch_size = batch_size
        # input [100, 1, 99], 10 output nodes, 3x3 kernel
        self.conv1 = nn.Conv1d(1, 20, 3)
        self.conv2 = nn.Conv1d(20, 40, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(2680, self.batch_size)
        self.fc2 = nn.Linear(self.batch_size, 128)
        self.fc3 = nn.Linear(128, num_classes)     # 4 output nodes
    
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = x.view(self.batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initializing model, loss function, and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training model on data
start = time.time()
for epoch in range(10):
    print("\nEpoch", epoch+1)
    running_loss = 0
    for i, (feats, labels) in enumerate(train_loader):
        if feats.shape[0] == batch_size:
            feats = feats.view([batch_size,1,input_length])

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
    catagories = [[0,0],[0,0],[0,0],[0,0]]
    for i, (test, labels) in enumerate(test_loader):
        if test.shape[0] == batch_size:
            test = test.view([batch_size, 1, input_length])

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