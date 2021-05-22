import time
import pandas as pd
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
"""
5 epochs = 37.5
"""
#get data from csvs
dif = "everything_2_"
x_train = pd.read_csv(dif + 'balanced-severity-x_training.csv',header=0)
y_train = pd.read_csv(dif + 'balanced-severity-y_training.csv',header=0)
x_train = pd.DataFrame.to_numpy(x_train)
y_train = pd.DataFrame.to_numpy(y_train.iloc[:,0])


x_test = pd.read_csv(dif + 'balanced-severity-x_test.csv',header=0)
y_test = pd.read_csv(dif + 'balanced-severity-y_test.csv',header=0)
x_test = pd.DataFrame.to_numpy(x_test)
y_test = pd.DataFrame.to_numpy(y_test.iloc[:,0])
print(x_test.shape)

#global variables
num_classes = 4
input_length = 16
train_data = TensorDataset(torch.tensor(x_train, dtype=torch.float), 
                           torch.tensor(y_train, dtype=torch.long))
test_data = TensorDataset(torch.tensor(x_test, dtype=torch.float), 
                          torch.tensor(y_test, dtype=torch.long))

torch.manual_seed(42)
batch_size = 100
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Building CNN model
# Building CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.batch_size = batch_size
        # input [100, 1, 99], 10 output nodes, 3x3 kernel
        
        self.conv1 = nn.Conv2d(1, 20, 3)
        self.norm1 = nn.BatchNorm1d(20)
        self.Tran = nn.ConvTranspose2d(20, 40, 3)
        self.conv2 = nn.Conv2d(40, 80, 3)
        self.conv3 = nn.Conv2d(80, 160, 3)
        self.norm2 = nn.BatchNorm1d(40)
        #nn.ConvTranspose1d(in_channels, out_channels, kernel_size)
        
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(39, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_classes)     # 4 output nodes
    
    def forward(self, x):
        # Max pooling over a (2, 2) window        
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        #x = self.Tran(x)
        #x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        #x = x.view(self.batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Initializing model, loss function, and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training model on data
start = time.time()
losses = []
validation_losses = []
for epoch in range(20):
    print("\nEpoch", epoch+1)
    running_loss = 0
    net.train()
    for i, (feats, labels) in enumerate(train_loader):
        if feats.shape[0] == batch_size:
            feats = feats.view([batch_size,1,5,5*input_length])
            optimizer.zero_grad()
            output = net(feats)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i * batch_size % 10000 == 0:
                print(i * batch_size, "/", len(train_loader) * batch_size, 
                    "\tLoss:", running_loss/(i*batch_size+1))
    epoch_loss = running_loss / len(train_loader.dataset)
    losses.append(epoch_loss)
    net.eval()
    valid_loss = 0.0
    for i, (feats, labels) in enumerate(test_loader):
        feats = feats.view([batch_size,1,input_length])
        output = net(feats)
        loss = criterion(output,labels)
    
    valid_loss += loss.item()
    validation_losses.append(valid_loss/len(test_loader.dataset))


print(round((time.time() - start), 2), "s")


plt.figure(figsize=(10,5))
plt.title("Validation loss and Trainign loss over 2D CNN Epochs")
plt.plot(np.array(losses),label="Training loss")
plt.plot(np.array(validation_losses),label="Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("2Dloss.png")

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