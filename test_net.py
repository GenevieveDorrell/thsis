import time
import pandas as pd
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt



#global variables
num_classes = 4
input_length = x_test.shape[1]
test_data = TensorDataset(torch.tensor(x_test, dtype=torch.float), 
                          torch.tensor(y_test, dtype=torch.long))

torch.manual_seed(42)
batch_size = 100
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
#pritn sixe of x after every layer
# Building CNN model
# Building CNN model
class linear_2L_Net(nn.Module):
    def __init__(self):
        super(linear_2L_Net, self).__init__()
        self.batch_size = batch_size
        # input [100, 1, 99], 10 output nodes, 3x3 kernel        
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(425, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_classes) # 4 output nodes #soft max                              
    
    def forward(self, x):
        # Max pooling over a (2, 2) window    
        x = x.view(self.batch_size, -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) 
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))      
        x = F.softmax(self.fc5(x),dim=1)
        return x

class Net_1D_CNN_3L(nn.Module):
    def __init__(self):
        super(Net_1D_CNN_3L, self).__init__()
        self.batch_size = batch_size
        # input [100, 1, 99], 10 output nodes, 3x3 kernel
        self.norm1 = nn.BatchNorm1d(1)

        self.conv1 = nn.Conv1d(1, 20, 3)
        
        self.conv2 = nn.Conv1d(20, 40, 3)
        self.conv3 = nn.Conv1d(40, 80, 3)
        self.conv4 = nn.Conv1d(80, 160, 3)
        self.norm2 = nn.BatchNorm1d(40)
        #nn.ConvTranspose1d(in_channels, out_channels, kernel_size)
        
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(8000, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, num_classes)     # 4 output nodes #soft max
    
    def forward(self, x):
        # Max pooling over a (2, 2) window    
    
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = F.max_pool1d(F.relu(self.conv3(x)), 2)
        x = F.max_pool1d(F.relu(self.conv4(x)), 2)
        x = x.view(self.batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.softmax(self.fc6(x),dim=1)
        return x

# recalling the model
PATH = '100_epochs\\torch_L3_cnn.n'
net = Net_1D_CNN_3L()
net.load_state_dict(torch.load(PATH))

#get data from csvs
dif = "everything_3_"

x_test = pd.read_csv(dif + 'balanced-severity-x_test.csv',header=0)
y_test = pd.read_csv(dif + 'balanced-severity-y_test.csv',header=0)
x_test = pd.DataFrame.to_numpy(x_test)
y_test = pd.DataFrame.to_numpy(y_test.iloc[:,0])
print(x_test.shape)


# Testing model with test dataset
y_true = []
y_pred = []
with torch.no_grad():
    count = 0
    catagories = [[0,0],[0,0],[0,0],[0,0]]
    for i, (test, labels) in enumerate(test_loader):
        if test.shape[0] == batch_size:
            test = test.view([batch_size, 1, input_length])

            test_out = net(test)
            pred = torch.max(test_out, 1)[1]
            for j, item in enumerate(pred):
                y_true.append(labels[j])
                y_pred.append(item)
                count += int(item == labels[j])
                catagories[item][0] += int(item == labels[j])
                catagories[labels[j]][1] += 1
                


    print("Accuracy: ", (count / len(x_test)) * 100)
    print(catagories)
    for i in range(num_classes):
        print(f"Acuracy of {i}: {catagories[i][0]/catagories[i][1]}")
    
# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_true, y_pred))
# Accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_true, y_pred))
# Recall
from sklearn.metrics import recall_score
print(recall_score(y_true, y_pred, average=None))
# Precision
from sklearn.metrics import precision_score
print(precision_score(y_true, y_pred, average=None))



#get data from csvs
dif = "fire_2017_L3"

x_test = pd.read_csv(dif + 'balanced-severity-x_training.csv',header=0)
y_test = pd.read_csv(dif + 'balanced-severity-y_training.csv',header=0)
x_test = pd.DataFrame.to_numpy(x_test)
y_test = pd.DataFrame.to_numpy(y_test.iloc[:,0])
print(x_test.shape)

#global variables
num_classes = 4
input_length = x_test.shape[1]
test_data = TensorDataset(torch.tensor(x_test, dtype=torch.float), 
                          torch.tensor(y_test, dtype=torch.long))

torch.manual_seed(42)
batch_size = 100
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Testing model with test dataset

y_true = []
y_pred = []
with torch.no_grad():
    count = 0
    catagories = [[0,0],[0,0],[0,0],[0,0]]
    for i, (test, labels) in enumerate(test_loader):
        if test.shape[0] == batch_size:
            test = test.view([batch_size, 1, input_length])

            test_out = net(test)
            pred = torch.max(test_out, 1)[1]
            for j, item in enumerate(pred):
                y_true.append(labels[j])
                y_pred.append(item)
                count += int(item == labels[j])
                catagories[item][0] += int(item == labels[j])
                catagories[labels[j]][1] += 1
                


    print("Accuracy: ", (count / len(x_test)) * 100)
    print(catagories)
    for i in range(num_classes):
        print(f"Acuracy of {i}: {catagories[i][0]/catagories[i][1]}")
    
# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_true, y_pred))
# Accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_true, y_pred))
# Recall
from sklearn.metrics import recall_score
print(recall_score(y_true, y_pred, average=None))
# Precision
from sklearn.metrics import precision_score
print(precision_score(y_true, y_pred, average=None))



