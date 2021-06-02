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
input_length = x_test.shape[1]
train_data = TensorDataset(torch.tensor(x_train, dtype=torch.float), 
                           torch.tensor(y_train, dtype=torch.long))
test_data = TensorDataset(torch.tensor(x_test, dtype=torch.float), 
                          torch.tensor(y_test, dtype=torch.long))

torch.manual_seed(42)
batch_size = 100
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
#pritn sixe of x after every layer
# Building CNN model
# Building CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.batch_size = batch_size
        # input [100, 1, 99], 10 output nodes, 3x3 kernel
        self.norm1 = nn.BatchNorm1d(1)

        self.conv1 = nn.Conv2d(1, 20, 3)
        
        self.conv2 = nn.Conv2d(20, 40, 3)
        self.conv3 = nn.Conv2d(40, 80, 3)
        self.norm2 = nn.BatchNorm1d(40)
        #nn.ConvTranspose1d(in_channels, out_channels, kernel_size)
        
        # an affine operation: y = Wx + b
        self.fc4 = nn.Linear(320, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, num_classes)     # 4 output nodes #soft max
    
    def forward(self, x):
        # Max pooling over a (2, 2) window    
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(self.batch_size, -1)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.softmax(self.fc6(x),dim=1)
        return x

# Initializing model, loss function, and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#42.85
# Training model on data
start = time.time()
losses = []
validation_losses = []
validation_accuracy = []
training_acuaracy = []
for epoch in range(100):
    print("\nEpoch", epoch+1)
    running_loss = 0
    net.train()
    count = 0
    for i, (feats, labels) in enumerate(train_loader):
        if feats.shape[0] == batch_size:
            feats = feats.view([batch_size,1,25,17])
            optimizer.zero_grad()
            output = net(feats)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            #calculate of epoch training accuracy
            pred = torch.max(output, 1)[1]
            for j, item in enumerate(pred):
                count += int(item == labels[j])
            if i * batch_size % 10000 == 0:
                print(i * batch_size, "/", len(train_loader) * batch_size, 
                    "\tLoss:", running_loss/(i*batch_size+1))
    epoch_loss = running_loss / len(train_loader.dataset)
    losses.append(epoch_loss)
    training_acuaracy.append(count/len(train_loader.dataset))


    net.eval()
    valid_loss = 0.0
    count = 0
    for i, (feats, labels) in enumerate(test_loader):
        feats = feats.view([batch_size,1,25,17])
        output = net(feats)
        valid_loss += criterion(output,labels).item()
        pred = torch.max(output, 1)[1]
        for j, item in enumerate(pred):
            count += int(item == labels[j])
    validation_accuracy.append(count/len(test_loader.dataset))
    validation_losses.append(valid_loss/len(test_loader.dataset))

#add in accuracy test graph
print(round((time.time() - start), 2), "s")



plt.figure(figsize=(20,10))
plt.title("Validation and Trainign Loss of 2D CNN",fontsize=44)
plt.plot(np.array(losses),label="Training Loss", color = 'mediumspringgreen',linewidth=4)
plt.plot(np.array(validation_losses),label="Validation Loss", color = 'deepskyblue',linewidth=4)
plt.xlabel("Epochs",fontsize=30)
plt.ylabel("Loss",fontsize=30)
plt.legend(fontsize=36,frameon=False)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().tick_params(labelsize=24,width=2)
plt.savefig("1DCNN_loss.png")

plt.figure(figsize=(20,10))
plt.title("Validation and Trainign accuracy of 2D CNN",fontsize=44)
plt.plot(np.array(training_acuaracy),label="Training Accuracy", color = 'mediumspringgreen',linewidth=4)
plt.plot(np.array(validation_accuracy),label="Validation Accuracy", color = 'deepskyblue',linewidth=4)
plt.xlabel("Epochs",fontsize=30)
plt.ylabel("Accuracy",fontsize=30)
plt.legend(fontsize=36,frameon=False)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().tick_params(labelsize=24,width=2)
plt.savefig("1DCNN_accuracy.png")

# Testing model with test dataset
y_true = []
y_pred = []
with torch.no_grad():
    count = 0
    catagories = [[0,0],[0,0],[0,0],[0,0]]
    for i, (test, labels) in enumerate(test_loader):
        if test.shape[0] == batch_size:
            test = test.view([batch_size, 1,25,17])

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


# Saving/loading model
PATH = '100_epochs/torch_2D_cnn.n'
torch.save(net.state_dict(), PATH)
net = Net()
net.load_state_dict(torch.load(PATH))