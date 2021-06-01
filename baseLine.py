from numpy.lib.function_base import average
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


#get data from csvs
dif = "everything_3_"
x_train = pd.read_csv(dif + 'balanced-severity-x_training.csv',header=0)
y_train = pd.read_csv(dif + 'balanced-severity-y_training.csv',header=0)
x_train = pd.DataFrame.to_numpy(x_train)
y_train = pd.DataFrame.to_numpy(y_train.iloc[:,0])


x_test = pd.read_csv(dif + 'balanced-severity-x_test.csv',header=0)
y_test = pd.read_csv(dif + 'balanced-severity-y_test.csv',header=0)
x_test = pd.DataFrame.to_numpy(x_test)
y_test = pd.DataFrame.to_numpy(y_test.iloc[:,0])

print(y_train.shape)
print(y_test.shape)
print(x_train.shape)
print(x_test.shape)
input_length = x_train.shape[1]
num_classes = 4

"""
#Logistic Regression
regressor = LogisticRegression(max_iter=1000000000)
regressor.fit(x_train, y_train)

rg_score = round(regressor.score(x_test, y_test) * 100, 2)
print(f"Logistic Regression score: {rg_score}%")

#Decisison Tree
dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)

dt_score = round(dtree.score(x_test, y_test) * 100, 2)
print(f"Decision Tree score: {dt_score}%")
"""
#Random Forest
rf_class = RandomForestClassifier()
rf_class.fit(x_train, y_train)

rf_score = round(rf_class.score(x_test, y_test) * 100, 2)
print(f"Random Forest score: {rf_score}%")
importance = rf_class.feature_importances_
# summarize feature importance
num_features = 17
averaged_importance = []
for i in range(num_features):
    averaged_importance.append(0)
    for j in range(i,len(importance),num_features):
        averaged_importance[i] += importance[j]
    averaged_importance[i] /= 25
# plot feature importance
feature_names = ["Land Sat Channel 1","Land Sat Channel 2","Land Sat Channel 3","Land Sat Channel 4","Land Sat Channel 5",
                    "Land Sat Channel 6", "Land Sat Channel 7", "Land Sat Channel 8", "Tree Cover", "Loss Year", "Slope", 
                    "Elevation", "Aspect", "Stream Distance", "Stand Age", "Stand Age 80", "Stand Age 200"]

pix_importance = []

for i in range(25):
    pix = 0
    for j in range(17):
        pix += importance[i*17 + j]
    pix_importance.append(pix/17)

pix_label = []
for i in range(5):
    for j in range(5):
        pix_label.append(f"({i+1}, {j+1})")

pix_label[13] = "Center Pixel"



plt.figure(figsize=(20,10))
plt.bar(feature_names, averaged_importance,color = 'mediumspringgreen')
plt.title("Discriptive Feature Importance for RF",fontsize=44)
plt.legend(fontsize=36,frameon=False)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().tick_params(labelsize=24,width=2)
plt.xticks(rotation=45,horizontalalignment='right')
plt.savefig("RF_feature_importance.png",bbox_inches="tight")

plt.figure(figsize=(20,10))
plt.bar([x for x in range(len(importance))], importance,color = 'deepskyblue')
plt.title("All Feature Importance for RF",fontsize=44)
plt.legend(fontsize=36,frameon=False)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().tick_params(labelsize=24,width=2)
plt.xticks(rotation=45,horizontalalignment='right')
plt.savefig("RF_all_feature_importance.png",bbox_inches="tight")

print(len(pix_importance))
plt.figure(figsize=(20,10))
plt.bar(pix_label, pix_importance,color = 'green')
plt.title("Pixel Importance for RF",fontsize=44)
plt.legend(fontsize=36,frameon=False)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().tick_params(labelsize=24,width=2)
plt.xticks(rotation=45,horizontalalignment='right')
plt.savefig("RF_pixel_importance.png",bbox_inches="tight")

y_pred = rf_class.predict(x_test)
y_true = y_test
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
