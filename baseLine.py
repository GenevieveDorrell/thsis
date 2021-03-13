import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from geoProc_latlon import get_numpy_from_tiffs
"""
Logistic Regression score: 25.0%
Decision Tree score: 26.08%
Random Forest score: 27.25%
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


y_train, y_test, x_train, x_test = get_numpy_from_tiffs([sev1_fp, sev2_fp, sev3_fp, sev4_fp, sev5_fp],[rgb1_fp, rgb2_fp, rgb3_fp, rgb4_fp, rgb5_fp], [treecover, lossyear, elevation, slope])
#y_train, y_test, x_train, x_test = get_numpy_from_tiffs([sev1_fp],[rgb1_fp], [treecover])#, slope])

print(y_train.shape)
print(y_test.shape)
print(x_train.shape)
print(x_test.shape)
input_length = x_train.shape[1]
num_classes = 4


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

#Random Forest
rf_class = RandomForestClassifier()
rf_class.fit(x_train, y_train)

rf_score = round(rf_class.score(x_test, y_test) * 100, 2)
print(f"Random Forest score: {rf_score}%")




