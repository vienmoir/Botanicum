# -*- coding: utf-8 -*-
from __future__ import division
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("nastya.csv")
### getting Xs and Ys #####
X = data.iloc[0:data.shape[0], 1:data.shape[1]]
y = data.iloc[0:data.shape[0], 0]
y = y.astype('category')

### NORMALIZATION #######
X_scaled = (X-X.min())/(X.max()-X.min())

### SPLITTING INTO TRAIN AND TEST DATA
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

####### KNN
knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train, y_train) # Fit the model using X as training data and y as target values
preds = knn.predict(X_test)	# Predict the class labels for the provided data

### accuracy 
acc = knn.score(X_test, y_test)
