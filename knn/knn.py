# -*- coding: utf-8 -*-
from __future__ import division
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv("all.csv")
### getting Xs and Ys #####
X = data.iloc[0:data.shape[0], 1:data.shape[1]]
y = data.iloc[0:data.shape[0], 0]
y = y.astype('category')

### NORMALIZATION #######
X_scaled = (X-X.min())/(X.max()-X.min())

### SPLITTING INTO TRAIN AND TEST DATA
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

####### KNN
knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train, y_train) # Fit the model using X as training data and y as target values
preds = knn.predict(X_test)	# Predict the class labels for the provided data

### accuracy 
acc = knn.score(X_test, y_test)

## Getting TOP results
#probabilities by classes, classes in lexicographic order.
probs = knn.predict_proba(X_test)
classes = np.unique(y)

# рабочий код 
ind = np.argpartition(probs[3], -3)[-3:]
probs[3][ind]
ind = [item for item in ind if probs[3][item] > 0]

if len(ind) > 3:
    ind = ind[len(ind)-3:len(ind)]
results = classes[ind]
# конец рабочего кода

k = 0
mistakes = []
# accuracy 
for i in range(1,len(probs)):
    ind = np.argpartition(probs[i], -3)[-3:]
    ind = [item for item in ind if probs[i][item] > 0]
    if len(ind) > 3:
        ind = ind[len(ind)-3:len(ind)]
    if y_test.iloc[i] in classes[ind]:
        k = k+1
    else:
        mistakes.append(i)

accuracy = k/len(probs)

# СОХРАНЯЕМ МИНИМУМЫ И МАКСИМУМЫ
mins = X.min()
maxs = X.max()
mms = pd.concat([maxs, mins], axis=1).T

# сохранение и загрузка для проверки
mms.to_pickle('mms.p')
mms = pd.read_pickle('mms.p',compression='infer')