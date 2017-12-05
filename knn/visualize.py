# -*- coding: utf-8 -*-
from __future__ import division
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_csv("all.csv")
#data = data.drop(['Mode','Vertical_symmetry','Horizontal_symmetry', 'Minimal_peak'], axis=1)
### getting Xs and Ys #####
X = data.iloc[0:data.shape[0], 1:data.shape[1]]
y = data.iloc[0:data.shape[0], 0]
y = y.astype('category')

### NORMALIZATION #######
X_scaled = (X-X.min())/(X.max()-X.min())

### SPLITTING INTO TRAIN AND TEST DATA
#X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=34)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=28)

####### KNN
knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train, y_train) # Fit the model using X as training data and y as target values
preds = knn.predict(X_test)	# Predict the class labels for the provided data

### accuracy 
acc = knn.score(X_test, y_test)

## Getting TOP results
#probabilities by classes, classes in lexicographic order.
probs = knn.predict_proba(X_test)
classes = np.unique(y)

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

# Очень простая но эффективная визуализация
# x, y - названия столбцов из датафрейма
betula_data = data.loc[(data['Type'].isin(['betula_pendula','betula_pubescens']))]

populus_data = data.loc[(data['Type'].isin(['populus_nigra','populus_tremula']))]


#sns.lmplot(x="Circularity", y="Eccentricity", data=data, fit_reg=False, hue='Type', legend=True)
sns.lmplot(x="Solidity", y="Maximal_distance", data=betula_data, fit_reg=False, hue='Type', legend=True)
sns.lmplot(x="Extent", y="Solidity", data=populus_data, fit_reg=False, hue='Type', legend=True)

#CONFUSION MATRIX
cm = confusion_matrix(y_test, preds)
classes = sorted(list(set(y)))
#print classes

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.BuPu):

#   #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plt.figure()
plot_confusion_matrix(cm, classes)
plt.show()
print acc
print accuracy