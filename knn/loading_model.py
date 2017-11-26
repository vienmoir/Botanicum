# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:09:04 2017

@author: Acer
"""
import pickle
import numpy as np
import pandas as pd

mms = pd.read_pickle('mms.p',compression='infer')
### здесь должен быть массив признаков, тип - dataframe (1,23)
example = data.iloc[0:1,0:data.shape[1]]
mms = pd.concat([mms,example],ignore_index=True)
example_scaled = ((mms-mms.min())/(mms.max()-mms.min())).loc[2:2,]

### загрузка модели
filename = 'knn_model_3.sav'
loaded_model = pickle.load(open(filename, 'rb'))
# если всё-таки 1 результат:
#result = loaded_model.predict(example)    

### загрузка названий классов
filehandler = open('classes.obj', 'r') 
classes = pickle.load(filehandler)

### TOP-3 (TOP-2 ИЛИ TOP-1)
prob = loaded_model.predict_proba(example_scaled)
ind = np.argpartition(prob[0], -3)[-3:]
ind = ind[np.argsort(prob[0][ind])]
ind = [item for item in ind if prob[0][item] > 0]
top1 = classes[ind[0]]
if len(ind) > 1:
    top2 = classes[ind[1]]
    if len(ind) > 2:
        top3 = classes[ind[2]]
