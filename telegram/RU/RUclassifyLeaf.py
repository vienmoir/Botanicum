# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:09:04 2017

@author: Acer
"""
import pickle
import numpy as np
import pandas as pd

def classify(data):
    data = data.drop(['Mode','Vertical_symmetry','Horizontal_symmetry', 'Minimal_peak'], axis=1)
    mms = pd.read_pickle('RUmms.p',compression='infer')
    example = data.iloc[0:1,0:data.shape[1]]
    mms = pd.concat([mms,example],ignore_index=True)
    example_scaled = ((mms-mms.min())/(mms.max()-mms.min())).loc[2:2,]

    ### загрузка модели
    filename = 'RUmodel.sav'
    loaded_model = pickle.load(open(filename, 'rb'))    
    
    ### загрузка названий классов
    filehandler = open('RUclasses.obj', 'r') 
    classes = pickle.load(filehandler)
    
    ### TOP-3 (TOP-2 ИЛИ TOP-1)
    prob = loaded_model.predict_proba(example_scaled)
    ind = np.argpartition(prob[0], -3)[-3:]
    ind = ind[np.argsort(prob[0][ind])]
    ind = [item for item in ind if prob[0][item] > 0]
    result2 = 0
    result3 = 0
    result1 = classes[ind[0]]
    if len(ind) > 1:
        result2 = classes[ind[1]]
        if len(ind) > 2:
            result3 = classes[ind[2]]
    return result1, result2, result3