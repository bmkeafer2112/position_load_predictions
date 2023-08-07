# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:05:48 2023

@author: bmkea
"""

import pickle 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class preprocess:
    def __init__(self):
        self.init = 'hello'
                
    def preprocess(self, filepath):
        #Read File
        position_df = pd.read_csv(filepath)
        #Remove movements (keep stationary positions)
        position_df = position_df[position_df['PATH'].str.len() < 5]
        #Remove timestamp and index value
        X = position_df.iloc[:,9:-1]
        y = position_df.iloc[:,-1:]        
        #Standard Scaler
        scaler = StandardScaler()
        #Get Column names for after transformation        
        col_names = X.columns
        col_names = col_names.tolist()
        # transform data
        scaled = pd.DataFrame(scaler.fit_transform(X), columns = col_names)    
        #Remove new rows with position lags (we don't want to show the model previous categorical position)        
        X = scaled
        return X, y
    
preprocess = preprocess()
X, y = preprocess.preprocess(r'C:\Users\bmkea\Documents\Denso_Test_cell\Python Scripts\Position_Prediction\denso_04_Sunday.csv')




#Read in Best Classification Model
# load the model from disk
loaded_model = pickle.load(open('best_model.sav', 'rb'))
result = loaded_model.score(X, y)
print(result)

