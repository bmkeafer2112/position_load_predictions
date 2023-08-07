# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 08:28:03 2023

@author: bmkea
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import *

#Read position CSV
load_df = pd.read_csv(r'C:\Users\bmkea\Documents\Denso_Test_cell\Denso_Test_Cell_Data\load_dataset.csv')


#Get X and y values
X = load_df.iloc[:,8:-4]
y = load_df.iloc[:,-3:-2]

#StandardScaler
scaler = StandardScaler()

col_names = X.columns
col_names = col_names.tolist()
# transform data
scaled = pd.DataFrame(scaler.fit_transform(X), columns = col_names)
X = scaled

tss = TimeSeriesSplit(n_splits = 2, max_train_size = (round(0.66 * len(load_df))))

for train_index, test_index in tss.split(X, y):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
#Create CV
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=21)


#MLP (Neural Network)
# =============================================================================
# mlp = MLPClassifier()
# mlp_params = {}
# mlp_params['hidden_layer_sizes'] = [(50,50,50)]#, (50,100,50), (100,)]
# mlp_params['activation'] = ['tanh', 'relu']
# mlp_params['solver'] =  ['sgd', 'adam']
# mlp_params['alpha'] = [0.0001, 0.05]
# mlp_params['learning_rate'] =  ['constant','adaptive']
# =============================================================================

mlp = MLPClassifier()
mlp_params = {}
mlp_params['hidden_layer_sizes'] = [(50,50,50)]#, (50,100,50), (100,)]
mlp_params['activation'] = ['relu']
mlp_params['solver'] =  ['adam']
mlp_params['alpha'] = [0.0001]
mlp_params['learning_rate'] =  ['adaptive']

# Train the grid search model
gs = GridSearchCV(estimator=mlp, param_grid=mlp_params, cv=3, n_jobs=-1, scoring='accuracy')
gs_result = gs.fit(X_train, y_train.values.ravel())

# Best performing model and its corresponding hyperparameters
print(gs_result.best_params_)

# ROC-AUC score for the best model
#print(gs_result.best_score_)

# Test data performance
#print("Test Precision:",precision_score(gs_result.predict(X_test), y_test, average=None))
#print("Test Recall:",recall_score(gs_result.predict(X_test), y_test, average=None))

pred  = gs.predict(X_test)
print(classification_report(y_test,pred))

#best_features = gs_result.best_estimator_.feature_importances_
#print(best_features)


