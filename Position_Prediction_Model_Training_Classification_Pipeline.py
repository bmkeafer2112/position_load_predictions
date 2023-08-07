# -*- coding: utf-8 -*-
"""
Created on Tue May  9 13:13:59 2023

@author: bmkea
"""
#Basic DS Libraries
import pandas as pd
import numpy as np
import pickle
#import matplotlib as plt


#Reporting/Evaluation
from sklearn.metrics import *

#Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


#For Processing/Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler


#Read position CSV
position_df = pd.read_csv(r'C:\Users\bmkea\Documents\Denso_Test_cell\Python Scripts\Position_Prediction\denso_01_Sunday.csv')
#df['A'].str.len() == 10)
position_df = position_df[position_df['PATH'].str.len() < 5]

#Remove timestamp and index value
X = position_df.iloc[:,9:-1]
y = position_df.iloc[:,-1:]

#Standard Scaler
scaler = StandardScaler()

col_names = X.columns
col_names = col_names.tolist()
# transform data
scaled = pd.DataFrame(scaler.fit_transform(X), columns = col_names)


#Create Lag values
#Iterate Over Columns
# =============================================================================
# for col in scaled.columns:
#     #Iterate over 10 lags (~ 1 second)
#     i = 1
#     while i < 11:
#         scaled[col +str('_lag_') + str(i)] = scaled[col].shift(i)
#         i += 1
# =============================================================================

#Remove new rows with position lags (we don't want to show the model previous categorical position)        
X = scaled

#Drop NANS and missingness
position_df = position_df.dropna()


#Split Dat for Train/Test, might try more gaps later
tss = TimeSeriesSplit(n_splits = 2, max_train_size = (round(0.66 * len(position_df))))

for train_index, test_index in tss.split(X):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
#Create CV
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=21)

estimators = {}

##Initialze the estimators
estimators['rf'] = RandomForestClassifier(random_state=21)
estimators['svc'] = SVC(probability=True, random_state=21)
estimators['mlp'] = MLPClassifier(random_state=21)
estimators['dt'] = DecisionTreeClassifier(random_state=21)
estimators['knn'] = KNeighborsClassifier()
estimators['gboost'] = GradientBoostingClassifier(random_state=21)
estimators['gauss'] = GaussianProcessClassifier(random_state=21)
estimators['adaboost'] = AdaBoostClassifier(random_state=21)
estimators['logreg'] = LogisticRegression()

#Create List for estimators and parameters to use
#estimator_list = ['rf', 'svc', 'mlp', 'dt', 'knn', 'gboost', 'adaboost', 'logreg']
estimator_list = ['mlp']
params = {}


#Initialize the hyperparameters for each classifiers dictionary
#RandomForestClassifier
rf_params = {}
#rf_params['n_estimators'] = [10, 50, 100, 250]
#rf_params['max_depth'] = [None, 5, 10, 20]
#rf_params['class_weight'] = [None, "balanced", "balanced_subsample"]
params['rf_params'] = rf_params


#SVC
svc_params = {}
params['svc_params'] = svc_params

#MLP (Neural Network)
mlp_params = {}
mlp_params['hidden_layer_sizes'] = [(50,50,50)]#, (50,100,50), (100,)]
mlp_params['activation'] = ['tanh', 'relu']
mlp_params['solver'] =  ['sgd', 'adam']
mlp_params['alpha'] = [0.0001, 0.05]
mlp_params['learning_rate'] =  ['constant','adaptive']
params['mlp_params'] = mlp_params

#Decision Tree
dt_params = {}
params['dt_params'] = mlp_params

#KNN
knn_params = {}
params['knn_params'] = knn_params

#Gradiant Boost
gboost_params = {}
params['gboost_params'] = gboost_params

#Gaussian
gauss_params = {}
params['gauss_params'] = gauss_params

#AdaBoost
adaboost_params = {}
params['adaboost_params'] = adaboost_params

#Gaussian
logreg_params = {}
params['logreg_params'] = logreg_params


# =============================================================================
# #Create Pipeline
# pipeline = Pipeline([('classifier', clf1)])
# params = [param1, param2]
# =============================================================================
for e in estimator_list: 
    key = e + '_params'
    # Train the grid search model
    gs = GridSearchCV(estimator=estimators[e], param_grid=params[key], cv=3, n_jobs=-1, scoring='accuracy')
    gs_result = gs.fit(X_train, y_train.values.ravel())
    filepath = r'C:\Users\bmkea\Documents\Denso_Test_cell\Python Scripts\Position_Prediction\best_model.sav'
    pickle.dump(gs, open(filepath, 'wb'))
    
    # Best performing model and its corresponding hyperparameters
    print(gs_result.best_params_)
    
    # ROC-AUC score for the best model
    #print(gs_result.best_score_)
    
    # Test data performance
    #print("Test Precision:",precision_score(gs_result.predict(X_test), y_test, average=None))
    #print("Test Recall:",recall_score(gs_result.predict(X_test), y_test, average=None))
    
    pred  = gs.predict(X_test)
    print(e)
    print(classification_report(y_test,pred))
    
    #best_features = gs_result.best_estimator_.feature_importances_
    #print(best_features)







# =============================================================================
# #Create Basic Model
# model_RF= RandomForestClassifier(n_estimators=100)
# model_GB = GradientBoostingClassifier() 
# 
# 
# # =============================================================================
# # #Random Forest Classifier
# # model_RF.fit(X_train, y_train)
# # pred_RF = model_RF.predict(X_test)
# # print(classification_report(y_test,pred_RF))
# # =============================================================================
# 
# 
# #Gradient Boosting Classifier
# model_GB.fit(X_train, y_train)
# pred_GB = model_GB.predict(X_test)
# print(classification_report(y_test,pred_GB))
# 
# =============================================================================



