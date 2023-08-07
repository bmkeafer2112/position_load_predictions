# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 09:08:57 2023

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
import psycopg2 as pg

class classification_model:
    def __init__(self):
        self.conn = pg.connect("dbname=postgres user=postgres password=admin host=192.168.187.7 port=5432")
        self.cur = self.conn.cursor()
        
    def LabelMotionPath(self, robot):
        whichRobot = robot
		#self.cur.execute("""SELECT * FROM positiondata_allrobots_programdt13ah3 WHERE "Robot_Name""")
        self.cur.execute("""SELECT * FROM positiondata_allrobots_programdt13ah3_4_hours WHERE "Robot_Name" = %s ORDER BY "Time_Stamp" """, [whichRobot])
		#self.cur.execute( """SELECT * FROM positiondata_allrobots_programdt13ah3 WHERE "Robot_Name" = %s ORDER BY "Time_Stamp""", [whichRobot])
        #self.cur.execute("""SELECT * FROM everything WHERE "Robot_Name" = %s AND "Time_Stamp" > Now() - interval '10 minutes' ORDER BY "Time_Stamp""", [whichRobot, limit])
        colnames = [desc[0] for desc in self.cur.description]
        records = self.cur.fetchall()
        dataFrameTP = pd.DataFrame(data = records, columns = colnames)
        dataFrameControllerTP = pd.read_csv(os.getcwd()+"\TP_Data.csv")
        dataFrameTP["Teach_Point"] = 999
        some_Number = 0

        for some_Number in range(400):
            some_Number_X = dataFrameControllerTP["TCP_Current_x"][some_Number]
            some_Number_Y = dataFrameControllerTP["TCP_Current_y"][some_Number]
            some_Number_Z = dataFrameControllerTP["TCP_Current_z"][some_Number]
            some_Number_RX = dataFrameControllerTP["TCP_Current_rx"][some_Number]
            some_Number_RY = dataFrameControllerTP["TCP_Current_ry"][some_Number]
            some_Number_RZ = dataFrameControllerTP["TCP_Current_rz"][some_Number]
	
            for i in range(len(dataFrameTP)):
                if math.isclose(dataFrameTP["TCP_Current_x"][i], some_Number_X, rel_tol = 0.1) & math.isclose(dataFrameTP["TCP_Current_y"][i], some_Number_Y, rel_tol = 0.1) & math.isclose(dataFrameTP["TCP_Current_z"][i], some_Number_Z, rel_tol = 0.1) & math.isclose(dataFrameTP["TCP_Current_rx"][i], some_Number_RX, rel_tol = 0.1) & math.isclose(dataFrameTP["TCP_Current_ry"][i], some_Number_RY, rel_tol = 0.1) & math.isclose(dataFrameTP["TCP_Current_rz"][i], some_Number_RZ, rel_tol = 0.1):
                    dataFrameTP["Teach_Point"][i] = some_Number

        dataFramePath = dataFrameTP
        PointFrom = 999
        PointTo = 999
        dataFramePath["PointFrom"] = "PointFrom"
        dataFramePath["PointTo"] = "PointTo"
        dataFramePath["PATH"] = "PointFrom_PointTo_Iteration"

        for j in range(len(dataFramePath)):
            if dataFramePath["Teach_Point"][j] != 999:
                PointFrom = dataFramePath["Teach_Point"][j]
            dataFramePath["PointFrom"][j] = PointFrom

        for k in range(len(dataFramePath)-1, 1, -1):
            if dataFramePath["Teach_Point"][k] != 999:
                PointTo = dataFramePath["Teach_Point"][k]
            dataFramePath["PointTo"][k] = PointTo

        for l in range(len(dataFramePath)):
            if dataFramePath["PointFrom"][l] == dataFramePath["PointTo"][l]:
                dataFramePath["PATH"][l] = ("P"+str(dataFramePath["PointFrom"][l]))
            else:
                dataFramePath["PATH"][l] = ("P"+str(dataFramePath["PointFrom"][l])+"->P"+str(dataFramePath["PointTo"][l]))

        dataFramePath = dataFramePath.drop(["Teach_Point", "PointFrom", "PointTo"], axis=1)
        dataFramePath = dataFramePath[~dataFramePath["PATH"].str.contains("PPointTo", na=False)]
        dataFramePath = dataFramePath[~dataFramePath["PATH"].str.contains("P999", na=False)]
        result = dataFramePath
        return result
    
    def preprocess_robot(self):
        self.cur.execute("""SELECT * FROM positiondata_allrobots_programdt13ah3_4_hours ORDER BY "Time_Stamp" """)
        colnames = [desc[0] for desc in self.cur.description]
        records = self.cur.fetchall()
        df = pd.DataFrame(data = records, columns = colnames)
        X = df.iloc[:,8:]
        y = df.iloc[:,1:2]
        return X, y
        
        
        
    def preprocess_position(self, filepath):
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
    
    def fit_robot(self):
        X, y = self.preprocess_robot()
        #Split Dat for Train/Test, might try more gaps later
        tss = TimeSeriesSplit(n_splits = 2, max_train_size = (round(0.66 * len(X))))
        
        for train_index, test_index in tss.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
        #Create CV
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=21)
        
        estimators = {}
        
        ##Initialze the estimators (this can be used to compare multiple models if needed)
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
        
        #MLP (Neural Network)
        mlp_params = {}
        mlp_params['hidden_layer_sizes'] = [(50,50,50)]#, (50,100,50), (100,)]
        mlp_params['activation'] = ['tanh', 'relu']
        mlp_params['solver'] =  ['sgd', 'adam']
        mlp_params['alpha'] = [0.0001, 0.05]
        mlp_params['learning_rate'] =  ['constant','adaptive']
        params['mlp_params'] = mlp_params

        
        for e in estimator_list: 
            key = e + '_params'
            # Train the grid search model
            gs = GridSearchCV(estimator=estimators[e], param_grid=params[key], cv=3, n_jobs=-1, scoring='accuracy')
            gs_result = gs.fit(X_train, y_train.values.ravel())
            filepath = r'C:\Users\bmkea\Documents\Denso_Test_cell\Python Scripts\Position_Prediction\best__robot_model.sav'
            pickle.dump(gs, open(filepath, 'wb'))
            
            # Best performing model and its corresponding hyperparameters
            results = gs_result.best_params_           
            pred  = gs.predict(X_test)
            report = classification_report(y_test,pred)
            return results, report, e
    
    def fit_position(self, filepath):
        X, y = self.preprocess_position(filepath)
        #Split Dat for Train/Test, might try more gaps later
        tss = TimeSeriesSplit(n_splits = 2, max_train_size = (round(0.66 * len(X))))
        
        for train_index, test_index in tss.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
        #Create CV
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=21)
        
        estimators = {}
        
        ##Initialze the estimators (this can be used to compare multiple models if needed)
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
        
        #MLP (Neural Network)
        mlp_params = {}
        mlp_params['hidden_layer_sizes'] = [(50,50,50)]#, (50,100,50), (100,)]
        mlp_params['activation'] = ['tanh', 'relu']
        mlp_params['solver'] =  ['sgd', 'adam']
        mlp_params['alpha'] = [0.0001, 0.05]
        mlp_params['learning_rate'] =  ['constant','adaptive']
        params['mlp_params'] = mlp_params

        
        for e in estimator_list: 
            key = e + '_params'
            # Train the grid search model
            gs = GridSearchCV(estimator=estimators[e], param_grid=params[key], cv=3, n_jobs=-1, scoring='accuracy')
            gs_result = gs.fit(X_train, y_train.values.ravel())
            filepath = r'C:\Users\bmkea\Documents\Denso_Test_cell\Python Scripts\Position_Prediction\best__position_model.sav'
            pickle.dump(gs, open(filepath, 'wb'))
            
            # Best performing model and its corresponding hyperparameters
            results = gs_result.best_params_           
            pred  = gs.predict(X_test)
            report = classification_report(y_test,pred)
            return results, report, e
    
    def predict_robot(self):
        loaded_model = pickle.load(open('best_model.sav', 'rb'))
        X, y = self.preprocess_robot()        
        score = loaded_model.score(X, y)
        result = loaded_model.predict(X)       
        return score, result
        
    def predict_position(self, pickled_model, data_to_predict):
        loaded_model = pickle.load(open('best_model.sav', 'rb'))
        X, y = self.preprocess_position(data_to_predict)        
        score = loaded_model.score(X, y)
        result = loaded_model.predict(X)       
        return score, result
    
model = classification_model()
#results, report, e = model.fit_position(r'C:\Users\bmkea\Documents\Denso_Test_cell\Python Scripts\Position_Prediction\denso_01_Sunday.csv')
results, report, e = model.fit_robot()
score, result = model.predict_robot()      