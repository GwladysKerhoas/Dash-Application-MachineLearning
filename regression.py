# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from time import time
from sklearn.model_selection import cross_val_predict
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


# Regression class to gather all the classification algorithm
class regression:

    # Linear regression algorithm
    def linear_reg(target,predictor):
        
        X = pd.DataFrame(predictor)
        
        # Normalization of quantitative explanatory variables
        transformer = Normalizer().fit(X)
        transformer
        transformer.transform(X)
        
        # Target variable
        y = pd.DataFrame(target)
        
        # Split train-test
        X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), train_size=0.7, random_state=10)
        
        scorer = make_scorer(r2_score) #Type of score
        regressionLineaire = LinearRegression() # Model instanciation
        params = {'fit_intercept':[True, False], 'normalize':[True, False]} # Paramaters to test
        clf = GridSearchCV(regressionLineaire, param_grid=params, cv=5,scoring=scorer, n_jobs=5)
        clf.fit(X_train, y_train) # We apply on the training data
        
        best_param = clf.best_params_ # Better setting
        best_score = round(clf.best_score_,2) # Best score
        
        # Instantiation of the model with identified optimal parameters
        regressionLineaire = LinearRegression(fit_intercept=best_param["fit_intercept"], normalize=best_param["normalize"]) 

        start = time()
        
        # Predictions
        y_pred = cross_val_predict(regressionLineaire, X_test, y_test, cv=5)
        r2 = round(r2_score(y_test, y_pred),2)
        mse = round(mean_squared_error(y_test, y_pred),2)
        
        # Best parameter
        labels_values = []
        labels_keys = []
        for j in best_param.keys():
            labels_keys.append(j)
        
        for j in best_param.values():
            labels_values.append(j)
        
        tab_param =[]
        tab_param.append(labels_keys)
        tab_param.append(labels_values)
        
        fig_table = ff.create_table(tab_param, height_constant=60)
    
        end = time()
        time_execution = round(end-start,2)
        
        # Scatter plot
        fig = go.Figure()
        absi = list(range(len(y_test)))
        
        fig.add_trace(go.Scatter(x=absi, y=y_pred[np.argsort(y_test)], mode='markers',name='predictions')) #Prédictions
        fig.add_trace(go.Scatter(x=absi, y=np.sort(y_test), mode='lines',name='reality')) #Données réelles

        return best_param, best_score, r2, mse, time_execution, fig, fig_table
    
    # Support Vector Regression algorithm
    def svr(target,predictor):
        X = pd.DataFrame(predictor)
        
        # Normalization of quantitative explanatory variables
        transformer = Normalizer().fit(X)
        transformer
        transformer.transform(X)
        
        # Target variable
        y = pd.DataFrame(target)
        
        # Split train-test
        X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), train_size=0.7, random_state=10)
        
        scorer = make_scorer(r2_score) #Type of score
        modelsvr = SVR() # Instanciation du modèle
        params = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 
                  'gamma':['scale', 'auto'],
                  'C' : [1,5,10],
                  'degree' : [3,8],
                  'coef0' : [0.01,10,0.5]} # Paramaters to test
        clf = GridSearchCV(modelsvr, param_grid=params, cv=5,scoring=scorer, n_jobs=5)
        clf.fit(X_train, y_train) # We apply on the training data
                
        best_param = clf.best_params_ # Better setting
        best_score = round(clf.best_score_,2) # Best score
        
        # Instantiation of the model with identified optimal parameters
        modelsvr = SVR(C=best_param["C"], coef0=best_param["coef0"], degree=best_param["degree"], gamma=best_param["gamma"], kernel=best_param["kernel"]) 
     
        start = time()
        
        # Predictions
        y_pred = cross_val_predict(modelsvr, X_test, y_test, cv=5)
        r2 = round(r2_score(y_test, y_pred),2)
        mse = round(mean_squared_error(y_test, y_pred),2)
        
        # Best parameter
        labels_values = []
        labels_keys = []
        for j in best_param.keys():
            labels_keys.append(j)
        
        for j in best_param.values():
            labels_values.append(j)
        
        tab_param =[]
        tab_param.append(labels_keys)
        tab_param.append(labels_values)
        
        fig_table = ff.create_table(tab_param, height_constant=60)
            
        end = time()
        time_execution = round(end-start,2)
        
        # Scatter plot
        fig = go.Figure()
        absi = list(range(len(y_test)))
        
        fig.add_trace(go.Scatter(x=absi, y=y_pred[np.argsort(y_test)], mode='markers',name='predictions')) #Prédictions
        fig.add_trace(go.Scatter(x=absi, y=np.sort(y_test), mode='lines',name='reality')) #Données réelles

        return best_param, best_score, r2, mse, time_execution, fig, fig_table
    
    # Decision tree regression algorithm
    def dec_tree_reg(target,predictor):
        X = pd.DataFrame(predictor)
        
        # Normalization of quantitative explanatory variables
        transformer = Normalizer().fit(X)
        transformer
        transformer.transform(X)
        
        # Target variable
        y = pd.DataFrame(target)
        
        # Split train-test
        X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), train_size=0.7, random_state=10)
        
        scorer = make_scorer(r2_score) # Type of score
        treeReg = DecisionTreeRegressor() # Model instanciation
        params = {'splitter':['best', 'random'],
                  'max_features':['auto', 'sqrt','log2']} # Paramaters to test
        clf = GridSearchCV(treeReg, param_grid=params, cv=5,scoring=scorer, n_jobs=5)
        clf.fit(X_train, y_train) # We apply on the training data
                        
        best_param = clf.best_params_ # Better setting
        best_score = round(clf.best_score_,2) # Best score
        
        # Instantiation of the model with identified optimal parameters
        treeReg = DecisionTreeRegressor(max_features=best_param["max_features"], splitter=best_param["splitter"])

        start = time()
        
        # Predictions
        y_pred = cross_val_predict(treeReg, X_test, y_test, cv=5)
        r2 = round(r2_score(y_test, y_pred),2)
        mse = round(mean_squared_error(y_test, y_pred),2)
        
        # Best parameter
        labels_values = []
        labels_keys = []
        for j in best_param.keys():
            labels_keys.append(j)
        
        for j in best_param.values():
            labels_values.append(j)
        
        tab_param =[]
        tab_param.append(labels_keys)
        tab_param.append(labels_values)
        
        fig_table = ff.create_table(tab_param, height_constant=60)
               
        end = time()
        time_execution = round(end-start,2)
        
        # Scatter plot
        fig = go.Figure()
        absi = list(range(len(y_test)))
        
        fig.add_trace(go.Scatter(x=absi, y=y_pred[np.argsort(y_test)], mode='markers',name='predictions')) #Prédictions
        fig.add_trace(go.Scatter(x=absi, y=np.sort(y_test), mode='lines',name='reality')) #Données réelles

        return best_param, best_score, r2, mse, time_execution, fig, fig_table
    
    



