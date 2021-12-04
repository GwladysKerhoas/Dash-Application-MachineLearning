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


class regression:

    def linear_reg(target,predictor):
        
        X = pd.DataFrame(predictor)
        #normalisation des variables explicatives quantitatives
        transformer = Normalizer().fit(X)
        transformer
        transformer.transform(X)
        #récupération variable cible
        y = pd.DataFrame(target)
        
        #split train-test
        X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), train_size=0.7, random_state=10)
        
        scorer = make_scorer(r2_score) #Type de score
        regressionLineaire = LinearRegression() #Instanciation du modèle
        params = {'fit_intercept':[True, False], 'normalize':[True, False]} #Paramètres à tester
        clf = GridSearchCV(regressionLineaire, param_grid=params, cv=5,scoring=scorer, n_jobs=5)
        clf.fit(X_train, y_train) #On applique sur les données d'apprentissage
        
        best_param = clf.best_params_ #Meilleur paramètrage
        best_score = round(clf.best_score_,2) #Meilleur score
        
        #Instanciation du modèle avec paramètres optimaux identifiés
        regressionLineaire = LinearRegression(fit_intercept=best_param["fit_intercept"], normalize=best_param["normalize"]) 

        start = time()
        
        #prédictions
        y_pred = cross_val_predict(regressionLineaire, X_test, y_test, cv=5)
        r2 = round(r2_score(y_test, y_pred),2)
        mse = round(mean_squared_error(y_test, y_pred),2)
        
        #metrics_class = metrics.classification_report(y_test, y_pred)
        
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
        
        #scatter plot
        fig = go.Figure()
        absi = list(range(len(y_test)))
        
        fig.add_trace(go.Scatter(x=absi, y=y_pred[np.argsort(y_test)], mode='markers',name='predictions')) #Prédictions
        fig.add_trace(go.Scatter(x=absi, y=np.sort(y_test), mode='lines',name='reality')) #Données réelles

        return best_param, best_score, r2, mse, time_execution, fig, fig_table
    
    
    def svr(target,predictor):
        X = pd.DataFrame(predictor)
        #normalisation des variables explicatives quantitatives
        transformer = Normalizer().fit(X)
        transformer
        transformer.transform(X)
        #récupération variable cible
        y = pd.DataFrame(target)
        
        #split train-test
        X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), train_size=0.7, random_state=10)
        
        scorer = make_scorer(r2_score) #Type de score
        modelsvr = SVR() #Instanciation du modèle
        params = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 
                  'gamma':['scale', 'auto'],
                  'C' : [1,5,10],
                  'degree' : [3,8],
                  'coef0' : [0.01,10,0.5]} #Paramètres à tester
        clf = GridSearchCV(modelsvr, param_grid=params, cv=5,scoring=scorer, n_jobs=5)
        clf.fit(X_train, y_train) #On applique sur les données d'apprentissage
                
        best_param = clf.best_params_ #Meilleur paramètrage
        best_score = round(clf.best_score_,2) #Meilleur score
        
        #Instaciation du modèle avec paramètres optimaux identifiés
        modelsvr = SVR(C=best_param["C"], coef0=best_param["coef0"], degree=best_param["degree"], gamma=best_param["gamma"], kernel=best_param["kernel"]) 
     
        start = time()
        
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
        
        #scatter plot
        fig = go.Figure()
        absi = list(range(len(y_test)))
        
        fig.add_trace(go.Scatter(x=absi, y=y_pred[np.argsort(y_test)], mode='markers',name='predictions')) #Prédictions
        fig.add_trace(go.Scatter(x=absi, y=np.sort(y_test), mode='lines',name='reality')) #Données réelles

        return best_param, best_score, r2, mse, time_execution, fig, fig_table
    
    
    def dec_tree_reg(target,predictor):
        X = pd.DataFrame(predictor)
        #normalisation des variables explicatives quantitatives
        transformer = Normalizer().fit(X)
        transformer
        transformer.transform(X)
        #récupération variable cible
        y = pd.DataFrame(target)
        
        #split train-test
        X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), train_size=0.7, random_state=10)
        
        scorer = make_scorer(r2_score) #Type de score
        treeReg = DecisionTreeRegressor() #Intaciation du modèle
        params = {'splitter':['best', 'random'],
                  'max_features':['auto', 'sqrt','log2']} #Paramètres à tester
        clf = GridSearchCV(treeReg, param_grid=params, cv=5,scoring=scorer, n_jobs=5)
        clf.fit(X_train, y_train) #On applique sur les données d'apprentissage
                        
        best_param = clf.best_params_ #Meilleur paramètrage
        best_score = round(clf.best_score_,2) #Meilleur score
        
        #Instaciation du modèle avec paramètres optimaux identifiés
        treeReg = DecisionTreeRegressor(max_features=best_param["max_features"], splitter=best_param["splitter"])

        start = time()
        
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
        
        #scatter plot
        fig = go.Figure()
        absi = list(range(len(y_test)))
        
        fig.add_trace(go.Scatter(x=absi, y=y_pred[np.argsort(y_test)], mode='markers',name='predictions')) #Prédictions
        fig.add_trace(go.Scatter(x=absi, y=np.sort(y_test), mode='lines',name='reality')) #Données réelles

        return best_param, best_score, r2, mse, time_execution, fig, fig_table
    
    



