#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:33:32 2021

@author: gwladyskerhoas
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from time import time
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score,roc_curve, auc
import plotly.express as px
import os
import joblib
import pickle
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
import pydot
import base64
import xml

class classification:

    def reg_log(target,predictor):
        
        X = pd.DataFrame(predictor)
        y = pd.DataFrame(target)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), train_size=0.7, random_state=10)
        
        scorer = make_scorer(accuracy_score) #Type de score
        regressionLogistique = LogisticRegression(solver='liblinear', max_iter=5000) #Intaciation du modèle
        params = {'C':[1,2,3], 'penalty':['l1', 'l2']} #Paramètres à tester
        clf = GridSearchCV(regressionLogistique, param_grid=params, cv=5,scoring=scorer, n_jobs=5)
        clf.fit(X_train, y_train) #On applique sur les données d'apprentissage
    
        best_param = clf.best_params_ #Meilleur paramètrage
        best_score = round(clf.best_score_,2) #Meilleur score
        
        regressionLogistique = LogisticRegression(solver='liblinear', C=best_param["C"], penalty=best_param["penalty"]) #Intaciation du modèle avec paramètres optimaux identifiés
    
        start = time()
    
        y_pred = cross_val_predict(regressionLogistique, X_test, y_test, cv=5)
        acc = round(accuracy_score(y_test, y_pred),2)
        
        #metrics_class = metrics.classification_report(y_test, y_pred)
    
        end = time()
        time_execution = round(end-start,2)
        
        # Confusion matrix
        labels = []
        distinct_value = np.unique(y)
        for j in range(len(distinct_value)):
            labels.append(distinct_value[j])
        confusion_mat = confusion_matrix(y_test,y_pred,labels=labels)
        matrix_confusion = ff.create_annotated_heatmap(z=confusion_mat, x=labels, y=labels, colorscale='Purples') 
        
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
        
        regressionLogistique.fit(X_train, y_train)
        
        # One hot encode the labels in order to plot them
        y_onehot = pd.get_dummies(y)
        
        y_scores = regressionLogistique.predict_proba(X)
        
        # Create an empty figure, and iteratively add new lines
        # every time we compute a new class
        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        
        for i in range(y_scores.shape[1]):
            y_true = y_onehot.iloc[:, i]
            y_score = y_scores[:, i]
        
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)
        
            name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
        
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700, height=500
        )
                
        return best_param, best_score, acc, matrix_confusion, time_execution, fig, fig_table
    
    
    def adl(target,predictor):
        X = pd.DataFrame(predictor)
        y = pd.DataFrame(target)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), train_size=0.7, random_state=10)
        
        scorer = make_scorer(accuracy_score) #Type de score
        adl = LinearDiscriminantAnalysis() #Intaciation du modèle
        params = {'solver':['svd','lsqr','eigen']} #Paramètres à tester
        clf = GridSearchCV(adl, param_grid=params, cv=5,scoring=scorer, n_jobs=5)
        clf.fit(X_train, y_train) #On applique sur les données d'apprentissage
    
        best_param = clf.best_params_ #Meilleur paramètrage
        best_score = round(clf.best_score_,2) #Meilleur score
        
        adl = LinearDiscriminantAnalysis(solver=best_param["solver"]) #Intaciation du modèle avec paramètres optimaux identifiés
    
        start = time()
    
        y_pred = cross_val_predict(adl, X_test, y_test, cv=5)
        acc = round(accuracy_score(y_test, y_pred),2)
        
        #metrics_class = metrics.classification_report(y_test, y_pred)
    
        end = time()
        time_execution = round(end-start,2)
        
        confusion_mat = confusion_matrix(y_test,y_pred)
        
        # Confusion matrix
        labels = []
        distinct_value = np.unique(y)
        for j in range(len(distinct_value)):
            labels.append(distinct_value[j])
        confusion_mat = confusion_matrix(y_test,y_pred,labels=labels)
        matrix_confusion = ff.create_annotated_heatmap(z=confusion_mat, x=labels, y=labels, colorscale='Purples') 
        
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
        
        adl.fit(X_train, y_train)

        # One hot encode the labels in order to plot them
        y_onehot = pd.get_dummies(y)
        y_scores = adl.predict_proba(X)
        
        # Create an empty figure, and iteratively add new lines
        # every time we compute a new class
        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        
        for i in range(y_scores.shape[1]):
            y_true = y_onehot.iloc[:, i]
            y_score = y_scores[:, i]
        
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)
        
            name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
        
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700, height=500
        )

        return best_param, best_score, acc, matrix_confusion, time_execution, fig, fig_table
        
    
    def arbre_de_decision(target,predictor):
        X = pd.DataFrame(predictor)
        y = pd.DataFrame(target)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), train_size=0.7, random_state=10)
        
        arbreFirst = DecisionTreeClassifier(min_samples_split=30,min_samples_leaf=10)
        arbreFirst.fit(X_train,y_train)
        
        #plot_tree(arbreFirst,feature_names = list(df.columns[:-1]),filled=True)
        #plt.show()
        
        scorer = make_scorer(accuracy_score) #Type de score

        params = {'criterion':['gini','entropy'],
                  'splitter':['best', 'random'],
                  'max_depth':[5, 8, 11, 13],
                  'min_samples_leaf':[3, 4, 5],
                  'min_samples_split':[8, 10, 12],
                 } #Paramètres à tester
        
        clf = GridSearchCV(arbreFirst, param_grid=params, cv=5,scoring=scorer,n_jobs=5)
        clf.fit(X_train, y_train)
        
        best_param = clf.best_params_ #Meilleur paramètrage
        best_score = round(clf.best_score_,2)
        
        dectreeclassif = DecisionTreeClassifier(criterion=best_param["criterion"], max_depth=best_param["max_depth"], min_samples_leaf=best_param["min_samples_leaf"], min_samples_split=best_param["min_samples_split"], splitter=best_param["splitter"]) #Instanciation du modèle avec paramètres optimaux identifiés

        start = time()
        
        y_pred = cross_val_predict(dectreeclassif, X_test, y_test, cv=5)
        acc = round(accuracy_score(y_test, y_pred),2)
        
        end = time()
        time_execution = round(end-start,2)
        
        confusion_mat = confusion_matrix(y_test,y_pred)
        
        # Confusion matrix
        labels = []
        distinct_value = np.unique(y)
        for j in range(len(distinct_value)):
            labels.append(distinct_value[j])
        confusion_mat = confusion_matrix(y_test,y_pred,labels=labels)
        matrix_confusion = ff.create_annotated_heatmap(z=confusion_mat, x=labels, y=labels, colorscale='Purples') 
        
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
        
        dectreeclassif.fit(X_train, y_train)

        # One hot encode the labels in order to plot them
        y_onehot = pd.get_dummies(y)
        y_scores = dectreeclassif.predict_proba(X)
        
        # Create an empty figure, and iteratively add new lines
        # every time we compute a new class
        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        
        for i in range(y_scores.shape[1]):
            y_true = y_onehot.iloc[:, i]
            y_score = y_scores[:, i]
        
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)
        
            name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
        
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700, height=500
        )
        
        #arbre
        dectreeclassif.fit(X_train, y_train)
        joblib.dump(dectreeclassif, open('model-random-split.joblib', 'wb'))
        pickle.dump(X_test.columns.values, open('feature_names.pickle', 'wb'))
        feature_names = pickle.load(open('feature_names.pickle', 'rb'))
        def svg_to_fig(svg_bytes, title=None, plot_bgcolor='white', x_lock=False, y_lock=False):
            svg_enc = base64.b64encode(svg_bytes)
            svg = f'data:image/svg+xml;base64, {svg_enc.decode()}'
            
            # Get the width and height
            xml_tree = xml.etree.ElementTree.fromstring(svg_bytes.decode())
            img_width = int(xml_tree.attrib['width'].strip('pt'))
            img_height = int(xml_tree.attrib['height'].strip('pt'))
        
            fig1 = go.Figure()
            # Add invisible scatter trace.
            # This trace is added to help the autoresize logic work.
            fig1.add_trace(
                go.Scatter(
                    x=[0, img_width],
                    y=[img_height, 0],
                    mode="markers",
                    marker_opacity=0,
                    hoverinfo="none",
                )
            )
            fig1.add_layout_image(
                dict(
                    source=svg,
                    x=0,
                    y=0,
                    xref="x",
                    yref="y",
                    sizex=img_width,
                    sizey=img_height,
                    opacity=1,
                    layer="below",
                )
            )
        
            # Adapt axes to the right width and height, lock aspect ratio
            fig1.update_xaxes(
                showgrid=False, 
                visible=False,
                range=[0, img_width]
            )
            fig1.update_yaxes(
                showgrid=False,
                visible=False,
                range=[img_height, 0],
            )
            
            if x_lock is True:
                fig1.update_xaxes(constrain='domain')
            if y_lock is True:
                fig1.update_yaxes(
                    scaleanchor="x",
                    scaleratio=1
                )
            
            fig1.update_layout(plot_bgcolor=plot_bgcolor)
        
            if title:
                fig1.update_layout(title=title)
        
            return fig1
        
        path = 'model-random-split.joblib'
        model = joblib.load(open(path, 'rb'))
        dot_data = export_graphviz(
            model, 
            out_file=None, 
            filled=True, 
            rounded=True, 
            feature_names=feature_names,
            class_names=model.classes_,
            proportion=True,
            rotate=True,
            precision=2
        )

        pydot_graph = pydot.graph_from_dot_data(dot_data)[0]
        svg_bytes = pydot_graph.create_svg()
        
        fig1 = svg_to_fig(svg_bytes, x_lock=True, y_lock=True)
        
        return best_param, best_score, acc, matrix_confusion, time_execution, fig, fig_table, fig1
    
    
# import pandas as pd
# df = pd.read_csv("iris_data.csv", ",", encoding="Latin-1")
# target = df["species"]
# predictor =  df["sepal_length"]

# # Séparation variable cible et variables prédictives
# print(classification.reg_log(target,predictor))
# #print(classification.adl(target,predictor))
# #print(classification.arbre_de_decision(target,predictor))







