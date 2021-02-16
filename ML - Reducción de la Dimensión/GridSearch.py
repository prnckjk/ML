# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 18:10:50 2020

@author: ASB
"""
# Grid Search


#Cómo imporar librería

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Creación del Dataframe

dataset = pd.read_csv('Social_Network_ads.csv')

X = dataset.iloc[:,[2,3]].values

y = dataset.iloc[:,4].values


# no se hace la separación entre entranamiento y test porque la muestra es muy pequeña y además, no se pueden separar datos porque no tendríamos la información completa.
#dividir el dataframe en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#ajustar el modelo en el conjunto de entrenamiento
#Crear el modelo de clasificación aquí
from sklearn.svm import SVC

classifier = SVC(kernel = "rbf", random_state=0)
classifier.fit(X_train, y_train)



#Predicción de los datos con el conjunto de testing

y_pred = classifier.predict(X_test)

#Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# Aplicar K-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

#Aplicar la mejora de Grid Search para optimizar el modelo y sus parámetros

from sklearn.model_selection import GridSearchCV

parameters = [{'C':[1, 10, 100, 10000],'kernel' : ['linear']},
              {'C':[1, 10, 100, 10000],'kernel' : ['rbf'], 'gamma': [0.5, 0.1, 0.001, 0.0001]}]

grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_