# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 20:21:04 2020

@author: ASB
"""
#XGBoost

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Creación del Dataframe

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values

y = dataset.iloc[:,13].values

#codificar datos categóricos
from sklearn import preprocessing
le_X_1 = preprocessing.LabelEncoder()
X[:,1] = le_X_1.fit_transform(X[:,1])
le_X_2 = preprocessing.LabelEncoder()
X[:,2] = le_X_2.fit_transform(X[:,2])

#Para utilizar one hot encoder y crear variables dummy
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

#Elimino la primera Columna que corresponde a Francia dado que si no se produciría multicolnealidad
X = X[:, 1:]

#No necesito hacerlo en la columna de género dado que ya está en 0/1 (hombre  o mujer)
"""ct_2 = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [2])], remainder='passthrough')
X = np.array(ct_2.fit_transform(X), dtype=np.float)"""


# no se hace la separación entre entranamiento y test porque la muestra es muy pequeña y además, no se pueden separar datos porque no tendríamos la información completa.
#dividir el dataframe en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Ajustar el modelo XGBoost al conjunto de entrenamiento

from xgboost import XGBClassifier

classifier = XGBClassifier()
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