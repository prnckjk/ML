# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 21:08:05 2019

@author: ASB
"""
# Plantilla Pre-procesado

#Cómo imporar librería

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Creación del Dataframe

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,3].values

# tratamiento de los Na

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
# [fila:columna] [:, 1:3] significa que coge todas filas, y las columnas de la dos. Hay que poner hasta el tres para que coja dos filas y no una
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])



#codificar datos categóricos
from sklearn import preprocessing
le_X = preprocessing.LabelEncoder()
X[:,0] = le_X.fit_transform(X[:,0])

#Para utilizar one hot encoder y crear variables dummy
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

le_y = preprocessing.LabelEncoder()
y = le_y.fit_transform(y)

#dividir el dataframe en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)   

