# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 21:37:41 2020

@author: ASB
"""
#regresion lineal multiple

#Cómo imporar librería

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Creación del Dataframe

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,4].values

#codificar datos categóricos
from sklearn import preprocessing
le_X = preprocessing.LabelEncoder()
X[:,3] = le_X.fit_transform(X[:,3])

#Para utilizar one hot encoder y crear variables dummy
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)


#No necesitamos transformar la y en este caso
"""le_y = preprocessing.LabelEncoder()
y = le_y.fit_transform(y)"""

#evitar la trampa de las variables dummy
X = X[:,1:]

#dividir el dataframe en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#ajustar el modelo de regresión lineal multiple con el conjunto de entrenamiento (modelo entrenado)

from sklearn.linear_model import LinearRegression

regre = LinearRegression()
regre.fit(X_train, y_train)

#Predicción de los resultados en el conjunto de testing

y_pred = regre.predict(X_test)

#Construir el modelo óptimo de Regresión Lineal Múltiple utilizando eliminación hacia atrás

import statsmodels.api as sm

#añade una columna de unos
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)


sl = 0.05

"""X_opt = X[:, [0,1,2,3,4,5]].tolist()
regre_ols = sm.OLS(y, X_opt.tolist()).fit()

regre_ols.summary()


X_opt = X[:, [0,1,3,4,5]].tolist()
regre_ols = sm.OLS(y, X.tolist()).fit()

regre_ols.summary()"""

X_opt= X[:,[0,1,2,3,4,5]]

regression_OLS=sm.OLS(endog = y, exog= X_opt).fit()

regression_OLS.summary()

X_opt= X[:,[0,1,3,4,5]]

regression_OLS=sm.OLS(endog = y, exog= X_opt).fit()

regression_OLS.summary()


X_opt= X[:,[0,3,4,5]]

regression_OLS=sm.OLS(endog = y, exog= X_opt).fit()

regression_OLS.summary()

X_opt= X[:,[0,3,5]]

regression_OLS=sm.OLS(endog = y, exog= X_opt).fit()

regression_OLS.summary()

X_opt= X[:,[0,3]]

regression_OLS=sm.OLS(endog = y, exog= X_opt).fit()

regression_OLS.summary()