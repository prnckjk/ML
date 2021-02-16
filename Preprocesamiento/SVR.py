# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 20:38:34 2020

@author: ASB
"""
#Cómo imporar librería

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Creación del Dataframe

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values

y = dataset.iloc[:,2].values


# no se hace la separación entre entranamiento y test porque la muestra es muy pequeña y además, no se pueden separar datos porque no tendríamos la información completa.
"""#dividir el dataframe en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))


#Ajustar la regresión con el dataset


#Crear aquí el modelo de Regresión
from sklearn.svm import SVR

regression = SVR(kernel = "rbf")

regression.fit(X, y)

#Predicción de nuestros modelos con SVR

y_pred = regression.predict(sc_X.transform(np.array([[6.5]])))
y_pred = sc_y.inverse_transform(y_pred)

#Visualización de los resultados del modelo SVR

#crea una secuencia de valores de valores intermedios
"""X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)"""



#Visualización de los resultados del modelo SVR con valores intermedios
plt.scatter(X, y, color = "red")
plt.plot(X, regression.predict(X), color = "green")
plt.title("Modelo de regresión")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo en $")
plt.show()