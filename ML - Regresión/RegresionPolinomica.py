# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 22:33:43 2020

@author: ASB
"""
#Regresion Polinomica

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

#ajustar la regresión lineal con el dataset

from sklearn.linear_model import LinearRegression

regre_lineal = LinearRegression()
regre_lineal.fit(X,y)


#ajustar la regresión polinómica con el dataset

from sklearn.preprocessing import PolynomialFeatures

regre_poli = PolynomialFeatures(degree = 4)
X_poli = regre_poli.fit_transform(X)
regre_poli_2 = LinearRegression()
regre_poli_2.fit(X_poli, y)

#Visualización de lo resultados del modelo lineal

plt.scatter(X, y, color = "red")
plt.plot(X, regre_lineal.predict(X), color = "blue")
plt.title("Modelo de regresión lineal")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo en $")
plt.show()

#Visualización de los resultados del modelo polinómico

#crea una secuencia de valores de valores intermedios
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)


plt.scatter(X, y, color = "red")
plt.plot(X, regre_poli_2.predict(regre_poli.fit_transform(X)), color = "blue")
plt.title("Modelo de regresión polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo en $")
plt.show()


#Visualización de los resultados del modelo polinómico con valores intermedios
plt.scatter(X, y, color = "red")
plt.plot(X_grid, regre_poli_2.predict(regre_poli.fit_transform(X_grid)), color = "green")
plt.title("Modelo de regresión polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo en $")
plt.show()

#Predicción de nuestros modelos

regre_lineal.predict([[6.5]])
regre_poli_2.predict(regre_poli.fit_transform([[6.5]]))