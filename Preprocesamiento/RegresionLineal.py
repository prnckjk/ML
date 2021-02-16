# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 19:07:47 2020

@author: ASB
"""
#Regresión Lineal Simple

#Cómo imporar librería

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Creación del Dataframe

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,1].values

#dividir el dataframe en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

"""#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)  """ 

# Creación del modelo de Regresión Lineal

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

#esta orden entrena el modelo de regresión

lr.fit(X_train, y_train)

#- ESTA ES LA PREDDICIÓN - Predecir el conjunto de test (El resultado son las Y que nos da el modelo según las X de test que le hemos pasado. Esas Y hay que compararlas con las Y de test que son las variables reales y que nos tendría que dar)

y_pred = lr.predict(X_test)

#Visualizar los resultados de entrenamiento

plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, lr.predict(X_train), color = "green")
plt.title("Sueldo Vs Años de Experiencia(Conjunto de Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo en $")
plt.show()

#Visualizar los resultados de test

plt.scatter(X_test, y_test, color = "blue")
plt.plot(X_train, lr.predict(X_train), color = "red")
plt.title("Sueldo Vs Años de Experiencia(Conjunto de Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo en $")
plt.show()

