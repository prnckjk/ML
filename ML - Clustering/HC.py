# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 23:42:27 2020

@author: ASB
"""
# Clustering Jerárquico

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Mall_Customers.csv")

X = dataset.iloc[:, [3,4]].values

# Utilizar el dendrograma para encontrar el número óptimo de clusters

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method = "ward"))

plt.title("Dendrograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclídea")
plt.show()

# Ajustar el Clustering Jerárquico a nuestro conjunto de datos

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")
y_hc = hc.fit_predict(X)

# Visualización de los Clusters

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = "red", label = ("Cautos"))
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = "blue", label = ("Estandard"))
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = "green", label = ("Objetivo"))
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = "cyan", label = ("Descuidados"))
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = "magenta", label = ("Conservadores"))

plt.title("Cluster de Clientes")
plt.xlabel("Ingresos anuales en miles de $")
plt.ylabel("Puntuación de gastos (1-100)")
plt.legend()
plt.show()