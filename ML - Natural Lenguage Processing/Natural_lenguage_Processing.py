# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 19:04:01 2020

@author: ASB
"""
#importar las librerías

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar el dataset

dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)


#Limpieza de texto

#Librerias textos
import re
import nltk

#Descarga el diccionario de palabras que no son útiles
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#lista reseñas limpias
corpus = []

for i in range(0,1000):

    #sustituye los caracteres que no necesitamos(como números)
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    #pasamos las palabras a minúsculas
    review = review.lower()
    # Creamos una lista de palabras
    review = review.split()
    
    #Quitamos las palabras que no son relevantes y las volvemos al "infinitivo" (loved, loving = love)
    ps = PorterStemmer()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    
    #Volvemos a juntar las palabras en forma de string
    
    review = ' '.join(review)
    corpus.append(review)
    
#Crear el Bag of words

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# no se hace la separación entre entranamiento y test porque la muestra es muy pequeña y además, no se pueden separar datos porque no tendríamos la información completa.
#dividir el dataframe en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#modelo Naïve Bayes

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Predicción de los datos con el conjunto de testing

y_pred = classifier.predict(X_test)

#Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
