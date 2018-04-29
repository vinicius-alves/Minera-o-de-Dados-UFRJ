
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from my_lib import *

#carregando os dados
iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33, random_state =0)

#criando o Classificador
clf = Perceptron()

#treinando
clf.fit(X_train,y_train)

#utilizando o modelo
y_hat = clf.predict(X_test)

acc = accuracy_score(y_test,y_hat)

print("ACC: ",acc)


'''

matrix_X_linhas = [[1,2,3],
				   [1,2,3]]

matrix_Y_linhas = [2,3,0]

for linha in matrix_X_linhas:
	linha.append(1) # para w0

matrix_X = np.matrix(matrix_X_linhas)

matrix_Y = np.matrix(matrix_Y_linhas).T

num_pontos = matrix_X.shape[0]
num_features = matrix_X.shape[1]

'''

