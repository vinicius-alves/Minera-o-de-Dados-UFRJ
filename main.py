
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import Perceptron
from my_lib import *

#carregando os dados
iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

random_state = 0

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33, random_state = random_state)

#criando o Classificador
clf = Perceptron(random_state = random_state)

#treinando
clf.fit(X_train,y_train)

#utilizando o modelo
y_hat = clf.predict(X_test)

acc = accuracy_score(y_test,y_hat)

print("\nACC: ",acc,"\n")


'''

W = [[  1.5   2.5  -4.9  -2.3][  2.8 -19.9   8.3  -5.7][-17.  -15.1  25.6  19.1]] 

'''

