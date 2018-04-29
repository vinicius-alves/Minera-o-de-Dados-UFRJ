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

'''
DEBUG

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

'''
NOT DEBUG
'''
acc_list =[]

for i in range(2000):

	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33)

	#criando o Classificador
	clf = Perceptron()

	#treinando
	clf.fit(X_train,y_train)

	#utilizando o modelo
	y_hat = clf.predict(X_test)

	#índice de acertos
	acc = accuracy_score(y_test,y_hat)
	acc_list.append(acc)

#print(acc_list)

print("\n","ACC MEAN: ","%.3f" % np.mean(acc_list))
print("ACC STD:  ","%.3f" % np.std(acc_list),"\n")

plt.hist(acc_list,bins =30)
plt.show()

