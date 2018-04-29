# -*- coding: utf-8 -*-
# Vinícius Almeida Alves

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from my_lib import Perceptron

#carregando os dados
iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

acc_list =[]

for i in range(2000):

	#divindo os dados
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33)

	#normalizando os dados
	scaler = MinMaxScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test  = scaler.transform(X_test)

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
print("\n")
print("ACC MEAN: ","%.3f" % np.mean(acc_list))
print("ACC STD:  ","%.3f" % np.std(acc_list))
print("\n")

plt.title(u"Histograma - ACC_Mean: " + str("%.2f" % np.mean(acc_list))+ " ACC_STD: " + str("%.2f" % np.std(acc_list)))
plt.ylabel("Quantity")
plt.xlabel("ACC")

plt.hist(acc_list,bins =30)
plt.show()