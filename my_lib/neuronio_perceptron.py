# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import accuracy_score

class NeuronioPerceptron:
	
	def __init__(self, learning_rate = 0.1, tol = 0.01, random_state = 0):
		self._weights = None
		self._b = None
		self._input_size = 0
		self._output_size = 0
		self._lr = learning_rate
		self._tol = tol
		self._random_state = random_state
		self._len_features = None
		self.coef_ = None


	def fit(self, X, y,input_size):
		self._input_size = input_size
		self._len_features = len(X[0])

		np.random.seed(self._random_state)
		self._weights = np.random.random_sample((self._len_features,))
		self._b = np.random.random_sample()
		
		acc = -np.inf

		while(acc<0.5):

			for i in range(self._input_size):
				Z = np.dot(self._weights,X[i]) + self._b 
				activation_value = self.activation_function(Z)

				if( not(activation_value) and y[i]) or (activation_value and not(y[i]) ):
					#ponto não corretamente classificado
					valor_desejado = 1 if y[i] else -1
					erro = np.abs( np.sign(Z) - valor_desejado )
					#variacao_w = self._lr*erro*x[i]
					print(np.transpose(X[i]))
					self._weights += self._lr*erro*np.transpose(X[i])
					self._b += self._lr*erro
					#print(self._weights)

			valor_previsto = self.activation_function(np.dot(self._weights,np.transpose(X)) + self._b) 
			acc = accuracy_score(y,valor_previsto)
			print(acc)

		self.coef_ = self._weights


	def predict(self,X, output_size):
		self._output_size = output_size
		Z = np.dot(self._weights,X.transpose()) + self._b
		return self.activation_function(Z)


	def predict_dist_hiperplano(self,X, output_size):
		self._output_size = output_size
		return self.dist_hiperplano(X)

	def dist_hiperplano(self,ponto):
		dividendo = np.abs(np.dot(self._weights,ponto) + self._b)
		divisor   = np.sqrt(np.sum(np.power(self._weights,2)))
		return dividendo / divisor

	def activation_function(self,Z):
		return Z>0
		