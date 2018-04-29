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
		
		erro_percentual = np.inf
		acc0 = 0

		while(erro_percentual>self._tol):

			for i in range(self._input_size):
				Z = np.dot(self._weights,X[i]) + self._b 
				activation_value = self.activation_function(Z)

				if( not(activation_value) and y[i]) or (activation_value and not(y[i]) ):
					#ponto não corretamente classificado
					valor_desejado = 1 if y[i] else -1
					erro = np.sign(Z) - valor_desejado 
					self._weights -= self._lr*erro*np.transpose(X[i])
					self._b -= self._lr*erro

			arr_valor_previsto = self.activation_function(np.dot(self._weights,np.transpose(X)) + self._b)
			acc1 = accuracy_score(y_true = y, y_pred =arr_valor_previsto, normalize = False)/self._input_size
			
			erro_percentual = np.abs(acc0 - acc1)/acc0
			acc0 = acc1

		#print("weights: ",self._weights)
		#print("acc1: ", acc1)
		self.coef_ = self._weights



	def predict(self,X, output_size):
		self._output_size = output_size
		Z = np.dot(self._weights,X.transpose()) + self._b
		return self.activation_function(Z)


	def predict_dist_hiperplano(self,X, output_size):
		self._output_size = output_size
		return self.dist_hiperplano(X)

	def dist_hiperplano(self,ponto):
		#sem abs, para manter sinal do produto escalar
		dividendo = np.dot(self._weights,ponto) + self._b
		divisor   = np.sqrt(np.sum(np.power(self._weights,2)))
		return dividendo / divisor

	def activation_function(self,Z):
		return Z>0
		