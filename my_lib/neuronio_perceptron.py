# -*- coding: utf-8 -*-

import numpy as np

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


	def fit(self, X, y,input_size):
		self._input_size = input_size
		self._len_features = len(X[0])

		np.random.seed(self._random_state)
		self._weights = np.random.random_sample((self._len_features,))
		self._b = np.random.random_sample()
		
		for i in range(self._input_size):
			y_previsto = np.dot(self._weights,X[i]) + self._b 

			if( (y_previsto<=0 and y[i]) or (y_previsto>0 and not(y[i])) ):
				#ponto não corretamente classificado
				valor_desejado = 1 if y[i] else -1
				erro = np.abs( np.sign(y_previsto) - valor_desejado )
				#variacao_w = self._lr*erro*x[i]
				self._weights += self._lr*erro#*X
				self._b += self._lr*erro

		print (self._weights)




	def predict(self,X, output_size):
		self._output_size = output_size
		#prevê uma saída
		self.predict_dist_hiperplano(X, self._output_size)
		return []

	def predict_dist_hiperplano(self,X, output_size):
		self._output_size = output_size

		for i in range(self._output_size):
			print("")


		#prevê uma saída
		return []

	def activation_function(self,Z):
		#mágica
		print ("")
		