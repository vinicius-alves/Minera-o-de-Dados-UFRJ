# -*- coding: utf-8 -*-

import numpy as np
from .neuronio_perceptron import *

class Perceptron:
	
	def __init__(self, learning_rate = 0.1, tol = 0.01, random_state = 0):
		self._weights = None
		self._input_size = 0
		self._output_size = 0
		self._lr = learning_rate
		self._tol = tol
		self._random_state = random_state
		self._neuronios = None
		self._classes = None
		self.coef_ = None 


	def fit(self, X, y):
		self._classes = np.unique(y)
		self._input_size = len(X)
		self._neuronios = []

		neuronio = None
		y_mapeado = None
		for i in range(1):#len(self._classes)):
			neuronio = NeuronioPerceptron(learning_rate = self._lr, 
				tol = self._tol, random_state = self._random_state)
			y_mapeado = y == self._classes[i]
			neuronio.fit(X,y_mapeado, self._input_size)
			self._neuronios.append(neuronio)




	def predict(self,X):
		self._output_size = len(X)

		for i in range(1):#len(self._classes)):
			neuronio = self._neuronios[i]
			neuronio.predict(X,self._output_size)


		#prevê uma saída
		return []

	